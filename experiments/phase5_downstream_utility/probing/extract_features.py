"""Feature extraction for Phase 5 probing — writes per-(run, task, aggregation) caches.

This is the encode-once-fit-many separation for Phase 5. Running this
script once per `(run_id, task_name, aggregation)` cell produces a
`.npz` with the encoded train+test features. `fit_probes.py` then
fits/evaluates probes against those features without ever touching the
SAE checkpoints again — so the probe training / selection / metric code
can be iterated on without a single re-encode.

Cache layout:
    results/feature_cache/
      {run_id}/
        {task_name}__{aggregation}.npz
          Z_train (n_train, d_feat) fp16
          Z_test  (n_test,  d_feat) fp16
          y_train (n_train,) int64
          y_test  (n_test,) int64

Runs on GPU. For each ckpt we load once, then iterate all tasks × both
aggregations. Idempotent — existing .npz are skipped unless --force.

Usage:
    PHASE5_REPO=/home/elysium/temp_xc TQDM_DISABLE=1 \
      .venv/bin/python experiments/phase5_downstream_utility/probing/extract_features.py \
      --aggregations last_position full_window
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase5_downstream_utility.probing.run_probing import (
    _load_task_cache,
    _load_model_for_run,
    _encode_for_probe,
)


REPO = Path(os.environ.get("PHASE5_REPO", Path(__file__).resolve().parents[3]))
PROBE_CACHE = REPO / "experiments/phase5_downstream_utility/results/probe_cache"
CKPT_DIR = REPO / "experiments/phase5_downstream_utility/results/ckpts"
FEATURE_CACHE = REPO / "experiments/phase5_downstream_utility/results/feature_cache"


def _save_features(
    out_path: Path,
    Z_train: np.ndarray, Z_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        Z_train=Z_train.astype(np.float16),
        Z_test=Z_test.astype(np.float16),
        y_train=np.asarray(y_train, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
    )


def extract_features(
    run_ids: list[str] | None = None,
    task_names: list[str] | None = None,
    aggregations: tuple[str, ...] = ("last_position", "full_window"),
    force: bool = False,
) -> None:
    device = torch.device("cuda")
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)

    task_dirs = [
        d for d in sorted(PROBE_CACHE.iterdir())
        if d.is_dir()
        and (d / "acts_anchor.npz").exists()
        and (d / "acts_mlc.npz").exists()
    ]
    if task_names:
        task_dirs = [d for d in task_dirs if d.name in task_names]
    task_cache = {d.name: _load_task_cache(d) for d in task_dirs}
    print(f"Tasks: {len(task_cache)}  aggregations: {aggregations}")

    if run_ids:
        ckpts = [(rid, CKPT_DIR / f"{rid}.pt") for rid in run_ids]
    else:
        ckpts = [(p.stem, p) for p in sorted(CKPT_DIR.glob("*.pt"))]

    for run_id, ckpt_path in ckpts:
        if not ckpt_path.exists():
            print(f"  {run_id}: ckpt missing, skip")
            continue

        # Fast-path: check whether every (task, aggregation) feature file
        # already exists. If so, skip the model load entirely.
        need_any = False
        for task_name in task_cache:
            for aggregation in aggregations:
                out = FEATURE_CACHE / run_id / f"{task_name}__{aggregation}.npz"
                if force or not out.exists():
                    need_any = True
                    break
            if need_any:
                break
        if not need_any:
            print(f"=== {run_id}: all cached, skip model load ===")
            continue

        try:
            model, arch, meta = _load_model_for_run(run_id, ckpt_path, device)
        except Exception as e:
            print(f"  {run_id}: load FAIL: {e}")
            continue

        print(f"=== {run_id} ({arch}) ===")
        for task_name, tc in task_cache.items():
            for aggregation in aggregations:
                # MLC is inherently last-position (probe cache is last-token-only)
                if arch == "mlc" and aggregation == "full_window":
                    continue
                out = FEATURE_CACHE / run_id / f"{task_name}__{aggregation}.npz"
                if out.exists() and not force:
                    continue
                try:
                    t0 = time.time()
                    Ztr = _encode_for_probe(
                        model, arch, meta, tc, "train", device,
                        aggregation=aggregation,
                    )
                    Zte = _encode_for_probe(
                        model, arch, meta, tc, "test", device,
                        aggregation=aggregation,
                    )
                    _save_features(
                        out, Ztr, Zte,
                        tc["train_labels"], tc["test_labels"],
                    )
                    print(
                        f"  {task_name}__{aggregation}: "
                        f"Z_tr={Ztr.shape} [{time.time() - t0:.1f}s] -> {out.name}"
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  {task_name}__{aggregation}: OOM — skip")
                except Exception as e:
                    print(f"  {task_name}__{aggregation} FAIL: {e}")
        del model
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument(
        "--aggregations", nargs="+",
        default=["last_position", "full_window"],
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    extract_features(
        run_ids=args.run_ids,
        task_names=args.tasks,
        aggregations=tuple(args.aggregations),
        force=args.force,
    )


if __name__ == "__main__":
    main()
