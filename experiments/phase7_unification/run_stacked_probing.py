"""Stacked-SAE control: probe a vanilla per-token SAE by CONCATenating
the last K position-encodings into a (K * d_sae,) feature vector.

Han's concern: TXC's leaderboard win at k_feat ∈ {5, 20} may simply
reflect having a d_sae-sized candidate pool fed by a window of T tokens,
vs a per-token SAE's d_sae pool. If we feed the per-token SAE's
last-K-positions concat (giving K*d_sae candidates), and it MATCHES or
BEATS TXC at the same k_feat, then TXC's structure is doing little
real work — the win is "more candidates, same per-feature quality".

Per-arch behaviour:
  - TopKSAE:       per-token encode → (N, S=32, d_sae). Last K positions
                   are concat into (N, K*d_sae). Probed at k_feat ∈ {5, 20}.
  - TemporalMatryoshkaBatchTopKSAE (T-SAE paper): same — also per-token
                   at inference. Run as second baseline.

Comparison: TXC at T=5 vs (TopKSAE × K=5 stacked) at the same k_feat.
Both get ~5 windows of context. Only difference is whether the structure
is one TXC feature or K vanilla SAE features concatenated.

Skipping rule: examples with fewer than K real tokens (first_real > S-K)
are dropped from both train and test before fitting the LR.

Output: `results/stacked_probing_results.jsonl` (separate from
leaderboard `probing_results.jsonl` — keep the control isolated).

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.run_stacked_probing \\
        --run_ids topk_sae__seed42 tsae_paper_k500__seed42 \\
        --K 2 5 --k_feat 5 20

Default behaviour (no args): runs both baselines × K ∈ {2, 5} × k_feat ∈ {5, 20}.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase5_downstream_utility.probing.run_probing import (
    sae_probe_metrics,
)
from experiments.phase7_unification._paths import (
    INDEX_PATH, OUT_DIR, banner,
)
from experiments.phase7_unification.run_probing_phase7 import (
    ACTIVE_PROBE_CACHE,
    _encode_per_token_z,
    _load_phase7_model,
    _load_task_cache_p7,
    flipped_auc,
)


STACKED_OUT_PATH = OUT_DIR / "stacked_probing_results.jsonl"
DEFAULT_RUN_IDS = ("topk_sae__seed42", "tsae_paper_k500__seed42")
DEFAULT_K_POSITIONS = (2, 5)            # how many last positions to concat
DEFAULT_K_FEAT = (5, 20)


def stack_last_K(z_full: np.ndarray, first_real: np.ndarray,
                 K: int, S: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Concat the last K positions of z_full per example.

    Args:
      z_full:     (N, S, d_sae) per-token SAE encodings.
      first_real: (N,) int — position of first real token per example.
                  Real tokens occupy [first_real[i], S-1].
      K:          number of trailing positions to stack.
      S:          tail length (= 32).

    Returns:
      (Z, valid)
        Z:      (N_valid, K*d_sae) — only rows where first_real <= S-K
                (i.e., at least K real tokens exist).
        valid:  (N,) bool mask of which examples were kept.
    """
    N, S_actual, d_sae = z_full.shape
    assert S_actual == S, f"z_full S={S_actual} != expected {S}"
    valid = first_real <= (S - K)
    if not valid.any():
        return np.zeros((0, K * d_sae), dtype=z_full.dtype), valid
    z_tail = z_full[:, S - K:S, :]                # (N, K, d_sae)
    Z = z_tail.reshape(N, K * d_sae)              # (N, K*d_sae)
    return Z[valid], valid


def run_one(run_id: str, ckpt_path: Path, meta: dict, task_dirs: list[Path],
            K_values: tuple[int, ...], k_feat_values: tuple[int, ...],
            out_f, device) -> int:
    """Encode each task with a per-token SAE, then for each K stack the
    last K positions and run the standard top-k-by-class-sep LR probe.

    Returns the number of (task, K, k_feat) rows written.
    """
    src_class = meta["src_class"]
    if src_class not in {"TopKSAE", "TemporalMatryoshkaBatchTopKSAE"}:
        print(f"  SKIP {run_id}: src_class={src_class} is not per-token; "
              f"stacked-SAE control only valid for per-token SAE families")
        return 0

    try:
        model, _ = _load_phase7_model(meta, ckpt_path, device)
    except Exception as e:
        print(f"  {run_id}: load FAIL {type(e).__name__}: {e}")
        return 0

    n_rows = 0
    print(f"  {run_id} src_class={src_class}")
    for task_dir in task_dirs:
        task_name = task_dir.name
        tc = _load_task_cache_p7(task_dir)
        if "anchor_train" not in tc:
            print(f"    {task_name}: no anchor cache, skip")
            continue
        ytr = tc["train_labels"]; yte = tc["test_labels"]
        try:
            z_train = _encode_per_token_z(model, tc["anchor_train"], device)  # (N, 32, d_sae)
            z_test  = _encode_per_token_z(model, tc["anchor_test"],  device)
        except Exception as e:
            print(f"    {task_name}: encode FAIL {type(e).__name__}: {e}")
            del tc; gc.collect(); continue

        first_real_tr = tc["train_first_real"]
        first_real_te = tc["test_first_real"]

        for K in K_values:
            Z_tr, valid_tr = stack_last_K(z_train, first_real_tr, K)
            Z_te, valid_te = stack_last_K(z_test,  first_real_te, K)
            n_drop_tr = int((~valid_tr).sum()); n_drop_te = int((~valid_te).sum())
            if Z_tr.shape[0] < 5 or Z_te.shape[0] < 5:
                continue
            ytr_v = ytr[valid_tr]; yte_v = yte[valid_te]
            for k_feat in k_feat_values:
                # NOTE: top-k-by-class-sep over K*d_sae = up to 92,160
                # candidates is fine — argpartition is O(F).
                auc, acc = sae_probe_metrics(Z_tr, ytr_v, Z_te, yte_v, k_feat)
                row = {
                    "run_id":      run_id,
                    "src_class":   src_class,
                    "arch_id":     meta.get("arch_id"),
                    "row":         meta.get("row"),
                    "group":       meta.get("group"),
                    "task_name":   task_name,
                    "dataset_key": tc["meta"]["dataset_key"],
                    "K_positions": int(K),
                    "k_feat":      int(k_feat),
                    "test_auc":    float(auc),
                    "test_auc_flip": float(flipped_auc(auc, task_name)),
                    "test_acc":    float(acc),
                    "n_train_eff": int(Z_tr.shape[0]),
                    "n_test_eff":  int(Z_te.shape[0]),
                    "n_drop_train": n_drop_tr,
                    "n_drop_test":  n_drop_te,
                    "d_sae":       int(z_train.shape[-1]),
                    "feature_dim": int(K * z_train.shape[-1]),
                    "stacked":     True,
                    **{k: meta.get(k) for k in (
                        "seed", "k_pos", "k_win",
                    ) if meta.get(k) is not None},
                }
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                n_rows += 1
            print(f"    {task_name:40s} K={K} kept={Z_tr.shape[0]}/{ytr.size}")
        # Free per-task tensors before next task.
        del tc, z_train, z_test
        try: del Z_tr, Z_te, valid_tr, valid_te
        except NameError: pass
        torch.cuda.empty_cache(); gc.collect()

    del model
    torch.cuda.empty_cache(); gc.collect()
    return n_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_ids", nargs="+", default=list(DEFAULT_RUN_IDS))
    p.add_argument("--task_names", nargs="+", default=None,
                   help="Restrict to these task names. Defaults to all 36.")
    p.add_argument("--K", nargs="+", type=int, default=list(DEFAULT_K_POSITIONS),
                   help="K-position values to stack. Default: 2 5")
    p.add_argument("--k_feat", nargs="+", type=int, default=list(DEFAULT_K_FEAT),
                   help="k_feat values for top-k-by-class-sep. Default: 5 20")
    args = p.parse_args()
    banner(__file__)

    if not INDEX_PATH.exists():
        raise SystemExit(f"no training index at {INDEX_PATH}")

    by_run = {}
    with INDEX_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            by_run[row["run_id"]] = row

    task_dirs = [d for d in sorted(ACTIVE_PROBE_CACHE.iterdir()) if d.is_dir()]
    if args.task_names:
        task_dirs = [d for d in task_dirs if d.name in args.task_names]
    task_dirs = [
        d for d in task_dirs
        if (d / "acts_anchor.npz").exists() and (d / "meta.json").exists()
    ]
    print(f"[stacked-probe] {len(task_dirs)} tasks, K={args.K}, k_feat={args.k_feat}")

    STACKED_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    total_rows = 0
    with STACKED_OUT_PATH.open("a") as out_f:
        for run_id in args.run_ids:
            if run_id not in by_run:
                print(f"  {run_id}: not in training_index, skip")
                continue
            meta = by_run[run_id]
            ckpt_path = Path(meta["ckpt"])
            if not ckpt_path.exists():
                print(f"  {run_id}: ckpt missing {ckpt_path}")
                continue
            n = run_one(run_id, ckpt_path, meta, task_dirs,
                        tuple(args.K), tuple(args.k_feat), out_f, device)
            total_rows += n
            print(f"  {run_id}: wrote {n} rows")
    print(f"Done. wrote {total_rows} rows to {STACKED_OUT_PATH}")


if __name__ == "__main__":
    main()
