"""Backfill ROC-AUC at top-S features into existing C7 leaderboard rows.

Why:
- Dmitry asked for ROC-AUC alongside PR-AUC for the sparse-probe table
  so the new (PR-AUC ≈ 0.16-0.24) and old internal-exploration
  (ROC-AUC ≈ 0.8-0.9) numbers can be compared apples-to-apples.
- We don't want to bump ``EVAL_PROTOCOL_VERSION`` because that would
  invalidate every C7 row and force a fresh judge re-run (~$$$).
- The sparse probe is the cheap part of the eval: it only needs the
  trained checkpoint + the cached ``sentence_acts_L10.npz`` (1.1 GB,
  already on disk). No subject-model generation, no LLM judge calls.

What it does:
For each canonical (n_steps=300_000) c7 leaderboard row that has
``pr_auc_S{S}`` keys but not ``roc_auc_S{S}``, this script:
  1. loads the SAE checkpoint at ``train_key``
  2. re-encodes the cached sentence-window activations
  3. runs :func:`compute_probe_metrics_at_S` (returns both PR-AUC + ROC-AUC)
  4. sanity-checks that the recomputed PR-AUC matches the stored one
     within 1e-3 (catches arch-version drift)
  5. patches ``roc_auc_S{S}`` into both:
       - the corresponding line of ``results/leaderboard.jsonl``
       - ``results/runs/<eval_key>/metrics.json``

Idempotent: cells that already have ``roc_auc_S8`` are skipped.

Usage::

    cd /workspace/aniket/temp_xc-final/purified
    .venv/bin/python -m scripts.c7_backfill_roc_auc           # all cells
    .venv/bin/python -m scripts.c7_backfill_roc_auc --dry-run # just print
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from temp_bench.case_studies.backtracking import (
    DEFAULT_PR_AUC_S_GRID, compute_probe_metrics_at_S,
)
from temp_bench.config import (
    act_cache_dir, checkpoint_dir, compute_act_cache_key, instantiate_arch,
    load_arch, load_datasource, purified_root, run_dir,
)
from temp_bench.cache import load_checkpoint_state_dict

DATASOURCE = "llama_3_1_8b_base_l10_ward_nousmirror"

PURIFIED = purified_root()
LEADERBOARD = PURIFIED / "results" / "leaderboard.jsonl"
SENTENCE_NPZ = (
    PURIFIED / "results" / "c7_backtracking" / "stage_a" / "sentence_acts_L10.npz"
)


def _train_cfg(row: dict) -> dict:
    cfg_path = checkpoint_dir(row["train_key"]) / "config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text()).get("training_cfg", {})


def _is_canonical(row: dict) -> bool:
    if row.get("component") != "c7":
        return False
    if row.get("eval_cfg", {}).get("_extended_mags"):
        return False
    return _train_cfg(row).get("n_steps") == 300_000


def _has_roc(row: dict) -> bool:
    return any(f"roc_auc_S{S}" in row.get("metrics", {}) for S in DEFAULT_PR_AUC_S_GRID)


def _load_sentence_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(SENTENCE_NPZ, allow_pickle=True)
    X = z["X"]                                            # (n, T, d)
    is_bt = z["is_bt"].astype(int)
    keys = z["keys"]
    qids = np.array([k.split("|")[0] for k in keys], dtype=object)
    return X, is_bt, qids


def _encode(arch: torch.nn.Module, X: np.ndarray, *, batch: int = 1024) -> np.ndarray:
    device = next(arch.parameters()).device
    dtype = next(arch.parameters()).dtype
    arch_T = getattr(arch.config, "T", None) or 1
    win_T = X.shape[1]
    sa = X
    if arch_T < win_T:
        sa = X[:, -arch_T:, :]
    elif arch_T > win_T:
        pad = arch_T - win_T
        sa = np.concatenate(
            [np.repeat(X[:, :1, :], pad, axis=1), X], axis=1
        )
    chunks: list[np.ndarray] = []
    n = sa.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.from_numpy(sa[i:i + batch]).to(device, dtype=dtype)
            z = arch.encode(xb).abs().float()
            if z.dim() == 3:
                z = z.amax(dim=1)
            chunks.append(z.detach().cpu().numpy())
            del z, xb
    return np.concatenate(chunks, axis=0)


def _instantiate(arch_name: str, state_dict: dict, *, component: str = "c7"):
    """Mirror experiments/c7_backtracking/run.py:_instantiate_from_state —
    pull d_in from the cached datasource layer_specs.json (single source
    of truth) rather than guessing from the state_dict shapes."""
    spec = load_arch(arch_name, component=component)
    ds = load_datasource(DATASOURCE)
    cache_dir = act_cache_dir(compute_act_cache_key(ds))
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    d_in = int(specs["d_model"])
    arch = instantiate_arch(spec, d_in=d_in)
    arch.load_state_dict(state_dict, strict=True)
    if torch.cuda.is_available():
        arch = arch.cuda()
    arch.eval()
    return arch


def _patch_leaderboard(eval_key: str, new_metrics: dict[str, float]) -> bool:
    if not LEADERBOARD.exists():
        return False
    lines = LEADERBOARD.read_text().splitlines()
    out: list[str] = []
    patched = False
    for line in lines:
        s = line.strip()
        if not s:
            out.append(line)
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            out.append(line)
            continue
        if r.get("eval_key") == eval_key:
            r.setdefault("metrics", {}).update(new_metrics)
            out.append(json.dumps(r))
            patched = True
        else:
            out.append(line)
    if patched:
        LEADERBOARD.write_text("\n".join(out) + ("\n" if lines and lines[-1] == "" else "\n"))
    return patched


def _patch_metrics_json(eval_key: str, new_metrics: dict[str, float]) -> bool:
    p = run_dir(eval_key) / "metrics.json"
    if not p.exists():
        return False
    blob = json.loads(p.read_text())
    metrics = blob.get("metrics", blob)
    metrics.update(new_metrics)
    if metrics is not blob:
        blob["metrics"] = metrics
    p.write_text(json.dumps(blob, indent=2))
    return True


def main(*, dry_run: bool = False) -> None:
    if not LEADERBOARD.exists():
        print(f"no leaderboard at {LEADERBOARD}")
        return
    if not SENTENCE_NPZ.exists():
        raise FileNotFoundError(
            f"sentence acts npz missing at {SENTENCE_NPZ}; rebuild via "
            "extract_labeled_sentence_acts() first"
        )
    rows = [
        json.loads(line) for line in LEADERBOARD.read_text().splitlines()
        if line.strip()
    ]
    canonical = [r for r in rows if _is_canonical(r)]
    todo = [r for r in canonical if not _has_roc(r)]
    print(f"[backfill] {len(canonical)} canonical c7 rows, "
          f"{len(canonical) - len(todo)} already have ROC-AUC, "
          f"{len(todo)} to backfill")
    if not todo:
        return

    print("[backfill] loading sentence acts npz (1.1 GB) …")
    X, y, qids = _load_sentence_data()
    print(f"[backfill] sentence acts: shape={X.shape}, n_pos={int(y.sum())} ({y.mean():.3f})")

    for r in todo:
        eval_key = r["eval_key"]
        train_key = r["train_key"]
        arch_name = r["arch"]
        bs = _train_cfg(r).get("batch_size", "?")
        ckpt = checkpoint_dir(train_key) / "model.safetensors"
        if not ckpt.exists():
            print(f"[backfill] SKIP {arch_name} bs={bs}: checkpoint missing at {ckpt}")
            continue
        print(f"[backfill] {arch_name} bs={bs} (train_key={train_key[:12]}, eval_key={eval_key[:12]})")
        state = load_checkpoint_state_dict(train_key)
        arch = _instantiate(arch_name, state, component="c7")
        feats = _encode(arch, X)
        del arch, state
        torch.cuda.empty_cache()

        probe = compute_probe_metrics_at_S(feats, y, qids, S_grid=DEFAULT_PR_AUC_S_GRID)
        # Sanity check: recomputed PR-AUC should match stored within 1e-3
        for S in DEFAULT_PR_AUC_S_GRID:
            stored = r["metrics"].get(f"pr_auc_S{S}")
            recomp = probe["pr_auc"][S]
            if stored is not None and abs(stored - recomp) > 1e-3:
                print(f"  WARN S={S}: stored PR-AUC={stored:.4f} recomputed={recomp:.4f} (drift > 1e-3)")
        new_metrics = {f"roc_auc_S{S}": float(probe["roc_auc"][S])
                       for S in DEFAULT_PR_AUC_S_GRID}
        line = "  " + " ".join(f"S{S}={new_metrics[f'roc_auc_S{S}']:.3f}"
                                for S in DEFAULT_PR_AUC_S_GRID)
        print(line)

        if dry_run:
            print("  (dry-run; not patched)")
            continue
        _patch_metrics_json(eval_key, new_metrics)
        _patch_leaderboard(eval_key, new_metrics)
        print("  patched leaderboard + metrics.json")

    print("[backfill] done")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="compute ROC-AUC but don't write")
    args = ap.parse_args()
    main(dry_run=args.dry_run)


if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")
    cli()
