"""Raw-activation concat probing — the most extreme "more features = better"
control for the stacked-SAE concat result.

Same protocol as run_stacked_probing.py but with NO SAE — just take the
raw `(d_in=2304)` Gemma activations at the last K positions, concat into
`(N, K * 2304)`, and run top-k-by-class-sep + L1 LR.

If raw-concat ALSO loses to TXC, then "more candidate features" is
firmly ruled out as the source of TXC's leaderboard win — there's
nothing left except feature quality and structural inductive bias.

Output: `results/raw_concat_probing_results.jsonl`.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.run_raw_concat_probing \\
        --K 2 5 --k_feat 5 20
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase5_downstream_utility.probing.run_probing import (
    sae_probe_metrics,
)
from experiments.phase7_unification._paths import OUT_DIR, banner
from experiments.phase7_unification.run_probing_phase7 import (
    ACTIVE_PROBE_CACHE,
    _load_task_cache_p7,
    flipped_auc,
)


RAW_OUT_PATH = OUT_DIR / "raw_concat_probing_results.jsonl"


def stack_last_K_raw(anchor: np.ndarray, first_real: np.ndarray,
                     K: int, S: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Concat last K positions of raw activations.

    Args:
      anchor:     (N, S=32, d_in=2304) raw Gemma residual-stream acts.
      first_real: (N,) int — pos of first real token.
      K:          last positions to concat.
      S:          tail length.

    Returns:
      (Z, valid)
        Z: (N_valid, K * d_in)
        valid: (N,) bool — kept rows.
    """
    N, S_actual, d_in = anchor.shape
    assert S_actual == S
    valid = first_real <= (S - K)
    if not valid.any():
        return np.zeros((0, K * d_in), dtype=anchor.dtype), valid
    raw_tail = anchor[:, S - K:S, :]              # (N, K, d_in)
    Z = raw_tail.reshape(N, K * d_in)             # (N, K * d_in)
    return Z[valid], valid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task_names", nargs="+", default=None)
    p.add_argument("--K", nargs="+", type=int, default=[2, 5])
    p.add_argument("--k_feat", nargs="+", type=int, default=[5, 20])
    args = p.parse_args()
    banner(__file__)

    task_dirs = [d for d in sorted(ACTIVE_PROBE_CACHE.iterdir()) if d.is_dir()]
    if args.task_names:
        task_dirs = [d for d in task_dirs if d.name in args.task_names]
    task_dirs = [
        d for d in task_dirs
        if (d / "acts_anchor.npz").exists() and (d / "meta.json").exists()
    ]
    print(f"[raw-concat] {len(task_dirs)} tasks, K={args.K}, k_feat={args.k_feat}")

    RAW_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with RAW_OUT_PATH.open("a") as out_f:
        for task_dir in task_dirs:
            task_name = task_dir.name
            tc = _load_task_cache_p7(task_dir)
            if "anchor_train" not in tc:
                continue
            ytr = tc["train_labels"]; yte = tc["test_labels"]
            for K in args.K:
                Z_tr, valid_tr = stack_last_K_raw(
                    tc["anchor_train"], tc["train_first_real"], K)
                Z_te, valid_te = stack_last_K_raw(
                    tc["anchor_test"], tc["test_first_real"], K)
                n_drop_tr = int((~valid_tr).sum())
                n_drop_te = int((~valid_te).sum())
                if Z_tr.shape[0] < 5 or Z_te.shape[0] < 5:
                    continue
                ytr_v = ytr[valid_tr]; yte_v = yte[valid_te]
                for k_feat in args.k_feat:
                    auc, acc = sae_probe_metrics(Z_tr, ytr_v, Z_te, yte_v, k_feat)
                    row = {
                        "arch_id":     "raw_concat",
                        "src_class":   "RAW_GEMMA",
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
                        "d_in":        int(tc["anchor_train"].shape[-1]),
                        "feature_dim": int(K * tc["anchor_train"].shape[-1]),
                        "stacked":     True,
                    }
                    out_f.write(json.dumps(row) + "\n")
                    out_f.flush()
                    n_rows += 1
                print(f"  {task_name:40s} K={K} kept={Z_tr.shape[0]}/{ytr.size}")
            del tc
            try: del Z_tr, Z_te, valid_tr, valid_te
            except NameError: pass
            gc.collect()

    print(f"\nDone. wrote {n_rows} rows to {RAW_OUT_PATH}")


if __name__ == "__main__":
    main()
