"""Concat-latent probing: fit a single probe on [TXC ∥ MLC] per example.

Part of the TXC/MLC complementarity analysis (handover
`2026-04-22-handover-batchtopk-tsweep.md` experiment iv, step 5).

Router analysis (analysis/router.py) picks ONE arch per task from task
metadata. This script asks a different question: if a single probe sees
BOTH archs' latents per example, can it exploit the complementarity
WITHIN each task? Emits the "concat probe" column for the
Complementarity table in summary.md.

For each task × aggregation:

1. Encode the task's train+test anchor/mlc activations through
   `agentic_txc_02` → Z_txc (N, d_sae=18432).
2. Encode through `agentic_mlc_08` → Z_mlc (N, d_sae=18432).
3. Concatenate → Z_concat (N, 2·d_sae = 36864).
4. Top-k-by-class-separation (k=5) on the train concat.
5. L1 logistic regression on the 5 selected features.
6. Report test AUC + accuracy.

Runs at seed=42. Takes ~3-5 min total on an A40 (GPU encode + CPU probe).
Writes `results/concat_probe_results.json`.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/workspace/temp_xc")
sys.path.insert(0, str(REPO))

from experiments.phase5_downstream_utility.probing.run_probing import (  # noqa: E402
    _encode_for_probe,
    _load_model_for_run,
    _load_task_cache,
    sae_probe_metrics,
)

PROBE_CACHE = REPO / "experiments/phase5_downstream_utility/results/probe_cache"
CKPT_DIR = REPO / "experiments/phase5_downstream_utility/results/ckpts"
OUT_JSON = REPO / "experiments/phase5_downstream_utility/results/concat_probe_results.json"

ARCH_A = "agentic_txc_02"
ARCH_B = "agentic_mlc_08"


def _encode_both(txc, txc_meta, mlc, mlc_meta, tc, split, device, agg):
    """Return (Z_txc_concat_mlc) of shape (N, d_txc + d_mlc)."""
    Z_txc = _encode_for_probe(txc, ARCH_A, txc_meta, tc, split, device,
                              aggregation=agg)
    Z_mlc = _encode_for_probe(mlc, ARCH_B, mlc_meta, tc, split, device,
                              aggregation=agg)
    return np.concatenate([Z_txc, Z_mlc], axis=1)


def run() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    rid_txc = f"{ARCH_A}__seed42"
    rid_mlc = f"{ARCH_B}__seed42"
    ckpt_txc = CKPT_DIR / f"{rid_txc}.pt"
    ckpt_mlc = CKPT_DIR / f"{rid_mlc}.pt"

    txc_model, _, txc_meta = _load_model_for_run(rid_txc, ckpt_txc, device)
    mlc_model, _, mlc_meta = _load_model_for_run(rid_mlc, ckpt_mlc, device)
    txc_model.eval()
    mlc_model.eval()

    task_dirs = sorted(d for d in PROBE_CACHE.iterdir() if d.is_dir())
    print(f"concat probe on {len(task_dirs)} tasks × 2 aggregations")

    results: dict[str, dict] = {}
    for agg in ["last_position", "mean_pool"]:
        print(f"\n=== aggregation = {agg} ===")
        per_task = {}
        for i, task_dir in enumerate(task_dirs):
            task = task_dir.name
            tc = _load_task_cache(task_dir)
            Z_tr = _encode_both(txc_model, txc_meta, mlc_model, mlc_meta,
                                tc, "train", device, agg)
            Z_te = _encode_both(txc_model, txc_meta, mlc_model, mlc_meta,
                                tc, "test", device, agg)
            y_tr = tc["train_labels"].astype(np.int32)
            y_te = tc["test_labels"].astype(np.int32)
            # flip polarity for winogrande/wsc matches existing probe convention
            FLIP = {"winogrande_correct_completion", "wsc_coreference"}
            auc, acc = sae_probe_metrics(Z_tr, y_tr, Z_te, y_te, k=5)
            if task in FLIP:
                auc = max(auc, 1.0 - auc)
            per_task[task] = {"auc": float(auc), "acc": float(acc),
                              "d_concat": int(Z_tr.shape[1])}
            print(f"  [{i+1:2d}/{len(task_dirs)}] {task:<40s} "
                  f"auc={auc:.4f} acc={acc:.4f}")

        mean_auc = st.mean(r["auc"] for r in per_task.values())
        mean_acc = st.mean(r["acc"] for r in per_task.values())
        print(f"  mean AUC = {mean_auc:.4f}   mean ACC = {mean_acc:.4f}")
        results[agg] = {
            "mean_auc": float(mean_auc),
            "mean_acc": float(mean_acc),
            "per_task": per_task,
        }

    return results


def main() -> None:
    out = run()
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # Compact summary
    print("\n=== CONCAT PROBE SUMMARY ===")
    for agg, r in out.items():
        print(f"  {agg:>14s}: mean_auc = {r['mean_auc']:.4f}  "
              f"mean_acc = {r['mean_acc']:.4f}")


if __name__ == "__main__":
    main()
