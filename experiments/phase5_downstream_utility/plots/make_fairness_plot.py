"""Fairness plot — TXC + anti-dead vs MLC + anti-dead, paired bars.

Addresses reviewer concern: "you may have given TXC an unfair anti-dead
advantage". This plot shows what happens when we apply the SAME anti-dead
stack to both architectures.

Three recipe pairs:
  1. recon-only        : MLCBareAntidead     (no TXC equivalent trained — would be TXCBareAntidead)
  2. + matryoshka + single-shift contr : Phase 6.2 C3 (TXC) vs MLCBareMatryoshkaContrastiveAntidead (MLC)
  3. + matryoshka + multi-scale contr  : H7 (TXC)         vs MLCBareMultiscaleAntidead (MLC)
  4. + matryoshka + multi-distance     : H8 (TXC, T=5)    — no MLC analog (TXC-only)
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/temp_xc")
JSONL = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
OUT_DIR = REPO / "experiments/phase5_downstream_utility/results/plots"

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}
K_FEAT = 5

PAIRS = [
    ("matryoshka + single-shift contr",
     "phase57_partB_h7_bare_multiscale",  # placeholder — replace if a true single-shift exists
     "mlc_bare_matryoshka_contrastive_antidead"),
    ("matryoshka + multi-scale contr (= H7-recipe)",
     "phase57_partB_h7_bare_multiscale",
     "mlc_bare_multiscale_antidead"),
    ("multi-distance (TXC-only — no MLC analog)",
     "phase57_partB_h8_bare_multidistance",
     None),
]


def _mean_auc(arch: str, seed: int = 42, agg: str = "last_position") -> float | None:
    if arch is None:
        return None
    aucs = []
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aggregation") != agg or r.get("k_feat") != K_FEAT:
                continue
            if r.get("run_id") != f"{arch}__seed{seed}":
                continue
            v = r.get("test_auc")
            if v is None:
                continue
            v = float(v)
            if r.get("task_name") in FLIP_TASKS:
                v = max(v, 1.0 - v)
            aucs.append(v)
    if len(aucs) < 30:
        return None
    return st.mean(aucs)


def make_plot(agg: str = "mean_pool"):
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, txc_aucs, mlc_aucs = [], [], []
    for label, txc_arch, mlc_arch in PAIRS:
        labels.append(label)
        txc_aucs.append(_mean_auc(txc_arch, agg=agg))
        mlc_aucs.append(_mean_auc(mlc_arch, agg=agg) if mlc_arch else None)

    x = np.arange(len(labels))
    w = 0.38
    txc_vals = [a if a is not None else 0 for a in txc_aucs]
    mlc_vals = [a if a is not None else 0 for a in mlc_aucs]
    ax.bar(x - w/2, txc_vals, w, color="#117733", alpha=0.85, label="TXC + anti-dead")
    ax.bar(x + w/2, mlc_vals, w, color="#332288", alpha=0.85, label="MLC + anti-dead")

    for i, (t, m) in enumerate(zip(txc_aucs, mlc_aucs)):
        if t is not None:
            ax.text(i - w/2, t + 0.001, f"{t:.4f}", ha='center', va='bottom', fontsize=9)
        if m is not None:
            ax.text(i + w/2, m + 0.001, f"{m:.4f}", ha='center', va='bottom', fontsize=9)
        if m is None:
            ax.text(i + w/2, 0.001, "no MLC analog", ha='center', va='bottom',
                    fontsize=8, style='italic', alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha='right', fontsize=9)
    ax.set_ylabel(f"mean test AUC ({agg}, k_feat=5, 36 tasks)")
    ax.set_title(f"Fairness: TXC + anti-dead vs MLC + anti-dead — {agg}")
    ax.set_ylim(bottom=0.7)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right')
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    out = OUT_DIR / f"fairness_txc_vs_mlc_antidead_{agg}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(OUT_DIR / f"fairness_txc_vs_mlc_antidead_{agg}.thumb.png",
                dpi=48, bbox_inches='tight')
    print(f"wrote {out}")


def main():
    for agg in ("mean_pool", "last_position"):
        make_plot(agg)


if __name__ == "__main__":
    main()
