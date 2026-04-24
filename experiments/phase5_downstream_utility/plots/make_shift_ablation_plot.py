"""Shift-ablation curve at T=5: AUC vs contrastive shift-set.

Mechanism study from the post-H8 handover. Eight variants:
  {1}, {1,2} (=H8), {1,2,3}, {1,2,3,4}, {1,2,4}, {2}, {4}, {1,2,3} uniform.

Goal is a CURVE not a winner. Whatever shape we observe is publishable
as an ablation.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path("/workspace/temp_xc")
JSONL = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
OUT_DIR = REPO / "experiments/phase5_downstream_utility/results/plots"

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}
K_FEAT = 5

VARIANTS = [
    ("{1}", "phase57_partB_h8a_shifts1", 1),
    ("{1,2} = H8", "phase57_partB_h8_bare_multidistance", 2),
    ("{1,2,3}", "phase57_partB_h8a_shifts123", 3),
    ("{1,2,3,4}", "phase57_partB_h8a_shifts1234", 4),
    ("{1,2,4}", "phase57_partB_h8a_shifts124", 3),
    ("{2}", "phase57_partB_h8a_shifts2", 1),
    ("{4}", "phase57_partB_h8a_shifts4", 1),
    ("{1,2,3} uniform", "phase57_partB_h8a_shifts123_uniform", 3),
]


def _mean_auc(arch: str, seed: int = 42, agg: str = "last_position") -> float | None:
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


def make_plot(agg: str = "mean_pool") -> None:
    labels, archs, n_shifts = zip(*VARIANTS)
    aucs = [_mean_auc(a, agg=agg) for a in archs]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(labels)))
    bar_colors = ["#4477aa" if "uniform" not in lbl else "#cc6677" for lbl in labels]
    bar_colors[1] = "#117733"  # H8 highlight
    bars = ax.bar(x, [(a if a is not None else 0) for a in aucs],
                  color=bar_colors, alpha=0.8)
    for i, (label, auc, bar) in enumerate(zip(labels, aucs, bars)):
        if auc is not None:
            ax.text(i, auc + 0.001, f"{auc:.4f}",
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold' if "H8" in label else 'normal')
        else:
            ax.text(i, 0.001, "—", ha='center', va='bottom', fontsize=9, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_xlabel("contrastive shifts")
    ax.set_ylabel(f"mean test AUC ({agg}, k_feat=5, 36 tasks)")
    ax.set_title(f"H8 shift-ablation curve at T=5 — {agg}")
    ax.set_ylim(bottom=min((a for a in aucs if a is not None), default=0.7) - 0.01,
                top=(max((a for a in aucs if a is not None), default=0.83) + 0.01))
    ax.grid(axis='y', alpha=0.3)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    out = OUT_DIR / f"shift_ablation_{agg}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(OUT_DIR / f"shift_ablation_{agg}.thumb.png", dpi=48, bbox_inches='tight')
    print(f"wrote {out}")


def main():
    for agg in ("mean_pool", "last_position"):
        make_plot(agg)


if __name__ == "__main__":
    main()
