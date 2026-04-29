"""Side-by-side base-vs-IT leaderboard plot.

Reads probing_results.jsonl for both subject models and renders a
2x2 panel:
  top-left:    BASE k_feat=5
  top-right:   BASE k_feat=20
  bottom-left: IT k_feat=5
  bottom-right: IT k_feat=20

Each arch is a row; mean ± σ_seeds error bar; same colour scheme as
build_leaderboard_2seed.py (per-token SAE blue, TFA grey, MLC purple,
TXC red).

Output: results/plots/phase7_leaderboard_base_vs_it.png + thumb.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.build_leaderboard_base_vs_it
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.phase7_unification._paths import OUT_DIR, PLOTS_DIR
from experiments.phase7_unification.build_leaderboard_2seed import (
    LEADERBOARD_ARCHS, load_seed_task_aucs, summarise, save_figure,
)


SUBJECTS = [
    ("google/gemma-2-2b", "BASE (gemma-2-2b L12)"),
    ("google/gemma-2-2b-it", "IT (gemma-2-2b-it L13)"),
]


def _color_for(arch: str) -> str:
    if arch in ("topk_sae", "tsae_paper_k20", "tsae_paper_k500"):
        return "#4472c4"  # blue
    if arch == "tfa_big":
        return "#888888"  # grey
    if arch in ("mlc", "agentic_mlc_08", "mlc_contrastive_alpha100_batchtopk"):
        return "#7030a0"  # purple
    return "#c00000"  # red (TXC family)


def _draw_panel(ax, rows, k_feat: int, title: str, missing_subjects: list[str]):
    rows_k = sorted(
        [r for r in rows if r["k_feat"] == k_feat and r["n_seeds"] > 0],
        key=lambda r: r["cross_seed_mean"],
    )
    archs = [r["arch_id"] for r in rows_k]
    means = [r["cross_seed_mean"] for r in rows_k]
    sigmas = [r["seed_only_std"] for r in rows_k]
    ypos = np.arange(len(archs))
    colors = [_color_for(a) for a in archs]
    if archs:
        ax.barh(ypos, means, xerr=sigmas, color=colors,
                error_kw={"ecolor": "k", "alpha": 0.6, "capsize": 3})
        ax.set_yticks(ypos); ax.set_yticklabels(archs, fontsize=8)
        for i, m in enumerate(means):
            ax.text(m + 0.005, i, f"{m:.4f}", va="center", fontsize=7)
        ax.set_xlim(0.65, 0.95 if k_feat == 5 else 0.97)
    else:
        ax.text(0.5, 0.5, "(no rows)", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel(f"mean test_auc_flip (k_feat={k_feat})")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    if missing_subjects:
        ax.text(
            0.02, 0.02,
            "missing: " + ", ".join(missing_subjects),
            transform=ax.transAxes, fontsize=7, color="grey",
        )


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)

    for row_idx, (subject_model, label) in enumerate(SUBJECTS):
        seed_task_aucs = load_seed_task_aucs(subject_model=subject_model)
        rows = summarise(seed_task_aucs)
        for col_idx, k_feat in enumerate((5, 20)):
            ax = axes[row_idx, col_idx]
            n_archs = sum(1 for r in rows if r["k_feat"] == k_feat and r["n_seeds"] > 0)
            title = f"{label}  k_feat={k_feat}  ({n_archs} archs)"
            # Note any archs with no rows for this subject (informational)
            covered = {r["arch_id"] for r in rows if r["k_feat"] == k_feat and r["n_seeds"] > 0}
            missing = [a for a in LEADERBOARD_ARCHS if a not in covered]
            _draw_panel(ax, rows, k_feat, title, missing)

    fig.suptitle(
        "Phase 7 leaderboard — BASE vs IT comparison, multi-seed mean ± σ_seeds, PAPER task set",
        fontsize=12, weight="bold",
    )
    out_path = PLOTS_DIR / "phase7_leaderboard_base_vs_it.png"
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
