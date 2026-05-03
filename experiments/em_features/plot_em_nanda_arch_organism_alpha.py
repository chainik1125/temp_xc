"""Arch x organism x alpha-regime panel plot for the EM-Nanda paper figure.

Visualizes the closed cross-cell table from EM_NANDA_BRIEF / em_nanda_synthesis:
two architectures (SAE arditi T=1, TXC paper k=100) x five organism-regime
columns (R1 5k mid-alpha, R1 10k mid-alpha, R1 30k mid-alpha, R32 10k std-alpha,
R32 10k ext-alpha). Each bar is a stage-4 single-feat best align value at 8
rollouts/cell, taken from the synthesis table; the horizontal reference line is
the prior Qwen-7B medical champion (align 58.47).

Numbers are hardcoded from the closed 8-cell table (synthesis 00:00 UTC entry).

    python -m experiments.em_features.plot_em_nanda_arch_organism_alpha \\
        --out docs/dmitry/results/em_features/plots/em_nanda_arch_organism_alpha_table.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLUMNS = [
    ("R1 5k\nmid-alpha",  "R1 5k mid-alpha"),
    ("R1 10k\nmid-alpha", "R1 10k mid-alpha"),
    ("R1 30k\nmid-alpha", "R1 30k mid-alpha"),
    ("R32 10k\nstd-alpha", "R32 10k std-alpha"),
    ("R32 10k\next-alpha", "R32 10k ext-alpha"),
]

SAE_ALIGN = [95.78, 94.69, 95.16, 54.61, 64.53]
TXC_ALIGN = [90.88, 90.23, 91.25, 52.50, 51.95]

SAE_ANNOT = ["f28663 a=-6", "f11086 a=-6", "f9135 a=-6",
             "f21224 a=-3", "f21224 a=-30"]
TXC_ANNOT = ["f15402 a=-2", "f14729 a=-1.75", "f4992 a=-1.5",
             "f15779 a=+1.50", "f718 a=-30"]

GOAL = 58.47  # Qwen-7B medical-champion single-feat align


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    n = len(COLUMNS)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.0))

    sae_color = "tab:blue"
    txc_color = "tab:orange"

    bars_sae = ax.bar(x - width / 2, SAE_ALIGN, width,
                      color=sae_color, edgecolor="black", linewidth=0.6,
                      label="SAE arditi T=1 (k=128)")
    bars_txc = ax.bar(x + width / 2, TXC_ALIGN, width,
                      color=txc_color, edgecolor="black", linewidth=0.6,
                      label="TXC paper k=100")

    # Goal line
    ax.axhline(GOAL, color="firebrick", linestyle="--", linewidth=1.4,
               alpha=0.85)
    ax.text(n - 0.55, GOAL + 1.2,
            f"Qwen-7B medical champion ({GOAL})",
            color="firebrick", fontsize=9, ha="right", va="bottom")

    # Bar value annotations
    for bar, val, meta in zip(bars_sae, SAE_ALIGN, SAE_ANNOT):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=9, color=sae_color, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 2.0,
                meta, ha="center", va="bottom",
                fontsize=7, color="black", rotation=90, alpha=0.7)
    for bar, val, meta in zip(bars_txc, TXC_ALIGN, TXC_ANNOT):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=9, color=txc_color, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 2.0,
                meta, ha="center", va="bottom",
                fontsize=7, color="black", rotation=90, alpha=0.7)

    # Arch-gap callouts above each pair
    for i, (s, t) in enumerate(zip(SAE_ALIGN, TXC_ALIGN)):
        gap = s - t
        ymax = max(s, t)
        ax.annotate(f"+{gap:.2f}", xy=(x[i], ymax + 5.5),
                    ha="center", va="bottom", fontsize=8.5, color="dimgray",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in COLUMNS], fontsize=10)
    ax.set_ylabel("Single-feat peak align "
                  "(Wang stage-4, 8 rollouts x 64 examples / cell)",
                  fontsize=10)
    ax.set_title(
        "EM Nanda - architecture x organism x alpha-regime "
        "(Qwen-14B + finance, layer 24)\n"
        "SAE arditi wins every cell; arch gap widens to +12.58 "
        "in R32 ext-alpha (the regime that asks for real causal work)",
        fontsize=11)
    ax.set_ylim(0, 118)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=10, framealpha=0.92)

    # Light banding to separate R1 from R32
    ax.axvspan(-0.5, 2.5, color="tab:green", alpha=0.05, zorder=0)
    ax.axvspan(2.5, n - 0.5, color="tab:red",   alpha=0.05, zorder=0)
    ax.text(1.0, 113, "R1 organism (mid-alpha)",
            ha="center", fontsize=10, color="tab:green", fontweight="bold")
    ax.text(3.5, 113, "R32 organism (harder; ~26% EM)",
            ha="center", fontsize=10, color="tab:red", fontweight="bold")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
