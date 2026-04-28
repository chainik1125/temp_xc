"""2-row × 4-col frontier comparison: SAE/Han 100k under cosine-k10, encoder-k10,
encoder-k1, and Wang-bundle-k30 procedures. Same layout idiom as plot_v2_frontier_grid.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="docs/dmitry/results/em_features (parent of v2_sweeps and wang)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    rows = []
    for r in d.get("rows", []):
        a = r.get("mean_alignment", r.get("mean_align"))
        c = r.get("mean_coherence", r.get("mean_coh"))
        if a is None or c is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(c, float) and math.isnan(c)):
            continue
        rows.append({"alpha": r["alpha"], "align": a, "coh": c})
    return rows


def main():
    args = parse_args()
    methods = [
        ("cosine k=10", "v2_sweeps/v2_qwen_l15_{arch}_step100000_frontier.json"),
        ("encoder Δz̄ k=10", "v2_sweeps/v2_qwen_l15_{arch}_step100000_encoder_frontier.json"),
        ("encoder Δz̄ k=1", "v2_sweeps/v2_qwen_l15_{arch}_step100000_encoder_k1_frontier.json"),
        ("Wang bundle k=30", "wang/{arch_short}_bundle30_frontier.json"),
    ]
    archs = [
        ("SAE arditi 100k", "sae_arditi_k128", "sae", "navy"),
        ("Han H8 100k", "han_champ", "han", "darkred"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    cmap = plt.cm.RdBu_r

    for row, (arch_label, arch_key, arch_short, color) in enumerate(archs):
        for col, (m_label, fname_fmt) in enumerate(methods):
            ax = axes[row, col]
            f = args.root / fname_fmt.format(arch=arch_key, arch_short=arch_short)
            rows = load_rows(f)
            if not rows:
                ax.set_title(f"{m_label}: NO DATA", fontsize=10)
                continue
            alphas = np.array([r["alpha"] for r in rows])
            aligns = np.array([r["align"] for r in rows])
            cohs = np.array([r["coh"] for r in rows])

            ax.scatter(cohs, aligns, c=alphas, cmap=cmap, norm=norm, s=64, edgecolors="k", linewidths=0.4)

            # Peak annotation (overall max align)
            peak_i = int(np.argmax(aligns))
            ax.annotate(f"α={alphas[peak_i]:+.1f}\n{aligns[peak_i]:.1f}",
                        (cohs[peak_i], aligns[peak_i]),
                        textcoords="offset points", xytext=(8, 8), fontsize=8,
                        color="darkgreen", fontweight="bold")

            # α=0 black star
            zero_i = int(np.argmin(np.abs(alphas)))
            if abs(alphas[zero_i]) < 0.01:
                ax.scatter([cohs[zero_i]], [aligns[zero_i]], s=200, c="black",
                           marker="*", edgecolors="white", linewidths=1.5, zorder=10)
                ax.annotate(f"α=0\n{aligns[zero_i]:.1f}",
                            (cohs[zero_i], aligns[zero_i]),
                            textcoords="offset points", xytext=(-12, -16), fontsize=8,
                            color="black", fontweight="bold", ha="right")

            ax.set_title(m_label, fontsize=11)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"{arch_label}\nmean alignment (Gemini)", fontsize=10)
            if row == 1:
                ax.set_xlabel("mean coherence (Gemini)")
            ax.text(0.03, 0.97, arch_label.split()[0], transform=ax.transAxes,
                    fontsize=12, fontweight="bold", ha="left", va="top",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))
            a_mu, a_sd = float(np.mean(aligns)), float(np.std(aligns))
            c_mu, c_sd = float(np.mean(cohs)), float(np.std(cohs))
            ax.text(0.03, 0.04,
                    f"align: mean={a_mu:.2f}, std={a_sd:.2f}\ncoh:   mean={c_mu:.2f}, std={c_sd:.2f}",
                    transform=ax.transAxes, fontsize=8, ha="left", va="bottom", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", alpha=0.9,
                              edgecolor="black", linewidth=0.6))

    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label="α")

    fig.suptitle("v2 100k frontier comparison: 4 attribution procedures × 2 archs (Gemini judge)", fontsize=13)
    fig.tight_layout(rect=[0.02, 0.02, 0.93, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
