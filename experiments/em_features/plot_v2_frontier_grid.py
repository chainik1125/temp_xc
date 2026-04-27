"""Plot a 2×3 frontier grid for v2 SAE and Han 100k checkpoints.

  Top row: v2 SAE arditi at step 50k, 80k, 100k
  Bottom row: v2 Han H8 champion at step 50k, 80k, 100k

Each panel scatters (mean_coherence, mean_alignment) for every α point in
the sweep, colored by α (red ↔ blue), with the α value labeled on points
that are local extrema or at the peak alignment.

    uv run python -m experiments.em_features.plot_v2_frontier_grid \\
        --results_dir /root/em_features/results \\
        --out /root/em_features/results/v2_frontier_grid.png
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
    p.add_argument("--results_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline_align", type=float, default=64.19,
                   help="OpenAI gpt-4o-mini baseline; for reference only — Gemini scores differently.")
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    rows = []
    for r in d.get("rows", []):
        a = r.get("mean_alignment")
        c = r.get("mean_coherence")
        alpha = r.get("alpha")
        if a is None or c is None or isinstance(a, float) and math.isnan(a) or isinstance(c, float) and math.isnan(c):
            continue
        rows.append({"alpha": alpha, "align": a, "coh": c})
    return rows


def main():
    args = parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=False, sharey=False)
    steps = [50000, 80000, 100000]
    archs = [
        ("SAE arditi v2 (k=128, d_sae=32k, T=1)",
         "v2_qwen_l15_sae_arditi_k128_step{}_frontier.json"),
        ("Han H8 champion v2 (k=128, d_sae=32k, T=5)",
         "v2_qwen_l15_han_champ_step{}_frontier.json"),
    ]

    # Shared colormap: α from -100 → +100 with 0 in the middle.
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    cmap = plt.cm.RdBu_r  # red = positive (more misaligned), blue = negative (more aligned suppression)

    for row, (arch_label, fname_fmt) in enumerate(archs):
        for col, step in enumerate(steps):
            ax = axes[row, col]
            rows = load_rows(args.results_dir / fname_fmt.format(step))
            if not rows:
                ax.set_title(f"step {step//1000}k — no data")
                ax.text(0.5, 0.5, "no valid rows", ha="center", va="center", transform=ax.transAxes)
                continue
            alphas = np.array([r["alpha"] for r in rows])
            aligns = np.array([r["align"] for r in rows])
            cohs = np.array([r["coh"] for r in rows])

            sc = ax.scatter(cohs, aligns, c=alphas, cmap=cmap, norm=norm, s=64, edgecolors="k", linewidths=0.4)

            # label peak alignment + a few extreme α points
            peak_i = int(np.argmax(aligns))
            ax.annotate(f"α={alphas[peak_i]:+.1f}\n{aligns[peak_i]:.1f}",
                        (cohs[peak_i], aligns[peak_i]),
                        textcoords="offset points", xytext=(8, 8), fontsize=8,
                        color="darkgreen", fontweight="bold")
            for idx in (int(np.argmin(alphas)), int(np.argmax(alphas))):
                if idx != peak_i:
                    ax.annotate(f"α={alphas[idx]:+.0f}",
                                (cohs[idx], aligns[idx]),
                                textcoords="offset points", xytext=(6, -10), fontsize=7,
                                color="gray")

            ax.axhline(args.baseline_align, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(0.02, args.baseline_align + 0.5, f"baseline (gpt-4o-mini): {args.baseline_align}",
                    transform=ax.get_yaxis_transform(), fontsize=7, color="gray")

            ax.set_xlabel("mean coherence (Gemini)")
            ax.set_ylabel("mean alignment (Gemini)")
            ax.set_title(f"step {step//1000}k  (n_α={len(rows)})", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Row title
        axes[row, 0].annotate(arch_label, xy=(-0.2, 0.5), xycoords="axes fraction",
                              ha="right", va="center", fontsize=11, fontweight="bold", rotation=90)

    # Shared colorbar on right
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label="α (steering coef)")

    fig.suptitle("v2 100k frontier: alignment vs coherence by α (Gemini judge, k=10 bundle)", fontsize=12)
    fig.tight_layout(rect=[0.03, 0, 0.92, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
