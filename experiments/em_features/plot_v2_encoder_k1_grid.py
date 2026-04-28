"""Same 2x3 frontier grid as plot_v2_frontier_grid.py but for the
encoder-side-attribution (Δz̄) sweeps. The α=0 point is in the grid itself,
so no separate _alpha0 file is needed.

    uv run python -m experiments.em_features.plot_v2_encoder_grid \\
        --results_dir /root/em_features/results \\
        --out /root/em_features/results/v2_encoder_k1_frontier_grid.png
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    steps = [50000, 80000, 100000]
    archs = [
        ("SAE arditi v2 (k=128, d_sae=32k, T=1)",
         "v2_qwen_l15_sae_arditi_k128_step{}_encoder_k1_frontier.json"),
        ("Han H8 champion v2 (k=128, d_sae=32k, T=5)",
         "v2_qwen_l15_han_champ_step{}_encoder_k1_frontier.json"),
    ]
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    cmap = plt.cm.RdBu_r

    for row, (arch_label, fname_fmt) in enumerate(archs):
        for col, step in enumerate(steps):
            ax = axes[row, col]
            rows = load_rows(args.results_dir / fname_fmt.format(step))
            if not rows:
                ax.set_title(f"step {step//1000}k — no data")
                continue
            alphas = np.array([r["alpha"] for r in rows])
            aligns = np.array([r["align"] for r in rows])
            cohs = np.array([r["coh"] for r in rows])

            sc = ax.scatter(cohs, aligns, c=alphas, cmap=cmap, norm=norm, s=64,
                            edgecolors="k", linewidths=0.4)

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

            # α=0 black star (in-grid point)
            zero_idx = np.argmin(np.abs(alphas))
            if abs(alphas[zero_idx]) < 0.01:
                ax.scatter([cohs[zero_idx]], [aligns[zero_idx]], s=200, c="black",
                           marker="*", edgecolors="white", linewidths=1.5, zorder=10)
                ax.annotate(f"α=0\nalign={aligns[zero_idx]:.1f}",
                            (cohs[zero_idx], aligns[zero_idx]),
                            textcoords="offset points", xytext=(-12, -16), fontsize=8,
                            color="black", fontweight="bold", ha="right")

            arch_short = "SAE arditi" if "sae" in fname_fmt else "Han H8"
            ax.set_xlabel("mean coherence (Gemini)")
            ax.set_ylabel("mean alignment (Gemini)")
            ax.set_title(f"{arch_short} — step {step//1000}k  (n_α={len(rows)})", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.text(0.03, 0.97, arch_short, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", ha="left", va="top",
                    color="navy" if "sae" in fname_fmt else "darkred",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))

            a_mu, a_sd = float(np.mean(aligns)), float(np.std(aligns))
            c_mu, c_sd = float(np.mean(cohs)), float(np.std(cohs))
            ax.text(0.03, 0.04,
                    f"align: mean={a_mu:.2f}, std={a_sd:.2f}\ncoh:   mean={c_mu:.2f}, std={c_sd:.2f}",
                    transform=ax.transAxes, fontsize=8, ha="left", va="bottom", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", alpha=0.9,
                              edgecolor="black", linewidth=0.6))

        axes[row, 0].annotate(arch_label, xy=(-0.2, 0.5), xycoords="axes fraction",
                              ha="right", va="center", fontsize=11, fontweight="bold", rotation=90)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label="α (steering coef)")

    fig.suptitle("v2 100k frontier (encoder-side Δz̄, k=1 single-feature steering (Wang-style))", fontsize=12)
    fig.tight_layout(rect=[0.03, 0, 0.92, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
