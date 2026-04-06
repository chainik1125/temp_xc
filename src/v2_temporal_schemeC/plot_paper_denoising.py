"""Generate paper-ready figure: global vs local feature recovery scatter.

Shows TXCDRv2 at multiple T values climbing above the per-token floor,
while Stacked SAE stays at the floor. Uses Experiment 1c2 data (sparse
regime, heterogeneous rho).

Usage:
  PYTHONPATH=/home/elysium/temp_xc python src/v2_temporal_schemeC/plot_paper_denoising.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from src.utils.plot import save_figure

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "results", "experiment1c2_sparse")
OUT_DIR = os.path.join(BASE, "results", "paper_figures")

TXCDR_T_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "results.json")) as f:
        data = json.load(f)
    models = data["models"]

    # ── Figure: global vs local scatter ──
    fig, ax = plt.subplots(figsize=(7, 6))

    # Reference lines
    lim = 0.85
    ax.plot([0, lim], [0, lim], "k-", alpha=0.15, lw=1, zorder=1)  # y=x

    # Per-token floor line: slope = corr(s,h) ≈ 0.77 in this regime
    # In correlation space, global = floor_ratio * local
    # We compute it from the Stacked SAE data
    stacked_locs, stacked_globs = [], []
    for name in ["Stacked-T2", "Stacked-T5"]:
        if name not in models:
            continue
        for r in models[name]:
            loc = r["mean_local_corr"]
            glob = r["mean_global_corr"]
            if loc > 0.01:
                stacked_locs.append(loc)
                stacked_globs.append(glob)
    if stacked_locs:
        floor_ratio = np.mean(stacked_globs) / np.mean(stacked_locs)
        ax.plot([0, lim], [0, lim * floor_ratio], color="gray", ls="--", lw=1,
                alpha=0.5, zorder=1, label=f"Per-token floor (ratio={floor_ratio:.2f})")

    # Stacked SAE (control)
    for name, marker, label in [("Stacked-T2", "s", "Stacked T=2"),
                                 ("Stacked-T5", "D", "Stacked T=5")]:
        if name not in models:
            continue
        locs = [r["mean_local_corr"] for r in models[name] if r["mean_local_corr"] > 0.01]
        globs = [r["mean_global_corr"] for r in models[name] if r["mean_local_corr"] > 0.01]
        ax.scatter(locs, globs, color="#9467bd", marker=marker, s=50, alpha=0.5,
                   label=label, zorder=3, edgecolors="white", linewidths=0.5)

    # TXCDRv2 at each T (color gradient)
    cmap = cm.get_cmap("YlOrRd", len(TXCDR_T_VALUES) + 2)
    markers = ["o", "s", "D", "^", "v", "<", "p", ">"]
    for i, T in enumerate(TXCDR_T_VALUES):
        name = f"TXCDRv2-T{T}"
        if name not in models or not models[name]:
            continue
        locs = [r["mean_local_corr"] for r in models[name] if r["mean_local_corr"] > 0.01]
        globs = [r["mean_global_corr"] for r in models[name] if r["mean_local_corr"] > 0.01]
        color = cmap(i + 2)
        ax.scatter(locs, globs, color=color, marker=markers[i % len(markers)],
                   s=60, alpha=0.8, label=f"TXCDRv2 T={T}", zorder=4,
                   edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Local correlation (latent vs noisy observation $s_i$)", fontsize=12)
    ax.set_ylabel("Global correlation (latent vs hidden state $h_i$)", fontsize=12)
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, "global_vs_local_scatter.png"))
    fig.savefig(os.path.join(OUT_DIR, "global_vs_local_scatter.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUT_DIR}/global_vs_local_scatter.{{png,pdf}}")

    # ── Figure 2: denoising ratio vs T at fixed k ──
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for k, color, marker in [(1, "#1f77b4", "o"), (3, "#d62728", "s"), (5, "#2ca02c", "^")]:
        Ts, rats = [], []
        for T in TXCDR_T_VALUES:
            name = f"TXCDRv2-T{T}"
            if name not in models:
                continue
            for r in models[name]:
                if r["k"] == k and r["mean_local_corr"] > 0.01:
                    Ts.append(T)
                    rats.append(r["mean_global_corr"] / r["mean_local_corr"])
                    break
        if Ts:
            ax.plot(Ts, rats, f"{marker}-", color=color, lw=2, ms=8, label=f"k = {k}")

    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, lw=1)
    # Show per-token floor
    if stacked_locs:
        ax.axhline(floor_ratio, color="gray", ls=":", alpha=0.4, lw=1,
                   label=f"Per-token floor ({floor_ratio:.2f})")

    ax.set_xlabel("Window size $T$", fontsize=12)
    ax.set_ylabel("Global / Local correlation ratio", fontsize=12)
    ax.set_xticks(TXCDR_T_VALUES)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, "denoising_ratio_vs_T.png"))
    fig.savefig(os.path.join(OUT_DIR, "denoising_ratio_vs_T.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUT_DIR}/denoising_ratio_vs_T.{{png,pdf}}")


if __name__ == "__main__":
    main()
