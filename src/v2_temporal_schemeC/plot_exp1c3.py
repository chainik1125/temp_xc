"""Generate Experiment 1c3 plots: local vs global AUC for coupled features.

Usage:
  PYTHONPATH=/home/elysium/temp_xc python src/v2_temporal_schemeC/plot_exp1c3.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.utils.plot import save_figure

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "results", "experiment1c3_coupled")
OUT_DIR = RESULTS_DIR

TXCDR_T_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]
_cmap = plt.colormaps["YlOrRd"]
_txcdr_colors = {T: _cmap((i + 2) / (len(TXCDR_T_VALUES) + 2))
                 for i, T in enumerate(TXCDR_T_VALUES)}
_txcdr_markers = {2: "o", 3: "s", 4: "D", 5: "^", 6: "v", 8: "p", 10: "h", 12: "*"}

STYLE = {
    "TopKSAE":     {"color": "#1f77b4", "marker": "x", "ls": "-"},
    "TFA":         {"color": "#17becf", "marker": "+", "ls": "-"},
    "TFA-pos":     {"color": "#2ca02c", "marker": "X", "ls": "-"},
    "Stacked T=2": {"color": "#9467bd", "marker": "o", "ls": "-"},
    "Stacked T=5": {"color": "#9467bd", "marker": "^", "ls": "--"},
}
for T in TXCDR_T_VALUES:
    STYLE[f"TXCDR T={T}"] = {
        "color": _txcdr_colors[T], "marker": _txcdr_markers[T], "ls": "-",
    }

SUPTITLE = r"Experiment 1c3: coupled features ($K\!=\!10$, $M\!=\!20$, $\rho\!=\!0.7$)"


def load_results():
    with open(os.path.join(RESULTS_DIR, "sweep_summary.json")) as f:
        return json.load(f)


def main():
    results = load_results()
    models = sorted(set(r["model"] for r in results))

    # ── Plot 1: NMSE, local AUC, global AUC vs k (3 panels) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for model in models:
        if model not in STYLE:
            continue
        s = STYLE[model]
        subset = sorted([r for r in results if r["model"] == model], key=lambda r: r["k"])
        if not subset:
            continue
        ks = [r["k"] for r in subset]
        axes[0].plot(ks, [r["nmse"] for r in subset], marker=s["marker"],
                     ls=s["ls"], color=s["color"], lw=2, ms=7, label=model)
        axes[1].plot(ks, [r["auc"] for r in subset], marker=s["marker"],
                     ls=s["ls"], color=s["color"], lw=2, ms=7, label=model)
        axes[2].plot(ks, [r["global_auc"] for r in subset], marker=s["marker"],
                     ls=s["ls"], color=s["color"], lw=2, ms=7, label=model)

    axes[0].set(xlabel="k", ylabel="NMSE", title="NMSE vs k", yscale="log")
    axes[1].set(xlabel="k", ylabel="Local AUC", title="Local AUC (vs emission features)")
    axes[2].set(xlabel="k", ylabel="Global AUC", title="Global AUC (vs hidden features)")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"{SUPTITLE}: NMSE and AUC", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, "exp1c3_topk_auc.png"))
    fig.savefig(os.path.join(OUT_DIR, "exp1c3_topk_auc.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  exp1c3_topk_auc")

    # ── Plot 2: global vs local AUC scatter ──
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.15, lw=1)

    for model in models:
        if model not in STYLE:
            continue
        s = STYLE[model]
        subset = [r for r in results if r["model"] == model]
        ax.scatter([r["auc"] for r in subset], [r["global_auc"] for r in subset],
                   color=s["color"], marker=s["marker"], s=60, alpha=0.7,
                   label=model, zorder=4, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Local AUC (vs emission features)", fontsize=12)
    ax.set_ylabel("Global AUC (vs hidden features)", fontsize=12)
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.suptitle(f"{SUPTITLE}: global vs local", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, "exp1c3_scatter.png"))
    fig.savefig(os.path.join(OUT_DIR, "exp1c3_scatter.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  exp1c3_scatter")

    # ── Plot 3: global AUC vs T at fixed k (TXCDR T-sweep) ──
    k_fixed = [1, 2, 3, 5]
    fig, axes = plt.subplots(1, len(k_fixed), figsize=(5 * len(k_fixed), 5), sharey=True)

    for ki, k in enumerate(k_fixed):
        ax = axes[ki]
        Ts, gaucs = [], []
        for T in TXCDR_T_VALUES:
            name = f"TXCDR T={T}"
            r = next((r for r in results if r["model"] == name and r["k"] == k), None)
            if r:
                Ts.append(T)
                gaucs.append(r["global_auc"])
        if Ts:
            ax.plot(Ts, gaucs, "o-", color="#d62728", lw=2, ms=8, label="TXCDR gAUC")

        # Stacked T=2 baseline at same k
        stacked = next((r for r in results if r["model"] == "Stacked T=2" and r["k"] == k), None)
        if stacked:
            ax.axhline(stacked["global_auc"], color="#9467bd", ls="--", alpha=0.6, lw=1.5,
                       label=f"Stacked T=2 ({stacked['global_auc']:.2f})")

        ax.set_xlabel("Window size T")
        if ki == 0:
            ax.set_ylabel("Global AUC")
        ax.set_title(f"k = {k}")
        ax.set_xticks([t for t in TXCDR_T_VALUES if t <= 14])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"{SUPTITLE}: TXCDR global AUC vs window size",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, "exp1c3_gauc_vs_T.png"))
    fig.savefig(os.path.join(OUT_DIR, "exp1c3_gauc_vs_T.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  exp1c3_gauc_vs_T")

    # ── Plot 4: gAUC gap (TXCDR T=2 - Stacked T=2) vs k ──
    fig, ax = plt.subplots(figsize=(7, 5))

    ks, gaps = [], []
    for k in sorted(set(r["k"] for r in results)):
        txcdr = next((r for r in results if r["model"] == "TXCDR T=2" and r["k"] == k), None)
        stacked = next((r for r in results if r["model"] == "Stacked T=2" and r["k"] == k), None)
        if txcdr and stacked:
            ks.append(k)
            gaps.append(txcdr["global_auc"] - stacked["global_auc"])

    ax.plot(ks, gaps, "o-", color="#d62728", lw=2, ms=8)
    ax.axhline(0, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel(r"$\Delta$gAUC (TXCDR T=2 $-$ Stacked T=2)", fontsize=12)
    ax.grid(True, alpha=0.2)

    plt.suptitle(f"{SUPTITLE}: TXCDR global AUC advantage", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, "exp1c3_gauc_gap.png"))
    fig.savefig(os.path.join(OUT_DIR, "exp1c3_gauc_gap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  exp1c3_gauc_gap")

    print(f"Done. All plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
