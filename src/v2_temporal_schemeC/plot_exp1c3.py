"""Generate Experiment 1c3 plots: local vs global AUC for coupled features.

Usage:
  PYTHONPATH=/home/elysium/temp_xc python src/v2_temporal_schemeC/plot_exp1c3.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plot import save_figure

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "results", "experiment1c3_coupled")
OUT_DIR = RESULTS_DIR

STYLE = {
    "Stacked T=2": {"color": "#9467bd", "marker": "o", "ls": "-"},
    "Stacked T=5": {"color": "#9467bd", "marker": "^", "ls": "--"},
    "TXCDR T=2":   {"color": "#d62728", "marker": "s", "ls": "-"},
    "TXCDR T=5":   {"color": "#d62728", "marker": "^", "ls": "--"},
}

SUPTITLE = r"Experiment 1c3: coupled features ($K\!=\!10$, $M\!=\!20$, $n_{\mathrm{parents}}\!=\!2$)"


def load_results():
    with open(os.path.join(RESULTS_DIR, "sweep_summary.json")) as f:
        return json.load(f)


def main():
    results = load_results()

    rho_values = sorted(set(r["rho"] for r in results))
    models = sorted(set(r["model"] for r in results))

    # ── Plot 1: local AUC and gAUC vs k, one column per rho ──
    fig, axes = plt.subplots(2, len(rho_values), figsize=(6 * len(rho_values), 10),
                             sharex=True)

    for ci, rho in enumerate(rho_values):
        for model in models:
            if model not in STYLE:
                continue
            s = STYLE[model]
            subset = [r for r in results if r["rho"] == rho and r["model"] == model]
            subset.sort(key=lambda r: r["k"])
            if not subset:
                continue
            ks = [r["k"] for r in subset]
            auc = [r["auc"] for r in subset]
            gauc = [r["global_auc"] for r in subset]

            axes[0, ci].plot(ks, auc, marker=s["marker"], ls=s["ls"],
                             color=s["color"], lw=2, ms=7, label=model)
            axes[1, ci].plot(ks, gauc, marker=s["marker"], ls=s["ls"],
                             color=s["color"], lw=2, ms=7, label=model)

        axes[0, ci].set_title(fr"$\rho = {rho}$", fontsize=13)
        axes[1, ci].set_xlabel("k")

    axes[0, 0].set_ylabel("Local AUC (vs emission features)")
    axes[1, 0].set_ylabel("Global AUC (vs hidden features)")

    for row in axes:
        for ax in row:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.set_ylim(0.35, 1.05)

    plt.suptitle(f"{SUPTITLE}: local vs global AUC", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, "exp1c3_auc_vs_k.png"))
    fig.savefig(os.path.join(OUT_DIR, "exp1c3_auc_vs_k.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  exp1c3_auc_vs_k")

    # ── Plot 2: gAUC vs local AUC scatter (all rho, all k) ──
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.15, lw=1)

    for model in models:
        if model not in STYLE:
            continue
        s = STYLE[model]
        subset = [r for r in results if r["model"] == model]
        auc = [r["auc"] for r in subset]
        gauc = [r["global_auc"] for r in subset]
        ax.scatter(auc, gauc, color=s["color"], marker=s["marker"],
                   s=70, alpha=0.7, label=model, zorder=4,
                   edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Local AUC (vs emission features)", fontsize=12)
    ax.set_ylabel("Global AUC (vs hidden features)", fontsize=12)
    ax.set_xlim(0.35, 1.0)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.suptitle(f"{SUPTITLE}", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, "exp1c3_scatter.png"))
    fig.savefig(os.path.join(OUT_DIR, "exp1c3_scatter.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  exp1c3_scatter")

    # ── Plot 3: gAUC gap (TXCDR - Stacked) vs k, one line per rho ──
    fig, ax = plt.subplots(figsize=(7, 5))
    rho_colors = {0.0: "#1f77b4", 0.7: "#ff7f0e", 0.9: "#d62728"}

    for rho in rho_values:
        ks, gaps = [], []
        for k in sorted(set(r["k"] for r in results)):
            txcdr = [r for r in results if r["rho"] == rho and r["k"] == k
                     and r["model"] == "TXCDR T=2"]
            stacked = [r for r in results if r["rho"] == rho and r["k"] == k
                       and r["model"] == "Stacked T=2"]
            if txcdr and stacked:
                ks.append(k)
                gaps.append(txcdr[0]["global_auc"] - stacked[0]["global_auc"])

        if ks:
            ax.plot(ks, gaps, "o-", color=rho_colors[rho], lw=2, ms=8,
                    label=fr"$\rho = {rho}$")

    ax.axhline(0, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel(r"$\Delta$gAUC (TXCDR T=2 $-$ Stacked T=2)", fontsize=12)
    ax.legend(fontsize=11)
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
