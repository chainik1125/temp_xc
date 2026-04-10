"""Generate Experiment 1c3 plots: three denoising metrics for coupled features.

Reads denoising_results.json (from run_exp1c3n_denoising.py) which contains:
  (i)   gAUC: decoder cosine similarity vs emission/hidden directions
  (ii)  Correlation ratio: single-latent Pearson corr with emissions vs hidden states
  (iii) Linear probe ratio: Ridge R² for z→emission vs z→hidden

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
RESULTS_DIR = os.path.join(BASE, "results", "experiment1c3_noisy_coupled")
OUT_DIR = RESULTS_DIR

TXCDR_T_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]
_cmap = plt.colormaps["YlOrRd"]
_txcdr_colors = {T: _cmap((i + 2) / (len(TXCDR_T_VALUES) + 2))
                 for i, T in enumerate(TXCDR_T_VALUES)}
_txcdr_markers = {2: "o", 3: "s", 4: "D", 5: "^", 6: "v", 8: "p", 10: "h", 12: "*"}

STYLE = {
    "Stacked-T2": {"color": "#9467bd", "marker": "o", "ls": "-"},
    "Stacked-T5": {"color": "#9467bd", "marker": "^", "ls": "--"},
}
for T in TXCDR_T_VALUES:
    STYLE[f"TXCDRv2-T{T}"] = {
        "color": _txcdr_colors[T], "marker": _txcdr_markers[T], "ls": "-",
    }

SUPTITLE = r"Experiment 1c3: noisy coupled features ($K\!=\!10$, $M\!=\!20$, $\rho\!=\!0.7$)"


def load_denoising():
    with open(os.path.join(RESULTS_DIR, "denoising_results.json")) as f:
        return json.load(f)


def _plot_metric_vs_k(data, emission_key, hidden_key, ylabel_e, ylabel_h, title, filename):
    """Plot emission metric and hidden metric vs k (2 panels)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name in sorted(data.keys()):
        if name not in STYLE or not data[name]:
            continue
        s = STYLE[name]
        results = sorted(data[name], key=lambda r: r["k"])
        ks = [r["k"] for r in results]
        axes[0].plot(ks, [r[emission_key] for r in results], marker=s["marker"],
                     ls=s["ls"], color=s["color"], lw=2, ms=7, label=name)
        axes[1].plot(ks, [r[hidden_key] for r in results], marker=s["marker"],
                     ls=s["ls"], color=s["color"], lw=2, ms=7, label=name)
    axes[0].set(xlabel="k", ylabel=ylabel_e, title=f"Emission (local) {title}")
    axes[1].set(xlabel="k", ylabel=ylabel_h, title=f"Hidden (global) {title}")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)
    plt.suptitle(f"{SUPTITLE}: {title}", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, filename))
    fig.savefig(os.path.join(OUT_DIR, filename.replace(".png", ".pdf")),
                bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_denoising()
    models = sorted(data.keys())

    # ── Plot 1: gAUC (emission vs hidden) ──
    _plot_metric_vs_k(data, "emission_auc", "hidden_auc",
                      "Local AUC", "Global AUC", "AUC", "exp1c3n_auc.png")
    print("  exp1c3n_auc")

    # ── Plot 2: Single-latent correlation (emission vs hidden) ──
    _plot_metric_vs_k(data, "emission_corr", "hidden_corr",
                      "Emission corr", "Hidden corr",
                      "single-latent correlation", "exp1c3n_corr.png")
    print("  exp1c3n_corr")

    # ── Plot 3: Linear probe R² (emission vs hidden) ──
    _plot_metric_vs_k(data, "emission_r2", "hidden_r2",
                      r"Emission $R^2$", r"Hidden $R^2$",
                      r"linear probe $R^2$", "exp1c3n_probe.png")
    print("  exp1c3n_probe")

    # ── Plot 4: All three ratios vs k (3 panels) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ratio_keys = [
        ("hidden_auc", "emission_auc", "gAUC ratio"),
        ("corr_ratio", None, "Correlation ratio"),
        ("probe_ratio", None, "Probe R² ratio"),
    ]
    for ai, (key, denom_key, title) in enumerate(ratio_keys):
        ax = axes[ai]
        for name in models:
            if name not in STYLE or not data[name]:
                continue
            s = STYLE[name]
            results = sorted(data[name], key=lambda r: r["k"])
            ks = [r["k"] for r in results]
            if denom_key:
                vals = [r[key] / r[denom_key] if r[denom_key] > 0.01 else 0
                        for r in results]
            else:
                vals = [r[key] for r in results]
            ax.plot(ks, vals, marker=s["marker"], ls=s["ls"],
                    color=s["color"], lw=2, ms=7, label=name)
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5, lw=1)
        ax.set(xlabel="k", ylabel="Hidden / Emission ratio", title=title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"{SUPTITLE}: hidden/emission ratios (>1 = global wins)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, "exp1c3n_ratios.png"))
    fig.savefig(os.path.join(OUT_DIR, "exp1c3n_ratios.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  exp1c3n_ratios")

    # ── Plot 5: Scatter — hidden vs emission (all three metrics side by side) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    scatter_pairs = [
        ("emission_auc", "hidden_auc", "Local AUC", "Global AUC", "AUC"),
        ("emission_corr", "hidden_corr", "Emission corr", "Hidden corr",
         "Single-latent correlation"),
        ("emission_r2", "hidden_r2", r"Emission $R^2$", r"Hidden $R^2$",
         "Linear probe"),
    ]
    for ai, (xkey, ykey, xlabel, ylabel, title) in enumerate(scatter_pairs):
        ax = axes[ai]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.15, lw=1)
        for name in models:
            if name not in STYLE or not data[name]:
                continue
            s = STYLE[name]
            xs = [r[xkey] for r in data[name]]
            ys = [r[ykey] for r in data[name]]
            ax.scatter(xs, ys, color=s["color"], marker=s["marker"],
                       s=60, alpha=0.7, label=name, zorder=4,
                       edgecolors="white", linewidths=0.5)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"{SUPTITLE}: hidden vs emission (above diagonal = global wins)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(OUT_DIR, "exp1c3n_scatter.png"))
    fig.savefig(os.path.join(OUT_DIR, "exp1c3n_scatter.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  exp1c3n_scatter")

    # ── Plot 6: gAUC vs T at fixed k ──
    k_fixed = [1, 2, 3, 5]
    fig, axes = plt.subplots(1, len(k_fixed), figsize=(5 * len(k_fixed), 5), sharey=True)
    for ki, k in enumerate(k_fixed):
        ax = axes[ki]
        Ts, gaucs = [], []
        for T in TXCDR_T_VALUES:
            nm = f"TXCDRv2-T{T}"
            r = next((r for r in data.get(nm, []) if r["k"] == k), None)
            if r:
                Ts.append(T)
                gaucs.append(r["hidden_auc"])
        if Ts:
            ax.plot(Ts, gaucs, "o-", color="#d62728", lw=2, ms=8, label="TXCDR gAUC")
        stk = next((r for r in data.get("Stacked-T2", []) if r["k"] == k), None)
        if stk:
            ax.axhline(stk["hidden_auc"], color="#9467bd", ls="--", alpha=0.6,
                       lw=1.5, label=f"Stacked T=2 ({stk['hidden_auc']:.2f})")
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
    save_figure(fig, os.path.join(OUT_DIR, "exp1c3n_gauc_vs_T.png"))
    fig.savefig(os.path.join(OUT_DIR, "exp1c3n_gauc_vs_T.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  exp1c3n_gauc_vs_T")

    print(f"Done. All plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
