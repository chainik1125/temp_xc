"""Generate Experiment 1c plots in the same style as plot_exp1_exp2.py.

Reads from results/experiment1c_noisy/results.json and overlays
γ=1 Experiment 1 curves from results/reproduction/*.json.

Usage:
  TQDM_DISABLE=1 PYTHONPATH=/home/elysium/temp_xc \
    python src/v2_temporal_schemeC/plot_exp1c.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.plot import save_figure

BASE = os.path.dirname(__file__)
EXP1C_DIR = os.path.join(BASE, "results", "experiment1c_noisy")
REPRO_DIR = os.path.join(BASE, "results", "reproduction")
OUT_DIR = EXP1C_DIR  # plots go alongside results

# Experiment 1c models: (results_key, display_name, color, marker, linestyle)
# Colors chosen to match Experiment 1 conventions where possible
EXP1C_MODELS = [
    ("TFA-pos",     "TFA-pos",       "#2ca02c", "X", "-"),    # green
    ("Stacked-T2",  "Stacked T=2",   "#9467bd", "o", "-"),    # purple solid
    ("Stacked-T5",  "Stacked T=5",   "#9467bd", "^", "--"),   # purple dashed
    ("TXCDRv2-T2",  "TXCDRv2 T=2",   "#e377c2", "o", "-"),   # pink solid
    ("TXCDRv2-T5",  "TXCDRv2 T=5",   "#e377c2", "^", "--"),  # pink dashed
]

# γ=1 overlay: same models from Experiment 1 reproduction
EXP1_OVERLAY = [
    ("TFA-pos",     "#2ca02c", "X"),
    ("Stacked-T2",  "#9467bd", "o"),
    ("Stacked-T5",  "#9467bd", "^"),
    ("TXCDRv2-T2",  "#e377c2", "o"),
    ("TXCDRv2-T5",  "#e377c2", "^"),
]


def load_exp1c():
    path = os.path.join(EXP1C_DIR, "results.json")
    with open(path) as f:
        return json.load(f)


def load_exp1_model(name):
    path = os.path.join(REPRO_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_exp1c()
    models_data = data["models"]

    # Load γ=1 overlay
    exp1 = {}
    for name, _, _ in EXP1_OVERLAY:
        d = load_exp1_model(name)
        if d:
            exp1[name] = d["topk"]

    # ══════════════════════════════════════════════════════════════════
    # Plot 1: All models — NMSE vs k, AUC vs k, NMSE vs AUC
    # ══════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for key, display, c, m, ls in EXP1C_MODELS:
        if key not in models_data or not models_data[key]:
            continue
        results = models_data[key]
        ks = [r["k"] for r in results]
        nmse = [r["nmse"] for r in results]
        auc = [r["auc"] for r in results]

        axes[0].plot(ks, nmse, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)
        axes[1].plot(ks, auc, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)
        axes[2].plot(nmse, auc, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)

    # γ=1 overlay (thin dotted)
    for name, c, m in EXP1_OVERLAY:
        if name not in exp1:
            continue
        topk = exp1[name]
        ks = [r["k"] for r in topk]
        nmse = [r["nmse"] for r in topk]
        auc = [r["auc"] for r in topk]
        axes[0].plot(ks, nmse, marker=m, linestyle=":", color=c, lw=1, ms=4,
                     alpha=0.4)
        axes[1].plot(ks, auc, marker=m, linestyle=":", color=c, lw=1, ms=4,
                     alpha=0.4)

    axes[0].set(xlabel="k", ylabel="NMSE", title="NMSE vs k", yscale="log")
    axes[1].set(xlabel="k", ylabel="AUC", title="Feature Recovery AUC vs k")
    axes[2].set(xlabel="NMSE", ylabel="AUC", title="NMSE vs AUC", xscale="log")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(r"Experiment 1c: TopK sweep ($\gamma = 0.25$, dotted = $\gamma = 1$)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(OUT_DIR, f"exp1c_topk_auc.{ext}")
        if ext == "png":
            save_figure(fig, path)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Plot 1 (all models) saved.")

    # ══════════════════════════════════════════════════════════════════
    # Plot 2: Windowed models only (Stacked vs TXCDR)
    # ══════════════════════════════════════════════════════════════════

    windowed = [m for m in EXP1C_MODELS if m[0] != "TFA-pos"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for key, display, c, m, ls in windowed:
        if key not in models_data or not models_data[key]:
            continue
        results = models_data[key]
        ks = [r["k"] for r in results]
        nmse = [r["nmse"] for r in results]
        auc = [r["auc"] for r in results]

        axes[0].plot(ks, nmse, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)
        axes[1].plot(ks, auc, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)
        axes[2].plot(nmse, auc, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)

    # γ=1 overlay
    for name, c, m in EXP1_OVERLAY:
        if name == "TFA-pos" or name not in exp1:
            continue
        topk = exp1[name]
        ks = [r["k"] for r in topk]
        nmse = [r["nmse"] for r in topk]
        auc = [r["auc"] for r in topk]
        axes[0].plot(ks, nmse, marker=m, linestyle=":", color=c, lw=1, ms=4,
                     alpha=0.4)
        axes[1].plot(ks, auc, marker=m, linestyle=":", color=c, lw=1, ms=4,
                     alpha=0.4)

    axes[0].set(xlabel="k", ylabel="NMSE", title="NMSE vs k", yscale="log")
    axes[1].set(xlabel="k", ylabel="AUC", title="Feature Recovery AUC vs k")
    axes[2].set(xlabel="NMSE", ylabel="AUC", title="NMSE vs AUC", xscale="log")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(r"Windowed models: Stacked SAE vs TXCDRv2 ($\gamma = 0.25$, dotted = $\gamma = 1$)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(OUT_DIR, f"exp1c_windowed_only.{ext}")
        if ext == "png":
            save_figure(fig, path)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Plot 2 (windowed only) saved.")

    # ══════════════════════════════════════════════════════════════════
    # Plot 3: Denoising — global/local ratio vs k
    # ══════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for key, display, c, m, ls in EXP1C_MODELS:
        if key not in models_data or not models_data[key]:
            continue
        results = models_data[key]
        ks, ratios, globals_, locals_ = [], [], [], []
        for r in results:
            loc = r["mean_local_corr"]
            glob = r["mean_global_corr"]
            if loc > 0.01:
                ks.append(r["k"])
                ratios.append(glob / loc)
                globals_.append(glob)
                locals_.append(loc)

        axes[0].plot(ks, ratios, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)
        axes[1].scatter(locals_, globals_, color=c, marker=m, s=60, alpha=0.7,
                        label=display, zorder=5)

    axes[0].axhline(1.0, color="gray", ls="--", alpha=0.5, lw=1)
    axes[0].axhline(0.5, color="gray", ls=":", alpha=0.4, lw=1)
    axes[0].set(xlabel="k", ylabel="Global / Local correlation ratio",
                title="Denoising ratio vs k\n(>1 = model denoises, 0.5 = no denoising)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Diagonal line for scatter
    axes[1].plot([0, 0.6], [0, 0.6], "k--", alpha=0.3, lw=1)
    axes[1].plot([0, 0.6], [0, 0.3], "k:", alpha=0.2, lw=1)  # ratio=0.5 line
    axes[1].set(xlabel="Local corr (vs noisy observation)",
                ylabel="Global corr (vs hidden state)",
                title="Global vs Local feature correlation\n(above diagonal = denoising)")
    axes[1].set_xlim(-0.02, 0.6)
    axes[1].set_ylim(-0.02, 0.5)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(r"Experiment 1c: Denoising analysis ($\gamma = 0.25$)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(OUT_DIR, f"exp1c_denoising.{ext}")
        if ext == "png":
            save_figure(fig, path)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Plot 3 (denoising) saved.")

    print(f"\nAll plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
