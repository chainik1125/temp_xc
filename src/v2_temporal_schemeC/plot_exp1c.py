"""Generate all Experiment 1c plots with consistent styling.

Reads from:
  results/experiment1c_noisy/results.json          (main experiment)
  results/experiment1c_noisy/linear_probe_results.json  (linear probe)
  results/reproduction/*.json                       (γ=1 overlay)

Outputs 5 figures to results/experiment1c_noisy/:
  exp1c_topk_auc.png       — all models: NMSE, AUC, NMSE-vs-AUC
  exp1c_windowed_only.png  — windowed models only
  exp1c_denoising.png      — single-latent correlation (local, global, ratio vs k)
  exp1c_linear_probe.png   — linear probe R² (local, global, ratio vs k)
  exp1c_probe_scatter.png  — both methods: global vs local scatter

Usage:
  PYTHONPATH=/home/elysium/temp_xc python src/v2_temporal_schemeC/plot_exp1c.py
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
OUT_DIR = EXP1C_DIR

# ── Style definitions ──

MODELS = [
    ("TFA-pos",     "TFA-pos",       "#2ca02c", "X", "-"),
    ("Stacked-T2",  "Stacked T=2",   "#9467bd", "o", "-"),
    ("Stacked-T5",  "Stacked T=5",   "#9467bd", "^", "--"),
    ("TXCDRv2-T2",  "TXCDRv2 T=2",   "#e377c2", "o", "-"),
    ("TXCDRv2-T5",  "TXCDRv2 T=5",   "#e377c2", "^", "--"),
]

EXP1_OVERLAY = [
    ("TFA-pos",     "#2ca02c", "X"),
    ("Stacked-T2",  "#9467bd", "o"),
    ("Stacked-T5",  "#9467bd", "^"),
    ("TXCDRv2-T2",  "#e377c2", "o"),
    ("TXCDRv2-T5",  "#e377c2", "^"),
]

SUPTITLE_PREFIX = r"Experiment 1c ($\gamma = 0.25$)"


# ── Data loading ──

def load_exp1c():
    with open(os.path.join(EXP1C_DIR, "results.json")) as f:
        return json.load(f)


def load_linear_probe():
    path = os.path.join(EXP1C_DIR, "linear_probe_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_exp1_model(name):
    path = os.path.join(REPRO_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, name):
    for ext in ["png", "pdf"]:
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        if ext == "png":
            save_figure(fig, path)
        else:
            fig.savefig(path, bbox_inches="tight")


# ── Plot 1: All models — NMSE, AUC, NMSE-vs-AUC ──

def plot_topk_auc(models_data, exp1):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for key, display, c, m, ls in MODELS:
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
    axes[1].set(xlabel="k", ylabel="AUC", title="Feature recovery AUC vs k")
    axes[2].set(xlabel="NMSE", ylabel="AUC", title="NMSE vs AUC", xscale="log")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{SUPTITLE_PREFIX}: NMSE and AUC "
                 r"(dotted = $\gamma = 1$)", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "exp1c_topk_auc")
    plt.close(fig)
    print("  exp1c_topk_auc")


# ── Plot 2: Windowed models only ──

def plot_windowed_only(models_data, exp1):
    windowed = [m for m in MODELS if m[0] != "TFA-pos"]
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
    axes[1].set(xlabel="k", ylabel="AUC", title="Feature recovery AUC vs k")
    axes[2].set(xlabel="NMSE", ylabel="AUC", title="NMSE vs AUC", xscale="log")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{SUPTITLE_PREFIX}: windowed models only "
                 r"(dotted = $\gamma = 1$)", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "exp1c_windowed_only")
    plt.close(fig)
    print("  exp1c_windowed_only")


# ── Plot 3: Single-latent correlation (3 panels: local, global, ratio vs k) ──

def plot_denoising(models_data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for key, display, c, m, ls in MODELS:
        if key not in models_data or not models_data[key]:
            continue
        results = models_data[key]
        ks, locals_, globals_, ratios = [], [], [], []
        for r in results:
            loc = r["mean_local_corr"]
            glob = r["mean_global_corr"]
            if loc > 0.01:
                ks.append(r["k"])
                locals_.append(loc)
                globals_.append(glob)
                ratios.append(glob / loc)

        axes[0].plot(ks, locals_, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)
        axes[1].plot(ks, globals_, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)
        axes[2].plot(ks, ratios, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)

    axes[0].set(xlabel="k", ylabel="Pearson correlation",
                title=r"Local: corr($z_j$, $s_i$) vs k")
    axes[1].set(xlabel="k", ylabel="Pearson correlation",
                title=r"Global: corr($z_j$, $h_i$) vs k")
    axes[2].set(xlabel="k", ylabel="Global / Local ratio",
                title="Denoising ratio vs k")
    axes[2].axhline(1.0, color="gray", ls="--", alpha=0.5, lw=1)
    axes[2].axhline(0.5, color="gray", ls=":", alpha=0.4, lw=1)

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{SUPTITLE_PREFIX}: single-latent correlation",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "exp1c_denoising")
    plt.close(fig)
    print("  exp1c_denoising")


# ── Plot 4: Linear probe R² (3 panels: local, global, ratio vs k) ──

def plot_linear_probe(probe_data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for key, display, c, m, ls in MODELS:
        if key not in probe_data or not probe_data[key]:
            continue
        results = probe_data[key]
        ks = [r["k"] for r in results]
        local_r2 = [r["mean_local_r2"] for r in results]
        global_r2 = [r["mean_global_r2"] for r in results]
        ratios = [r["ratio"] for r in results]

        axes[0].plot(ks, local_r2, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)
        axes[1].plot(ks, global_r2, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)
        axes[2].plot(ks, ratios, marker=m, linestyle=ls, color=c, lw=2, ms=7,
                     label=display)

    axes[0].set(xlabel="k", ylabel=r"$R^2$",
                title=r"Local probe: $z \to s_i$ vs k")
    axes[1].set(xlabel="k", ylabel=r"$R^2$",
                title=r"Global probe: $z \to h_i$ vs k")
    axes[2].set(xlabel="k", ylabel="Global / Local R² ratio",
                title="Denoising ratio vs k")
    axes[2].axhline(1.0, color="gray", ls="--", alpha=0.5, lw=1)
    axes[2].axhline(0.25, color="gray", ls=":", alpha=0.4, lw=1)

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{SUPTITLE_PREFIX}: linear probe",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "exp1c_linear_probe")
    plt.close(fig)
    print("  exp1c_linear_probe")


# ── Plot 5: Combined scatter (single-latent + linear probe side by side) ──

def plot_scatter(models_data, probe_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: single-latent correlation
    ax = axes[0]
    ax.plot([0, 0.6], [0, 0.6], "k--", alpha=0.3, lw=1)
    ax.plot([0, 0.6], [0, 0.3], "k:", alpha=0.2, lw=1)  # ratio=0.5 line
    for key, display, c, m, ls in MODELS:
        if key not in models_data or not models_data[key]:
            continue
        results = models_data[key]
        locals_ = [r["mean_local_corr"] for r in results if r["mean_local_corr"] > 0.01]
        globals_ = [r["mean_global_corr"] for r in results if r["mean_local_corr"] > 0.01]
        ax.scatter(locals_, globals_, color=c, marker=m, s=60, alpha=0.7,
                   label=display, zorder=5)
    ax.set(xlabel="Local corr (vs noisy obs)",
           ylabel="Global corr (vs hidden state)",
           title="Single-latent correlation")
    ax.set_xlim(-0.02, 0.6)
    ax.set_ylim(-0.02, 0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: linear probe R²
    ax = axes[1]
    ax.plot([0, 0.55], [0, 0.55], "k--", alpha=0.3, lw=1)
    ax.plot([0, 0.55], [0, 0.55 * 0.25], "k:", alpha=0.2, lw=1)  # ratio=0.25 line
    if probe_data is not None:
        for key, display, c, m, ls in MODELS:
            if key not in probe_data or not probe_data[key]:
                continue
            results = probe_data[key]
            local_r2 = [r["mean_local_r2"] for r in results]
            global_r2 = [r["mean_global_r2"] for r in results]
            ax.scatter(local_r2, global_r2, color=c, marker=m, s=60, alpha=0.7,
                       label=display, zorder=5)
    ax.set(xlabel=r"Local $R^2$ ($z \to s_i$)",
           ylabel=r"Global $R^2$ ($z \to h_i$)",
           title="Linear probe")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.02, 0.25)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{SUPTITLE_PREFIX}: global vs local feature recovery",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "exp1c_probe_scatter")
    plt.close(fig)
    print("  exp1c_probe_scatter")


# ── Main ──

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_exp1c()
    models_data = data["models"]
    probe_data = load_linear_probe()

    exp1 = {}
    for name, _, _ in EXP1_OVERLAY:
        d = load_exp1_model(name)
        if d:
            exp1[name] = d["topk"]

    print("Generating Experiment 1c plots:")
    plot_topk_auc(models_data, exp1)
    plot_windowed_only(models_data, exp1)
    plot_denoising(models_data)
    if probe_data is not None:
        plot_linear_probe(probe_data)
    plot_scatter(models_data, probe_data)
    print(f"Done. All plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
