"""Generate Experiment 1 and 2 plots from reproduction results.

Reads ONLY from results/reproduction/*.json (unified schema).

Usage:
  TQDM_DISABLE=1 python src/v2_temporal_schemeC/plot_exp1_exp2.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotting.save_figure import save_figure

BASE = os.path.dirname(__file__)
REPRO_DIR = os.path.join(BASE, "results", "reproduction")
OUT_DIR = os.path.join(BASE, "results", "auc_and_crosscoder")

# Model display config: (json_filename, display_name, color, marker, linestyle)
# Convention: same method = same color, T=2 = solid/circle, T=5 = dashed/triangle
MODELS = [
    # Per-token models (no window)
    ("SAE",           "Shared SAE",    "#1f77b4", "o", "-"),    # blue
    ("TFA",           "TFA",           "#ff7f0e", "s", "-"),    # orange
    ("TFA-shuf",      "TFA-shuf",      "#ff7f0e", "s", "--"),   # orange dashed
    ("TFA-pos",       "TFA-pos",       "#2ca02c", "X", "-"),    # green
    ("TFA-pos-shuf",  "TFA-pos-shuf",  "#2ca02c", "X", "--"),   # green dashed
    # Stacked SAE (same color, different marker/style per T)
    ("Stacked-T2",    "Stacked T=2",   "#9467bd", "o", "-"),    # purple solid
    ("Stacked-T5",    "Stacked T=5",   "#9467bd", "^", "--"),   # purple dashed
    # TXCDR original (same color, different marker/style per T)
    ("TXCDR-T2",      "TXCDR T=2",     "#d62728", "o", "-"),    # red solid
    ("TXCDR-T5",      "TXCDR T=5",     "#d62728", "^", "--"),   # red dashed
    # TXCDRv2 fair (same color, different marker/style per T)
    ("TXCDRv2-T2",    "TXCDRv2 T=2",   "#e377c2", "o", "-"),   # pink solid
    ("TXCDRv2-T5",    "TXCDRv2 T=5",   "#e377c2", "^", "--"),  # pink dashed
]


def load_model(filename):
    path = os.path.join(REPRO_DIR, f"{filename}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pareto_min(xs, ys):
    pts = sorted(zip(xs, ys))
    fx, fy = [], []
    best = float("inf")
    for x, y in pts:
        if y < best:
            fx.append(x); fy.append(y); best = y
    return fx, fy


def pareto_max(xs, ys):
    pts = sorted(zip(xs, ys))
    fx, fy = [], []
    best = -float("inf")
    for x, y in pts:
        if y > best:
            fx.append(x); fy.append(y); best = y
    return fx, fy


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all models
    data = {}
    for filename, display, c, m, ls in MODELS:
        d = load_model(filename)
        if d:
            data[filename] = d
            print(f"  Loaded {filename}: {len(d['topk'])} topk, {len(d['l1'])} l1")
        else:
            print(f"  MISSING: {filename}")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: TopK sweep
    # ══════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for filename, display, c, m, ls in MODELS:
        if filename not in data:
            continue
        topk = data[filename]["topk"]
        ks = [r["k"] for r in topk]
        nmse = [r["nmse"] for r in topk]
        auc = [r["auc"] for r in topk]

        axes[0].plot(ks, nmse, marker=m, linestyle=ls, color=c, lw=2, ms=7, label=display)
        axes[1].plot(ks, auc, marker=m, linestyle=ls, color=c, lw=2, ms=7, label=display)
        axes[2].plot(nmse, auc, marker=m, linestyle=ls, color=c, lw=2, ms=7, label=display)

    axes[0].set(xlabel="k", ylabel="NMSE", title="NMSE vs k", yscale="log")
    axes[1].set(xlabel="k", ylabel="AUC", title="Feature Recovery AUC vs k")
    axes[2].set(xlabel="NMSE", ylabel="AUC", title="NMSE vs AUC", xscale="log")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Experiment 1: TopK sweep", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(OUT_DIR, f"exp1_topk_auc.{ext}")
        if ext == "png":
            save_figure(fig, path)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Experiment 1 plots saved.")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 1 WINDOWED-ONLY: Stacked SAE vs TXCDR vs TXCDRv2
    # ══════════════════════════════════════════════════════════════════

    windowed = [m for m in MODELS if m[0].startswith(("Stacked", "TXCDR"))]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for filename, display, c, m, ls in windowed:
        if filename not in data:
            continue
        topk = data[filename]["topk"]
        ks = [r["k"] for r in topk]
        nmse = [r["nmse"] for r in topk]
        auc = [r["auc"] for r in topk]

        axes[0].plot(ks, nmse, marker=m, linestyle=ls, color=c, lw=2, ms=7, label=display)
        axes[1].plot(ks, auc, marker=m, linestyle=ls, color=c, lw=2, ms=7, label=display)
        axes[2].plot(nmse, auc, marker=m, linestyle=ls, color=c, lw=2, ms=7, label=display)

    axes[0].set(xlabel="k", ylabel="NMSE", title="NMSE vs k", yscale="log")
    axes[1].set(xlabel="k", ylabel="AUC", title="Feature Recovery AUC vs k")
    axes[2].set(xlabel="NMSE", ylabel="AUC", title="NMSE vs AUC", xscale="log")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Experiment 1: Windowed models (decoder-averaged AUC)", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(OUT_DIR, f"exp1_windowed_only.{ext}")
        if ext == "png":
            save_figure(fig, path)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Experiment 1 windowed-only plots saved.")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: ReLU+L1 Pareto
    # ══════════════════════════════════════════════════════════════════

    # Models with L1 data (exclude shuffled variants)
    l1_models = [(f, d, c, m, ls) for f, d, c, m, ls in MODELS
                 if f in data and len(data[f]["l1"]) > 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for filename, display, c, m, ls in l1_models:
        l1_data = data[filename]["l1"]
        novel_l0 = [r["novel_l0"] for r in l1_data]
        total_l0 = [r["total_l0"] for r in l1_data]
        nmse = [r["nmse"] for r in l1_data]
        auc = [r["auc"] for r in l1_data]

        # Novel L0 (solid)
        ax = axes[0]
        ax.scatter(novel_l0, nmse, color=c, alpha=0.3, s=30)
        fx, fy = pareto_min(novel_l0, nmse)
        ax.plot(fx, fy, marker=m, linestyle="-", color=c, lw=2, ms=7, label=f"{display} (novel)")

        # Total L0 (dashed) — only for TFA variants that have pred_l0 > 0
        if any(r["pred_l0"] > 0 for r in l1_data):
            fx, fy = pareto_min(total_l0, nmse)
            ax.plot(fx, fy, marker=m, linestyle="--", color=c, lw=1.5, ms=5, alpha=0.7,
                    label=f"{display} (total)")

        # L0 vs AUC
        ax = axes[1]
        ax.scatter(novel_l0, auc, color=c, alpha=0.3, s=30)
        fx, fy = pareto_max(novel_l0, auc)
        ax.plot(fx, fy, marker=m, linestyle="-", color=c, lw=2, ms=7, label=f"{display} (novel)")
        if any(r["pred_l0"] > 0 for r in l1_data):
            fx, fy = pareto_max(total_l0, auc)
            ax.plot(fx, fy, marker=m, linestyle="--", color=c, lw=1.5, ms=5, alpha=0.7,
                    label=f"{display} (total)")

        # NMSE vs AUC
        ax = axes[2]
        pts = sorted(zip(nmse, auc))
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=m, linestyle="-", color=c, lw=1.5, ms=6, alpha=0.8, label=display)

    axes[0].set(xlabel="L0", ylabel="NMSE", title="L0 vs NMSE Pareto", yscale="log")
    axes[1].set(xlabel="L0", ylabel="AUC", title="L0 vs AUC Pareto")
    axes[2].set(xlabel="NMSE", ylabel="AUC", title="NMSE vs AUC", xscale="log")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Experiment 2: ReLU+L1 Pareto", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(OUT_DIR, f"exp2_pareto_auc.{ext}")
        if ext == "png":
            save_figure(fig, path)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Experiment 2 plots saved.")


if __name__ == "__main__":
    main()
