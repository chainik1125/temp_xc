#!/usr/bin/env python3
"""
viz.py — Post-sweep visualization for v8 (stacked SAE baseline).

Produces:
  1. 3-way trade-off scatter: L0 vs Reconstruction Loss, colored by AUC
  2. TXCDR advantage heatmap: AUC(TXCDR) - AUC(StackedSAE) per (k, T) at each rho
  3. Advantage vs correlation: how TXCDR advantage grows with rho
  4. Convergence curves: AUC over training steps
  5. IID vs Markov comparison panel
  6. Delta loss heatmaps: reconstruction loss difference per (k, T) at each rho
  7. k analysis: how k affects AUC and loss for each model/rho
  8. Summary table

Usage:
    python viz.py                     # read from logs/
    python viz.py --log-dir my_logs   # custom log dir
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def load_all_logs(log_dir: str) -> dict:
    """
    Load all JSON logs into a structured dict.

    Returns:
        {(model, rho, k, T): [{"step": ..., "auc": ..., "loss": ..., "window_l0": ...}, ...]}
    """
    results = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        basename = os.path.basename(path).replace(".json", "")
        if basename == "sweep_summary":
            continue
        # Naming: {model}__{rho_label}__k{k}__T{T}
        parts = basename.split("__")
        if len(parts) != 4:
            continue
        model = parts[0]
        # Parse rho from e.g. "rho0p9"
        rho_match = re.match(r"rho(\d+)p(\d+)", parts[1])
        if not rho_match:
            continue
        rho = float(f"{rho_match.group(1)}.{rho_match.group(2)}")
        k = int(parts[2].replace("k", ""))
        T = int(parts[3].replace("T", ""))

        with open(path) as f:
            history = json.load(f)
        results[(model, rho, k, T)] = history
    return results


def get_final(history: list[dict], key: str, default=0.0):
    """Last value of a metric from a training history."""
    if not history:
        return default
    return history[-1].get(key, default)


# ─── 1. 3-way trade-off scatter ─────────────────────────────────────────────────

def plot_tradeoff_scatter(results: dict, viz_dir: str):
    """
    Scatter: Window L0 (x) vs Reconstruction Loss (y), color = AUC.
    One panel per rho, markers distinguish model type.
    """
    rhos = sorted(set(r for _, r, _, _ in results.keys()))
    n_rho = len(rhos)
    fig, axes = plt.subplots(1, n_rho, figsize=(5 * n_rho, 5), squeeze=False)

    for col, rho in enumerate(rhos):
        ax = axes[0, col]
        for model_type, marker, label in [
            ("stacked_sae", "s", "Stacked SAE"),
            ("txcdr", "^", "TXCDR"),
        ]:
            l0s, losses, aucs, ks_plot = [], [], [], []
            for (m, r, k, T), hist in results.items():
                if m != model_type or r != rho:
                    continue
                l0s.append(get_final(hist, "window_l0"))
                losses.append(get_final(hist, "loss"))
                aucs.append(get_final(hist, "auc"))
                ks_plot.append(k)

            if not l0s:
                continue
            sc = ax.scatter(l0s, losses, c=aucs, marker=marker, s=80,
                           cmap="viridis", vmin=0, vmax=1, edgecolors="k",
                           linewidths=0.5, label=label, zorder=5)
            for x, y, k_val in zip(l0s, losses, ks_plot):
                ax.annotate(f"k={k_val}", (x, y), fontsize=6,
                           textcoords="offset points", xytext=(4, 4))

        ax.set_xlabel("Window L0 (total active latents)")
        ax.set_ylabel("Reconstruction Loss")
        rho_label = "IID" if rho == 0.0 else f"Markov ρ={rho}"
        ax.set_title(f"{rho_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("3-Way Trade-off: Sparsity vs Loss (color = AUC)", fontsize=13, y=1.02)
    cbar = plt.colorbar(sc, ax=axes.ravel().tolist(), label="AUC", shrink=0.8, pad=0.04)
    plt.tight_layout()
    path = os.path.join(viz_dir, "tradeoff_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 2. Advantage heatmap per rho ───────────────────────────────────────────────

def plot_advantage_heatmaps(results: dict, viz_dir: str):
    """
    For each rho: k×T heatmap of AUC(TXCDR) - AUC(StackedSAE).
    """
    rhos = sorted(set(r for _, r, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    n_rho = len(rhos)
    fig, axes = plt.subplots(1, n_rho, figsize=(5 * n_rho, 4), squeeze=False)

    for col, rho in enumerate(rhos):
        ax = axes[0, col]
        mat = np.full((len(ks), len(Ts)), np.nan)
        for i, k in enumerate(ks):
            for j, T in enumerate(Ts):
                txcdr_key = ("txcdr", rho, k, T)
                sae_key = ("stacked_sae", rho, k, T)
                if txcdr_key in results and sae_key in results:
                    diff = get_final(results[txcdr_key], "auc") - get_final(results[sae_key], "auc")
                    mat[i, j] = diff

        vabs = max(abs(np.nanmin(mat)) if not np.all(np.isnan(mat)) else 0.01,
                   abs(np.nanmax(mat)) if not np.all(np.isnan(mat)) else 0.01,
                   0.01)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")

        for i in range(len(ks)):
            for j in range(len(Ts)):
                val = mat[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vabs * 0.6 else "black"
                    ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)
                else:
                    ax.text(j, i, "skip", ha="center", va="center",
                            fontsize=8, color="gray")

        ax.set_xticks(range(len(Ts)))
        ax.set_xticklabels([str(t) for t in Ts])
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([str(k) for k in ks])
        ax.set_xlabel("T")
        ax.set_ylabel("k")
        rho_label = "IID" if rho == 0.0 else f"Markov ρ={rho}"
        ax.set_title(f"ΔAUC — {rho_label}")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label="ΔAUC (TXCDR − Stacked SAE)",
                        shrink=0.8, pad=0.08)
    fig.suptitle("TXCDR Advantage over Stacked SAE (positive = TXCDR wins)", fontsize=12, y=1.02)
    path = os.path.join(viz_dir, "advantage_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 3. Advantage vs correlation level ──────────────────────────────────────────

def plot_advantage_vs_rho(results: dict, viz_dir: str):
    """
    Line plot: mean TXCDR advantage (AUC) vs rho, one line per (k, T).
    Shows how temporal correlation benefits the crosscoder.
    """
    rhos = sorted(set(r for _, r, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(ks) * len(Ts)))
    idx = 0
    for T in Ts:
        for k in ks:
            advantages = []
            valid_rhos = []
            for rho in rhos:
                txcdr_key = ("txcdr", rho, k, T)
                sae_key = ("stacked_sae", rho, k, T)
                if txcdr_key in results and sae_key in results:
                    diff = get_final(results[txcdr_key], "auc") - get_final(results[sae_key], "auc")
                    advantages.append(diff)
                    valid_rhos.append(rho)
            if valid_rhos:
                ax.plot(valid_rhos, advantages, "o-", color=colors[idx],
                       label=f"k={k}, T={T}", lw=1.5, ms=5)
            idx += 1

    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("ρ (lag-1 autocorrelation)", fontsize=12)
    ax.set_ylabel("ΔAUC (TXCDR − Stacked SAE)", fontsize=12)
    ax.set_title("TXCDR advantage vs temporal correlation", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(viz_dir, "advantage_vs_rho.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 4. Convergence curves ──────────────────────────────────────────────────────

def plot_convergence_curves(results: dict, viz_dir: str):
    """AUC convergence for each (rho, k) with Stacked SAE vs TXCDR(T) overlaid."""
    rhos = sorted(set(r for _, r, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    colors_model = {"stacked_sae": "steelblue", "txcdr": "firebrick"}

    for rho in rhos:
        rho_label = "iid" if rho == 0.0 else f"rho{rho}"
        fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4), squeeze=False)
        for col, k in enumerate(ks):
            ax = axes[0, col]
            for T in Ts:
                for model_type in ["stacked_sae", "txcdr"]:
                    key = (model_type, rho, k, T)
                    if key not in results:
                        continue
                    hist = results[key]
                    steps = [h["step"] for h in hist]
                    aucs = [h["auc"] for h in hist]
                    ls = "--" if model_type == "stacked_sae" else "-"
                    label_name = "StackedSAE" if model_type == "stacked_sae" else "TXCDR"
                    ax.plot(steps, aucs, color=colors_model[model_type],
                           lw=1.5, ls=ls, alpha=0.5 + 0.5 * (T / max(Ts)),
                           label=f"{label_name} T={T}")

            ax.set_title(f"k={k}")
            ax.set_xlabel("Step")
            if col == 0:
                ax.set_ylabel("AUC")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=6, loc="lower right")
            ax.grid(True, alpha=0.3)

        title_label = "IID (ρ=0)" if rho == 0.0 else f"Markov (ρ={rho})"
        fig.suptitle(f"Convergence — {title_label}", fontsize=12)
        plt.tight_layout()
        path = os.path.join(viz_dir, f"convergence_{rho_label}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ─── 5. IID vs Markov comparison panel ──────────────────────────────────────────

def plot_iid_vs_markov(results: dict, viz_dir: str):
    """
    Side-by-side comparison: IID (rho=0) vs highest rho.
    Shows raw AUC heatmaps for both models.
    """
    rhos = sorted(set(r for _, r, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    if 0.0 not in rhos or len(rhos) < 2:
        return

    rho_iid = 0.0
    rho_markov = max(r for r in rhos if r > 0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, (rho, rho_name) in enumerate([(rho_iid, "IID (ρ=0)"), (rho_markov, f"Markov (ρ={rho_markov})")]):
        for col_idx, (model_type, model_name) in enumerate([("stacked_sae", "Stacked SAE"), ("txcdr", "TXCDR")]):
            ax = axes[row, col_idx]
            mat = np.full((len(ks), len(Ts)), np.nan)
            for i, k in enumerate(ks):
                for j, T in enumerate(Ts):
                    key = (model_type, rho, k, T)
                    if key in results:
                        mat[i, j] = get_final(results[key], "auc")

            im = ax.imshow(mat, cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=1)
            for i in range(len(ks)):
                for j in range(len(Ts)):
                    val = mat[i, j]
                    if not np.isnan(val):
                        color = "white" if val < 0.5 else "black"
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                                fontsize=8, color=color)

            ax.set_xticks(range(len(Ts)))
            ax.set_xticklabels([str(t) for t in Ts])
            ax.set_yticks(range(len(ks)))
            ax.set_yticklabels([str(k) for k in ks])
            ax.set_xlabel("T")
            ax.set_ylabel("k")
            ax.set_title(f"{model_name} — {rho_name}")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label="AUC", shrink=0.6, pad=0.08)
    fig.suptitle("IID vs Markov: Feature Recovery (AUC)", fontsize=13, y=1.01)
    path = os.path.join(viz_dir, "iid_vs_markov.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 6. Delta loss heatmaps ───────────────────────────────────────────────────

def plot_delta_loss_heatmaps(results: dict, viz_dir: str):
    """
    For each rho: k×T heatmap of Loss(TXCDR) - Loss(StackedSAE).
    Negative means TXCDR achieves lower reconstruction loss.
    """
    rhos = sorted(set(r for _, r, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    n_rho = len(rhos)
    fig, axes = plt.subplots(1, n_rho, figsize=(5 * n_rho, 4), squeeze=False)

    for col, rho in enumerate(rhos):
        ax = axes[0, col]
        mat = np.full((len(ks), len(Ts)), np.nan)
        for i, k in enumerate(ks):
            for j, T in enumerate(Ts):
                txcdr_key = ("txcdr", rho, k, T)
                sae_key = ("stacked_sae", rho, k, T)
                if txcdr_key in results and sae_key in results:
                    diff = get_final(results[txcdr_key], "loss") - get_final(results[sae_key], "loss")
                    mat[i, j] = diff

        vabs = max(abs(np.nanmin(mat)) if not np.all(np.isnan(mat)) else 0.01,
                   abs(np.nanmax(mat)) if not np.all(np.isnan(mat)) else 0.01,
                   0.01)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax.imshow(mat, cmap="RdBu", norm=norm, aspect="auto", origin="lower")

        for i in range(len(ks)):
            for j in range(len(Ts)):
                val = mat[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vabs * 0.6 else "black"
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)
                else:
                    ax.text(j, i, "skip", ha="center", va="center",
                            fontsize=8, color="gray")

        ax.set_xticks(range(len(Ts)))
        ax.set_xticklabels([str(t) for t in Ts])
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([str(k) for k in ks])
        ax.set_xlabel("T")
        ax.set_ylabel("k")
        rho_label = "IID" if rho == 0.0 else f"Markov ρ={rho}"
        ax.set_title(f"ΔLoss — {rho_label}")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        label="ΔLoss (TXCDR − Stacked SAE)", shrink=0.8, pad=0.08)
    fig.suptitle("Reconstruction Loss Difference (negative = TXCDR lower loss)", fontsize=12, y=1.02)
    path = os.path.join(viz_dir, "delta_loss_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 7. k analysis ──────────────────────────────────────────────────────────────

def plot_k_analysis(results: dict, viz_dir: str):
    """
    How k affects AUC and Loss for each model type.
    Two rows: AUC (top), Loss (bottom). One column per rho.
    Lines for each (model, T) combination.
    """
    rhos = sorted(set(r for _, r, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    n_rho = len(rhos)
    fig, axes = plt.subplots(2, n_rho, figsize=(5 * n_rho, 8), squeeze=False)

    style_map = {
        "stacked_sae": {"color_base": "steelblue", "ls": "--", "name": "StackedSAE"},
        "txcdr": {"color_base": "firebrick", "ls": "-", "name": "TXCDR"},
    }

    for col, rho in enumerate(rhos):
        ax_auc = axes[0, col]
        ax_loss = axes[1, col]

        for model_type, style in style_map.items():
            for t_idx, T in enumerate(Ts):
                auc_vals, loss_vals, valid_ks = [], [], []
                for k in ks:
                    key = (model_type, rho, k, T)
                    if key in results:
                        auc_vals.append(get_final(results[key], "auc"))
                        loss_vals.append(get_final(results[key], "loss"))
                        valid_ks.append(k)
                if not valid_ks:
                    continue
                alpha = 0.5 + 0.5 * (t_idx / max(len(Ts) - 1, 1))
                label = f"{style['name']} T={T}"
                ax_auc.plot(valid_ks, auc_vals, f"o{style['ls']}", color=style["color_base"],
                            alpha=alpha, lw=1.5, ms=5, label=label)
                ax_loss.plot(valid_ks, loss_vals, f"o{style['ls']}", color=style["color_base"],
                             alpha=alpha, lw=1.5, ms=5, label=label)

        rho_label = "IID" if rho == 0.0 else f"Markov ρ={rho}"
        ax_auc.set_title(f"{rho_label}")
        ax_auc.set_ylabel("AUC")
        ax_auc.set_ylim(0, 1.05)
        ax_auc.set_xscale("log", base=2)
        ax_auc.set_xticks(ks)
        ax_auc.set_xticklabels([str(k) for k in ks])
        ax_auc.legend(fontsize=6, loc="lower right")
        ax_auc.grid(True, alpha=0.3)

        ax_loss.set_xlabel("k (dictionary size multiplier)")
        ax_loss.set_ylabel("Reconstruction Loss")
        ax_loss.set_xscale("log", base=2)
        ax_loss.set_xticks(ks)
        ax_loss.set_xticklabels([str(k) for k in ks])
        ax_loss.legend(fontsize=6, loc="upper right")
        ax_loss.grid(True, alpha=0.3)

    fig.suptitle("Effect of k on AUC and Reconstruction Loss", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(viz_dir, "k_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 8. Summary table ───────────────────────────────────────────────────────────

def print_summary_table(results: dict, viz_dir: str):
    """Print and save a text summary table."""
    rhos = sorted(set(r for _, r, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    lines = []
    header = f"{'Rho':>5} {'Model':<12} {'k':>3} {'T':>3} {'L0':>5} {'AUC':>8} {'Loss':>8} {'R@0.9':>6}"
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for rho in rhos:
        for k in ks:
            for T in Ts:
                for model in ["stacked_sae", "txcdr"]:
                    key = (model, rho, k, T)
                    if key not in results:
                        continue
                    h = results[key][-1] if results[key] else {}
                    lines.append(
                        f"{rho:>5.1f} {model:<12} {k:>3} {T:>3} "
                        f"{h.get('window_l0', 0):>5.0f} "
                        f"{h.get('auc', 0):>8.4f} {h.get('loss', 0):>8.4f} "
                        f"{h.get('recovery_90', 0):>6.2f}"
                    )

    lines.append("=" * len(header))

    # IID vs Markov summary
    lines.append("\n── TXCDR ADVANTAGE SUMMARY ──")
    for rho in rhos:
        advantages = []
        for k in ks:
            for T in Ts:
                txcdr_key = ("txcdr", rho, k, T)
                sae_key = ("stacked_sae", rho, k, T)
                if txcdr_key in results and sae_key in results:
                    diff = get_final(results[txcdr_key], "auc") - get_final(results[sae_key], "auc")
                    advantages.append(diff)
        if advantages:
            rho_name = "IID" if rho == 0.0 else f"ρ={rho}"
            mean_adv = np.mean(advantages)
            max_adv = max(advantages)
            min_adv = min(advantages)
            lines.append(f"  [{rho_name}] mean ΔAUC={mean_adv:+.4f}  "
                         f"range=[{min_adv:+.4f}, {max_adv:+.4f}]")

    text = "\n".join(lines)
    print(text)

    path = os.path.join(viz_dir, "summary_table.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"\n  Saved: {path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Post-sweep visualization (v8)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory containing JSON logs")
    parser.add_argument("--viz-dir", type=str, default="viz_outputs_v8",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.viz_dir, exist_ok=True)

    print("Loading logs...")
    results = load_all_logs(args.log_dir)
    if not results:
        print(f"  No logs found in {args.log_dir}/. Run sweep.py first.")
        sys.exit(1)

    print(f"  Loaded {len(results)} run histories\n")

    print("── Generating visualizations ──")
    plot_tradeoff_scatter(results, args.viz_dir)
    plot_advantage_heatmaps(results, args.viz_dir)
    plot_advantage_vs_rho(results, args.viz_dir)
    plot_convergence_curves(results, args.viz_dir)
    plot_iid_vs_markov(results, args.viz_dir)
    plot_delta_loss_heatmaps(results, args.viz_dir)
    plot_k_analysis(results, args.viz_dir)
    print()
    print_summary_table(results, args.viz_dir)

    print(f"\n  All plots saved to {args.viz_dir}/")


if __name__ == "__main__":
    main()
