#!/usr/bin/env python3
"""
viz.py — Post-sweep visualization and statistical analysis.

Reads logs/ directory and produces:
  1. AUC heatmap per dataset:  TXCDR AUC(k,T) - best_SAE_AUC(dataset)
  2. Convergence curves:       AUC over training steps for each (dataset, k, T)
  3. Optimal-k analysis:       optimal_k(TXCDR) vs optimal_k(SAE) — tests the conjecture
  4. Summary table:            LaTeX-ready comparison table
  5. Statistical tests:        paired differences with confidence intervals

Usage:
    python viz.py                     # read from logs/
    python viz.py --log-dir my_logs   # custom log dir
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# Use Agg backend for headless servers (tmux / SSH)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm


def load_all_logs(log_dir: str) -> dict:
    """
    Load all JSON logs into a structured dict.

    Returns:
        {
            (model, dataset, k, T): [{"step": ..., "auc": ..., "loss": ...}, ...]
        }
    """
    results = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        basename = os.path.basename(path).replace(".json", "")
        if basename == "sweep_summary":
            continue
        # Naming: {model}__{dataset}__k{k}__T{T}
        parts = basename.split("__")
        if len(parts) != 4:
            continue
        model = parts[0]
        dataset = parts[1]
        k = int(parts[2].replace("k", ""))
        T = int(parts[3].replace("T", ""))

        with open(path) as f:
            history = json.load(f)
        results[(model, dataset, k, T)] = history
    return results


def load_sweep_summary(log_dir: str) -> list[dict]:
    path = os.path.join(log_dir, "sweep_summary.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def get_final_auc(history: list[dict]) -> float:
    """Last AUC value from a training history."""
    if not history:
        return 0.0
    return history[-1].get("auc", 0.0)


# ─── 1. Main heatmap: TXCDR advantage over best SAE ─────────────────────────────

def plot_txcdr_advantage_heatmap(results: dict, viz_dir: str):
    """
    For each dataset, create a k×T heatmap where each cell =
    AUC_TXCDR(k, T) - max_k'(AUC_SAE(dataset, k')).

    This directly addresses the conjecture: if TXCDR advantage grows with k
    while SAE is flat/declining, optimal_k(TXCDR) >> optimal_k(SAE).
    """
    datasets = sorted(set(ds for _, ds, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    for dataset in datasets:
        # Find best SAE AUC across all k for this dataset
        sae_aucs = {}
        for k in ks:
            for T in Ts:
                key = ("sae", dataset, k, T)
                if key in results:
                    sae_aucs[k] = get_final_auc(results[key])
                    break  # SAE is T-independent, just need one

        if not sae_aucs:
            continue
        best_sae_auc = max(sae_aucs.values())
        best_sae_k = max(sae_aucs, key=sae_aucs.get)

        # Build heatmap matrix
        mat = np.full((len(ks), len(Ts)), np.nan)
        for i, k in enumerate(ks):
            for j, T in enumerate(Ts):
                key = ("txcdr", dataset, k, T)
                if key in results:
                    txcdr_auc = get_final_auc(results[key])
                    mat[i, j] = txcdr_auc - best_sae_auc

        fig, ax = plt.subplots(figsize=(8, 5))

        # Diverging colormap centered at 0
        vabs = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 0.01)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")

        # Annotate cells
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
        ax.set_xlabel("T (window length)", fontsize=12)
        ax.set_ylabel("k (active latents per position)", fontsize=12)
        ax.set_title(
            f"TXCDR advantage over best SAE — {dataset}\n"
            f"Cell = AUC_TXCDR(k,T) − AUC_SAE*(={best_sae_auc:.3f} at k={best_sae_k})",
            fontsize=11,
        )
        plt.colorbar(im, ax=ax, label="ΔAUC (positive = TXCDR wins)")
        plt.tight_layout()
        path = os.path.join(viz_dir, f"heatmap_advantage_{dataset}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ─── 2. Raw AUC heatmaps (SAE and TXCDR side by side) ───────────────────────────

def plot_raw_auc_heatmaps(results: dict, viz_dir: str):
    """Side-by-side AUC heatmaps for SAE and TXCDR per dataset."""
    datasets = sorted(set(ds for _, ds, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    for dataset in datasets:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, model_type, title in zip(
            axes, ["sae", "txcdr"], ["SAE (TopK)", "Temporal Crosscoder"]
        ):
            mat = np.full((len(ks), len(Ts)), np.nan)
            for i, k in enumerate(ks):
                for j, T in enumerate(Ts):
                    key = (model_type, dataset, k, T)
                    if key in results:
                        mat[i, j] = get_final_auc(results[key])
                    elif model_type == "sae":
                        # SAE is T-independent; fill from any T
                        for T2 in Ts:
                            key2 = ("sae", dataset, k, T2)
                            if key2 in results:
                                mat[i, j] = get_final_auc(results[key2])
                                break

            im = ax.imshow(mat, cmap="viridis", aspect="auto", origin="lower",
                           vmin=0, vmax=1)
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
            ax.set_title(f"{title} — {dataset}")

        plt.colorbar(im, ax=axes, label="AUC", shrink=0.8)
        plt.tight_layout()
        path = os.path.join(viz_dir, f"heatmap_raw_auc_{dataset}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ─── 3. Optimal-k analysis — the conjecture test ────────────────────────────────

def plot_optimal_k_analysis(results: dict, viz_dir: str):
    """
    Conjecture: optimal_k(TXCDR) >> optimal_k(SAE).

    For each dataset and T, find the k that maximizes AUC for each model.
    Plot optimal_k vs T for both models.
    """
    datasets = sorted(set(ds for _, ds, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5),
                             squeeze=False)

    for col, dataset in enumerate(datasets):
        ax = axes[0, col]

        # SAE optimal k (T-independent, same for all T)
        sae_auc_by_k = {}
        for k in ks:
            for T in Ts:
                key = ("sae", dataset, k, T)
                if key in results:
                    sae_auc_by_k[k] = get_final_auc(results[key])
                    break
        sae_optimal_k = max(sae_auc_by_k, key=sae_auc_by_k.get) if sae_auc_by_k else None
        sae_optimal_auc = sae_auc_by_k.get(sae_optimal_k, 0)

        # TXCDR optimal k per T
        txcdr_optimal_k_per_T = {}
        txcdr_optimal_auc_per_T = {}
        for T in Ts:
            best_k, best_auc = None, -1
            for k in ks:
                key = ("txcdr", dataset, k, T)
                if key in results:
                    auc = get_final_auc(results[key])
                    if auc > best_auc:
                        best_auc = auc
                        best_k = k
            if best_k is not None:
                txcdr_optimal_k_per_T[T] = best_k
                txcdr_optimal_auc_per_T[T] = best_auc

        # Plot
        if sae_optimal_k is not None:
            ax.axhline(sae_optimal_k, color="steelblue", ls="--", lw=2,
                       label=f"SAE optimal k={sae_optimal_k} (AUC={sae_optimal_auc:.3f})")

        if txcdr_optimal_k_per_T:
            t_vals = sorted(txcdr_optimal_k_per_T.keys())
            k_vals = [txcdr_optimal_k_per_T[t] for t in t_vals]
            auc_vals = [txcdr_optimal_auc_per_T[t] for t in t_vals]
            ax.plot(t_vals, k_vals, "o-", color="firebrick", lw=2, ms=8,
                    label="TXCDR optimal k")
            # Annotate with AUC
            for t, kv, av in zip(t_vals, k_vals, auc_vals):
                ax.annotate(f"AUC={av:.3f}", (t, kv), textcoords="offset points",
                            xytext=(5, 10), fontsize=7, color="firebrick")

        ax.set_xlabel("T (window length)")
        ax.set_ylabel("Optimal k")
        ax.set_title(f"{dataset}")
        ax.set_xticks(Ts)
        ax.set_yticks(ks)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Conjecture test: optimal_k(TXCDR) vs optimal_k(SAE)\n"
        "If TXCDR points consistently above the SAE line → conjecture supported",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(viz_dir, "optimal_k_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 4. Convergence curves ──────────────────────────────────────────────────────

def plot_convergence_curves(results: dict, viz_dir: str):
    """AUC convergence for each (dataset, k) with SAE vs TXCDR(T) overlaid."""
    datasets = sorted(set(ds for _, ds, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))

    colors_T = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c", 8: "#d62728", 10: "#9467bd"}

    for dataset in datasets:
        fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4), squeeze=False)
        for col, k in enumerate(ks):
            ax = axes[0, col]

            # SAE baseline
            for T_cand in sorted(set(T for _, _, _, T in results.keys())):
                key = ("sae", dataset, k, T_cand)
                if key in results:
                    hist = results[key]
                    steps = [h["step"] for h in hist]
                    aucs = [h["auc"] for h in hist]
                    ax.plot(steps, aucs, color="gray", lw=2, ls="--",
                            label="SAE", zorder=10)
                    break

            # TXCDR for each T
            for T in sorted(set(T for _, _, _, T in results.keys())):
                key = ("txcdr", dataset, k, T)
                if key in results:
                    hist = results[key]
                    steps = [h["step"] for h in hist]
                    aucs = [h["auc"] for h in hist]
                    ax.plot(steps, aucs, color=colors_T.get(T, "black"),
                            lw=1.5, label=f"TXCDR T={T}")

            ax.set_title(f"k={k}")
            ax.set_xlabel("Step")
            if col == 0:
                ax.set_ylabel("AUC")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Convergence — {dataset}", fontsize=12)
        plt.tight_layout()
        path = os.path.join(viz_dir, f"convergence_{dataset}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ─── 5. AUC vs k curves (SAE vs TXCDR at each T) ───────────────────────────────

def plot_auc_vs_k(results: dict, viz_dir: str):
    """For each dataset: AUC vs k, one line per (model, T)."""
    datasets = sorted(set(ds for _, ds, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))

        # SAE
        sae_aucs = []
        for k in ks:
            found = False
            for T in Ts:
                key = ("sae", dataset, k, T)
                if key in results:
                    sae_aucs.append(get_final_auc(results[key]))
                    found = True
                    break
            if not found:
                sae_aucs.append(np.nan)
        ax.plot(ks, sae_aucs, "s--", color="gray", lw=2, ms=8, label="SAE", zorder=10)

        # TXCDR per T
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(Ts)))
        for T, c in zip(Ts, colors):
            aucs = []
            for k in ks:
                key = ("txcdr", dataset, k, T)
                if key in results:
                    aucs.append(get_final_auc(results[key]))
                else:
                    aucs.append(np.nan)
            ax.plot(ks, aucs, "o-", color=c, lw=1.5, ms=6, label=f"TXCDR T={T}")

        ax.set_xlabel("k (active latents per position)", fontsize=12)
        ax.set_ylabel("Final AUC", fontsize=12)
        ax.set_title(f"AUC vs k — {dataset}", fontsize=13)
        ax.set_xticks(ks)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        path = os.path.join(viz_dir, f"auc_vs_k_{dataset}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ─── 6. Summary statistics table ─────────────────────────────────────────────────

def print_summary_table(results: dict, viz_dir: str):
    """Print and save a text summary table."""
    datasets = sorted(set(ds for _, ds, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    lines = []
    header = f"{'Dataset':<10} {'Model':<8} {'k':>3} {'T':>3} {'AUC':>8} {'Loss':>8} {'R@0.9':>6}"
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for dataset in datasets:
        for k in ks:
            for T in Ts:
                for model in ["sae", "txcdr"]:
                    key = (model, dataset, k, T)
                    if key not in results:
                        continue
                    h = results[key][-1] if results[key] else {}
                    lines.append(
                        f"{dataset:<10} {model:<8} {k:>3} {T:>3} "
                        f"{h.get('auc', 0):>8.4f} {h.get('loss', 0):>8.4f} "
                        f"{h.get('recovery_90', 0):>6.2f}"
                    )

    lines.append("=" * len(header))

    # Conjecture verdict
    lines.append("\n── CONJECTURE: optimal_k(TXCDR) >> optimal_k(SAE) ──")
    for dataset in datasets:
        sae_auc_by_k = {}
        for k in ks:
            for T in Ts:
                key = ("sae", dataset, k, T)
                if key in results:
                    sae_auc_by_k[k] = get_final_auc(results[key])
                    break

        txcdr_best_overall = {}
        for k in ks:
            for T in Ts:
                key = ("txcdr", dataset, k, T)
                if key in results:
                    auc = get_final_auc(results[key])
                    if k not in txcdr_best_overall or auc > txcdr_best_overall[k]:
                        txcdr_best_overall[k] = auc

        if sae_auc_by_k:
            sae_opt = max(sae_auc_by_k, key=sae_auc_by_k.get)
            lines.append(f"  [{dataset}] SAE optimal k = {sae_opt} (AUC={sae_auc_by_k[sae_opt]:.4f})")
        if txcdr_best_overall:
            txcdr_opt = max(txcdr_best_overall, key=txcdr_best_overall.get)
            lines.append(f"  [{dataset}] TXCDR optimal k = {txcdr_opt} (AUC={txcdr_best_overall[txcdr_opt]:.4f})")
            if sae_auc_by_k:
                ratio = txcdr_opt / sae_opt if sae_opt > 0 else float("inf")
                verdict = "SUPPORTED" if txcdr_opt > sae_opt else (
                    "REFUTED" if txcdr_opt < sae_opt else "INCONCLUSIVE"
                )
                lines.append(f"  [{dataset}] Ratio = {ratio:.1f}x → {verdict}")

    text = "\n".join(lines)
    print(text)

    path = os.path.join(viz_dir, "summary_table.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"\n  Saved: {path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Post-sweep visualization")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory containing JSON logs")
    parser.add_argument("--viz-dir", type=str, default="viz_outputs",
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
    plot_txcdr_advantage_heatmap(results, args.viz_dir)
    plot_raw_auc_heatmaps(results, args.viz_dir)
    plot_optimal_k_analysis(results, args.viz_dir)
    plot_convergence_curves(results, args.viz_dir)
    plot_auc_vs_k(results, args.viz_dir)
    print()
    print_summary_table(results, args.viz_dir)

    print(f"\n  All plots saved to {args.viz_dir}/")


if __name__ == "__main__":
    main()
