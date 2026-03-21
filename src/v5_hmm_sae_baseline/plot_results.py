"""Plotting script for the HMM SAE baseline experiment."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logging import log

RESULTS_DIR = Path("results/v5_hmm_sae_baseline")


def load_results() -> list[dict]:
    """Load experiment results from JSON."""
    with open(RESULTS_DIR / "results.json") as f:
        return json.load(f)


def plot_validation_autocorrelation(results: list[dict]) -> None:
    """Plot empirical vs theoretical autocorrelation for each configuration.

    Creates a grid of subplots: rows = emission configs, columns = lambda values.
    """
    by_label: dict[str, list] = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r)

    labels = list(by_label.keys())
    n_labels = len(labels)
    lam_values = sorted(set(r["lam"] for r in results))
    n_lams = len(lam_values)

    fig, axes = plt.subplots(
        n_labels, n_lams, figsize=(3 * n_lams, 3 * n_labels),
        squeeze=False, sharex=True, sharey=True,
    )

    for row, label in enumerate(labels):
        group = sorted(by_label[label], key=lambda r: r["lam"])
        for col, r in enumerate(group):
            ax = axes[row, col]
            lags = np.arange(len(r["theory_autocorr"]))
            ax.plot(lags, r["theory_autocorr"], "r-", linewidth=1.5, label="theory")
            ax.plot(lags, r["empirical_autocorr"], "b.", markersize=3, label="per-chain")
            if "pooled_autocorr" in r:
                ax.plot(lags, r["pooled_autocorr"], "g.", markersize=3, label="pooled")
            ax.set_ylim(-0.1, 1.1)
            if row == 0:
                ax.set_title(f"$\\lambda={r['lam']}$", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            if row == n_labels - 1:
                ax.set_xlabel("lag")
            if row == 0 and col == n_lams - 1:
                ax.legend(fontsize=7)

    fig.suptitle("Empirical vs Theoretical Autocorrelation", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "validation_autocorrelation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("done", "saved validation_autocorrelation.png")


def plot_tradeoff_curves(results: list[dict]) -> None:
    """Plot reconstruction loss and feature AUC vs lambda for each amplitude level."""
    by_label: dict[str, list] = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(by_label)))

    for (label, group), color in zip(sorted(by_label.items()), colors):
        group = sorted(group, key=lambda r: r["lam"])
        lams = [r["lam"] for r in group]
        recon = [r["recon_loss"] for r in group]
        auc = [r["auc"] for r in group]

        ax1.plot(lams, recon, "o-", color=color, label=label, markersize=4)
        ax2.plot(lams, auc, "o-", color=color, label=label, markersize=4)

    ax1.set_xlabel("$\\lambda$")
    ax1.set_ylabel("Reconstruction Loss (MSE)")
    ax1.set_title("Reconstruction Loss vs $\\lambda$")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("$\\lambda$")
    ax2.set_ylabel("Feature Recovery AUC")
    ax2.set_title("Feature AUC vs $\\lambda$")
    ax2.legend(fontsize=8)

    fig.suptitle("SAE Baseline: Trade-off Curves (TopK k=1)", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tradeoff_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("done", "saved tradeoff_curves.png")


def plot_convergence(results: list[dict]) -> None:
    """Plot training convergence curves for a representative subset."""
    by_label: dict[str, list] = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r)

    # Pick one lambda value to show convergence across gamma levels
    target_lam = 0.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(by_label)))

    for (label, group), color in zip(sorted(by_label.items()), colors):
        for r in group:
            if abs(r["lam"] - target_lam) < 1e-6 and "history" in r:
                epochs = [h["epoch"] for h in r["history"]]
                losses = [h["recon_loss"] for h in r["history"]]
                l0s = [h["l0"] for h in r["history"]]
                ax1.plot(epochs, losses, "-", color=color, label=label, linewidth=1.5)
                ax2.plot(epochs, l0s, "-", color=color, label=label, linewidth=1.5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Reconstruction Loss")
    ax1.set_title(f"Training Loss ($\\lambda={target_lam}$)")
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("L0")
    ax2.set_title(f"L0 Sparsity ($\\lambda={target_lam}$)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("done", "saved convergence.png")


def main() -> None:
    """Generate all plots from saved results."""
    results = load_results()
    plot_validation_autocorrelation(results)
    plot_tradeoff_curves(results)
    plot_convergence(results)
    log("done", "all plots generated")


if __name__ == "__main__":
    main()
