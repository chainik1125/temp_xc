"""Validation plots for coupled-feature data generation.

Generates plots to verify the coupled-feature pipeline is working correctly:
1. Coupling matrix heatmap (which hidden states control which emissions)
2. Emission co-occurrence heatmap (empirical correlation between emissions)
3. Hidden-to-emission cosine similarity (do hidden features align with emissions?)
4. Sparsity comparison (hidden states vs emissions over time)

Usage:
    python -m src.data_generation.plot_coupled [--output-dir results/coupled_validation]
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

matplotlib.use("Agg")

from src.data_generation.configs import (
    CoupledDataGenerationConfig,
    CouplingConfig,
    SequenceConfig,
    TransitionConfig,
)
from src.data_generation.coupled_dataset import generate_coupled_dataset
from src.data_generation.coupling import compute_hidden_features


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_coupling_matrix(C: torch.Tensor, save_path: str) -> None:
    """Heatmap of the binary coupling matrix C (M x K)."""
    M, K = C.shape
    fig, ax = plt.subplots(figsize=(max(4, K * 0.5), max(4, M * 0.35)))
    sns.heatmap(
        C.numpy(),
        cmap="Blues",
        vmin=0, vmax=1,
        xticklabels=[f"h{i}" for i in range(K)],
        yticklabels=[f"e{j}" for j in range(M)],
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        cbar_kws={"label": "Connected"},
    )
    ax.set_xlabel("Hidden states")
    ax.set_ylabel("Emission features")
    ax.set_title(f"Coupling matrix ({M} emissions, {K} hidden, {int(C.sum(1).mean())} parents/emission)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_emission_cooccurrence(support: torch.Tensor, save_path: str) -> None:
    """Heatmap of empirical emission co-occurrence (correlation between emissions).

    Args:
        support: (n_seq, M, T) binary emission support.
    """
    # Flatten across sequences and time: (n_seq*T, M)
    flat = support.permute(0, 2, 1).reshape(-1, support.shape[1])
    corr = np.corrcoef(flat.numpy().T)

    M = support.shape[1]
    fig, ax = plt.subplots(figsize=(max(6, M * 0.35), max(5, M * 0.35)))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        vmin=-1, vmax=1, center=0,
        xticklabels=[f"e{j}" for j in range(M)],
        yticklabels=[f"e{j}" for j in range(M)],
        ax=ax,
    )
    ax.set_title("Emission co-occurrence (Pearson correlation)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_emission_cosine(
    hidden_features: torch.Tensor,
    emission_features: torch.Tensor,
    coupling_matrix: torch.Tensor,
    save_path: str,
) -> None:
    """Heatmap of cosine similarity between hidden and emission feature directions.

    Args:
        hidden_features: (K, d)
        emission_features: (M, d)
        coupling_matrix: (M, K) for annotation
    """
    # Normalize
    hf = hidden_features / hidden_features.norm(dim=1, keepdim=True).clamp(min=1e-8)
    ef = emission_features / emission_features.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos_sim = (hf @ ef.T).numpy()  # (K, M)

    K, M = cos_sim.shape
    fig, ax = plt.subplots(figsize=(max(6, M * 0.4), max(4, K * 0.5)))
    sns.heatmap(
        cos_sim,
        cmap="RdBu_r",
        vmin=-1, vmax=1, center=0,
        xticklabels=[f"e{j}" for j in range(M)],
        yticklabels=[f"h{i}" for i in range(K)],
        ax=ax,
    )
    # Mark coupled pairs with dots
    for j in range(M):
        for i in range(K):
            if coupling_matrix[j, i] > 0.5:
                ax.plot(j + 0.5, i + 0.5, "k.", markersize=4)
    ax.set_xlabel("Emission features")
    ax.set_ylabel("Hidden features")
    ax.set_title("Cosine similarity (hidden vs emission dirs, dots = coupled)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_traces(
    hidden_states: torch.Tensor,
    support: torch.Tensor,
    coupling_matrix: torch.Tensor,
    save_path: str,
    seq_idx: int = 0,
    max_features: int = 5,
) -> None:
    """Time traces showing hidden states and their coupled emissions.

    Shows the first few hidden states alongside the emissions they control,
    illustrating how coupled emissions co-activate with their parent hidden states.
    """
    K = hidden_states.shape[1]
    n_show = min(max_features, K)
    T = hidden_states.shape[2]

    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        ax = axes[i]
        # Plot hidden state
        h = hidden_states[seq_idx, i, :].numpy()
        ax.fill_between(range(T), 0, h, alpha=0.3, color="steelblue", label=f"h{i}")

        # Plot emissions controlled by this hidden state
        children = (coupling_matrix[:, i] > 0.5).nonzero(as_tuple=True)[0]
        colors = plt.cm.Set2(np.linspace(0, 1, len(children)))
        for idx, j in enumerate(children):
            s = support[seq_idx, j, :].numpy()
            ax.plot(range(T), s * (0.8 - 0.15 * idx), ".", color=colors[idx],
                    markersize=2, alpha=0.7, label=f"e{j.item()}")

        ax.set_ylabel(f"h{i}")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc="upper right", fontsize=7, ncol=len(children) + 1)

    axes[-1].set_xlabel("Position t")
    fig.suptitle("Temporal traces: hidden states (shaded) and emissions (dots)", y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sparsity_summary(
    hidden_states: torch.Tensor,
    support: torch.Tensor,
    coupling_matrix: torch.Tensor,
    save_path: str,
) -> None:
    """Bar chart comparing hidden state and emission sparsity."""
    K = hidden_states.shape[1]
    M = support.shape[1]

    h_sparsity = hidden_states.mean(dim=(0, 2)).numpy()  # per hidden state
    e_sparsity = support.mean(dim=(0, 2)).numpy()  # per emission

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(range(K), h_sparsity, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Hidden state index")
    ax1.set_ylabel("Firing probability")
    ax1.set_title(f"Hidden state sparsity (mean={h_sparsity.mean():.3f})")
    ax1.axhline(h_sparsity.mean(), color="red", linestyle="--", alpha=0.5)

    ax2.bar(range(M), e_sparsity, color="coral", alpha=0.8)
    ax2.set_xlabel("Emission feature index")
    ax2.set_ylabel("Firing probability")
    ax2.set_title(f"Emission sparsity (mean={e_sparsity.mean():.3f})")
    ax2.axhline(e_sparsity.mean(), color="red", linestyle="--", alpha=0.5)

    fig.suptitle("Sparsity comparison: hidden states vs emissions")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate coupled-feature validation plots")
    parser.add_argument("--output-dir", type=str, default="results/coupled_validation",
                        help="Output directory for plots")
    parser.add_argument("--K", type=int, default=10, help="Number of hidden states")
    parser.add_argument("--M", type=int, default=20, help="Number of emission features")
    parser.add_argument("--n-parents", type=int, default=2, help="Parents per emission")
    parser.add_argument("--rho", type=float, default=0.6, help="Autocorrelation")
    args = parser.parse_args()

    _ensure_dir(args.output_dir)

    lam = 1.0 - args.rho
    cfg = CoupledDataGenerationConfig(
        transition=TransitionConfig.from_reset_process(lam=lam, p=0.1),
        coupling=CouplingConfig(
            K_hidden=args.K,
            M_emission=args.M,
            n_parents=args.n_parents,
            emission_mode="or",
        ),
        sequence=SequenceConfig(T=128, n_sequences=200),
        hidden_dim=64,
        seed=42,
    )

    print(f"Generating coupled dataset: K={args.K}, M={args.M}, "
          f"n_parents={args.n_parents}, rho={args.rho}")
    result = generate_coupled_dataset(cfg)

    print("Generating plots...")

    plot_coupling_matrix(
        result["coupling_matrix"],
        os.path.join(args.output_dir, "coupling_matrix.png"),
    )
    print("  coupling_matrix.png")

    plot_emission_cooccurrence(
        result["support"],
        os.path.join(args.output_dir, "emission_cooccurrence.png"),
    )
    print("  emission_cooccurrence.png")

    plot_hidden_emission_cosine(
        result["hidden_features"],
        result["emission_features"],
        result["coupling_matrix"],
        os.path.join(args.output_dir, "hidden_emission_cosine.png"),
    )
    print("  hidden_emission_cosine.png")

    plot_temporal_traces(
        result["hidden_states"],
        result["support"],
        result["coupling_matrix"],
        os.path.join(args.output_dir, "temporal_traces.png"),
    )
    print("  temporal_traces.png")

    plot_sparsity_summary(
        result["hidden_states"],
        result["support"],
        result["coupling_matrix"],
        os.path.join(args.output_dir, "sparsity_summary.png"),
    )
    print("  sparsity_summary.png")

    print(f"\nDone. Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
