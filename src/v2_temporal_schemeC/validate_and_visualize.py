"""Validate Scheme C data generation and create visualizations.

Checks:
1. Empirical marginal activation probability matches pi.
2. Empirical lag-1 autocorrelation matches rho.
3. Empirical lag-k autocorrelation matches rho^k for several lags.

Visualizations:
- Activation grid: rows = features, columns = timesteps, color = magnitude.
- Empirical vs theoretical autocorrelation across lags.
- Marginal activation probability: empirical vs target.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.seed import set_seed
from src.v2_temporal_schemeC.markov_data_generation import (
    generate_markov_activations,
    theoretical_autocorrelation,
)


def compute_empirical_autocorrelation(
    support: torch.Tensor, max_lag: int = 10
) -> torch.Tensor:
    """Compute empirical autocorrelation of binary support across time.

    Args:
        support: Binary tensor, shape (batch_size, seq_len, n_features).
        max_lag: Maximum lag to compute.

    Returns:
        Autocorrelation tensor, shape (max_lag, n_features).
    """
    batch_size, seq_len, n_features = support.shape
    autocorrs = torch.zeros(max_lag, n_features, device=support.device)

    # Mean and variance per feature (across batch and time)
    mean = support.mean(dim=(0, 1))  # (n_features,)
    var = support.var(dim=(0, 1), unbiased=False)  # (n_features,)

    for lag in range(1, max_lag + 1):
        # Covariance at this lag
        x_t = support[:, :seq_len - lag, :]  # (batch, T-lag, n_features)
        x_t_lag = support[:, lag:, :]  # (batch, T-lag, n_features)
        cov = ((x_t - mean) * (x_t_lag - mean)).mean(dim=(0, 1))
        autocorrs[lag - 1] = cov / (var + 1e-12)

    return autocorrs


def run_validation(
    pi: torch.Tensor,
    rho: torch.Tensor,
    seq_len: int = 100,
    batch_size: int = 10000,
    max_lag: int = 10,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run validation checks on generated data.

    Returns a dict with empirical and theoretical values.
    """
    set_seed(seed)
    n_features = pi.shape[0]

    activations, support = generate_markov_activations(
        batch_size=batch_size,
        seq_len=seq_len,
        pi=pi,
        rho=rho,
        device=device,
    )

    # Check marginal activation probability
    empirical_pi = support.mean(dim=(0, 1))  # (n_features,)

    # Check autocorrelations
    empirical_autocorr = compute_empirical_autocorrelation(support, max_lag)

    # Theoretical autocorrelations
    theoretical_autocorr = torch.zeros(max_lag, n_features)
    for lag in range(1, max_lag + 1):
        theoretical_autocorr[lag - 1] = theoretical_autocorrelation(rho, lag)

    return {
        "pi_target": pi,
        "pi_empirical": empirical_pi,
        "rho_target": rho,
        "autocorr_empirical": empirical_autocorr,
        "autocorr_theoretical": theoretical_autocorr,
        "activations": activations,
        "support": support,
    }


def plot_activation_grid(
    activations: torch.Tensor,
    title: str = "Feature activations",
    save_path: str | None = None,
) -> None:
    """Plot activation grid: rows=features, columns=timesteps.

    Shows a single sequence from the batch.

    Args:
        activations: shape (batch_size, seq_len, n_features). Uses first sequence.
        title: Plot title.
        save_path: If provided, save the figure.
    """
    data = activations[0].cpu().numpy().T  # (n_features, seq_len)

    fig, ax = plt.subplots(figsize=(14, max(3, data.shape[0] * 0.4)))
    im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([f"f{i}" for i in range(data.shape[0])])
    plt.colorbar(im, ax=ax, label="Activation magnitude")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_support_grid(
    support: torch.Tensor,
    n_sequences: int = 5,
    title: str = "Support patterns",
    save_path: str | None = None,
) -> None:
    """Plot support (on/off) patterns for multiple sequences side by side.

    Args:
        support: Binary tensor, shape (batch_size, seq_len, n_features).
        n_sequences: Number of sequences to show.
        title: Plot title.
        save_path: If provided, save the figure.
    """
    n_features = support.shape[2]
    n_sequences = min(n_sequences, support.shape[0])

    fig, axes = plt.subplots(
        1, n_sequences, figsize=(3 * n_sequences, max(3, n_features * 0.4)),
        sharey=True,
    )
    if n_sequences == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        data = support[i].cpu().numpy().T  # (n_features, seq_len)
        ax.imshow(data, aspect="auto", cmap="Greys", interpolation="nearest",
                  vmin=0, vmax=1)
        ax.set_xlabel("Timestep")
        if i == 0:
            ax.set_ylabel("Feature")
            ax.set_yticks(range(n_features))
            ax.set_yticklabels([f"f{j}" for j in range(n_features)])
        ax.set_title(f"Seq {i}")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_autocorrelation_comparison(
    empirical: torch.Tensor,
    theoretical: torch.Tensor,
    rho: torch.Tensor,
    title: str = "Autocorrelation: empirical vs theoretical",
    save_path: str | None = None,
) -> None:
    """Plot empirical vs theoretical autocorrelation across lags.

    Args:
        empirical: shape (max_lag, n_features).
        theoretical: shape (max_lag, n_features).
        rho: target rho per feature, shape (n_features,).
        title: Plot title.
        save_path: If provided, save the figure.
    """
    max_lag = empirical.shape[0]
    n_features = empirical.shape[1]
    lags = np.arange(1, max_lag + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))

    for i in range(n_features):
        color = colors[i]
        label_emp = f"f{i} (rho={rho[i]:.2f}) emp"
        label_th = f"f{i} (rho={rho[i]:.2f}) theory"
        ax.plot(lags, empirical[:, i].cpu().numpy(), "o-", color=color,
                label=label_emp, markersize=4)
        ax.plot(lags, theoretical[:, i].cpu().numpy(), "--", color=color,
                label=label_th, alpha=0.7)

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_marginal_probability(
    pi_target: torch.Tensor,
    pi_empirical: torch.Tensor,
    title: str = "Marginal activation probability",
    save_path: str | None = None,
) -> None:
    """Bar plot comparing target and empirical activation probabilities.

    Args:
        pi_target: shape (n_features,).
        pi_empirical: shape (n_features,).
        title: Plot title.
        save_path: If provided, save the figure.
    """
    n_features = pi_target.shape[0]
    x = np.arange(n_features)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, n_features * 0.8), 5))
    ax.bar(x - width / 2, pi_target.cpu().numpy(), width, label="Target", alpha=0.8)
    ax.bar(x + width / 2, pi_empirical.cpu().numpy(), width, label="Empirical", alpha=0.8)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Activation probability")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"f{i}" for i in range(n_features)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    results_dir = "src/v2_temporal_schemeC/results/validation"
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Scenario 1: Variety of rho values at fixed pi ---
    print("=" * 60)
    print("Scenario 1: Varying rho at fixed pi=0.2")
    print("=" * 60)

    n_features = 6
    pi = torch.full((n_features,), 0.2)
    rho = torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9, 0.95])

    results = run_validation(
        pi=pi, rho=rho, seq_len=200, batch_size=20000,
        max_lag=15, seed=42, device=device,
    )

    # Print numerical checks
    print("\nMarginal activation probability:")
    for i in range(n_features):
        print(f"  f{i}: target={pi[i]:.3f}, empirical={results['pi_empirical'][i]:.4f}")

    print("\nLag-1 autocorrelation:")
    for i in range(n_features):
        emp = results["autocorr_empirical"][0, i].item()
        th = results["autocorr_theoretical"][0, i].item()
        print(f"  f{i}: target={th:.4f}, empirical={emp:.4f}, diff={abs(emp-th):.4f}")

    # Plots
    plot_activation_grid(
        results["activations"],
        title="Activations (varying rho, pi=0.2)",
        save_path=os.path.join(results_dir, "activation_grid_varying_rho.png"),
    )
    plot_support_grid(
        results["support"], n_sequences=5,
        title="Support patterns (varying rho, pi=0.2)",
        save_path=os.path.join(results_dir, "support_grid_varying_rho.png"),
    )
    plot_autocorrelation_comparison(
        results["autocorr_empirical"],
        results["autocorr_theoretical"],
        rho,
        title="Autocorrelation: varying rho, pi=0.2",
        save_path=os.path.join(results_dir, "autocorr_varying_rho.png"),
    )
    plot_marginal_probability(
        results["pi_target"],
        results["pi_empirical"],
        title="Marginal probability: varying rho, pi=0.2",
        save_path=os.path.join(results_dir, "marginal_prob_varying_rho.png"),
    )

    # --- Scenario 2: Varying pi at fixed rho ---
    print("\n" + "=" * 60)
    print("Scenario 2: Varying pi at fixed rho=0.7")
    print("=" * 60)

    pi2 = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    rho2 = torch.full((6,), 0.7)

    results2 = run_validation(
        pi=pi2, rho=rho2, seq_len=200, batch_size=20000,
        max_lag=15, seed=42, device=device,
    )

    print("\nMarginal activation probability:")
    for i in range(6):
        print(f"  f{i}: target={pi2[i]:.3f}, empirical={results2['pi_empirical'][i]:.4f}")

    print("\nLag-1 autocorrelation:")
    for i in range(6):
        emp = results2["autocorr_empirical"][0, i].item()
        th = results2["autocorr_theoretical"][0, i].item()
        print(f"  f{i}: target={th:.4f}, empirical={emp:.4f}, diff={abs(emp-th):.4f}")

    plot_activation_grid(
        results2["activations"],
        title="Activations (varying pi, rho=0.7)",
        save_path=os.path.join(results_dir, "activation_grid_varying_pi.png"),
    )
    plot_support_grid(
        results2["support"], n_sequences=5,
        title="Support patterns (varying pi, rho=0.7)",
        save_path=os.path.join(results_dir, "support_grid_varying_pi.png"),
    )
    plot_autocorrelation_comparison(
        results2["autocorr_empirical"],
        results2["autocorr_theoretical"],
        rho2,
        title="Autocorrelation: varying pi, rho=0.7",
        save_path=os.path.join(results_dir, "autocorr_varying_pi.png"),
    )
    plot_marginal_probability(
        results2["pi_target"],
        results2["pi_empirical"],
        title="Marginal probability: varying pi, rho=0.7",
        save_path=os.path.join(results_dir, "marginal_prob_varying_pi.png"),
    )

    # --- Scenario 3: Longer sequences to show persistence visually ---
    print("\n" + "=" * 60)
    print("Scenario 3: Long sequence visualization (T=500)")
    print("=" * 60)

    pi3 = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
    rho3 = torch.tensor([0.0, 0.5, 0.8, 0.95, 0.99])

    results3 = run_validation(
        pi=pi3, rho=rho3, seq_len=500, batch_size=5000,
        max_lag=15, seed=42, device=device,
    )

    plot_support_grid(
        results3["support"], n_sequences=3,
        title="Long sequences (T=500, pi=0.1, varying rho)",
        save_path=os.path.join(results_dir, "support_long_sequences.png"),
    )

    print("\nAll validation complete. Results saved to:", results_dir)


if __name__ == "__main__":
    main()
