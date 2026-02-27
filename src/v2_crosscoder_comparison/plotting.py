"""Plotting utilities for crosscoder comparison experiment."""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.v2_crosscoder_comparison.eval import EvalResult

matplotlib.use("Agg")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


def plot_pareto_frontiers(
    results_by_arch: dict[str, list[EvalResult]],
    rho: float,
    save_path: str,
) -> None:
    """Plot Pareto frontiers (L0 vs MSE) for each architecture.

    Args:
        results_by_arch: Dict mapping arch_type -> list of EvalResults.
        rho: The rho value for this plot.
        save_path: Path to save the PNG.
    """
    _ensure_dir(save_path)
    colors = {"naive_sae": "steelblue", "stacked_sae": "coral", "crosscoder": "seagreen"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for arch_type, results in results_by_arch.items():
        if not results:
            continue
        l0s = [r.l0 for r in results]
        mses = [r.mse for r in results]
        color = colors.get(arch_type, "gray")
        ax.scatter(l0s, mses, c=color, alpha=0.4, s=20)

        # Pareto frontier line
        from src.v2_crosscoder_comparison.eval import compute_pareto_frontier

        pareto = compute_pareto_frontier(results)
        if pareto:
            p_l0 = [r.l0 for r in pareto]
            p_mse = [r.mse for r in pareto]
            ax.plot(p_l0, p_mse, "o-", color=color, linewidth=2, label=arch_type)

    ax.set_xlabel("L0")
    ax.set_ylabel("MSE")
    ax.set_title(f"Pareto Frontiers (rho={rho:.1f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_recovery_heatmap(
    cos_sim_matrix: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """Plot decoder-feature cosine similarity heatmap.

    Args:
        cos_sim_matrix: (d_sae, num_features) array.
        title: Plot title.
        save_path: Path to save the PNG.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(max(8, cos_sim_matrix.shape[1] * 0.3),
                                     max(6, cos_sim_matrix.shape[0] * 0.3)))
    im = ax.imshow(cos_sim_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_xlabel("True Features")
    ax.set_ylabel("SAE Latents")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_crosscoder_advantage(
    rho_values: list[float],
    advantage: list[float],
    save_path: str,
) -> None:
    """Plot crosscoder advantage (cos_sim difference) vs rho.

    Args:
        rho_values: List of rho values.
        advantage: Crosscoder cos_sim - best baseline cos_sim at each rho.
        save_path: Path to save the PNG.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rho_values, advantage, "o-", color="seagreen", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Cross-position correlation (rho)")
    ax.set_ylabel("Crosscoder advantage (mean_max_cos_sim)")
    ax.set_title("Crosscoder Advantage vs Correlation Strength")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_rho_sweep_summary(
    results_by_rho: dict[float, dict[str, list[EvalResult]]],
    save_path: str,
) -> None:
    """Plot summary of mean_max_cos_sim across architectures vs rho.

    Args:
        results_by_rho: Dict mapping rho -> {arch_type -> [EvalResult]}.
        save_path: Path to save the PNG.
    """
    _ensure_dir(save_path)
    colors = {"naive_sae": "steelblue", "stacked_sae": "coral", "crosscoder": "seagreen"}

    fig, ax = plt.subplots(figsize=(8, 5))

    arch_types = set()
    for arch_results in results_by_rho.values():
        arch_types.update(arch_results.keys())

    for arch_type in sorted(arch_types):
        rho_vals = []
        means = []
        stds = []
        for rho in sorted(results_by_rho.keys()):
            if arch_type in results_by_rho[rho]:
                results = results_by_rho[rho][arch_type]
                cos_sims = [r.mean_max_cos_sim for r in results]
                rho_vals.append(rho)
                means.append(np.mean(cos_sims))
                stds.append(np.std(cos_sims))

        color = colors.get(arch_type, "gray")
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(rho_vals, means, "o-", color=color, linewidth=2, label=arch_type)
        ax.fill_between(
            rho_vals,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Cross-position correlation (rho)")
    ax.set_ylabel("Mean Max Cosine Similarity")
    ax.set_title("Feature Recovery vs Cross-Position Correlation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
