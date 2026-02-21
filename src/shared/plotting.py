"""Plotting utilities for SAE experiments.

All plots are saved as both .png (static) and .html (interactive plotly).
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch

matplotlib.use("Agg")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


def plot_decoder_feature_heatmap(
    cos_sim_matrix: torch.Tensor,
    title: str,
    save_path: str,
) -> None:
    """Plot decoder-feature cosine similarity as a heatmap.

    Args:
        cos_sim_matrix: Shape (d_sae, num_features) cosine similarity matrix.
        title: Plot title.
        save_path: Base path (without extension) for saving.
    """
    _ensure_dir(save_path)
    data = cos_sim_matrix.numpy() if isinstance(cos_sim_matrix, torch.Tensor) else cos_sim_matrix

    # Matplotlib/seaborn version
    fig, ax = plt.subplots(figsize=(max(8, data.shape[1] * 0.4), max(6, data.shape[0] * 0.4)))
    sns.heatmap(
        data,
        center=0,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=data.shape[0] <= 10,
        fmt=".2f" if data.shape[0] <= 10 else "",
        ax=ax,
        xticklabels=[f"f{i}" for i in range(data.shape[1])],
        yticklabels=[f"l{i}" for i in range(data.shape[0])],
    )
    ax.set_xlabel("True Features")
    ax.set_ylabel("SAE Latents")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150)
    plt.close()

    # Plotly version
    fig_plotly = go.Figure(
        data=go.Heatmap(
            z=data,
            x=[f"f{i}" for i in range(data.shape[1])],
            y=[f"l{i}" for i in range(data.shape[0])],
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Cosine Sim"),
        )
    )
    fig_plotly.update_layout(
        title=title,
        xaxis_title="True Features",
        yaxis_title="SAE Latents",
        yaxis=dict(autorange="reversed"),
    )
    fig_plotly.write_html(f"{save_path}.html")


def plot_correlation_matrix(
    corr_matrix: torch.Tensor,
    title: str,
    save_path: str,
) -> None:
    """Plot a correlation matrix as a diverging heatmap.

    Args:
        corr_matrix: Shape (g, g) correlation matrix.
        title: Plot title.
        save_path: Base path (without extension).
    """
    _ensure_dir(save_path)
    data = corr_matrix.numpy() if isinstance(corr_matrix, torch.Tensor) else corr_matrix
    n = data.shape[0]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.3), max(5, n * 0.3)))
    sns.heatmap(
        data,
        center=0,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=n <= 10,
        fmt=".2f" if n <= 10 else "",
        ax=ax,
        xticklabels=[f"f{i}" for i in range(n)],
        yticklabels=[f"f{i}" for i in range(n)],
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150)
    plt.close()

    fig_plotly = go.Figure(
        data=go.Heatmap(
            z=data,
            x=[f"f{i}" for i in range(n)],
            y=[f"f{i}" for i in range(n)],
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )
    fig_plotly.update_layout(
        title=title,
        yaxis=dict(autorange="reversed"),
    )
    fig_plotly.write_html(f"{save_path}.html")


def plot_cdec_vs_l0(
    results: dict[float, list[float]],
    true_l0: float,
    save_path: str,
    title: str = "Decoder Pairwise Cosine Similarity vs L0",
) -> None:
    """Plot c_dec vs L0 with error bands.

    Args:
        results: Dict mapping k value -> list of c_dec values (one per seed).
        true_l0: The true L0 of the toy model (shown as vertical line).
        save_path: Base path (without extension).
        title: Plot title.
    """
    _ensure_dir(save_path)
    k_values = sorted(results.keys())
    means = [np.mean(results[k]) for k in k_values]
    stds = [np.std(results[k]) for k in k_values]
    means_arr = np.array(means)
    stds_arr = np.array(stds)

    # Matplotlib version
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, means, "o-", color="steelblue", linewidth=2)
    ax.fill_between(
        k_values,
        means_arr - stds_arr,
        means_arr + stds_arr,
        alpha=0.2,
        color="steelblue",
    )
    ax.axvline(true_l0, color="gray", linestyle="--", label=f"True L0 = {true_l0}")
    ax.set_xlabel("SAE L0 (k)")
    ax.set_ylabel("$c_{\\mathrm{dec}}$")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150)
    plt.close()

    # Plotly version
    fig_plotly = go.Figure()
    fig_plotly.add_trace(
        go.Scatter(
            x=k_values,
            y=means,
            mode="lines+markers",
            name="c_dec",
            line=dict(color="steelblue"),
        )
    )
    fig_plotly.add_trace(
        go.Scatter(
            x=k_values + k_values[::-1],
            y=(means_arr + stds_arr).tolist() + (means_arr - stds_arr)[::-1].tolist(),
            fill="toself",
            fillcolor="rgba(70,130,180,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
        )
    )
    fig_plotly.add_vline(
        x=true_l0, line_dash="dash", line_color="gray",
        annotation_text=f"True L0 = {true_l0}",
    )
    fig_plotly.update_layout(
        title=title,
        xaxis_title="SAE L0 (k)",
        yaxis_title="c_dec",
    )
    fig_plotly.write_html(f"{save_path}.html")


def plot_sparsity_reconstruction_tradeoff(
    learned_results: dict[float, list[float]],
    gt_results: dict[float, list[float]],
    true_l0: float,
    save_path: str,
    title: str = "Sparsity-Reconstruction Tradeoff",
) -> None:
    """Plot variance explained vs L0 for learned and ground-truth SAEs.

    Args:
        learned_results: Dict mapping k -> list of VE values for learned SAEs.
        gt_results: Dict mapping k -> list of VE values for ground-truth SAEs.
        true_l0: True L0 of the toy model.
        save_path: Base path (without extension).
        title: Plot title.
    """
    _ensure_dir(save_path)

    def _plot_data(results):
        ks = sorted(results.keys())
        means = [np.mean(results[k]) for k in ks]
        stds = [np.std(results[k]) for k in ks]
        return ks, np.array(means), np.array(stds)

    lk, lm, ls = _plot_data(learned_results)
    gk, gm, gs = _plot_data(gt_results)

    # Matplotlib
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lk, lm, "o-", color="steelblue", linewidth=2, label="Learned SAE")
    ax.fill_between(lk, lm - ls, lm + ls, alpha=0.2, color="steelblue")
    ax.plot(gk, gm, "s--", color="coral", linewidth=2, label="Ground-truth SAE")
    ax.fill_between(gk, gm - gs, gm + gs, alpha=0.2, color="coral")
    ax.axvline(true_l0, color="gray", linestyle="--", label=f"True L0 = {true_l0}")
    ax.set_xlabel("SAE L0 (k)")
    ax.set_ylabel("Variance Explained")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150)
    plt.close()

    # Plotly
    fig_plotly = go.Figure()
    fig_plotly.add_trace(
        go.Scatter(x=lk, y=lm.tolist(), mode="lines+markers", name="Learned SAE",
                   line=dict(color="steelblue"))
    )
    fig_plotly.add_trace(
        go.Scatter(
            x=lk + lk[::-1],
            y=(lm + ls).tolist() + (lm - ls)[::-1].tolist(),
            fill="toself", fillcolor="rgba(70,130,180,0.2)",
            line=dict(color="rgba(255,255,255,0)"), showlegend=False,
        )
    )
    fig_plotly.add_trace(
        go.Scatter(x=gk, y=gm.tolist(), mode="lines+markers", name="Ground-truth SAE",
                   line=dict(color="coral", dash="dash"))
    )
    fig_plotly.add_trace(
        go.Scatter(
            x=gk + gk[::-1],
            y=(gm + gs).tolist() + (gm - gs)[::-1].tolist(),
            fill="toself", fillcolor="rgba(255,127,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"), showlegend=False,
        )
    )
    fig_plotly.add_vline(
        x=true_l0, line_dash="dash", line_color="gray",
        annotation_text=f"True L0 = {true_l0}",
    )
    fig_plotly.update_layout(
        title=title,
        xaxis_title="SAE L0 (k)",
        yaxis_title="Variance Explained",
    )
    fig_plotly.write_html(f"{save_path}.html")
