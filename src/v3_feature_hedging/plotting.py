"""Plotting for the feature hedging experiment.

Wraps shared plotting with better readability for large (50×50) heatmaps:
capped figure size, spaced tick labels, and larger fonts.
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
    """Plot decoder-feature cosine similarity heatmap.

    For matrices larger than 20×20, caps figure size and uses spaced ticks
    so that titles and labels remain readable.
    """
    _ensure_dir(save_path)
    data = cos_sim_matrix.numpy() if isinstance(cos_sim_matrix, torch.Tensor) else cos_sim_matrix
    n = max(data.shape)

    w = min(max(8, data.shape[1] * 0.4), 14)
    h = min(max(6, data.shape[0] * 0.4), 12)
    fig, ax = plt.subplots(figsize=(w, h))

    if n > 20:
        tick_step = max(1, n // 10)
        xtick_pos = list(range(0, data.shape[1], tick_step))
        ytick_pos = list(range(0, data.shape[0], tick_step))
        xlabels = [f"f{i}" for i in xtick_pos]
        ylabels = [f"l{i}" for i in ytick_pos]
    else:
        xtick_pos = ytick_pos = None
        xlabels = [f"f{i}" for i in range(data.shape[1])]
        ylabels = [f"l{i}" for i in range(data.shape[0])]

    sns.heatmap(
        data,
        center=0,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=n <= 10,
        fmt=".2f" if n <= 10 else "",
        ax=ax,
        xticklabels=xlabels if n <= 20 else False,
        yticklabels=ylabels if n <= 20 else False,
    )

    if n > 20:
        ax.set_xticks([t + 0.5 for t in xtick_pos])
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_yticks([t + 0.5 for t in ytick_pos])
        ax.set_yticklabels(ylabels, fontsize=10)

    ax.set_xlabel("True Features", fontsize=12)
    ax.set_ylabel("SAE Latents", fontsize=12)
    ax.set_title(title, fontsize=14)
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
    """Plot a correlation matrix heatmap with capped size for large matrices."""
    _ensure_dir(save_path)
    data = corr_matrix.numpy() if isinstance(corr_matrix, torch.Tensor) else corr_matrix
    n = data.shape[0]

    w = min(max(6, n * 0.3), 14)
    h = min(max(5, n * 0.3), 12)
    fig, ax = plt.subplots(figsize=(w, h))

    if n > 20:
        tick_step = max(1, n // 10)
        tick_pos = list(range(0, n, tick_step))
        labels = [f"f{i}" for i in tick_pos]
    else:
        tick_pos = None
        labels = [f"f{i}" for i in range(n)]

    sns.heatmap(
        data,
        center=0,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=n <= 10,
        fmt=".2f" if n <= 10 else "",
        ax=ax,
        xticklabels=labels if n <= 20 else False,
        yticklabels=labels if n <= 20 else False,
    )

    if n > 20:
        ax.set_xticks([t + 0.5 for t in tick_pos])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks([t + 0.5 for t in tick_pos])
        ax.set_yticklabels(labels, fontsize=10)

    ax.set_title(title, fontsize=14)
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
