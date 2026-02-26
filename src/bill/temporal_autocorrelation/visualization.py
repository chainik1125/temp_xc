from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_magnitude_vs_autocorrelation(
    mean_magnitude: np.ndarray,
    mean_autocorrelation_lag1: np.ndarray,
    output_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot of mean activation magnitude vs. lag-1 autocorrelation.

    This is the main result plot: we're looking for features in the
    low-magnitude, high-autocorrelation quadrant.
    """
    valid = ~(np.isnan(mean_magnitude) | np.isnan(mean_autocorrelation_lag1))
    mag = mean_magnitude[valid]
    ac = mean_autocorrelation_lag1[valid]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mag, ac, alpha=0.3, s=4)
    ax.set_xlabel("Mean activation magnitude (when active)")
    ax.set_ylabel("Mean lag-1 autocorrelation")
    ax.set_title("Activation magnitude vs. temporal autocorrelation")
    ax.set_xscale("log")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_autocorrelation_histogram(
    real_autocorrelation_lag1: np.ndarray,
    shuffled_autocorrelation_lag1: np.ndarray,
    output_path: Path | None = None,
) -> plt.Figure:
    """Histogram of lag-1 autocorrelation, real vs. shuffled baseline."""
    real = real_autocorrelation_lag1[~np.isnan(real_autocorrelation_lag1)]
    shuffled = shuffled_autocorrelation_lag1[~np.isnan(shuffled_autocorrelation_lag1)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-0.2, 1.0, 80)
    ax.hist(real, bins=bins, alpha=0.6, label="Real", density=True)
    ax.hist(shuffled, bins=bins, alpha=0.6, label="Shuffled", density=True)
    ax.set_xlabel("Lag-1 autocorrelation")
    ax.set_ylabel("Density")
    ax.set_title("Temporal autocorrelation: real vs. shuffled baseline")
    ax.legend()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_autocorrelation_decay(
    mean_autocorrelation: np.ndarray,
    feature_indices: list[int],
    max_lag: int,
    output_path: Path | None = None,
) -> plt.Figure:
    """Autocorrelation decay curves at lags 1..max_lag for selected features."""
    lags = np.arange(1, max_lag + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in feature_indices:
        ac = mean_autocorrelation[idx]
        if np.isnan(ac).all():
            continue
        ax.plot(lags, ac, marker="o", markersize=3, label=f"Feature {idx}")

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation decay for selected features")
    ax.set_xticks(lags)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_token_activation_heatmap(
    feature_acts: np.ndarray,
    tokens: list[str],
    feature_idx: int,
    output_path: Path | None = None,
) -> plt.Figure:
    """Heatmap of a single feature's activation across token positions.

    Args:
        feature_acts: Array of shape [T] — activations for one feature, one sequence.
        tokens: List of token strings of length T.
        feature_idx: Feature index (for title).
        output_path: Optional save path.
    """
    T = len(feature_acts)
    # Show at most 100 tokens for readability
    display_len = min(T, 100)
    acts_display = feature_acts[:display_len]
    tokens_display = tokens[:display_len]

    fig, ax = plt.subplots(figsize=(max(12, display_len * 0.15), 2))
    acts_2d = acts_display.reshape(1, -1)
    sns.heatmap(
        acts_2d,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=tokens_display,
        yticklabels=False,
        cbar_kws={"label": "Activation"},
    )
    ax.set_title(f"Feature {feature_idx} activation across tokens")
    plt.xticks(rotation=90, fontsize=6)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ljung_box_pvalues(
    p_values: np.ndarray,
    output_path: Path | None = None,
) -> plt.Figure:
    """Histogram of Ljung-Box p-values across features.

    Under the null (no temporal structure), p-values should be uniform.
    A spike near 0 indicates widespread temporal structure.
    """
    valid = p_values[~np.isnan(p_values)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid, bins=50, range=(0, 1), edgecolor="black", linewidth=0.3)
    ax.axhline(
        y=len(valid) / 50,
        color="red",
        linestyle="--",
        linewidth=1,
        label="Expected under null",
    )
    ax.set_xlabel("Ljung-Box p-value")
    ax.set_ylabel("Count")
    ax.set_title("Ljung-Box test p-values across features")
    ax.legend()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
