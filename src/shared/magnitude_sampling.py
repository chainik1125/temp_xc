"""Magnitude sampling for binary feature activations."""

import torch

from src.shared.feature_generation import get_correlated_features
from src.utils.device import DEFAULT_DEVICE


def get_training_batch(
    batch_size: int,
    firing_probabilities: torch.Tensor,
    correlation_matrix: torch.Tensor,
    mean_magnitudes: torch.Tensor | None = None,
    std_magnitudes: torch.Tensor | None = None,
    device: torch.device = DEFAULT_DEVICE,
) -> torch.Tensor:
    """Generate a batch of feature activations.

    Combines correlated binary firing indicators with magnitude sampling:
    magnitude_i = ReLU(mean_i + std_i * epsilon_i) where epsilon ~ N(0,1).

    Args:
        batch_size: Number of samples.
        firing_probabilities: Per-feature firing probabilities, shape (g,).
        correlation_matrix: PSD correlation matrix, shape (g, g).
        mean_magnitudes: Mean magnitudes per feature. Defaults to all ones.
        std_magnitudes: Std of magnitude noise per feature. Defaults to zeros.
        device: Torch device.

    Returns:
        Feature activation tensor of shape (batch_size, g).
    """
    firing_features = get_correlated_features(
        batch_size, firing_probabilities, correlation_matrix, device
    )

    if mean_magnitudes is None:
        mean_magnitudes = torch.ones_like(firing_probabilities)
    if std_magnitudes is None:
        std_magnitudes = torch.zeros_like(firing_probabilities)

    mean_magnitudes = mean_magnitudes.to(device)
    std_magnitudes = std_magnitudes.to(device)

    firing_magnitude_delta = torch.normal(
        torch.zeros_like(firing_probabilities)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(device),
        std_magnitudes.unsqueeze(0).expand(batch_size, -1).to(device),
    )
    firing_magnitude_delta[firing_features == 0] = 0

    return (firing_features * (mean_magnitudes + firing_magnitude_delta)).relu()
