"""Gaussian copula for generating correlated binary features."""

import torch
from scipy.stats import norm
from torch.distributions import MultivariateNormal

from src.utils.device import DEFAULT_DEVICE


def get_correlated_features(
    batch_size: int,
    firing_probabilities: torch.Tensor,
    correlation_matrix: torch.Tensor,
    device: torch.device = DEFAULT_DEVICE,
) -> torch.Tensor:
    """Generate correlated binary features via Gaussian copula.

    Samples from MultivariateNormal(0, corr_matrix), then thresholds each
    dimension at norm.ppf(1 - p_i) to get binary features that respect both
    marginal firing probabilities and pairwise correlations.

    Args:
        batch_size: Number of samples.
        firing_probabilities: Per-feature firing probabilities, shape (g,).
        correlation_matrix: PSD correlation matrix, shape (g, g).
        device: Torch device.

    Returns:
        Binary tensor of shape (batch_size, g).
    """
    num_features = firing_probabilities.shape[0]

    thresholds = torch.tensor(
        [norm.ppf(1 - p.item()) for p in firing_probabilities], device=device
    )

    mvn = MultivariateNormal(
        loc=torch.zeros(num_features, device=device),
        covariance_matrix=correlation_matrix.to(device),
    )

    gaussian_samples = mvn.sample((batch_size,))
    binary_features = (gaussian_samples > thresholds.unsqueeze(0)).float()
    return binary_features
