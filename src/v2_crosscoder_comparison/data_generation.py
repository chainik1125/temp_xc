"""Two-position data generation with controllable cross-position correlation (rho)."""

import torch

from src.shared.correlation import _fix_correlation_matrix
from src.shared.magnitude_sampling import get_training_batch
from src.v2_crosscoder_comparison.configs import DataConfig


def build_cross_position_correlation_matrix(
    num_features: int,
    rho: float,
) -> torch.Tensor:
    """Build a block correlation matrix for two-position features.

    Structure: [[I, rho*I], [rho*I, I]] over 2*num_features dimensions.
    Clamps rho to 0.999 to avoid singular matrix.

    Args:
        num_features: Number of features per position.
        rho: Cross-position correlation strength.

    Returns:
        PSD correlation matrix of shape (2*num_features, 2*num_features).
    """
    rho = min(rho, 0.999)
    n = num_features
    matrix = torch.eye(2 * n)
    # Fill off-diagonal blocks with rho * I
    matrix[:n, n:] = rho * torch.eye(n)
    matrix[n:, :n] = rho * torch.eye(n)

    if rho > 0.99:
        matrix = _fix_correlation_matrix(matrix)

    return matrix


def generate_two_position_batch(
    batch_size: int,
    data_cfg: DataConfig,
    correlation_matrix: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate a batch of two-position feature activations.

    Uses the Gaussian copula via get_training_batch over 2*num_features dimensions,
    then reshapes to (batch, 2, num_features).

    Args:
        batch_size: Number of samples.
        data_cfg: Data configuration.
        correlation_matrix: PSD correlation matrix, shape (2*num_features, 2*num_features).
        device: Torch device.

    Returns:
        Feature activations of shape (batch, 2, num_features).
    """
    n = data_cfg.num_features

    # Firing probabilities: same for both positions
    firing_probs = torch.full((2 * n,), data_cfg.firing_prob)

    # Magnitude parameters: same for both positions
    mean_mags = torch.full((2 * n,), data_cfg.mean_magnitude)
    std_mags = torch.full((2 * n,), data_cfg.std_magnitude)

    # Generate as flat (batch, 2*n) then reshape
    flat_batch = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=firing_probs,
        correlation_matrix=correlation_matrix,
        mean_magnitudes=mean_mags,
        std_magnitudes=std_mags,
        device=device,
    )

    # Reshape to (batch, 2, num_features)
    return flat_batch.view(batch_size, 2, n)
