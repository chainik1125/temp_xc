"""Temporal feature generation with global (persistent) and local (per-position) features.

Global features fire with the same Bernoulli draw for all positions in a sequence.
Local features fire independently per position.
"""

import torch

from src.shared.feature_generation import get_correlated_features
from src.utils.device import DEFAULT_DEVICE


def generate_temporal_features(
    batch_size: int,
    seq_len: int,
    global_firing_probs: torch.Tensor,
    local_firing_probs: torch.Tensor,
    global_corr_matrix: torch.Tensor | None = None,
    local_corr_matrix: torch.Tensor | None = None,
    device: torch.device = DEFAULT_DEVICE,
) -> torch.Tensor:
    """Generate binary temporal feature activations.

    Global features: same Bernoulli draw tiled across all positions.
    Local features: independent Bernoulli draw per position.

    Args:
        batch_size: Number of sequences.
        seq_len: Number of positions (T).
        global_firing_probs: Firing probabilities for global features, shape (n_global,).
        local_firing_probs: Firing probabilities for local features, shape (n_local,).
        global_corr_matrix: Correlation matrix for global features. Identity if None.
        local_corr_matrix: Correlation matrix for local features. Identity if None.
        device: Torch device.

    Returns:
        Binary tensor of shape (batch, T, n_global + n_local).
    """
    n_global = global_firing_probs.shape[0]
    n_local = local_firing_probs.shape[0]

    if global_corr_matrix is None:
        global_corr_matrix = torch.eye(n_global)
    if local_corr_matrix is None:
        local_corr_matrix = torch.eye(n_local)

    # Global features: sample once, tile to all positions
    global_features = get_correlated_features(
        batch_size, global_firing_probs, global_corr_matrix, device
    )  # (batch, n_global)
    global_features = global_features.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, T, n_global)

    # Local features: sample independently per position
    local_features = get_correlated_features(
        batch_size * seq_len, local_firing_probs, local_corr_matrix, device
    )  # (batch*T, n_local)
    local_features = local_features.view(batch_size, seq_len, n_local)  # (batch, T, n_local)

    # Concatenate: (batch, T, n_global + n_local)
    return torch.cat([global_features, local_features], dim=-1)


def generate_temporal_batch(
    batch_size: int,
    seq_len: int,
    global_firing_probs: torch.Tensor,
    local_firing_probs: torch.Tensor,
    global_corr_matrix: torch.Tensor | None = None,
    local_corr_matrix: torch.Tensor | None = None,
    mean_magnitudes: torch.Tensor | None = None,
    std_magnitudes: torch.Tensor | None = None,
    device: torch.device = DEFAULT_DEVICE,
) -> torch.Tensor:
    """Generate temporal feature activations with magnitude sampling.

    Args:
        batch_size: Number of sequences.
        seq_len: Number of positions (T).
        global_firing_probs: Firing probabilities for global features, shape (n_global,).
        local_firing_probs: Firing probabilities for local features, shape (n_local,).
        global_corr_matrix: Optional global feature correlation matrix.
        local_corr_matrix: Optional local feature correlation matrix.
        mean_magnitudes: Mean magnitudes per feature, shape (num_features,). Defaults to ones.
        std_magnitudes: Std of magnitude noise per feature, shape (num_features,). Defaults to zeros.
        device: Torch device.

    Returns:
        Feature activation tensor of shape (batch, T, num_features).
    """
    num_features = global_firing_probs.shape[0] + local_firing_probs.shape[0]

    firing_features = generate_temporal_features(
        batch_size, seq_len,
        global_firing_probs, local_firing_probs,
        global_corr_matrix, local_corr_matrix,
        device,
    )  # (batch, T, num_features), binary

    if mean_magnitudes is None:
        mean_magnitudes = torch.ones(num_features, device=device)
    if std_magnitudes is None:
        std_magnitudes = torch.zeros(num_features, device=device)

    mean_magnitudes = mean_magnitudes.to(device)
    std_magnitudes = std_magnitudes.to(device)

    # Magnitude sampling: ReLU(mean + std * epsilon)
    noise = torch.randn_like(firing_features) * std_magnitudes.unsqueeze(0).unsqueeze(0)
    noise[firing_features == 0] = 0

    return (firing_features * (mean_magnitudes.unsqueeze(0).unsqueeze(0) + noise)).relu()
