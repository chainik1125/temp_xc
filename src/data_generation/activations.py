"""Activation coefficient generation: support x magnitudes."""

import torch


def generate_activations(
    support: torch.Tensor,
    magnitudes: torch.Tensor,
) -> torch.Tensor:
    """Combine binary support with magnitudes to produce activation coefficients.

    a_{i,t} = s_{i,t} * m_{i,t}

    Args:
        support: Binary tensor of shape (..., k, T).
        magnitudes: Non-negative tensor of shape (..., k, T).

    Returns:
        Activation coefficient tensor of same shape as inputs.
    """
    return support * magnitudes
