"""Magnitude sampling for feature activations."""

import torch

from src.data.toy.configs import MagnitudeConfig


def sample_magnitudes(
    k: int,
    T: int,
    config: MagnitudeConfig,
    rng: torch.Generator,
) -> torch.Tensor:
    """Sample magnitude values for k features over T timesteps.

    Args:
        k: Number of features.
        T: Sequence length.
        config: Magnitude sampling configuration.
        rng: Torch random number generator for reproducibility.

    Returns:
        Non-negative tensor of shape (k, T).
    """
    if config.distribution == "half_normal":
        if config.mu != 0.0:
            raise ValueError(
                f"half_normal requires mu=0 (got mu={config.mu}). "
                "Use distribution='folded_normal' for non-zero mu."
            )
        return torch.randn(k, T, generator=rng).abs() * config.sigma
    elif config.distribution == "folded_normal":
        raw = torch.randn(k, T, generator=rng) * config.sigma + config.mu
        return raw.abs()
    else:
        raise ValueError(f"Unknown magnitude distribution: {config.distribution}")
