"""Support sequence generation wrapping shared temporal_support module."""

import torch

from src.data_generation.configs import TransitionConfig
from src.shared.temporal_support import generate_support_markov


def generate_support(
    k: int,
    T: int,
    config: TransitionConfig,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate binary support sequences from a transition config.

    Args:
        k: Number of independent features (chains).
        T: Sequence length (number of timesteps).
        config: Transition matrix configuration.
        rng: Torch random number generator for reproducibility.

    Returns:
        Binary tensor of shape (k, T) where 1 = active, 0 = inactive.
    """
    return generate_support_markov(
        k=k,
        T=T,
        transition_matrix=config.matrix,
        stationary_on_prob=config.stationary_on_prob,
        rng=rng,
    )
