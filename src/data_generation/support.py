"""Support sequence generation: hidden Markov chain + stochastic emission."""

import torch

from src.data_generation.configs import EmissionConfig, TransitionConfig
from src.shared.temporal_support import generate_support_markov


def generate_hidden_states(
    k: int,
    T: int,
    config: TransitionConfig,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate hidden state sequences from a transition config.

    Args:
        k: Number of independent features (chains).
        T: Sequence length (number of timesteps).
        config: Transition matrix configuration.
        rng: Torch random number generator for reproducibility.

    Returns:
        Binary tensor of shape (k, T) where 1 = state B, 0 = state A.
    """
    return generate_support_markov(
        k=k,
        T=T,
        transition_matrix=config.matrix,
        stationary_on_prob=config.stationary_on_prob,
        rng=rng,
    )


def apply_emission(
    hidden_states: torch.Tensor,
    config: EmissionConfig,
    rng: torch.Generator,
) -> torch.Tensor:
    """Sample observed support from hidden states via stochastic emission.

    s_t | z_t = A ~ Bernoulli(p_A)
    s_t | z_t = B ~ Bernoulli(p_B)

    With p_A=0, p_B=1 (defaults), the output equals the hidden states.

    Args:
        hidden_states: Binary tensor of shape (k, T).
        config: Emission probabilities.
        rng: Torch random number generator for reproducibility.

    Returns:
        Binary tensor of shape (k, T) where 1 = feature fires.
    """
    if config.p_A == 0.0 and config.p_B == 1.0:
        return hidden_states.clone()

    k, T = hidden_states.shape
    noise = torch.rand(k, T, generator=rng)
    emission_probs = torch.where(
        hidden_states > 0.5,
        torch.tensor(config.p_B),
        torch.tensor(config.p_A),
    )
    return (noise < emission_probs).float()


def generate_support(
    k: int,
    T: int,
    transition_config: TransitionConfig,
    emission_config: EmissionConfig,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate hidden states and observed support.

    Args:
        k: Number of independent features (chains).
        T: Sequence length (number of timesteps).
        transition_config: Transition matrix configuration.
        emission_config: Emission probabilities.
        rng: Torch random number generator for reproducibility.

    Returns:
        Tuple of (hidden_states, support), each shape (k, T).
        hidden_states: the raw Markov chain states.
        support: the observed (emitted) binary support.
    """
    hidden_states = generate_hidden_states(k, T, transition_config, rng)
    support = apply_emission(hidden_states, emission_config, rng)
    return hidden_states, support
