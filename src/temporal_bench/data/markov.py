"""Markov chain support generation for synthetic temporal data.

Each feature follows an independent two-state Markov chain parameterized by
(pi, rho) where:
  - pi: stationary probability of being ON
  - rho: lag-1 autocorrelation (0 = iid, 1 = fully persistent)

Transition probabilities derived from (pi, rho):
  - P(off -> on) = beta = pi * (1 - rho)
  - P(on -> on)  = alpha = rho * (1 - pi) + pi
"""

from __future__ import annotations

import torch


def pi_rho_to_transition(
    pi: float, rho: float
) -> tuple[float, float]:
    """Convert (pi, rho) to Markov transition probabilities (alpha, beta).

    Returns:
        alpha: P(on -> on) = stay-on probability
        beta:  P(off -> on) = turn-on probability
    """
    beta = pi * (1.0 - rho)
    alpha = rho * (1.0 - pi) + pi
    return alpha, beta


def generate_markov_support(
    n_features: int,
    T: int,
    pi: float,
    rho: float,
    n_sequences: int = 1,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate binary support sequences from independent Markov chains.

    Each feature independently follows a two-state Markov chain.

    Args:
        n_features: Number of features (k).
        T: Sequence length.
        pi: Stationary firing probability (same for all features).
        rho: Lag-1 autocorrelation (same for all features).
        n_sequences: Number of independent sequences.
        generator: Optional torch random generator for reproducibility.

    Returns:
        support: (n_sequences, n_features, T) binary tensor.
    """
    alpha, beta = pi_rho_to_transition(pi, rho)
    support = torch.zeros(n_sequences, n_features, T)

    # Initialize from stationary distribution
    support[:, :, 0] = (
        torch.rand(n_sequences, n_features, generator=generator) < pi
    ).float()

    # Generate transitions
    for t in range(1, T):
        u = torch.rand(n_sequences, n_features, generator=generator)
        was_on = support[:, :, t - 1]
        # P(on at t | on at t-1) = alpha
        # P(on at t | off at t-1) = beta
        threshold = was_on * alpha + (1.0 - was_on) * beta
        support[:, :, t] = (u < threshold).float()

    return support
