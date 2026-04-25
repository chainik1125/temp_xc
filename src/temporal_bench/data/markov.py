"""Markov chain support generation for synthetic temporal data.

Each feature follows an independent two-state Markov chain parameterized by
(pi, rho) where:
  - pi: stationary probability of being ON
  - rho: lag-1 autocorrelation (0 = iid, 1 = fully persistent)

Transition probabilities derived from (pi, rho):
  - P(off -> on) = beta = pi * (1 - rho)
  - P(on -> on)  = alpha = rho * (1 - pi) + pi

Stochastic HMM emissions (report §2.1.1): the hidden chain h is generated as
above; the observed emission s is then sampled via
  s | h=0 ~ Bernoulli(p_A),  s | h=1 ~ Bernoulli(p_B).
Setting p_A=0, p_B=1 recovers the deterministic case s = h.
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

    # Pre-generate all random numbers at once (avoids per-step torch.rand overhead)
    all_u = torch.rand(n_sequences, n_features, T, generator=generator)

    # Initialize from stationary distribution
    support[:, :, 0] = (all_u[:, :, 0] < pi).float()

    # Generate transitions (loop is unavoidable due to Markov dependency,
    # but each step is a single vectorized op over all sequences and features)
    for t in range(1, T):
        was_on = support[:, :, t - 1]
        threshold = was_on * alpha + (1.0 - was_on) * beta
        support[:, :, t] = (all_u[:, :, t] < threshold).float()

    return support


def generate_markov_support_hetero(
    rhos: torch.Tensor,
    T: int,
    pi: float,
    n_sequences: int = 1,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate binary support with per-feature lag-1 autocorrelations.

    Same two-state Markov chain as :func:`generate_markov_support`, but each
    feature can have its own rho. Used for the heterogeneous-persistence
    setup in the report's Fig 8/9 experiments (10 features each at
    rho in {0.1, 0.4, 0.7, 0.95}).

    Args:
        rhos: (n_features,) tensor of lag-1 autocorrelations, one per feature.
        T: Sequence length.
        pi: Stationary firing probability (shared across features).
        n_sequences: Number of independent sequences.
        generator: Optional torch random generator for reproducibility.

    Returns:
        support: (n_sequences, n_features, T) binary tensor.
    """
    rhos = rhos.float()
    n_features = rhos.shape[0]
    # Per-feature transition probabilities, broadcast to (1, n_features, 1)
    beta = (pi * (1.0 - rhos)).view(1, n_features, 1)
    alpha = (rhos * (1.0 - pi) + pi).view(1, n_features, 1)

    support = torch.zeros(n_sequences, n_features, T)
    all_u = torch.rand(n_sequences, n_features, T, generator=generator)

    support[:, :, 0] = (all_u[:, :, 0] < pi).float()
    for t in range(1, T):
        was_on = support[:, :, t - 1 : t]  # (n_seq, n_feat, 1)
        threshold = was_on * alpha + (1.0 - was_on) * beta  # (n_seq, n_feat, 1)
        support[:, :, t : t + 1] = (all_u[:, :, t : t + 1] < threshold).float()

    return support


def emit(
    h: torch.Tensor,
    p_A: float,
    p_B: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Emit observed support s from hidden chain h via stochastic emissions.

    s | h=0 ~ Bernoulli(p_A), s | h=1 ~ Bernoulli(p_B). When p_A=0 and p_B=1
    the mapping is deterministic and s = h. Values of p_A > 0 introduce
    false positives; p_B < 1 introduces false negatives.

    Args:
        h: Binary hidden-state tensor of arbitrary shape.
        p_A: Emission probability given h=0.
        p_B: Emission probability given h=1.
        generator: Optional torch random generator for reproducibility.

    Returns:
        s: Binary observation tensor of the same shape as h.
    """
    if p_A == 0.0 and p_B == 1.0:
        return h.clone()
    probs = torch.where(
        h > 0,
        torch.full_like(h, float(p_B)),
        torch.full_like(h, float(p_A)),
    )
    u = torch.rand(h.shape, generator=generator)
    return (u < probs).float()
