"""Temporal support generation via two-state Markov chains.

Provides three entry points:

1. ``generate_support_markov`` — general version accepting a 2x2 transition matrix
   (shared across all features) or per-feature (alpha, beta) vectors.
2. ``generate_support_reset`` — convenience wrapper using the reset process
   parameterization (lam, p).
3. ``generate_support_per_feature`` — per-feature (pi, rho) parameterization,
   following Han's convention from v2_temporal_schemeC.

The reset process transition matrix is:

    T(lam) = (1 - lam) * I + lam * R_S

where R_S has every row equal to the stationary distribution [1-p, p].
This gives transition probabilities:
  - P(on  -> on)  = 1 - lam*(1-p)
  - P(off -> on)  = lam*p
  - Stationary probability: p (for all lam)
  - Autocorrelation: (1 - lam)^|tau|

lam=0 is perfect memory; lam=1 is i.i.d. Bernoulli(p).
"""

import torch


def generate_support_markov(
    k: int,
    T: int,
    transition_matrix: torch.Tensor,
    stationary_on_prob: float | torch.Tensor,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate binary support sequences via 2-state Markov chains.

    Runs k independent two-state Markov chains for T timesteps each.
    States: 0 = off, 1 = on.

    The transition matrix is indexed as P[from_state, to_state]:
      - P[0, 0] = P(off -> off),  P[0, 1] = P(off -> on)
      - P[1, 0] = P(on  -> off),  P[1, 1] = P(on  -> on)

    Supports both shared parameters (single 2x2 matrix) and per-feature
    parameters (alpha/beta vectors extracted from the matrix rows).

    Args:
        k: Number of independent features (chains).
        T: Sequence length (number of timesteps).
        transition_matrix: 2x2 row-stochastic transition matrix (shared across
            features). For per-feature control, use generate_support_per_feature.
        stationary_on_prob: Stationary probability of being in the ON state.
            Scalar (shared) or tensor of shape (k,) (per-feature).
        rng: Torch random number generator for reproducibility.

    Returns:
        Binary tensor of shape (k, T) where 1 = active, 0 = inactive.
    """
    alpha = transition_matrix[1, 1].item()  # P(on -> on)
    beta = transition_matrix[0, 1].item()  # P(off -> on)

    support = torch.zeros(k, T)

    # Initialize from stationary distribution
    if isinstance(stationary_on_prob, torch.Tensor):
        init_probs = stationary_on_prob
    else:
        init_probs = stationary_on_prob
    support[:, 0] = (torch.rand(k, generator=rng) < init_probs).float()

    for t in range(1, T):
        u = torch.rand(k, generator=rng)
        prev = support[:, t - 1]
        support[:, t] = torch.where(
            prev == 1,
            (u < alpha).float(),
            (u < beta).float(),
        )

    return support


def generate_support_per_feature(
    k: int,
    T: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    stationary_on_prob: torch.Tensor,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate binary support with per-feature transition probabilities.

    Each feature i has its own (alpha_i, beta_i) transition probabilities:
      - alpha_i = P(on -> on) for feature i
      - beta_i = P(off -> on) for feature i

    Args:
        k: Number of independent features (chains).
        T: Sequence length (number of timesteps).
        alpha: P(on -> on) per feature, shape (k,).
        beta: P(off -> on) per feature, shape (k,).
        stationary_on_prob: Stationary probability per feature, shape (k,).
        rng: Torch random number generator for reproducibility.

    Returns:
        Binary tensor of shape (k, T) where 1 = active, 0 = inactive.
    """
    support = torch.zeros(k, T)

    # Initialize from per-feature stationary distribution
    support[:, 0] = (torch.rand(k, generator=rng) < stationary_on_prob).float()

    for t in range(1, T):
        u = torch.rand(k, generator=rng)
        prev = support[:, t - 1]
        support[:, t] = torch.where(
            prev == 1,
            (u < alpha).float(),
            (u < beta).float(),
        )

    return support


def generate_support_reset(
    k: int,
    T: int,
    p: float,
    lam: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate binary support sequences via the reset process.

    Runs k independent two-state Markov chains for T timesteps each.
    Each chain has stationary probability p of being active, with temporal
    autocorrelation controlled by lam.

    Args:
        k: Number of independent features (chains).
        T: Sequence length (number of timesteps).
        p: Stationary firing probability (sparsity).
        lam: Mixing parameter in [0, 1]. 0 = perfect memory, 1 = i.i.d.
        rng: Torch random number generator for reproducibility.

    Returns:
        Binary tensor of shape (k, T) where 1 = active, 0 = inactive.
    """
    alpha = 1 - lam * (1 - p)  # P(stay on | on)
    beta = lam * p  # P(turn on | off)

    support = torch.zeros(k, T)

    # Initialize from stationary distribution
    support[:, 0] = (torch.rand(k, generator=rng) < p).float()

    for t in range(1, T):
        u = torch.rand(k, generator=rng)
        prev = support[:, t - 1]
        # If previously on: stay on with probability alpha
        # If previously off: turn on with probability beta
        support[:, t] = torch.where(
            prev == 1,
            (u < alpha).float(),
            (u < beta).float(),
        )

    return support


def per_feature_from_pi_rho(
    pi: torch.Tensor, rho: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (pi, rho) parameterization to (alpha, beta) transition probs.

    Uses Han's convention from v2_temporal_schemeC:
        beta  = p01 = pi * (1 - rho)    (off -> on)
        alpha = 1 - p10 = 1 - (1-pi)*(1-rho) = pi + rho*(1-pi)  (on -> on)

    This is equivalent to the reset process with lam = 1 - rho, p = pi.

    Args:
        pi: Marginal activation probability per feature, shape (k,).
        rho: Lag-1 autocorrelation per feature, shape (k,).

    Returns:
        Tuple of (alpha, beta), each shape (k,).
        alpha: P(on -> on) per feature.
        beta: P(off -> on) per feature.
    """
    beta = pi * (1 - rho)
    alpha = 1 - (1 - pi) * (1 - rho)
    return alpha, beta
