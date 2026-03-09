"""Temporal support generation via two-state Markov chains.

Provides two entry points:

1. ``generate_support_reset`` — convenience wrapper using the reset process
   parameterization (lam, p).
2. ``generate_support_markov`` — general version accepting any 2x2 transition matrix.

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
    stationary_on_prob: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate binary support sequences via an arbitrary 2x2 Markov chain.

    Runs k independent two-state Markov chains for T timesteps each.
    States: 0 = off, 1 = on.

    The transition matrix is indexed as P[from_state, to_state]:
      - P[0, 0] = P(off -> off),  P[0, 1] = P(off -> on)
      - P[1, 0] = P(on  -> off),  P[1, 1] = P(on  -> on)

    Args:
        k: Number of independent features (chains).
        T: Sequence length (number of timesteps).
        transition_matrix: 2x2 row-stochastic transition matrix.
        stationary_on_prob: Stationary probability of being in the ON state,
            used to initialize the chains.
        rng: Torch random number generator for reproducibility.

    Returns:
        Binary tensor of shape (k, T) where 1 = active, 0 = inactive.
    """
    alpha = transition_matrix[1, 1].item()  # P(on -> on)
    beta = transition_matrix[0, 1].item()  # P(off -> on)

    support = torch.zeros(k, T)

    # Initialize from stationary distribution
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
