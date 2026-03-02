"""Temporal support generation via the reset process (two-state Markov chain).

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
