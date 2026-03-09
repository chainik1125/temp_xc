"""Transition matrix construction, validation, and analysis utilities."""

import torch


def build_transition_matrix(lam: float, p: float) -> torch.Tensor:
    """Construct the reset process transition matrix.

    T(lam) = (1 - lam) * I + lam * R_S

    where R_S has every row equal to [1-p, p].

    Args:
        lam: Mixing parameter in [0, 1]. 0 = perfect memory, 1 = i.i.d.
        p: Stationary firing probability.

    Returns:
        2x2 row-stochastic transition matrix.
    """
    alpha = 1 - lam * (1 - p)  # P(on -> on)
    beta = lam * p  # P(off -> on)
    return torch.tensor([
        [1 - beta, beta],
        [1 - alpha, alpha],
    ])


def validate_transition_matrix(P: torch.Tensor) -> None:
    """Validate that P is a valid 2x2 row-stochastic transition matrix.

    Args:
        P: Matrix to validate.

    Raises:
        ValueError: If P is not valid.
    """
    if P.shape != (2, 2):
        raise ValueError(f"Transition matrix must be 2x2, got {P.shape}")
    if (P < -1e-6).any():
        raise ValueError(f"Transition matrix has negative entries: {P}")
    row_sums = P.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones(2), atol=1e-5):
        raise ValueError(f"Rows must sum to 1, got sums {row_sums}")


def stationary_distribution(P: torch.Tensor) -> torch.Tensor:
    """Compute the stationary distribution of a 2x2 transition matrix.

    For a 2-state chain with P[0,1] = beta and P[1,0] = 1-alpha:
      pi_on = beta / (beta + (1 - alpha))

    Args:
        P: 2x2 row-stochastic transition matrix.

    Returns:
        Tensor [pi_off, pi_on].
    """
    beta = P[0, 1].item()
    one_minus_alpha = P[1, 0].item()
    denom = beta + one_minus_alpha
    if denom < 1e-12:
        # Absorbing: both states are absorbing, return uniform
        return torch.tensor([0.5, 0.5])
    pi_on = beta / denom
    return torch.tensor([1 - pi_on, pi_on])


def theoretical_autocorrelation(P: torch.Tensor, max_lag: int) -> torch.Tensor:
    """Compute theoretical autocorrelation at each lag for a 2-state Markov chain.

    For a 2-state chain, Corr(s_t, s_{t+tau}) = (alpha - beta)^tau
    where alpha = P[1,1] and beta = P[0,1].

    Args:
        P: 2x2 row-stochastic transition matrix.
        max_lag: Maximum lag to compute.

    Returns:
        Tensor of shape (max_lag + 1,) with autocorrelation at each lag.
    """
    alpha = P[1, 1].item()
    beta = P[0, 1].item()
    eigenvalue = alpha - beta
    return torch.tensor([eigenvalue**tau for tau in range(max_lag + 1)])


def expected_holding_times(P: torch.Tensor) -> dict[str, float]:
    """Compute expected holding times in each state.

    The holding time in state i is geometrically distributed with parameter
    1 - P[i, i], so the expected holding time is 1 / (1 - P[i, i]).

    Args:
        P: 2x2 row-stochastic transition matrix.

    Returns:
        Dict with keys 'on' and 'off' giving expected holding times.
    """
    p_leave_off = 1 - P[0, 0].item()
    p_leave_on = 1 - P[1, 1].item()
    return {
        "off": 1.0 / p_leave_off if p_leave_off > 1e-12 else float("inf"),
        "on": 1.0 / p_leave_on if p_leave_on > 1e-12 else float("inf"),
    }
