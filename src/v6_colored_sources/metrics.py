"""Empirical estimators and recovery metrics for colored-source experiments."""

from __future__ import annotations

import math

import torch


def empirical_lag_covariance(x: torch.Tensor, lag: int) -> torch.Tensor:
    """Pooled empirical lag covariance C_lag = E[x_{t+lag} x_t^T].

    Args:
        x: (n_seq, T, d) observations.
        lag: Non-negative lag.

    Returns:
        (d, d) covariance estimate.
    """
    if lag < 0:
        raise ValueError(f"lag must be non-negative, got {lag}")
    n_seq, T, d = x.shape
    if lag >= T:
        raise ValueError(f"lag={lag} >= T={T}")

    x_lead = x[:, lag:, :]              # (n_seq, T-lag, d), x_{t+lag}
    x_lag = x[:, : T - lag, :]          # (n_seq, T-lag, d), x_t

    n_pairs = n_seq * (T - lag)
    # outer products averaged: (1 / n_pairs) * sum outer(x_lead, x_lag)
    flat_lead = x_lead.reshape(n_pairs, d)
    flat_lag = x_lag.reshape(n_pairs, d)
    return (flat_lead.T @ flat_lag) / n_pairs


def squared_axis_recovery(F: torch.Tensor, F_hat: torch.Tensor) -> float:
    """Sign-invariant squared-cosine recovery.

    Implements (1/N) sum_i max_j |<f_i, f_hat_j>|^2.
    Both F (N, d) and F_hat (H, d) are expected to have unit-norm rows.

    Args:
        F: (N, d) ground-truth directions.
        F_hat: (H, d) recovered directions.

    Returns:
        Scalar in [0, 1].
    """
    F = F / F.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    F_hat = F_hat / F_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos = F @ F_hat.T          # (N, H)
    return (cos ** 2).max(dim=1).values.mean().item()


def chance_adjusted_recovery(S: float, N: int, H: int) -> float:
    """S_adj = max(0, (S - log(H)/N) / (1 - log(H)/N)).

    Implements the chance-adjusted recovery metric from the proposal: subtracts the
    asymptotic Rec score for random unit vectors (~ log(H)/N) and rescales so that
    a perfect score still maps to 1.
    """
    s_chance = math.log(H) / N
    if s_chance >= 1.0:
        return 0.0
    return max(0.0, (S - s_chance) / (1.0 - s_chance))


def random_dictionary_recovery(
    F: torch.Tensor, H: int, *, n_trials: int, generator: torch.Generator
) -> float:
    """Average squared recovery for H random unit vectors over n_trials draws."""
    N, d = F.shape
    scores = []
    for _ in range(n_trials):
        R = torch.randn(H, d, generator=generator, dtype=F.dtype)
        scores.append(squared_axis_recovery(F, R))
    return float(sum(scores) / len(scores))
