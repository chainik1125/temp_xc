"""Population covariances and the spectral oracle baseline."""

from __future__ import annotations

import torch

from .metrics import empirical_lag_covariance


def population_C_lag(
    F: torch.Tensor, rho: torch.Tensor, sigma: float, lag: int, D: int
) -> torch.Tensor:
    """Population covariance E[x_{t+lag} x_t^T] for the delayed colored-source model.

    Args:
        F: (N, d) orthonormal basis (rows orthonormal).
        rho: (N,) per-coord autocorrelations.
        sigma: Observation noise.
        lag: Non-negative lag in time.
        D: Delay parameter of the AR process.

    Returns:
        (d, d) covariance.
    """
    N, d = F.shape
    if lag == 0:
        return F.T @ F + (sigma ** 2) * torch.eye(d, dtype=F.dtype)
    if lag % D != 0:
        return torch.zeros(d, d, dtype=F.dtype)
    power = lag // D
    return F.T @ torch.diag(rho ** power) @ F


def spectral_oracle(x: torch.Tensor, lag_D: int, n_components: int) -> torch.Tensor:
    """Eigenvectors of the symmetrized empirical lag-D covariance.

    The proposal's "temporal spectral oracle": eigendecompose
    C_sym = (Ĉ_D + Ĉ_D^T) / 2 and return the top-n_components eigenvectors,
    sorted by |eigenvalue| descending. Each row of the output is a recovered
    direction.

    Args:
        x: (n_seq, T, d) observations.
        lag_D: The delay D (lag at which colored-source covariance is informative).
        n_components: Number of directions to return.

    Returns:
        (n_components, d) eigenvector rows.
    """
    C_hat = empirical_lag_covariance(x, lag_D)
    C_sym = 0.5 * (C_hat + C_hat.T)
    eigvals, eigvecs = torch.linalg.eigh(C_sym)
    order = torch.argsort(eigvals.abs(), descending=True)
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[order]
    top = eigvecs[:, :n_components].T.contiguous()  # (n_components, d)
    return top
