"""Data generation for delayed temporally-colored sources (Experiment 4)."""

from __future__ import annotations

import torch

from .configs import ColoredSourceConfig, rho_schedule


def generate_orthonormal_basis(N: int, d: int, *, generator: torch.Generator) -> torch.Tensor:
    """Sample F in R^{N x d} with rows orthonormal: F F^T = I_N.

    Uses QR of an iid Gaussian matrix (Haar-distributed columns of Q). Exact
    orthonormality is required for the impossibility theorem — do not use the
    gradient-based src.shared.orthogonalize, which is only approximate.

    Args:
        N: Number of feature directions.
        d: Ambient dimension. Must satisfy N <= d.
        generator: torch RNG.

    Returns:
        F of shape (N, d) with F @ F.T == I_N.
    """
    if N > d:
        raise ValueError(f"N={N} > d={d}: cannot have more orthonormal rows than dimensions.")
    A = torch.randn(d, d, generator=generator, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)  # Q in R^{d x d}, Q^T Q = I
    F = Q[:, :N].T.contiguous()  # (N, d), rows orthonormal
    return F


def sample_ar_chains(
    N: int,
    n_seq: int,
    T_chain: int,
    rho: torch.Tensor,
    D: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample latent z ~ delayed AR(1) per coordinate.

    For each residue class r in {0, ..., D-1}, the subsequence
    z[:, r, i], z[:, r+D, i], z[:, r+2D, i], ... is an independent AR(1) with
    autocorrelation rho[i] and unit stationary variance:

        z[:, r,         i] ~ N(0, 1)
        z[:, r + k*D,   i] = rho[i] * z[:, r + (k-1)*D, i] + sqrt(1 - rho[i]^2) * eta

    Different residue classes are independent. Stationary init is exact, so no
    burn-in is needed.

    Args:
        N: Number of latent coordinates.
        n_seq: Number of independent chains.
        T_chain: Chain length.
        rho: (N,) autocorrelations, distinct entries.
        D: Delay (>= 1).
        generator: torch RNG.

    Returns:
        z of shape (n_seq, T_chain, N), float64.
    """
    if D < 1:
        raise ValueError(f"D must be >= 1, got {D}")
    if rho.shape != (N,):
        raise ValueError(f"rho must have shape ({N},), got {tuple(rho.shape)}")

    rho = rho.to(dtype=torch.float64)
    innov_scale = torch.sqrt(torch.clamp(1.0 - rho * rho, min=0.0))  # (N,)

    z = torch.empty(n_seq, T_chain, N, dtype=torch.float64)
    # Initialize positions 0..D-1 from the stationary distribution N(0, I).
    init_len = min(D, T_chain)
    z[:, :init_len, :] = torch.randn(
        n_seq, init_len, N, generator=generator, dtype=torch.float64
    )

    # For each subsequent position t >= D, recurse from position t - D.
    for t in range(D, T_chain):
        eta = torch.randn(n_seq, N, generator=generator, dtype=torch.float64)
        z[:, t, :] = rho * z[:, t - D, :] + innov_scale * eta

    return z


def generate_dataset(cfg: ColoredSourceConfig) -> dict:
    """Sample F, z, and observations x = z @ F + sigma * eps.

    Args:
        cfg: ColoredSourceConfig.

    Returns:
        dict with keys
            features: F (N, d), float64
            z:        (n_seq, T_chain, N), float64
            x:        (n_seq, T_chain, d), float64
            rho:      (N,), float64
            config:   the input cfg
    """
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)

    F = generate_orthonormal_basis(cfg.N, cfg.d, generator=generator)
    rho = rho_schedule(cfg.N, cfg.rho_min, cfg.rho_max)
    z = sample_ar_chains(
        cfg.N, cfg.n_seq, cfg.T_chain, rho, cfg.D, generator=generator
    )
    eps = torch.randn(
        cfg.n_seq, cfg.T_chain, cfg.d, generator=generator, dtype=torch.float64
    )
    x = z @ F + cfg.sigma * eps

    return {"features": F, "z": z, "x": x, "rho": rho, "config": cfg}
