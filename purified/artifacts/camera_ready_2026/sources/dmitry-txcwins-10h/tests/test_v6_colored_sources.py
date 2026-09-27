"""Unit tests for the v6 colored-source synthetic experiment."""

from __future__ import annotations

import math

import torch

from src.v6_colored_sources.colored_sources import (
    generate_dataset,
    generate_orthonormal_basis,
    sample_ar_chains,
)
from src.v6_colored_sources.configs import ColoredSourceConfig, eigengap, rho_schedule
from src.v6_colored_sources.metrics import (
    chance_adjusted_recovery,
    empirical_lag_covariance,
    squared_axis_recovery,
)
from src.v6_colored_sources.theory import population_C_lag, spectral_oracle
from src.v6_colored_sources.validation import run_gates


def test_orthonormal_basis_is_exact() -> None:
    gen = torch.Generator(device="cpu").manual_seed(0)
    F = generate_orthonormal_basis(N=8, d=16, generator=gen)
    assert F.shape == (8, 16)
    gram = F @ F.T
    assert torch.allclose(gram, torch.eye(8, dtype=F.dtype), atol=1e-10)


def test_rho_schedule_distinct_and_in_range() -> None:
    rho = rho_schedule(N=16, rho_min=0.1, rho_max=0.9)
    assert rho.min().item() >= 0.1 - 1e-9
    assert rho.max().item() <= 0.9 + 1e-9
    diffs = torch.diff(rho).abs()
    assert diffs.min().item() > 0
    assert eigengap(rho) > 0


def test_ar_chain_stationary_variance() -> None:
    """Each coordinate should have unit variance at every t under correct stationary init."""
    gen = torch.Generator(device="cpu").manual_seed(1)
    rho = torch.linspace(0.1, 0.9, 4, dtype=torch.float64)
    z = sample_ar_chains(N=4, n_seq=2000, T_chain=8, rho=rho, D=1, generator=gen)
    var = z.var(dim=(0, 1), unbiased=False)
    assert torch.allclose(var, torch.ones_like(var), atol=0.05), var


def test_lag_covariance_diagonal_in_true_basis() -> None:
    """C_1 should be approximately diagonal in F's basis with entries ~ rho_i (D=1)."""
    cfg = ColoredSourceConfig(N=32, d=32, D=1, sigma=0.0, n_seq=512, T_chain=64, seed=2)
    data = generate_dataset(cfg)
    x, F, rho = data["x"], data["features"], data["rho"]
    C_1 = empirical_lag_covariance(x, 1)
    # Project into the F basis: F @ C_1 @ F.T should be diag(rho)
    proj = F @ C_1 @ F.T
    diag = torch.diagonal(proj)
    off = proj - torch.diag(diag)
    assert torch.allclose(diag, rho, atol=0.1), (diag, rho)
    assert off.abs().max().item() < 0.1


def test_short_lag_zero_with_delay() -> None:
    """For D > 1, lags 1..D-1 give covariance with much smaller op-norm than lag D."""
    cfg = ColoredSourceConfig(N=32, d=32, D=4, sigma=0.0, n_seq=512, T_chain=128, seed=3)
    data = generate_dataset(cfg)
    x = data["x"]
    norm0 = torch.linalg.matrix_norm(empirical_lag_covariance(x, 0), ord=2).item()
    norm_D = torch.linalg.matrix_norm(empirical_lag_covariance(x, cfg.D), ord=2).item()
    short_norms = [
        torch.linalg.matrix_norm(empirical_lag_covariance(x, lag), ord=2).item()
        for lag in (1, 2, 3)
    ]
    assert max(short_norms) < 0.1 * norm0, short_norms
    assert norm_D > 5 * max(short_norms)


def test_population_C_lag_zero_for_short_lags() -> None:
    F = generate_orthonormal_basis(N=8, d=8, generator=torch.Generator().manual_seed(4))
    rho = rho_schedule(8, 0.1, 0.9)
    for lag in (1, 2, 3):
        C = population_C_lag(F, rho, sigma=0.1, lag=lag, D=4)
        assert torch.allclose(C, torch.zeros_like(C))


def test_population_C_lag_at_D_recovers_basis() -> None:
    F = generate_orthonormal_basis(N=8, d=8, generator=torch.Generator().manual_seed(5))
    rho = rho_schedule(8, 0.1, 0.9)
    C = population_C_lag(F, rho, sigma=0.1, lag=4, D=4)
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (C + C.T))
    eigvecs_top = eigvecs.T  # rows
    rec = squared_axis_recovery(F, eigvecs_top)
    assert rec > 0.99, rec


def test_squared_axis_recovery_perfect_and_chance() -> None:
    F = generate_orthonormal_basis(N=8, d=8, generator=torch.Generator().manual_seed(6))
    assert squared_axis_recovery(F, F) > 1.0 - 1e-6
    # Permuted F is still a perfect match (sign-invariant, axis-invariant)
    F_perm = F[torch.tensor([3, 1, 0, 2, 5, 4, 7, 6])]
    assert squared_axis_recovery(F, F_perm) > 1.0 - 1e-6
    # Sign-flipped is perfect
    assert squared_axis_recovery(F, -F) > 1.0 - 1e-6


def test_chance_adjusted_recovery_zero_at_chance() -> None:
    s_chance = math.log(8) / 8
    assert chance_adjusted_recovery(s_chance, N=8, H=8) == 0.0
    assert chance_adjusted_recovery(1.0, N=8, H=8) > 0.99
    # Below chance clamps to 0
    assert chance_adjusted_recovery(0.0, N=8, H=8) == 0.0


def test_spectral_oracle_recovers_basis_at_lag_1() -> None:
    cfg = ColoredSourceConfig(N=16, d=16, D=1, sigma=0.0, n_seq=256, T_chain=128, seed=7)
    data = generate_dataset(cfg)
    F_hat = spectral_oracle(data["x"], lag_D=1, n_components=cfg.N)
    rec = squared_axis_recovery(data["features"], F_hat)
    assert rec > 0.85, rec


def test_run_gates_passes_at_default() -> None:
    """Smoke test that all five gates pass at a small, fast config.

    Uses N=16 with a wide rho range and big sample to keep eigengap-driven
    oracle recovery achievable in ~1 second.
    """
    cfg = ColoredSourceConfig(
        N=16, d=16, D=2, sigma=0.1, rho_min=0.1, rho_max=0.9,
        n_seq=128, T_chain=512, seed=8,
    )
    results = run_gates(cfg)
    failed = [r for r in results if not r.passed]
    assert not failed, [(r.name, r.details) for r in failed]
