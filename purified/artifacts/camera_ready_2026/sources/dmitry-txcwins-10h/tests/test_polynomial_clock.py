"""Unit tests for the polynomial-clock generator and oracles."""

from __future__ import annotations

import math

import torch

from src.v6_colored_sources.polynomial_clock import (
    PolynomialClockConfig,
    all_polynomial_atoms,
    enumerate_coefficient_grid,
    evaluate_polynomial,
    generate_alphabet_basis,
    generate_polynomial_clock_dataset,
    is_prime,
)
from src.v6_colored_sources.polynomial_clock_oracles import (
    interpolation_oracle,
    lagrange_leading_coeff,
    nearest_template_oracle,
    run_gates,
    time_shuffle,
)


def test_alphabet_basis_is_orthonormal() -> None:
    gen = torch.Generator().manual_seed(0)
    U = generate_alphabet_basis(7, 16, generator=gen)
    assert U.shape == (7, 16)
    gram = U @ U.T
    assert torch.allclose(gram, torch.eye(7, dtype=U.dtype), atol=1e-10)


def test_evaluate_polynomial_matches_naive() -> None:
    # P(t) = 3 + 2t + t^2 mod 5; check at t = 0, 1, 2, 3, 4
    coeffs = torch.tensor([3, 2, 1], dtype=torch.long)
    t = torch.arange(5)
    out = evaluate_polynomial(coeffs.unsqueeze(0), t.unsqueeze(0), q=5)
    expected = torch.tensor([(3 + 2 * int(ti) + int(ti) ** 2) % 5 for ti in t]).long()
    assert torch.equal(out.squeeze(0), expected)


def test_lagrange_leading_coeff_recovers_y() -> None:
    """Sample many random degree-2 polynomials over F_11 and verify the
    interpolation oracle recovers Y from values at t=0,1,2."""
    q = 11
    h = 2
    gen = torch.Generator().manual_seed(7)
    Y = torch.randint(0, q, (200,), generator=gen)
    B = torch.randint(0, q, (200, h), generator=gen)
    coeffs = torch.cat([B, Y.unsqueeze(-1)], dim=-1)
    t = torch.arange(h + 1).unsqueeze(0)
    Q = evaluate_polynomial(coeffs.unsqueeze(1), t, q)  # (200, 3)
    Y_hat = lagrange_leading_coeff(Q, q)
    assert torch.equal(Y_hat, Y)


def test_short_window_information_zero_in_F_q() -> None:
    """Empirical version of the local-impossibility theorem at h=2, q=5.

    Group windows by their (Q_0, Q_1) prefix (W=h=2) and check that the
    distribution of Y within each group is uniform over F_q.
    """
    q = 5
    h = 2
    cfg = PolynomialClockConfig(h=h, q=q, d=8, sigma=0.0,
                                 n_seq=10000, T_chain=4, seed=42)
    data = generate_polynomial_clock_dataset(cfg)
    Q = data["Q"][:, :h]                                        # (10000, 2)
    Y = data["Y"]
    keys = Q[:, 0] * q + Q[:, 1]                                # 25 unique groups
    for key in torch.unique(keys):
        mask = keys == key
        if mask.sum() < 50:
            continue
        Y_in_group = Y[mask]
        counts = torch.bincount(Y_in_group, minlength=q).float()
        # Expected uniform over q outcomes; chi-square style allowance
        expected = mask.sum().float() / q
        max_dev = (counts - expected).abs().max().item()
        assert max_dev < 4 * math.sqrt(expected.item()), (
            f"Y conditional on (Q_0, Q_1)={int(key)} not uniform: counts={counts}"
        )


def test_atom_norms_and_margin() -> None:
    """Each ``G_β`` should be unit-norm. Pairs of atoms agreeing on at most
    h evaluation points have inner product at most h/(h+1)."""
    h = 2
    q = 5
    gen = torch.Generator().manual_seed(0)
    U = generate_alphabet_basis(q, 16, generator=gen)
    atoms = all_polynomial_atoms(U.float(), h, q, W=h + 1)         # (M, W, d)
    norms = atoms.reshape(atoms.shape[0], -1).norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    flat = atoms.reshape(atoms.shape[0], -1)                       # (M, W*d)
    inner = flat @ flat.T
    M = atoms.shape[0]
    off_diag = inner - torch.eye(M)
    max_off = off_diag.max().item()
    margin_bound = h / (h + 1)
    assert max_off <= margin_bound + 1e-6, (
        f"max off-diagonal inner product {max_off} exceeded margin bound {margin_bound}"
    )


def test_nearest_template_oracle_perfect_when_noiseless() -> None:
    cfg = PolynomialClockConfig(h=2, q=7, d=16, sigma=0.0,
                                 n_seq=200, T_chain=4, seed=3)
    data = generate_polynomial_clock_dataset(cfg)
    x_window = data["x"][:, : cfg.h + 1, :].float()
    Y_hat = nearest_template_oracle(x_window, data["alphabet"].float(), cfg.h, cfg.q)
    assert torch.equal(Y_hat, data["Y"])


def test_interpolation_oracle_perfect() -> None:
    cfg = PolynomialClockConfig(h=2, q=11, d=16, sigma=0.0,
                                 n_seq=300, T_chain=4, seed=5)
    data = generate_polynomial_clock_dataset(cfg)
    Q_window = data["Q"][:, : cfg.h + 1]
    Y_hat = interpolation_oracle(Q_window, cfg.q)
    assert torch.equal(Y_hat, data["Y"])


def test_shuffle_destroys_signal() -> None:
    cfg = PolynomialClockConfig(h=2, q=11, d=32, sigma=0.0,
                                 n_seq=2048, T_chain=4, seed=11)
    data = generate_polynomial_clock_dataset(cfg)
    x_window = data["x"][:, : cfg.h + 1, :].float()
    gen = torch.Generator().manual_seed(99)
    x_shuf = time_shuffle(x_window, gen)
    Y_hat = nearest_template_oracle(x_shuf, data["alphabet"].float(), cfg.h, cfg.q)
    acc = (Y_hat == data["Y"]).float().mean().item()
    # Shuffled windows can still hit some accuracy because some time
    # permutations correspond to *other* valid polynomials whose leading
    # coefficient happens to coincide with Y. The relative degradation —
    # shuffled accuracy substantially smaller than the unshuffled
    # 1.0 ceiling — is the meaningful signal here.
    Y_clean = nearest_template_oracle(x_window, data["alphabet"].float(), cfg.h, cfg.q)
    acc_clean = (Y_clean == data["Y"]).float().mean().item()
    assert acc < 0.6 * acc_clean, (
        f"shuffle didn't destroy signal enough: shuf={acc:.3f} clean={acc_clean:.3f}"
    )


def test_run_gates_at_default_passes() -> None:
    results = run_gates()
    failed = [r for r in results if not r.passed]
    assert not failed, [(r.name, r.details) for r in failed]


def test_is_prime_handles_edge_cases() -> None:
    assert is_prime(2)
    assert is_prime(7)
    assert is_prime(31)
    assert is_prime(11)
    assert not is_prime(1)
    assert not is_prime(4)
    assert not is_prime(9)
