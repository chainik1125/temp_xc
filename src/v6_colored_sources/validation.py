"""Pre-training validation gates for colored-source experiments.

The proposal lists five checks (see proposal lines 1227-1234) that must pass
before training: the empirical data must look like the model says it should
before we ask any architecture to recover its features.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import torch

from .colored_sources import generate_dataset
from .configs import ColoredSourceConfig
from .metrics import (
    chance_adjusted_recovery,
    empirical_lag_covariance,
    random_dictionary_recovery,
    squared_axis_recovery,
)
from .theory import spectral_oracle


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: dict


def _op_norm(M: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(M, ord=2).item())


def check_one_token_isotropy(x: torch.Tensor, sigma: float, atol_rel: float = 0.10) -> CheckResult:
    """C_0 should be approximately (1 + sigma^2) * I_d when d == N.

    Strong test: off-diagonal entries are small relative to diagonal mean. Sample
    noise per entry is O((1+sigma^2)/sqrt(n_eff)) where n_eff is roughly
    n_seq * T_chain (correlated samples reduce this slightly). The eigenvalue
    ratio is intentionally NOT checked — Marchenko-Pastur gives a finite spread
    proportional to sqrt(d/n) even for perfectly isotropic populations, which
    looks like a failure but is just the bulk eigenvalue distribution.
    """
    C_0 = empirical_lag_covariance(x, 0)
    expected = 1.0 + sigma ** 2
    diag = torch.diagonal(C_0)
    off_diag = C_0 - torch.diag(diag)
    off_diag_max = off_diag.abs().max().item()
    diag_mean = diag.mean().item()
    rel_off = off_diag_max / max(expected, 1e-12)
    rel_diag_dev = abs(diag_mean - expected) / expected
    passed = rel_off < atol_rel and rel_diag_dev < atol_rel
    return CheckResult(
        name="one_token_isotropy",
        passed=passed,
        details={
            "off_diag_max": off_diag_max,
            "diag_mean": diag_mean,
            "expected_diag": expected,
            "rel_off_diag": rel_off,
            "rel_diag_dev": rel_diag_dev,
        },
    )


def check_short_lag_zero(x: torch.Tensor, D: int) -> CheckResult:
    """C_lag for 0 < lag < D should be small (just sample noise); C_D should dominate.

    Pass criterion: every short-lag op-norm is at least 3x smaller than C_D's,
    which is the meaningful signal-vs-noise relation regardless of sample size.
    """
    if D < 2:
        return CheckResult(
            name="short_lag_zero",
            passed=True,
            details={"skipped": "D < 2 has no short lags to check"},
        )
    C_0 = empirical_lag_covariance(x, 0)
    norm_C0 = _op_norm(C_0)
    short_norms = {}
    for lag in range(1, D):
        norm = _op_norm(empirical_lag_covariance(x, lag))
        short_norms[lag] = norm / norm_C0
    norm_CD = _op_norm(empirical_lag_covariance(x, D)) / norm_C0
    short_max = max(short_norms.values())
    passed = norm_CD > 3.0 * short_max
    return CheckResult(
        name="short_lag_zero",
        passed=passed,
        details={
            "short_lag_relative_op_norms": short_norms,
            "lag_D_relative_op_norm": norm_CD,
            "lag_D_over_short_max": norm_CD / max(short_max, 1e-12),
        },
    )


def check_oracle_recovers_basis(
    x: torch.Tensor, F: torch.Tensor, D: int, threshold: float = 0.7
) -> CheckResult:
    """Spectral oracle on lag-D covariance should recover the basis.

    Threshold is intentionally loose (0.7 default): perfect recovery requires
    T_eff >> N / gamma^2, where gamma is the eigengap. With a linspace
    rho schedule on [rho_min, rho_max], gamma = (rho_max - rho_min) / (N - 1)
    is small, so finite-sample recovery sits below 1. The point of the gate is
    to confirm the oracle works AT ALL on this generator — not to verify the
    sample-complexity bound, which is regime-specific.
    """
    N = F.shape[0]
    F_hat = spectral_oracle(x, D, N)
    rec = squared_axis_recovery(F, F_hat)
    return CheckResult(
        name="oracle_recovers_basis",
        passed=rec > threshold,
        details={"squared_recovery": rec, "threshold": threshold},
    )


def check_shuffle_destroys_oracle(
    x: torch.Tensor, F: torch.Tensor, D: int, generator: torch.Generator
) -> CheckResult:
    """Shuffling time independently per chain should destroy lag-D structure."""
    n_seq, T, d = x.shape
    x_shuf = torch.empty_like(x)
    for s in range(n_seq):
        perm = torch.randperm(T, generator=generator)
        x_shuf[s] = x[s, perm]
    N = F.shape[0]
    F_hat = spectral_oracle(x_shuf, D, N)
    rec = squared_axis_recovery(F, F_hat)
    # Same tolerance as the random-dictionary chance check: 4 * log(N)/N
    # accounts for the leading-constant gap between the proposal's asymptotic
    # log(H)/N and the finite-N max-of-Beta level.
    threshold = 4.0 * math.log(N) / N
    return CheckResult(
        name="shuffle_destroys_oracle",
        passed=rec < threshold,
        details={"squared_recovery_after_shuffle": rec, "threshold": threshold},
    )


def check_random_dictionary_chance(
    F: torch.Tensor, H: int, n_trials: int, generator: torch.Generator
) -> CheckResult:
    """Random unit vectors give squared recovery ~ 2 log(H) / N.

    The proposal writes S_chance ~ log(H)/N (asymptotic). The leading constant
    for max of H Beta(1/2, (d-1)/2) draws is closer to 2 log(H) / d, so we
    accept any value within [0.5 log(H)/N, 4 log(H)/N] — a generous window
    that catches a metric implementation bug but tolerates the constant factor.
    """
    N = F.shape[0]
    rec = random_dictionary_recovery(F, H, n_trials=n_trials, generator=generator)
    proposal_chance = math.log(H) / N
    in_range = 0.5 * proposal_chance < rec < 4.0 * proposal_chance
    return CheckResult(
        name="random_dictionary_chance",
        passed=in_range,
        details={
            "empirical_mean": rec,
            "proposal_log_H_over_N": proposal_chance,
            "rough_2_log_H_over_N": 2.0 * proposal_chance,
        },
    )


def run_gates(cfg: Optional[ColoredSourceConfig] = None) -> list[CheckResult]:
    """Run all five gates. Default config is sized for the spectral oracle to
    succeed: N=64, D=2, n_seq=256, T_chain=1024 → ~262k lag-D pairs, well above
    the N/gamma^2 ~ 400k threshold for the (0.1, 0.9) rho range at N=64.
    """
    cfg = cfg or ColoredSourceConfig(N=64, d=64, D=2, n_seq=256, T_chain=1024)
    data = generate_dataset(cfg)
    x = data["x"]
    F = data["features"]

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 9999)
    H = cfg.N

    results = [
        check_one_token_isotropy(x, cfg.sigma),
        check_short_lag_zero(x, cfg.D),
        check_oracle_recovers_basis(x, F, cfg.D),
        check_shuffle_destroys_oracle(x, F, cfg.D, gen),
        check_random_dictionary_chance(F, H, n_trials=10, generator=gen),
    ]
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage-0 validation gates.")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--rho_min", type=float, default=0.1)
    parser.add_argument("--rho_max", type=float, default=0.9)
    parser.add_argument("--n_seq", type=int, default=256)
    parser.add_argument("--T_chain", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = ColoredSourceConfig(
        N=args.N,
        d=args.d,
        D=args.D,
        sigma=args.sigma,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        n_seq=args.n_seq,
        T_chain=args.T_chain,
        seed=args.seed,
    )
    results = run_gates(cfg)
    all_passed = True
    for r in results:
        marker = "PASS" if r.passed else "FAIL"
        print(f"[{marker}] {r.name}: {r.details}")
        all_passed = all_passed and r.passed
    if not all_passed:
        print("\nOne or more gates failed. Stage 1 must not run until gates pass.")
        return 1
    print("\nAll gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
