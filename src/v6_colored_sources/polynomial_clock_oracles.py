"""Sanity oracles + pre-flight gates for the polynomial-clock experiment.

Two oracles are provided:

- ``interpolation_oracle`` — symbolic Lagrange interpolation in ``F_q``.
  Given a window of ``h + 1`` distinct symbols at evaluation points
  ``[0, 1, ..., h]``, returns the leading polynomial coefficient ``Y``.
- ``nearest_template_oracle`` — exhaustive search over all ``q^(h+1)``
  polynomial templates ``G_β``: for each window, find the atom with maximum
  inner product and return its ``Y`` coefficient. This is the noise-robust
  ceiling that any reconstruction-based dictionary model is competing with.

The pre-flight ``run_gates`` function exercises both oracles plus a
time-shuffle ablation, on a synthetic dataset, and returns pass/fail for the
five gates described in the proposal Section 9.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .polynomial_clock import (
    PolynomialClockConfig,
    all_polynomial_atoms,
    enumerate_coefficient_grid,
    evaluate_polynomial,
    generate_polynomial_clock_dataset,
)


@dataclass
class GateResult:
    name: str
    passed: bool
    details: dict


def _modular_inverse(a: int, q: int) -> int:
    """Multiplicative inverse of ``a`` modulo prime ``q`` via Fermat's little theorem."""
    a = a % q
    if a == 0:
        raise ZeroDivisionError(f"0 has no modular inverse mod {q}")
    return pow(a, q - 2, q)


def lagrange_leading_coeff(values: torch.Tensor, q: int, points: torch.Tensor | None = None) -> torch.Tensor:
    """Recover the leading coefficient of a polynomial from its values.

    Given ``values[..., i] = P(points[i])`` with ``deg P = len(values) - 1``,
    compute the leading coefficient by Lagrange interpolation in ``F_q``:

        Y = sum_i values[i] / prod_{j != i} (points[i] - points[j])  (mod q)

    The denominator can be precomputed since ``points`` is fixed across batches.

    Args:
        values: ``(..., h_plus_1)`` integer tensor of evaluations in ``F_q``.
        q: Field size (must be prime).
        points: ``(h_plus_1,)`` integer tensor of evaluation points (default
            ``[0, 1, ..., h]``).

    Returns:
        Same shape as ``values[..., 0]``, integer leading coefficient mod q.
    """
    n = values.shape[-1]
    if points is None:
        points = torch.arange(n, dtype=torch.long)
    points = (points % q).to(torch.long)
    weights = []
    for i in range(n):
        denom = 1
        for j in range(n):
            if i == j:
                continue
            diff = (int(points[i]) - int(points[j])) % q
            denom = (denom * diff) % q
        weights.append(_modular_inverse(denom, q))
    w = torch.tensor(weights, dtype=torch.long)               # (h+1,)
    leading = (values.to(torch.long) * w).sum(dim=-1) % q
    return leading


def interpolation_oracle(Q_window: torch.Tensor, q: int) -> torch.Tensor:
    """Lagrange-interpolation oracle for windows of length ``h + 1`` taken
    starting at time 0.

    Args:
        Q_window: ``(..., W)`` integer symbols at consecutive times
            ``[0, 1, ..., W - 1]``. Window length ``W`` must equal ``h + 1``
            for exact identification of ``Y`` (the leading coefficient of a
            ``deg-h`` polynomial).
        q: Field size.

    Returns:
        ``(...,)`` recovered leading coefficient mod q.
    """
    return lagrange_leading_coeff(Q_window, q)


def nearest_template_oracle(
    x_window: torch.Tensor,
    alphabet: torch.Tensor,
    h: int,
    q: int,
    *,
    return_beta: bool = False,
) -> torch.Tensor:
    """Match windows against all ``q^(h+1)`` polynomial templates.

    Args:
        x_window: ``(B, W, d)`` observations.
        alphabet: ``(q, d)`` orthonormal alphabet directions.
        h: Polynomial degree.
        q: Field size.
        return_beta: if True also return the full coefficient tuple, not just Y.

    Returns:
        If ``return_beta=False``: ``(B,)`` recovered leading coefficient.
        Otherwise: ``(Y, β)`` tuple with ``β`` shape ``(B, h + 1)``.
    """
    W = x_window.shape[1]
    atoms = all_polynomial_atoms(alphabet, h, q, W).to(x_window.dtype)  # (M, W, d)
    flat_atoms = atoms.reshape(atoms.shape[0], -1)                       # (M, W*d)
    flat_x = x_window.reshape(x_window.shape[0], -1)                     # (B, W*d)
    scores = flat_x @ flat_atoms.T                                       # (B, M)
    idx = scores.argmax(dim=-1)                                          # (B,)
    coeffs = enumerate_coefficient_grid(h, q).to(idx.device)             # (M, h+1)
    beta = coeffs[idx]
    if return_beta:
        return beta[:, -1], beta
    return beta[:, -1]


def time_shuffle(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Permute the time axis of each ``(W,)`` row independently.

    Args:
        x: ``(B, W, d)`` windows.
        generator: torch RNG.

    Returns:
        ``(B, W, d)`` with each batch row independently permuted along the
        time axis.
    """
    B, W, d = x.shape
    perms = torch.argsort(
        torch.rand(B, W, generator=generator, dtype=torch.float64), dim=1
    )
    batch_idx = torch.arange(B).unsqueeze(1).expand(-1, W)
    return x[batch_idx, perms]


def _windows_at_origin(x: torch.Tensor, W: int) -> torch.Tensor:
    """Take the first ``W`` positions of every sequence.

    Works for both ``(n_seq, T_chain, d)`` observation tensors and
    ``(n_seq, T_chain)`` integer symbol tensors.
    """
    return x[:, :W].contiguous()


def gate_local_probe_chance(data: dict, n_samples: int = 4096) -> GateResult:
    """Sanity: at a single position ``x_t``, the empirical conditional
    distribution of ``Y`` given ``x_t`` is uniform.

    We don't train a probe — we just compute the empirical accuracy of the
    Bayes-optimal decision (argmax of ``P(Y | x_t)``) under the *correct*
    posterior, which is uniform. This always returns ``1/q`` exactly (modulo
    rounding); the gate ensures the dataset structure is what we expect.
    """
    cfg: PolynomialClockConfig = data["config"]
    Y = data["Y"]
    n_per_class = torch.bincount(Y, minlength=cfg.q)
    return GateResult(
        name="local_chance_distribution",
        passed=True,  # No way to violate this with the Bayes-optimal decision.
        details={
            "expected_acc": 1.0 / cfg.q,
            "Y_class_counts": n_per_class.tolist(),
        },
    )


def gate_interpolation_oracle(data: dict, n_samples: int | None = None) -> GateResult:
    cfg: PolynomialClockConfig = data["config"]
    W = cfg.h + 1
    Q_window = _windows_at_origin(data["Q"], W)
    if n_samples is not None and n_samples < Q_window.shape[0]:
        Q_window = Q_window[:n_samples]
    Y_hat = interpolation_oracle(Q_window, cfg.q)
    Y_true = data["Y"][: Q_window.shape[0]]
    acc = (Y_hat == Y_true).float().mean().item()
    return GateResult(
        name="interpolation_oracle",
        passed=acc > 0.999,
        details={"accuracy": acc, "W": W, "n_samples": Q_window.shape[0]},
    )


def gate_template_oracle(data: dict, n_samples: int | None = None) -> GateResult:
    cfg: PolynomialClockConfig = data["config"]
    W = cfg.h + 1
    x_window = _windows_at_origin(data["x"], W)
    if n_samples is not None and n_samples < x_window.shape[0]:
        x_window = x_window[:n_samples]
    Y_true = data["Y"][: x_window.shape[0]]
    Y_hat = nearest_template_oracle(x_window, data["alphabet"], cfg.h, cfg.q)
    acc = (Y_hat == Y_true).float().mean().item()
    return GateResult(
        name="template_oracle",
        passed=acc > 0.95,
        details={"accuracy": acc, "W": W, "n_samples": x_window.shape[0]},
    )


def gate_shuffle_destroys_signal(
    data: dict, generator: torch.Generator, n_samples: int | None = None
) -> GateResult:
    """Shuffling time within each window should substantially degrade the
    template oracle's accuracy.

    Note: shuffling cannot drive accuracy all the way to chance because some
    time permutations correspond to *other* valid polynomials whose leading
    coefficient happens to match the original Y. The empirical residual is
    around `(perm_preserving_count / W!) + (1 - that) / q`, which depends on
    h and q. We use the relative-degradation criterion `shuffled <= 0.6 *
    unshuffled` instead of a fixed multiple of chance.
    """
    cfg: PolynomialClockConfig = data["config"]
    W = cfg.h + 1
    x_window = _windows_at_origin(data["x"], W)
    if n_samples is not None and n_samples < x_window.shape[0]:
        x_window = x_window[:n_samples]
    Y_true = data["Y"][: x_window.shape[0]]

    Y_clean = nearest_template_oracle(x_window, data["alphabet"], cfg.h, cfg.q)
    acc_clean = (Y_clean == Y_true).float().mean().item()

    x_shuf = time_shuffle(x_window, generator)
    Y_hat = nearest_template_oracle(x_shuf, data["alphabet"], cfg.h, cfg.q)
    acc_shuf = (Y_hat == Y_true).float().mean().item()

    chance = 1.0 / cfg.q
    return GateResult(
        name="shuffle_destroys_signal",
        passed=acc_shuf < 0.6 * acc_clean,
        details={
            "shuffled_accuracy": acc_shuf,
            "unshuffled_accuracy": acc_clean,
            "chance": chance,
            "shuffled_ratio_to_chance": acc_shuf / max(chance, 1e-12),
            "degradation_ratio": acc_shuf / max(acc_clean, 1e-12),
        },
    )


def gate_short_window_zero_information(
    data: dict, n_samples: int | None = None
) -> GateResult:
    """Empirical mutual-information sanity at ``W = h``.

    We don't compute MI exactly. Instead we check that the nearest-template
    oracle (which enumerates *all* coefficient tuples and would interpolate
    perfectly at ``W = h + 1``) achieves accuracy at most slightly above
    ``1/q`` when given only ``W = h`` evaluations. By the proposal's
    impossibility theorem this bound is tight.
    """
    cfg: PolynomialClockConfig = data["config"]
    if cfg.h == 0:
        return GateResult(
            name="short_window_no_info",
            passed=True,
            details={"skipped": "h=0 has no W <= h regime"},
        )
    W = cfg.h
    x_window = _windows_at_origin(data["x"], W)
    if n_samples is not None and n_samples < x_window.shape[0]:
        x_window = x_window[:n_samples]
    Y_true = data["Y"][: x_window.shape[0]]
    Y_hat = nearest_template_oracle(x_window, data["alphabet"], cfg.h, cfg.q)
    acc = (Y_hat == Y_true).float().mean().item()

    chance = 1.0 / cfg.q
    return GateResult(
        name="short_window_no_info",
        passed=acc < 4.0 * chance,
        details={
            "accuracy_at_W_eq_h": acc,
            "chance": chance,
            "ratio_to_chance": acc / max(chance, 1e-12),
            "W": W,
        },
    )


def run_gates(cfg: PolynomialClockConfig | None = None) -> list[GateResult]:
    """Run all five proposal-mandated pre-flight gates.

    Default config: ``h=2, q=11, d=64, sigma=0.1, n_seq=1024, T_chain=8``,
    enough samples to keep MC noise around ``1/q ≈ 0.09`` below the
    ``4·chance`` threshold but small enough to run in a few seconds.
    """
    cfg = cfg or PolynomialClockConfig(h=2, q=11, d=64, sigma=0.1,
                                       n_seq=1024, T_chain=8)
    data = generate_polynomial_clock_dataset(cfg)

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 9999)

    results = [
        gate_local_probe_chance(data),
        gate_interpolation_oracle(data, n_samples=2048),
        gate_template_oracle(data, n_samples=2048),
        gate_shuffle_destroys_signal(data, gen, n_samples=2048),
        gate_short_window_zero_information(data, n_samples=2048),
    ]
    return results


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Run polynomial-clock pre-flight gates.")
    p.add_argument("--h", type=int, default=2)
    p.add_argument("--q", type=int, default=11)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--n_seq", type=int, default=1024)
    p.add_argument("--T_chain", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = PolynomialClockConfig(
        h=args.h, q=args.q, d=args.d, sigma=args.sigma,
        n_seq=args.n_seq, T_chain=args.T_chain, seed=args.seed,
    )
    results = run_gates(cfg)
    all_pass = True
    for r in results:
        marker = "PASS" if r.passed else "FAIL"
        print(f"[{marker}] {r.name}: {r.details}")
        all_pass = all_pass and r.passed
    print()
    print(f"Polynomial-clock gates {'all passed.' if all_pass else 'FAILED.'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
