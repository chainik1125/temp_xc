"""Polynomial clock HMM: a finite-field synthetic experiment with an exact
local-impossibility theorem AND a constructive sparse-reconstruction
solution.

For prime ``q``, sample a hidden target ``Y ~ Unif(F_q)`` and ``h`` nuisance
coefficients ``B_0, ..., B_{h-1} ~ Unif(F_q)``, then emit the symbol stream

    Q_t = B_0 + B_1 t + ... + B_{h-1} t^{h-1} + Y t^h    (mod q)

with each symbol observed via a fixed orthonormal direction ``u_a``:

    x_t = u_{Q_t} + sigma * eps_t.

Properties (proposal Section 4–5):

- Any window of length ``W <= h`` carries zero mutual information about
  ``Y`` — in the noiseless case the conditional symbol distribution is
  identically uniform regardless of ``Y``.
- A window of length ``W = h + 1`` uniquely identifies ``Y`` via Lagrange
  interpolation in ``F_q``.
- For every ``β = (B_0, ..., B_{h-1}, Y) ∈ F_q^{h+1}`` the temporal atom

      G_β = (1 / sqrt(h + 1)) * stack(u_{P_β(0)}, ..., u_{P_β(h)})

  is a unit-norm template of shape ``(h + 1, d)`` with reconstruction-loss
  margin ``1 / (h + 1)``: the correct atom matches ``x_window`` with
  inner product ``1``; any wrong polynomial agrees on at most ``h``
  evaluations so its score is at most ``h / (h + 1)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolynomialClockConfig:
    """Parameters for one stage of the polynomial-clock experiment.

    Attributes:
        h: Polynomial degree. ``W = h + 1`` is the threshold window length.
        q: Field size; must be prime for Lagrange interpolation. The alphabet
            is ``F_q = {0, ..., q - 1}``.
        d: Ambient observation dimension. Must satisfy ``q <= d`` so the
            alphabet directions can be orthonormal.
        sigma: Observation noise standard deviation.
        n_seq: Number of independent episodes (each episode has its own
            random ``Y, B`` and runs the clock for ``T_chain`` steps).
        T_chain: Number of clock ticks per episode. Must be at least
            ``W_max`` for the longest window we'll evaluate.
        seed: RNG seed for reproducibility.
    """

    h: int = 1
    q: int = 31
    d: int = 64
    sigma: float = 0.1
    n_seq: int = 1024
    T_chain: int = 16
    seed: int = 0

    def __post_init__(self) -> None:
        if self.q < 2:
            raise ValueError(f"q must be >= 2; got {self.q}")
        if self.h < 1:
            raise ValueError(f"h must be >= 1; got {self.h}")
        if self.q > self.d:
            raise ValueError(f"need q <= d so alphabet fits; got q={self.q}, d={self.d}")
        if self.T_chain < self.h + 1:
            raise ValueError(
                f"T_chain={self.T_chain} too short for h+1={self.h + 1}"
            )

    @property
    def num_atoms(self) -> int:
        """Number of distinct polynomial atoms G_β = q^(h+1)."""
        return self.q ** (self.h + 1)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, int(math.isqrt(n)) + 1, 2):
        if n % d == 0:
            return False
    return True


def generate_alphabet_basis(q: int, d: int, *, generator: torch.Generator) -> torch.Tensor:
    """Sample ``q`` orthonormal rows in ``R^d`` via QR of a Haar-random matrix.

    Returns a tensor of shape ``(q, d)`` with ``U @ U.T = I_q``.
    """
    if q > d:
        raise ValueError(f"q={q} > d={d}")
    A = torch.randn(d, d, generator=generator, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :q].T.contiguous()


def evaluate_polynomial(coeffs: torch.Tensor, t: torch.Tensor, q: int) -> torch.Tensor:
    """Evaluate polynomials at integer points modulo q.

    Args:
        coeffs: ``(..., h_plus_1)`` integer tensor in ``F_q``. ``coeffs[..., k]``
            is the coefficient of ``t^k``. The leading entry ``coeffs[..., -1]``
            is ``Y`` for our convention.
        t: ``(...,)`` integer tensor; values are reduced mod q internally.
        q: Field size.

    Returns:
        Same broadcast shape as ``t``, integer modulo ``q``.
    """
    t_mod = (t % q).to(torch.long)
    powers = torch.stack(
        [torch.pow(t_mod, k) % q for k in range(coeffs.shape[-1])], dim=-1
    )
    return (coeffs * powers).sum(dim=-1) % q


def generate_polynomial_clock_dataset(cfg: PolynomialClockConfig) -> dict:
    """Sample episodes and return the full dataset.

    Returns a dict with:

        x:         (n_seq, T_chain, d) observations
        Q:         (n_seq, T_chain) integer symbols in F_q
        Y:         (n_seq,)         leading coefficient (the target)
        B:         (n_seq, h)       nuisance coefficients (B_0..B_{h-1})
        coeffs:    (n_seq, h+1)     concatenated [B_0, ..., B_{h-1}, Y]
        alphabet:  (q, d)           orthonormal alphabet directions
        config:    the input cfg
    """
    if not is_prime(cfg.q):
        # We allow non-prime q but warn — the impossibility theorem assumes
        # F_q is a field. Lagrange interpolation requires invertibility of
        # difference matrices, which only holds for prime q (or prime-power
        # extension fields, which we don't implement).
        raise ValueError(
            f"q={cfg.q} must be prime for the impossibility theorem to hold"
        )

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    alphabet = generate_alphabet_basis(cfg.q, cfg.d, generator=gen)

    Y = torch.randint(0, cfg.q, (cfg.n_seq,), generator=gen)
    B = torch.randint(0, cfg.q, (cfg.n_seq, cfg.h), generator=gen)
    coeffs = torch.cat([B, Y.unsqueeze(-1)], dim=-1)  # (n_seq, h+1)

    t_grid = torch.arange(cfg.T_chain, dtype=torch.long).unsqueeze(0)  # (1, T)
    Q = evaluate_polynomial(coeffs.unsqueeze(1), t_grid, cfg.q)  # (n_seq, T)

    x_clean = alphabet[Q]                                          # (n_seq, T, d)
    eps = torch.randn(cfg.n_seq, cfg.T_chain, cfg.d, generator=gen, dtype=torch.float64)
    x = x_clean + cfg.sigma * eps

    return {
        "x": x,
        "Q": Q,
        "Y": Y,
        "B": B,
        "coeffs": coeffs,
        "alphabet": alphabet,
        "config": cfg,
    }


def enumerate_coefficient_grid(h: int, q: int) -> torch.Tensor:
    """Return all ``q^(h+1)`` polynomial coefficient tuples in row-major order.

    Output shape: ``(q^(h+1), h+1)``. Row index ``i`` decodes as the base-``q``
    representation of ``i`` with the least-significant digit at column 0
    (i.e. ``B_0``) and the most-significant digit at column ``h`` (i.e. ``Y``).
    """
    grid = torch.cartesian_prod(*[torch.arange(q) for _ in range(h + 1)])
    return grid.to(torch.long)


def all_polynomial_atoms(alphabet: torch.Tensor, h: int, q: int, W: int) -> torch.Tensor:
    """Build the full bank of unit-norm temporal atoms ``G_β`` for windows of
    length ``W``.

    For each coefficient tuple ``β ∈ F_q^{h+1}``, evaluate the polynomial at
    ``t = 0, ..., W - 1`` and stack the corresponding alphabet rows, scaled
    by ``1/sqrt(W)`` so each atom has unit Frobenius norm.

    Args:
        alphabet: ``(q, d)`` orthonormal alphabet rows.
        h: Polynomial degree.
        q: Field size.
        W: Window length. Typically ``W = h + 1`` for the proposal's atoms,
            but ``W > h + 1`` is allowed (atoms are still well-defined; the
            margin argument applies for ``W >= h + 1``).

    Returns:
        ``(q^(h+1), W, d)`` tensor of unit-norm atoms.
    """
    coeffs = enumerate_coefficient_grid(h, q)                       # (M, h+1)
    M = coeffs.shape[0]
    t_grid = torch.arange(W, dtype=torch.long).unsqueeze(0)          # (1, W)
    Q = evaluate_polynomial(coeffs.unsqueeze(1), t_grid, q)          # (M, W)
    atoms = alphabet[Q]                                              # (M, W, d)
    return atoms / math.sqrt(W)
