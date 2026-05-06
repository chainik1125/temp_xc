"""Multi-lane Reed-Solomon HMM: secret-sharing with reconstruction-loss
separation between local-alphabet and temporal-trajectory codes.

Spec at `docs/aniket/experiments/synthetic/notes/multilane_rs_hmm_txc_proposal.tex`.

For a prime field F_q, privacy degree h, and m parallel lanes, sample
a shared secret `Y ~ Unif(F_q)` and per-lane nuisance coefficients
`B_{ell, 0}, ..., B_{ell, h-1} ~ Unif(F_q)` for each lane. Each lane runs
its own degree-h polynomial

    P_ell(z) = B_{ell, 0} + B_{ell, 1} z + ... + B_{ell, h-1} z^{h-1} + Y z^h

at the same evaluation points {alpha_0, ..., alpha_{W-1}}. At phase phi
the HMM emits an m-symbol vector

    Q_{phi, ell} = P_ell(alpha_phi)  (mod q)

and observes

    x_phi = (1/sqrt(m)) sum_ell u_{ell, Q_{phi, ell}} + sigma * eps

with `u_{ell, a}` orthonormal across (lane, symbol) pairs.

Privacy: any window of W' <= h consecutive phases is independent of `Y`
(each lane is locally private; lanes are conditionally independent).

Decodability: any one lane with W' >= h+1 evaluations identifies `Y` via
Lagrange interpolation; m lanes give redundant witnesses.

Reconstruction loss separation: at the threshold `W = h+1`, a local
alphabet decomposition with k active features over the window has
reconstruction error >= 1 - min(k, mW)/(mW). The TXC trajectory
solution with k_total = m active lane-trajectory atoms achieves the
noise floor exactly. So matching TXC's reconstruction loss with a local
code requires k_total >= mW = m(h+1) active features — a factor-W
budget gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .polynomial_clock import (
    enumerate_coefficient_grid,
    evaluate_polynomial,
    is_prime,
)


@dataclass(frozen=True)
class MultilaneRSConfig:
    """Parameters for a multi-lane Reed-Solomon HMM episode.

    Attributes:
        h: Privacy degree; any W <= h consecutive phases carry zero
            information about `Y`.
        q: Prime field size. Alphabet is F_q = {0, 1, ..., q-1}.
        m: Number of parallel lanes (each runs its own degree-h polynomial
            with shared leading coefficient `Y`).
        d: Ambient observation dimension. Default `max(m * q, 64)` so the
            `m * q` lane-symbol directions can be packed orthonormally.
        sigma: Observation noise standard deviation.
        n_seq: Number of independent episodes.
        T_chain: Chain length per episode.
        seed: RNG seed.
    """

    h: int = 1
    q: int = 11
    m: int = 32
    d: int = 512
    sigma: float = 0.1
    n_seq: int = 4096
    T_chain: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if not is_prime(self.q):
            raise ValueError(f"q={self.q} must be prime")
        if self.h < 1:
            raise ValueError(f"h={self.h} must be >= 1")
        if self.m < 1:
            raise ValueError(f"m={self.m} must be >= 1")
        if self.d < self.m * self.q:
            raise ValueError(
                f"d={self.d} too small to fit m*q={self.m * self.q} orthonormal lane-symbol directions"
            )
        if self.T_chain < self.h + 1:
            raise ValueError(
                f"T_chain={self.T_chain} must be >= h+1={self.h + 1}"
            )

    @property
    def num_atoms_per_lane(self) -> int:
        """Number of distinct lane-trajectory atoms G_{ell, beta} per lane."""
        return self.q ** (self.h + 1)

    @property
    def total_temporal_atoms(self) -> int:
        """Lane-level dictionary size required for full coverage: m * q^(h+1)."""
        return self.m * self.num_atoms_per_lane


def generate_lane_alphabet_basis(
    m: int, q: int, d: int, *, generator: torch.Generator
) -> torch.Tensor:
    """Sample `m * q` orthonormal directions in R^d via QR.

    Returns a tensor of shape `(m, q, d)` where row `(ell, a)` is `u_{ell, a}`,
    orthonormal across both indices: `<u_{ell, a}, u_{ell', a'}>` =
    indicator of `(ell, a) == (ell', a')`.
    """
    if m * q > d:
        raise ValueError(f"need d >= m*q={m*q}; got d={d}")
    A = torch.randn(d, d, generator=generator, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    flat = Q[:, : m * q].T.contiguous()        # (m*q, d)
    return flat.reshape(m, q, d)


def generate_multilane_dataset(
    cfg: MultilaneRSConfig,
    *,
    alphabet: torch.Tensor | None = None,
) -> dict:
    """Sample episodes and return the full multilane dataset.

    If `alphabet` is provided (shape `(m, q, d)`), reuse it instead of
    regenerating one from `cfg.seed`. This is essential when generating a
    held-out test set: the test split must share the *task* alphabet with
    the train split and only differ in episode-level coefficients
    `(Y, B)` and observation noise.

    Returns a dict with:

        x:        (n_seq, T_chain, d) observations
        Q:        (n_seq, T_chain, m) integer per-lane symbols in F_q
        Y:        (n_seq,)              shared leading coefficient (target)
        B:        (n_seq, m, h)         per-lane nuisance coefficients
        coeffs:   (n_seq, m, h+1)       concat [B[..., 0..h-1], Y broadcast]
        alphabet: (m, q, d)             orthonormal lane-symbol directions
        config:   the input cfg
    """
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    if alphabet is None:
        alphabet = generate_lane_alphabet_basis(cfg.m, cfg.q, cfg.d, generator=gen)
    else:
        if alphabet.shape != (cfg.m, cfg.q, cfg.d):
            raise ValueError(
                f"alphabet shape {tuple(alphabet.shape)} does not match "
                f"(m={cfg.m}, q={cfg.q}, d={cfg.d})"
            )

    Y = torch.randint(0, cfg.q, (cfg.n_seq,), generator=gen)         # (n_seq,)
    B = torch.randint(0, cfg.q, (cfg.n_seq, cfg.m, cfg.h), generator=gen)
    # coeffs[..., 0..h-1] = B; coeffs[..., h] = Y
    Y_b = Y.unsqueeze(-1).unsqueeze(-1).expand(-1, cfg.m, 1)         # (n_seq, m, 1)
    coeffs = torch.cat([B, Y_b], dim=-1)                              # (n_seq, m, h+1)

    # Evaluate every lane polynomial at t = 0, 1, ..., T_chain - 1.
    t_grid = torch.arange(cfg.T_chain, dtype=torch.long).view(1, 1, cfg.T_chain)
    Q = evaluate_polynomial(coeffs.unsqueeze(2), t_grid, cfg.q)       # (n_seq, m, T_chain)
    Q = Q.permute(0, 2, 1).contiguous()                               # (n_seq, T_chain, m)

    # Build x_phi = (1/sqrt(m)) sum_ell alphabet[ell, Q[seq, t, ell]]
    n_seq, T, m = Q.shape
    lane_idx = torch.arange(m).view(1, 1, m).expand(n_seq, T, m)
    sym_idx = Q
    selected = alphabet[lane_idx, sym_idx]                            # (n_seq, T, m, d)
    x_clean = selected.sum(dim=2) / math.sqrt(cfg.m)                  # (n_seq, T, d)

    eps = torch.randn(n_seq, T, cfg.d, generator=gen, dtype=torch.float64)
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


def all_lane_atoms(
    alphabet: torch.Tensor, h: int, q: int, W: int
) -> torch.Tensor:
    """Build the full bank of lane-level temporal atoms `G_{ell, beta}`.

    For each lane `ell` and coefficient tuple `beta in F_q^{h+1}`, evaluate the
    lane polynomial at `t = 0, ..., W - 1` and stack the corresponding alphabet
    rows scaled by `1/sqrt(W)`.

    Args:
        alphabet: `(m, q, d)` orthonormal lane-symbol directions.
        h: Privacy degree.
        q: Field size.
        W: Window length.

    Returns:
        `(m, q^(h+1), W, d)` tensor. Each `(ell, beta_idx, :, :)` slice is a
        unit-Frobenius-norm length-`W` template.
    """
    m, _, d = alphabet.shape
    coeffs = enumerate_coefficient_grid(h, q)                         # (M, h+1)
    M = coeffs.shape[0]
    t_grid = torch.arange(W, dtype=torch.long).view(1, W)             # (1, W)
    Q_per_beta = evaluate_polynomial(coeffs.unsqueeze(1), t_grid, q)  # (M, W)
    # Same per-lane: each lane sees the same polynomial Q values but with its
    # own alphabet directions u_{ell, a}.
    atoms = alphabet[:, Q_per_beta, :]                                # (m, M, W, d)
    return atoms / math.sqrt(W)


def lane_alphabet_lower_bound(
    *, m: int, W: int, k_window: int
) -> float:
    """Predicted reconstruction error for a local-alphabet code.

    From Proposition (Local alphabet reconstruction lower bound) in the
    proposal:

        ||X - X_hat||^2 >= 1 - min(k_window, m*W) / (m*W)

    Args:
        m: number of lanes.
        W: window length.
        k_window: total active alphabet atoms across the window.

    Returns:
        Squared signal error in [0, 1] (1 = total miss, 0 = perfect).
    """
    return 1.0 - min(k_window, m * W) / (m * W)


def isotropy_check(data: dict) -> dict:
    """Sanity for the multilane dataset: per-position activations are
    `(1/sqrt(m)) * sum of m orthonormal directions + noise`. The squared
    norm of the clean signal should be exactly 1, so `E ||x_phi||^2 =
    1 + sigma^2 * d` (signal energy plus noise energy)."""
    cfg: MultilaneRSConfig = data["config"]
    x = data["x"]
    norms_sq = (x ** 2).sum(dim=-1)                                   # (n_seq, T)
    expected = 1.0 + cfg.sigma ** 2 * cfg.d
    return {
        "mean_norm_sq": float(norms_sq.mean().item()),
        "expected": expected,
        "rel_err": float(abs(norms_sq.mean().item() - expected) / expected),
    }
