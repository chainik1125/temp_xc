"""Synthetic data generators for § 4 (TempBench synthetic).

Two generators map to the paper's two synthetic benchmarks:

- :func:`markov_chain_support` — N independent Markov chains with
  optional Bernoulli emission noise (the "denoising" bench).
- :func:`coupled_hmm`         — K hidden chains drive M emissions through
  an OR-gated coupling matrix (the "coupling" bench).

Both return ``(N, seq_len, d_in)`` tensors of activations PLUS auxiliary
arrays the synthetic evaluator uses for ground-truth feature recovery:

    @dataclass
    class SyntheticData:
        x:                  (N, seq_len, d_in)    activations
        emission_features:  (M_or_N, d_in)         local feature directions
        hidden_features:    (K, d_in)              global feature directions (coupled only)
        support:            (N, seq_len, M_or_N)   binary activation indicators
        hidden_support:     (N, seq_len, K)        binary hidden-chain indicators (coupled)

Each generator also exposes a thin :func:`build_refill` factory that
returns a callable suitable for ``ActivationBuffer`` / ``WindowBuffer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from temp_bench.core.schemas import DataSourceSpec


# ── Result containers ──────────────────────────────────────────────────


@dataclass
class SyntheticData:
    """One full synthetic dataset, materialised."""

    x: torch.Tensor                      # (N, T_seq, d_in)
    emission_features: torch.Tensor      # (M, d_in)
    hidden_features: torch.Tensor | None # (K, d_in) — None for the markov bench
    support: torch.Tensor                # (N, T_seq, M)
    hidden_support: torch.Tensor | None  # (N, T_seq, K) — None for markov
    seq_len: int
    d_in: int


# ── Markov chain + Bernoulli emission noise (§ 4 denoising) ────────────


def markov_chain_support(
    *,
    n_features: int = 20,
    d_in: int = 40,
    seq_len: int = 64,
    n_seqs: int = 4096,
    rho: float = 0.7,
    pi: float = 0.5,
    p_A: float = 0.0,
    p_B: float = 0.625,
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
) -> SyntheticData:
    """N independent 2-state Markov chains with Bernoulli emission noise.

    Hidden state h_i(t) ∈ {0, 1} follows a Markov chain with
    P(h(t+1)=1 | h(t)=1) = ρ + (1-ρ)π and P(h(t+1)=1 | h(t)=0) = (1-ρ)π.

    Emission s_i(t) ∈ {0, 1}: P(s=1 | h=1) = p_B,  P(s=1 | h=0) = p_A.

    Activation x(t) = Σ s_i(t) · m_i · f_i where f_i are orthogonal
    unit vectors in R^{d_in} and m_i are folded-normal magnitudes.

    Tests denoising: temporal models that aggregate the noisy s into
    the underlying h recover features better than per-token SAEs.
    """
    rng = np.random.default_rng(seed)

    # Orthogonal feature directions (random Gaussian then QR).
    if n_features > d_in:
        raise ValueError(f"n_features ({n_features}) > d_in ({d_in})")
    raw = rng.standard_normal((d_in, d_in))
    Q, _ = np.linalg.qr(raw)
    features = Q[:n_features]                                     # (M, d_in)

    # Magnitudes: folded normal with given mean/std.
    magnitudes = np.abs(rng.normal(magnitude_mean, magnitude_std, size=n_features))

    # Markov hidden states (n_seqs, seq_len, n_features).
    h = np.zeros((n_seqs, seq_len, n_features), dtype=np.float32)
    p11 = rho + (1 - rho) * pi
    p01 = (1 - rho) * pi
    h[:, 0, :] = (rng.random((n_seqs, n_features)) < pi).astype(np.float32)
    for t in range(1, seq_len):
        prev = h[:, t-1, :]
        p_on = prev * p11 + (1 - prev) * p01
        h[:, t, :] = (rng.random((n_seqs, n_features)) < p_on).astype(np.float32)

    # Bernoulli emission noise: s = h*p_B + (1-h)*p_A
    p_s = h * p_B + (1 - h) * p_A
    s = (rng.random((n_seqs, seq_len, n_features)) < p_s).astype(np.float32)

    # Activations: x(t) = Σ s_i(t) m_i f_i
    coeffs = s * magnitudes[None, None, :]                     # (N, T, M)
    x = coeffs @ features                                       # (N, T, d_in)

    return SyntheticData(
        x=torch.from_numpy(x.astype(np.float32)),
        emission_features=torch.from_numpy(features.astype(np.float32)),
        hidden_features=None,                                   # no separate hidden
        support=torch.from_numpy(s),
        hidden_support=torch.from_numpy(h),
        seq_len=seq_len,
        d_in=d_in,
    )


# ── Coupled HMM (§ 4 coupling) ─────────────────────────────────────────


def coupled_hmm(
    *,
    K_hidden: int = 10,
    M_emissions: int = 20,
    n_parents: int = 2,
    d_in: int = 256,
    seq_len: int = 64,
    n_seqs: int = 4096,
    rho: float = 0.7,
    pi: float = 0.05,
    p_B: float = 1.0,
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
) -> SyntheticData:
    """K hidden chains drive M emissions through an OR-gated coupling.

    Coupling matrix C ∈ {0,1}^{M x K} with ``n_parents`` ones per emission
    row (sampled without replacement). Emission j fires iff ANY of its
    parents is active AND a Bernoulli(p_B) coin lands 1.

    Tests global vs local feature recovery: TXC-style window encoders
    are predicted to align with the K hidden directions; per-token SAEs
    align with the M emission directions.
    """
    rng = np.random.default_rng(seed)

    if n_parents > K_hidden:
        raise ValueError(f"n_parents ({n_parents}) > K_hidden ({K_hidden})")
    if M_emissions > d_in:
        raise ValueError(f"M_emissions ({M_emissions}) > d_in ({d_in})")

    # Coupling: for each emission, pick n_parents hidden parents.
    C = np.zeros((M_emissions, K_hidden), dtype=np.float32)
    for j in range(M_emissions):
        parents = rng.choice(K_hidden, size=n_parents, replace=False)
        C[j, parents] = 1.0

    # Emission features: orthogonal unit vectors in R^{d_in}.
    raw = rng.standard_normal((d_in, d_in))
    Q, _ = np.linalg.qr(raw)
    f_emission = Q[:M_emissions]                              # (M, d_in)

    # Hidden features: normalised mean of children emission features.
    # f_hidden[k] = normalize(mean of f_emission[j] for j: C[j,k]==1).
    f_hidden = np.zeros((K_hidden, d_in), dtype=np.float32)
    for k in range(K_hidden):
        children = (C[:, k] == 1)
        f_hidden[k] = f_emission[children].mean(axis=0)
    norms = np.linalg.norm(f_hidden, axis=1, keepdims=True) + 1e-12
    f_hidden = f_hidden / norms

    # Hidden Markov state.
    h = np.zeros((n_seqs, seq_len, K_hidden), dtype=np.float32)
    p11 = rho + (1 - rho) * pi
    p01 = (1 - rho) * pi
    h[:, 0, :] = (rng.random((n_seqs, K_hidden)) < pi).astype(np.float32)
    for t in range(1, seq_len):
        prev = h[:, t-1, :]
        p_on = prev * p11 + (1 - prev) * p01
        h[:, t, :] = (rng.random((n_seqs, K_hidden)) < p_on).astype(np.float32)

    # OR-gated emission firing.
    parent_active = (h @ C.T) > 0                              # (N, T, M) bool
    fires = (rng.random((n_seqs, seq_len, M_emissions)) < p_B).astype(np.float32)
    s = parent_active.astype(np.float32) * fires               # (N, T, M)

    # Magnitudes per emission.
    magnitudes = np.abs(rng.normal(magnitude_mean, magnitude_std, size=M_emissions))

    coeffs = s * magnitudes[None, None, :]                     # (N, T, M)
    x = coeffs @ f_emission                                    # (N, T, d_in)

    return SyntheticData(
        x=torch.from_numpy(x.astype(np.float32)),
        emission_features=torch.from_numpy(f_emission.astype(np.float32)),
        hidden_features=torch.from_numpy(f_hidden.astype(np.float32)),
        support=torch.from_numpy(s),
        hidden_support=torch.from_numpy(h),
        seq_len=seq_len,
        d_in=d_in,
    )


# ── Refill-source factory (used by ActivationBuffer / WindowBuffer) ────


_GENERATORS = {
    "markov":  markov_chain_support,
    "coupled": coupled_hmm,
}


def _generate(spec: DataSourceSpec, *, seed: int) -> SyntheticData:
    """Look up the generator on a DataSourceSpec and call it."""
    if spec.generator is None:
        raise ValueError(
            f"Synthetic datasource {spec.name!r} missing 'generator' field."
        )
    # Convention: 'generator' is either a short name from _GENERATORS,
    # or a "module:fn" path that we resolve via importlib.
    if spec.generator in _GENERATORS:
        fn = _GENERATORS[spec.generator]
    else:
        from temp_bench.core.config import import_by_path
        fn = import_by_path(spec.generator)
    params = dict(spec.params or {})
    params["seed"] = seed
    return fn(**params)


# A module-level cache: regenerating the same synthetic dataset on every
# refill call is wasteful; we materialise once per (spec, seed) and slice.
_SYNTHETIC_CACHE: dict[tuple, SyntheticData] = {}


def _cached_synthetic(spec: DataSourceSpec, *, seed: int) -> SyntheticData:
    key = (spec.name, seed)
    if key not in _SYNTHETIC_CACHE:
        _SYNTHETIC_CACHE[key] = _generate(spec, seed=seed)
    return _SYNTHETIC_CACHE[key]


def build_refill(spec: DataSourceSpec, *, seed: int) -> Callable[[int], torch.Tensor]:
    """Return a callable ``(n_seqs) -> (n_seqs, seq_len, d_in)``.

    The buffer asks for ``n_seqs`` at a time; we sample WITH REPLACEMENT
    from the materialised dataset's sequences.
    """
    data = _cached_synthetic(spec, seed=seed)
    n_total, seq_len, _d_in = data.x.shape
    rng = np.random.default_rng(seed)

    def refill(n: int) -> torch.Tensor:
        idx = rng.integers(0, n_total, size=n)
        return data.x[idx].clone()
    return refill


def materialise(spec: DataSourceSpec, *, seed: int) -> SyntheticData:
    """Public accessor for the cached dataset (used by synthetic eval)."""
    return _cached_synthetic(spec, seed=seed)
