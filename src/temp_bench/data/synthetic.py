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

import math
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
    support: torch.Tensor | None         # (N, T_seq, M) — None for symbolic benches
    hidden_support: torch.Tensor | None  # (N, T_seq, K) — None for markov
    seq_len: int
    d_in: int
    # Optional bag of bench-specific ground-truth labels (e.g. the signed-
    # motion bench stores its hidden sign + phase here). Backward-compatible:
    # the markov/coupled generators leave this None, and evaluators that
    # don't need it ignore it.
    extra: dict | None = None


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


# ── AC-only signed motion (FrequencyBench § 5) ─────────────────────────


def signed_motion(
    *,
    M: int = 19,
    v: int = 9,
    d_in: int = 40,
    seq_len: int = 64,
    n_seqs: int = 4096,
    sigma: float = 0.0,
    seed: int = 0,
) -> SyntheticData:
    """AC-only signed-motion bench (FrequencyBench § 5).

    Each sequence has a hidden sign ``S ∈ {-1, +1}`` and a hidden phase
    ``B ~ Unif(Z_M)``. The emitted symbol walks the cyclic group::

        Q_t = (B + S · v · t)  mod M           t = 0 … seq_len-1

    and the activation is the embedding of that symbol::

        x_t = u_{Q_t} + sigma · noise

    where ``{u_0, …, u_{M-1}}`` are random orthonormal directions in R^d_in.

    Why this is an AC-only / order-sensitive test:

    - **Per-token marginal carries zero info about S.** Because B is
      uniform, ``Q_t | S=s`` is Unif(Z_M) for either sign, so
      ``I(S; Q_t) = 0`` exactly. By the data-processing inequality, for
      ANY per-token encoder ``Z = φ(x_t)`` the chain ``S → Q_t → x_t → Z``
      gives ``I(S; Z) = 0`` — no per-token SAE can read the sign off a
      single token at any width/sparsity/nonlinearity.
    - More strongly, the sign is an *interaction* term: it lives in the
      step ``Q_{t+1} − Q_t = S·v (mod M)``. A LINEAR reader of per-token
      codes (one block of features per position) can only form an additive
      score ``Σ_t h_t(Q_t)``; summed over the M phases the +v and −v orbits
      have identical totals, so additive scores cannot separate them.
    - **Window encoders can solve it.** A T-window latent that learns a
      zero-mean (AC) filter pair ``(−u_prev, +u_curr)`` exposes the step
      direction directly, so the sign becomes linearly decodable.

    Ground truths exposed (in ``extra``):
        sign_labels:  (n_seqs,)  hidden S ∈ {-1, +1}
        phase_labels: (n_seqs,)  hidden B ∈ Z_M
    plus ``emission_features`` = the M orthonormal alphabet directions.
    """
    rng = np.random.default_rng(seed)
    if math.gcd(int(v), int(M)) != 1:
        raise ValueError(
            f"v ({v}) must be coprime to M ({M}) so the orbit covers Z_M."
        )
    if M > d_in:
        raise ValueError(f"M ({M}) > d_in ({d_in})")

    # Orthonormal alphabet directions u_a ∈ R^{d_in}.
    raw = rng.standard_normal((d_in, d_in))
    Q, _ = np.linalg.qr(raw)
    alphabet = Q[:M]                                                # (M, d_in)

    # Hidden labels.
    S = rng.choice([-1, 1], size=n_seqs).astype(np.int64)           # (n_seqs,)
    B = rng.integers(0, M, size=n_seqs).astype(np.int64)            # (n_seqs,)

    # Symbol trajectory  Q_t = B + S·v·t (mod M).
    t = np.arange(seq_len)[None, :]                                 # (1, T)
    qmat = (B[:, None] + S[:, None] * v * t) % M                    # (n_seqs, T)

    # Embed: x_t = u_{Q_t}.
    x = alphabet[qmat]                                             # (n_seqs, T, d_in)
    if sigma > 0:
        x = x + sigma * rng.standard_normal(x.shape)
    x = x.astype(np.float32)

    return SyntheticData(
        x=torch.from_numpy(x),
        emission_features=torch.from_numpy(alphabet.astype(np.float32)),
        hidden_features=None,                                       # sign is not a direction
        support=None,                                               # symbolic, no support tensor
        hidden_support=None,
        seq_len=seq_len,
        d_in=d_in,
        extra={
            "sign_labels": torch.from_numpy(S),
            "phase_labels": torch.from_numpy(B),
            "M": int(M),
            "v": int(v),
        },
    )


# ── Self-exciting (Hawkes-style) backtracking bench (autoresearch #1) ──


def _self_exciting_intercept(
    w: np.ndarray, base_rate: float, seq_len: int,
    *, n_tune: int = 4000, n_iter: int = 32, tune_seed: int = 12345,
) -> float:
    """Bisect the intercept ``a`` so the trend-off stationary base rate ≈ target.

    Tuned with a FIXED rng (seed-independent) so the generative *process*
    ``(a, w)`` is identical across data seeds — only the random draws differ.
    """
    K = len(w)
    rng = np.random.default_rng(tune_seed)
    lo, hi = -8.0, 2.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        b = np.zeros((n_tune, seq_len), dtype=np.float64)
        for i in range(seq_len):
            hist = np.zeros(n_tune)
            for l in range(K):
                j = i - 1 - l
                if j >= 0:
                    hist += w[l] * b[:, j]
            p = 1.0 / (1.0 + np.exp(-(mid + hist)))
            b[:, i] = (rng.random(n_tune) < p).astype(np.float64)
        if float(b.mean()) < base_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def self_exciting(
    *,
    K: int = 2,
    tau: float = 2.0,
    alpha: float = 3.06,
    base_rate: float = 0.12,
    d_in: int = 64,
    K_c: int = 19,
    n_c: int = 3,
    mag_bt: float = 2.5,
    mag_content: float = 1.0,
    sigma: float = 0.0,
    seq_len: int = 64,
    n_seqs: int = 4096,
    seed: int = 0,
) -> SyntheticData:
    """Self-exciting (discrete Hawkes / logistic-AR) backtracking bench.

    Mirrors the *measured* self-excitation of real CoT backtracking
    (``synthetic/backtracking/measurement.md``). Two layers:

    **Layer 1 — self-exciting event dynamics.** A binary event stream ``b`` with
    hidden conditional intensity ``λ``::

        λ_i = σ( a + Σ_{l=1..K} w_l · b_{i-l} ) ,   b_i ~ Bernoulli(λ_i)

    where ``w_l = α · κ_l``, ``κ_l = exp(-l/τ)`` normalised, and ``a`` is tuned
    so the (trend-off) base rate ≈ ``base_rate``. ``K = 2`` is set by held-out
    model selection on the real labels (``backtracking_kernel_order.py``); the
    position trend is 0 (headline isolates pure self-excitation, the ``N2``
    control). The hidden intensity ``λ_i`` is a deterministic function of the
    *past* events — a per-token encoder sees only the single Bernoulli sample
    ``b_i`` (DPI floor ``corr ≤ √(Var λ/Var b)``), a window encoder sees the
    history that *sets* ``λ_i``.

    **Layer 2 — emission.** Over a fixed orthonormal dictionary
    ``{u_bt, u_1..u_{K_c}}`` (so ``F = 1 + K_c`` feature directions)::

        x_i = b_i · m · u_bt  +  Σ_{j ∈ content_i} m_j · u_j  +  σ · ε_i

    ``u_bt`` (the backtracking feature) fires iff ``b_i = 1`` with a *dominant*
    magnitude (``mag_bt > mag_content``) so it is the top-1 atom on firing
    tokens and is therefore recovered even at ``k_pos = 1`` — both per-token and
    window archs encode ``b`` equally; the per-token→window gap is purely about
    history access, not feature recovery. ``content_i`` is a random size-``n_c``
    subset of the ``K_c`` content directions (filler).

    Ground truths exposed (in ``extra``):
        lambda_labels: (n_seqs, seq_len)  hidden intensity λ (regression target)
        b_labels:      (n_seqs, seq_len)  the event stream
    plus ``emission_features`` = the ``F`` orthonormal directions (``u_bt`` +
    content), the cosine-AUC targets.
    """
    if K_c + 1 > d_in:
        raise ValueError(f"F = K_c+1 ({K_c + 1}) > d_in ({d_in})")
    if n_c > K_c:
        raise ValueError(f"n_c ({n_c}) > K_c ({K_c})")

    # Fixed self-excitation kernel + seed-independent intercept.
    l = np.arange(1, K + 1)
    kappa = np.exp(-l / tau)
    kappa = kappa / kappa.sum()
    w = alpha * kappa                                              # effective lag weights
    a = _self_exciting_intercept(w, base_rate, seq_len)

    rng = np.random.default_rng(seed)

    # Layer 1: simulate the event stream + intensity (batched over sequences).
    b = np.zeros((n_seqs, seq_len), dtype=np.float64)
    lam = np.zeros((n_seqs, seq_len), dtype=np.float64)
    for i in range(seq_len):
        hist = np.zeros(n_seqs)
        for lg in range(K):
            j = i - 1 - lg
            if j >= 0:
                hist += w[lg] * b[:, j]
        p = 1.0 / (1.0 + np.exp(-(a + hist)))
        lam[:, i] = p
        b[:, i] = (rng.random(n_seqs) < p).astype(np.float64)

    # Orthonormal dictionary: u_bt + K_c content directions (rows of an
    # orthogonal matrix → mutually orthonormal, as in signed_motion).
    raw = rng.standard_normal((d_in, d_in))
    Qd, _ = np.linalg.qr(raw)
    u_bt = Qd[0]                                                   # (d_in,)
    content = Qd[1:1 + K_c]                                        # (K_c, d_in)
    features = np.concatenate([u_bt[None, :], content], axis=0)    # (F, d_in)

    # Layer 2: emission.
    x = np.zeros((n_seqs, seq_len, d_in), dtype=np.float32)
    mbt = np.abs(rng.normal(mag_bt, 0.3 * mag_bt, size=(n_seqs, seq_len)))
    x += ((b * mbt)[:, :, None] * u_bt[None, None, :]).astype(np.float32)
    # each token lights a random size-n_c subset of the content directions
    pick = np.argsort(rng.random((n_seqs, seq_len, K_c)), axis=-1)[:, :, :n_c]
    cmag = np.abs(rng.normal(mag_content, 0.3 * mag_content,
                             size=(n_seqs, seq_len, n_c))).astype(np.float32)
    chosen = content.astype(np.float32)[pick]                     # (n_seqs, seq_len, n_c, d_in)
    x += (cmag[..., None] * chosen).sum(axis=2)
    if sigma > 0:
        x = x + (sigma * rng.standard_normal(x.shape)).astype(np.float32)

    return SyntheticData(
        x=torch.from_numpy(x),
        emission_features=torch.from_numpy(features.astype(np.float32)),
        hidden_features=None,                                      # λ is a latent, not a direction
        support=None,
        hidden_support=None,
        seq_len=seq_len,
        d_in=d_in,
        extra={
            "lambda_labels": torch.from_numpy(lam.astype(np.float32)),
            "b_labels": torch.from_numpy(b.astype(np.float32)),
            "intercept": float(a),
            "kernel_w": [float(v) for v in w],
            "K": int(K), "alpha": float(alpha), "tau": float(tau),
            "base_rate_realized": float(b.mean()),
        },
    )


# ── Refill-source factory (used by ActivationBuffer / WindowBuffer) ────


_GENERATORS = {
    "markov":  markov_chain_support,
    "coupled": coupled_hmm,
    "signed_motion": signed_motion,
    "self_exciting": self_exciting,
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
