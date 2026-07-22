"""Synthetic data generators for § 4 + the synthetic-benchmark program.

Eight generators, registered in :data:`_GENERATORS` (name → fn) at the bottom of
this module. The first two are the paper's § 4 benches; the rest were developed
by the synthetic-benchmark program (``experiments/explorations/synthetic/``):

- :func:`markov_chain_support` — N independent Markov chains with optional
  Bernoulli emission noise (§ 4 "denoising" bench).
- :func:`coupled_hmm` — K hidden chains drive M emissions through an OR-gated
  coupling matrix (§ 4 "coupling" bench).
- :func:`signed_motion` — cyclic walk with a hidden ±sign; an AC-only /
  order-sensitive latent (the signed_motion bench).
- :func:`self_exciting` — discrete Hawkes / logistic-AR intensity; a bursty
  self-exciting latent (the backtracking bench).
- :func:`semi_markov_modes` — geometric-dwell modes + change-point structure;
  a dual DC (mode) / AC (time-since-switch) substrate (the changepoint bench).
- :func:`cyclic_tones` — cyclic walk at a hidden velocity, circle- or
  random-embedded; a periodic / frequency latent (the frequency bench).
- :func:`assumption_consequence` — 3-state {N,A,C} Markov discourse chain,
  the fitted g7 mirror of real reasoning traces; a directed-transition AC
  latent (the assumption_consequence bench, expansion stage 6).
- :func:`hedging_drift` — hierarchical AR(1) confidence stream (per-trace
  level + trend + AR residual), the fitted C3 mirror; a slow-drift DC latent
  (the hedging_drift bench, expansion stage 6).
- :func:`recipe_instruction_phase_runs` — hierarchical categorical phase
  process (per-doc propensities + empirical dwell + tilted jump chain), the
  fitted C4 ``hier_categorical`` mirror; an interaction/equality latent
  ``e_t = [c_t = c_{t-1}]`` (the recipe_instruction_phase_runs bench,
  expansion stage 6 #3).

Each datasource in ``configs/data.yaml`` names one generator; the reverse index
(generator → datasource(s) → bench) is at :data:`_GENERATORS`.

All return ``(N, seq_len, d_in)`` tensors of activations PLUS auxiliary
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

import json
import math
from dataclasses import dataclass
from pathlib import Path
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


# ── Cyclic-tone frequency bench (autoresearch #3 — periodic axis) ──────


def cyclic_tones(
    *,
    M: int = 101,
    omega: tuple = (0, 1, 2, 4, 8, 16, 24, 32, 40, 50),
    embedding: str = "circle",
    d_in: int = 128,
    sigma: float = 0.10,
    seq_len: int = 64,
    n_seqs: int = 4096,
    seed: int = 0,
) -> SyntheticData:
    """Cyclic-tone frequency bench — the periodic axis (frequency/bench_spec.md).

    Port of Dmitry's FrequencyBench (``origin/dmitry-spectral-sprint2``) onto our
    fair backbone. A symbol walks a cyclic alphabet ``Z_M`` at a **hidden
    velocity ``Y``**::

        Q_t = (B + Y·t) mod M,   B ~ Unif(Z_M),   Y ~ Unif(Ω)
        x_t = u_{Q_t} + sigma · ε_t

    The per-sequence label is the **index into Ω** (the velocity *class*), NOT
    the velocity value. Two embeddings (settled in gating, spec § 3):

    - **circle (headline):** ``u_a = R·[cos 2πa/M, sin 2πa/M]``, ``R`` a random
      ``d_in×2`` isometry. Velocity ``Y`` becomes a temporal **tone** at
      ``f = Y/M`` cycles/token; recovering ``Y`` is single-tone spectral
      estimation (ML decoder = periodogram peak-pick). The ``M`` symbol atoms
      lie on a **2-D circle** (NOT orthonormal); the reconstruction codebook is
      those ``M`` atoms and ``d_sae`` is anchored on ``M`` (not 2). ``eauc`` is
      ill-defined (adjacent atoms nearly parallel) → capability is read off
      ``NMSE`` + codebook recovery.

    - **random (symmetry null):** ``M`` orthonormal directions in ``d_in ≥ M``
      (``F = M``, like signed_motion). For prime ``M`` and an exchangeable frame
      the relabel ``a ↦ c·a`` maps ``Y ↦ cY`` bijectively over ``Z_M^*`` → all
      nonzero velocities are one orbit → **flat** frequency response (the
      ratio-invariance null; verifies "frequency" ≠ "symbol identity").

    Why velocity is invisible per-token & to a linear window reader:

    - **Per-token:** ``B`` uniform ⇒ ``Q_t | Y`` is ``Unif(Z_M)`` for every ``Y``
      ⇒ ``I(Y; x_t) = 0`` exactly (DPI: no per-token encoder recovers ``Y``).
    - **Raw-linear window:** ``E[x_t | Y] ≈ 0`` (the circle is centred, ``B``
      uniform), so the velocity lives in the **2nd moment** (the phase
      progression), not the class-conditional mean — a linear probe on the raw
      concatenated tile is at chance too. A window *win* on a trained code is
      nonlinear feature-learning, not linear access (the untrained control
      checks the nonlinear-access residual).

    Ground truths exposed (``extra``):
        velocity_labels: (n_seqs, seq_len)  int64 — Ω class index (const/seq)
        velocity:        (n_seqs, seq_len)  int64 — Y value (const/seq)
        phase_labels:    (n_seqs,)          int64 — hidden B ∈ Z_M
        circle_plane:    (d_in, 2)          — R, the circle plane (None if random)
        omega, M, embedding, sigma, freq_by_class (Y/M per Ω class)
    plus ``emission_features`` = the ``M`` symbol atoms ``{u_a}``.
    """
    if embedding not in ("circle", "random"):
        raise ValueError(f"embedding must be 'circle' or 'random', got {embedding!r}")
    if embedding == "random" and d_in < M:
        raise ValueError(f"random embedding needs d_in ({d_in}) >= M ({M})")
    omega = tuple(int(y) for y in omega)
    if any(y < 0 or y >= M for y in omega):
        raise ValueError(f"omega values must be in [0, M={M}); got {omega}")

    rng = np.random.default_rng(seed)

    # Embedding: the M symbol atoms {u_a} in R^{d_in}.
    if embedding == "circle":
        A = rng.standard_normal((d_in, 2))
        R, _ = np.linalg.qr(A)                              # (d_in, 2), isometry
        ang = 2 * np.pi * np.arange(M) / M
        V = np.stack([np.cos(ang), np.sin(ang)], axis=1)    # (M, 2)
        U = (V @ R.T).astype(np.float32)                    # (M, d_in), unit-norm rows
        circle_plane = R.astype(np.float32)
    else:
        raw = rng.standard_normal((d_in, M))
        Qd, _ = np.linalg.qr(raw)
        U = Qd.T[:M].astype(np.float32)                     # (M, d_in) orthonormal
        circle_plane = None

    # Hidden latents.
    lab = rng.integers(0, len(omega), size=n_seqs).astype(np.int64)   # Ω class idx
    Y = np.asarray(omega, dtype=np.int64)[lab]                        # velocity value
    B = rng.integers(0, M, size=n_seqs).astype(np.int64)             # phase

    # Symbol trajectory Q_t = (B + Y·t) mod M, then embed.
    t = np.arange(seq_len)[None, :]
    Q = (B[:, None] + Y[:, None] * t) % M                            # (n_seqs, seq_len)
    x = U[Q]                                                          # (n_seqs, seq_len, d_in)
    if sigma > 0:
        x = x + (sigma * rng.standard_normal(x.shape)).astype(np.float32)
    x = x.astype(np.float32)

    # Per-sequence labels broadcast to (n_seqs, seq_len) so the tiled leading-edge
    # evaluator (shared with changepoint) extracts the constant velocity per tile.
    vel_lab = np.broadcast_to(lab[:, None], (n_seqs, seq_len)).copy()
    vel_val = np.broadcast_to(Y[:, None], (n_seqs, seq_len)).copy()

    extra = {
        "velocity_labels": torch.from_numpy(vel_lab),
        "velocity": torch.from_numpy(vel_val),
        "phase_labels": torch.from_numpy(B),
        "omega": list(omega),
        "M": int(M),
        "embedding": embedding,
        "sigma": float(sigma),
        "freq_by_class": [float(y) / M for y in omega],
    }
    if circle_plane is not None:
        extra["circle_plane"] = torch.from_numpy(circle_plane)

    return SyntheticData(
        x=torch.from_numpy(x),
        emission_features=torch.from_numpy(U),          # M symbol atoms (codebook)
        hidden_features=None,                           # velocity is not a direction
        support=None,
        hidden_support=None,
        seq_len=seq_len,
        d_in=d_in,
        extra=extra,
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
    (``experiments/explorations/synthetic/backtracking/measurement.md``). Two layers:

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


# ── Semi-Markov modes / change-point bench (autoresearch #2) ───────────


def semi_markov_modes(
    *,
    K_m: int = 8,
    C: int = 12,
    spread: int = 3,
    d_in: int = 64,
    dwell: str = "geometric",
    dwell_mean: float = 1.73,
    mag_mode: float = 2.5,
    mag_content: float = 1.0,
    sigma: float = 0.0,
    seq_len: int = 64,
    n_seqs: int = 4096,
    seed: int = 0,
) -> SyntheticData:
    """Semi-Markov mode process — the dual-latent change-point bench.

    Mirrors the *measured* topic dwell (``experiments/explorations/synthetic/
    topic_switching/measurement.md``: dwell ≈ geometric, mean run 1.73). Two
    layers (``changepoint/bench_spec.md`` § 2 + amendments A1/A2):

    **Layer 1 — mode dynamics.** A categorical mode ``m_t ∈ {0..K_m-1}`` dwells
    in its state for a duration drawn from the **persistence knob** and then
    jumps, uniformly over the other modes (``Π`` uniform — the § 8 (i)
    rebalance, by design). The geometric setting (the only one currently
    ungated) is exactly a first-order Markov chain with per-step switch
    probability ``p_switch = 1/dwell_mean``. Three latents are exposed:

    - ``mode_labels`` ``m_t`` — categorical, **DC** (stamped into every token
      of the dwell → readable per-token).
    - ``time_since_switch`` ``τ_t`` — scalar, the **primary AC latent**
      (amendment A2): tokens since the last boundary, with the sequence start
      counted as a renewal (``τ_0 = 0``; ``τ_t = 0 ⟺`` boundary at ``t``).
      Needs counting beyond adjacency; provably unreadable per-token
      (``E[τ|m_t]`` is constant in ``m_t`` by Π-symmetry).
    - ``changepoint_labels`` ``c_t = [m_t ≠ m_{t-1}]`` — binary, the
      simple-floor AC companion (``c_0 = 0``).

    **Layer 2 — emission.** Over a fixed orthonormal dictionary of
    ``F = K_m + C`` directions::

        x_t = m · u^m_{m_t}  +  Σ_{j ∈ content_t} m_j · u^c_j  +  σ · ε_t

    ``u^m_{m_t}`` is the **mode-signature** direction (dominant magnitude
    ``mag_mode > mag_content`` so it is the top-1 atom and survives
    ``k_pos = 1`` — the DC readout is clean); ``content_t`` is a random
    size-``spread`` subset of the ``C`` content directions, **mode-independent**
    (amendment A1) so ``x_t ⊥ past | m_t`` exactly and the per-token AC floor
    is a DPI statement, not an estimate.

    Feature-direction sets are reported separately (spec § 3): the ``C``
    content directions go in ``emission_features`` (→ ``eauc``) and the
    ``K_m`` mode-signature directions in ``hidden_features`` (→ ``gauc``),
    never pooled.
    """
    if dwell != "geometric":
        raise ValueError(
            f"dwell={dwell!r}: only the 'geometric' persistence-knob setting is "
            "ungated (changepoint/bench_spec.md amendment A1); heavy-tailed / "
            "absorbing variants stay gated on a validated real measurement."
        )
    if K_m < 2:
        raise ValueError(f"K_m ({K_m}) must be >= 2")
    if K_m + C > d_in:
        raise ValueError(f"F = K_m+C ({K_m + C}) > d_in ({d_in})")
    if spread > C:
        raise ValueError(f"spread ({spread}) > C ({C})")
    if dwell_mean <= 1.0:
        raise ValueError(f"dwell_mean ({dwell_mean}) must be > 1")
    p_switch = 1.0 / dwell_mean

    rng = np.random.default_rng(seed)

    # Layer 1: geometric-dwell semi-Markov modes (= Markov-1, uniform Π).
    m = np.zeros((n_seqs, seq_len), dtype=np.int64)
    m[:, 0] = rng.integers(0, K_m, n_seqs)
    for t in range(1, seq_len):
        switch = rng.random(n_seqs) < p_switch
        jump = 1 + rng.integers(0, K_m - 1, n_seqs)   # uniform over OTHER modes
        m[:, t] = np.where(switch, (m[:, t - 1] + jump) % K_m, m[:, t - 1])
    c = np.zeros((n_seqs, seq_len), dtype=np.float32)
    c[:, 1:] = (m[:, 1:] != m[:, :-1]).astype(np.float32)
    tau = np.zeros((n_seqs, seq_len), dtype=np.float32)
    for t in range(1, seq_len):
        tau[:, t] = np.where(c[:, t] == 1, 0.0, tau[:, t - 1] + 1.0)

    # Orthonormal dictionary: K_m mode-signature + C content directions.
    raw = rng.standard_normal((d_in, d_in))
    Qd, _ = np.linalg.qr(raw)
    U_m = Qd[:K_m].astype(np.float32)                 # (K_m, d_in)
    U_c = Qd[K_m:K_m + C].astype(np.float32)          # (C, d_in)

    # Layer 2: emission.
    mm = np.abs(rng.normal(mag_mode, 0.3 * mag_mode,
                           size=(n_seqs, seq_len))).astype(np.float32)
    x = mm[..., None] * U_m[m]                        # (n_seqs, seq_len, d_in)
    pick = np.argsort(rng.random((n_seqs, seq_len, C)), axis=-1)[:, :, :spread]
    cmag = np.abs(rng.normal(mag_content, 0.3 * mag_content,
                             size=(n_seqs, seq_len, spread))).astype(np.float32)
    x += (cmag[..., None] * U_c[pick]).sum(axis=2)
    if sigma > 0:
        x = x + (sigma * rng.standard_normal(x.shape)).astype(np.float32)

    return SyntheticData(
        x=torch.from_numpy(x),
        emission_features=torch.from_numpy(U_c),       # content set → eauc
        hidden_features=torch.from_numpy(U_m),         # mode-signature set → gauc
        support=None,
        hidden_support=None,
        seq_len=seq_len,
        d_in=d_in,
        extra={
            "mode_labels": torch.from_numpy(m),
            "changepoint_labels": torch.from_numpy(c),
            "time_since_switch": torch.from_numpy(tau),
            "K_m": int(K_m),
            "dwell": dwell,
            "dwell_mean": float(dwell_mean),
            "p_switch": float(p_switch),
            "base_switch_rate_realized": float(c.mean()),
        },
    )


# ── Assumption→consequence directed-grammar bench (expansion C1/C2 g7) ─


# Canonical g7 mirror (assumption_consequence/mirror_params_g7.json, frozen
# 2026-07-14): 3-state {N, A, C} Markov chain fit on 207 strict-labeled
# (ctx=0) R1-Distill traces, held-out validated, gate-8 PASS. Full precision
# — the YAML datasource pins the same values explicitly.
_AC_G7_P = (
    (0.7912243453644727, 0.058386411889596604, 0.15038924274593066),
    (0.38893617021276594, 0.24765957446808512, 0.36340425531914894),
    (0.3607421875, 0.0455078125, 0.59375),
)
_AC_G7_PI = (0.6434792380738327, 0.06635949879193122, 0.2901612631342361)


def assumption_consequence(
    *,
    P: tuple = _AC_G7_P,
    pi: tuple = _AC_G7_PI,
    d_in: int = 64,
    K_c: int = 17,
    n_c: int = 3,
    mag_state: float = 2.5,
    mag_content: float = 1.0,
    sigma: float = 0.0,
    seq_len: int = 64,
    n_seqs: int = 4096,
    seed: int = 0,
) -> SyntheticData:
    """Assumption→consequence directed-grammar bench (expansion C1, g7 re-exam).

    Mirrors the *measured* directed discourse grammar of real R1-Distill
    reasoning traces (``experiments/explorations/synthetic/assumption_consequence/
    bench_spec.md``): assumptions precede their consequences — directed
    asymmetry 0.297 under the strict per-sentence labeler, ≫ nulls. Two layers:

    **Layer 1 — 3-state Markov discourse chain.** Per sequence, a state stream
    ``s_i ∈ {0:N, 1:A, 2:C}`` (neither / assumption / consequence) from the
    fitted g7 transition matrix (``P[A→C] = 0.363`` vs unconditional C rate
    0.290 — the directed edge). Matched: the forward transition matrix (hence
    the directed asymmetry, marginal, dwell geometry). Deliberately NOT
    matched: order-2+ structure (the real stream has it — a documented
    fidelity limit; on this order-1 mirror the one-step conditional given
    ``s_i`` is the Bayes ceiling for next-state prediction).

    **Layer 2 — emission** over a fixed orthonormal dictionary of
    ``F = 3 + K_c`` directions, the backtracking/changepoint pattern::

        x_i = m · u_{s_i}  +  Σ_{j ∈ content_i} m_j · u_j  +  σ · ε_i

    ``u_{s_i}`` is the discourse-state direction (dominant magnitude
    ``mag_state > mag_content`` → top-1 atom, survives ``k_pos = 1``);
    ``content_i`` is a random size-``n_c`` subset of the ``K_c`` content
    directions, state-independent (so ``x_i ⊥ past | s_i`` exactly).

    Direction sets reported separately: the 3 state directions go in
    ``hidden_features`` (→ ``gauc``), the ``K_c`` content directions in
    ``emission_features`` (→ ``eauc``).

    Ground truths exposed (in ``extra``):
        state_labels:      (n_seqs, seq_len)  s_i ∈ {0,1,2} (DC probe target)
        next_state_labels: (n_seqs, seq_len)  s_{i+1} (AC probe target;
                           last column = -1, an invalid-sentinel to mask)
        P, pi:             the generating chain (evaluator computes the
                           Bayes-balanced oracle rule from these)
    """
    P_arr = np.asarray(P, dtype=np.float64)
    pi_arr = np.asarray(pi, dtype=np.float64)
    n_states = P_arr.shape[0]
    if P_arr.shape != (n_states, n_states) or pi_arr.shape != (n_states,):
        raise ValueError(f"P {P_arr.shape} / pi {pi_arr.shape} malformed")
    if not (np.allclose(P_arr.sum(1), 1.0, atol=1e-6)
            and np.isclose(pi_arr.sum(), 1.0, atol=1e-6)):
        raise ValueError("P rows / pi must sum to 1")
    if n_states + K_c > d_in:
        raise ValueError(f"F = n_states+K_c ({n_states + K_c}) > d_in ({d_in})")
    if n_c > K_c:
        raise ValueError(f"n_c ({n_c}) > K_c ({K_c})")

    rng = np.random.default_rng(seed)

    # Layer 1: Markov state stream (vectorised inverse-CDF sampling).
    cumP = np.cumsum(P_arr, axis=1)
    s = np.zeros((n_seqs, seq_len), dtype=np.int64)
    s[:, 0] = rng.choice(n_states, size=n_seqs, p=pi_arr)
    for t in range(1, seq_len):
        u = rng.random(n_seqs)
        s[:, t] = (u[:, None] > cumP[s[:, t - 1]]).sum(axis=1)
    next_s = np.full_like(s, -1)
    next_s[:, :-1] = s[:, 1:]

    # Orthonormal dictionary: n_states state + K_c content directions.
    raw = rng.standard_normal((d_in, d_in))
    Qd, _ = np.linalg.qr(raw)
    U_s = Qd[:n_states].astype(np.float32)                # state-signature dirs
    U_c = Qd[n_states:n_states + K_c].astype(np.float32)  # content dirs

    # Layer 2: emission (the changepoint/backtracking pattern).
    ms = np.abs(rng.normal(mag_state, 0.3 * mag_state,
                           size=(n_seqs, seq_len))).astype(np.float32)
    x = ms[..., None] * U_s[s]                            # (n_seqs, seq_len, d_in)
    pick = np.argsort(rng.random((n_seqs, seq_len, K_c)), axis=-1)[:, :, :n_c]
    cmag = np.abs(rng.normal(mag_content, 0.3 * mag_content,
                             size=(n_seqs, seq_len, n_c))).astype(np.float32)
    x += (cmag[..., None] * U_c[pick]).sum(axis=2)
    if sigma > 0:
        x = x + (sigma * rng.standard_normal(x.shape)).astype(np.float32)

    fwd = float((s[:, 1:][s[:, :-1] == 1] == 2).mean())   # P(C@t+1 | A@t)
    return SyntheticData(
        x=torch.from_numpy(x),
        emission_features=torch.from_numpy(U_c),          # content set → eauc
        hidden_features=torch.from_numpy(U_s),            # state set → gauc
        support=None,
        hidden_support=None,
        seq_len=seq_len,
        d_in=d_in,
        extra={
            "state_labels": torch.from_numpy(s),
            "next_state_labels": torch.from_numpy(next_s),
            "P": [[float(v) for v in row] for row in P_arr],
            "pi": [float(v) for v in pi_arr],
            "n_states": int(n_states),
            "fwd_rate_realized": fwd,
        },
    )


# ── Hedging-drift DC confidence-persistence bench (expansion C1/C3) ────


# Empirical per-trace levels of the canonical hier_ar1 mirror
# (hedging_drift/mirror_params_hier.json, C3 rider, gate-8 PASS on
# ACF(2)+ACF(4)): 210 per-trace mean deviations, heavy tails preserved,
# sampled with replacement per generated sequence.
_HEDGING_LEVELS_HIER = (
    0.11815517434222396, -0.017228059174385805, -0.22657062895090546,
    -0.11099083229997792, 0.18454048216020563, -0.21503262855213431,
    0.005637268987687844, 0.28184509652355794, 0.1098856046434224,
    -0.1633641634406751, -0.004463741113322262, -0.023699658187811907,
    -0.15888342902844152, -0.07842041888395734, -0.015826265528714828,
    0.08218300090909914, 0.26426804101986473, -0.015826265528714828,
    -0.20638331070652144, -0.02466576131534247, 0.11535403536195293,
    -0.15646196289288952, -0.08437705529594106, -0.03112458257512287,
    0.1310201002562563, -0.27458253910133, -0.11380232396817762,
    0.2111875998632353, 0.4169121302306601, 0.31120307819102944,
    0.14834714310496086, 0.23406442851211073, -0.2541535919847598,
    0.2441562375348173, -0.18387596174636148, -0.030641453840830474,
    -0.1733641634406751, 0.1218056034074863, -0.010594432940559607,
    -0.16789635580161477, -0.33736457050425933, -0.13663185794100358,
    0.05614231949273834, -0.0179201260598149, -0.13701596115600834,
    -0.04381585513040103, 0.21287957928217854, 0.4351925856302975,
    -0.02105153944965656, 0.0790186228559182, -0.07922479895566382,
    0.04466115649954761, -0.03112458257512287, 0.19984065234415144,
    -0.008136076827996429, 0.29769230258709084, -0.07710159406937576,
    -0.14631689330469969, 0.005624127443074163, -0.20713177569984983,
    0.21553542259489006, -0.08437705529594106, 0.024484391337194267,
    -0.13099923087045648, 0.17053850530876705, -0.03508844251415327,
    0.07877348023281193, -0.060115408656076987, 0.06501674532632187,
    0.2447666562504754, -0.29562050578635946, -0.20782978237902128,
    0.1745637449940456, -0.012830092970420594, 0.1870600022868231,
    -0.02466576131534247, 0.05759384027681657, 0.057641839491176156,
    -0.03158915426570104, -0.023078371619039948, -0.014564751214332365,
    -0.018316588007011494, -0.23991012195956538, -0.19965022663803303,
    -0.014564751214332365, -0.06163711508052495, 0.1218056034074863,
    0.1240394738776498, 0.29996322809327464, -0.289873712219839,
    0.05127497922850602, -0.1091792030270794, 0.10653203561314982,
    -0.005585759146635736, -0.014564751214332365, 0.20254451886239266,
    0.022683535317375315, -0.04869406832452692, -0.20638331070652144,
    0.13406277078710685, 0.011615229017977555, 0.1218056034074863,
    0.06883535672491403, -0.039149921340344784, -0.20782978237902128,
    0.051274979228506004, -0.24677205871698396, -0.2232257569332134,
    0.26426804101986473, -0.18995453383989117, 0.22929191957280018,
    0.2540986476945504, -0.026316893304699676, -0.039781504304042606,
    -0.30631689330469974, 0.08650837510398181, -0.07299832170486396,
    0.019778916236380045, -0.01848650027009571, -0.06907541296077026,
    -0.09842130205692153, -0.08782829799415166, 0.23866907607530474,
    -0.03610837999373732, 0.003358176045566786, 0.06849634718720155,
    -0.004463741113322262, -0.15864698363643165, 0.003358176045566786,
    -0.09351309523301324, 0.07342937104909451, 0.01837815400230678,
    0.09713724351736687, -0.07966106168396772, -0.02105153944965656,
    0.19410534558317583, 0.1218056034074863, -0.2150326285521344,
    -0.013819112359780693, 0.25406435289658413, -0.03145268901418209,
    0.06577192079198917, 0.1218056034074863, 0.3612030781910295,
    0.08124344666226109, -0.00853976994778159, -0.08035474916829638,
    -0.0019605277530293096, 0.05685089425712289, -0.06963325572162928,
    0.1663771337798823, 0.13481457985646755, 0.024462723207075984,
    -0.043364163440675056, 0.26426804101986473, -0.06017105243813966,
    0.012442418594542228, -0.19026028410816048, -0.02083753227220303,
    0.2447666562504754, -0.1659357347486851, -0.045272214555809986,
    -0.04986393446740889, 0.2447666562504754, 0.04575297343013075,
    0.023972751296285174, -0.1402214910157469, -0.015826265528714828,
    0.3201060951650829, -0.1733641634406751, -0.03945498554625225,
    0.07663583655932495, 0.10714882384404538, 0.16485500712996787,
    -0.133566973332239, -0.19204412280500788, 0.6904509230352205,
    0.9789484605503436, 0.3920238907284889, -0.15616148646975103,
    0.015528580207536127, 0.37679295444682775, 0.08770328125321593,
    -0.015789840779288807, 0.0627927189073616, -0.17360132700966685,
    0.0314664661634081, -0.05660616730990104, 0.13329921401561856,
    0.04220565024594305, 0.1381462566338799, -0.014564751214332365,
    0.4464394288709064, -0.15505143764868912, -0.20638331070652144,
    -0.12984569699606685, -0.6784855902897917, -0.3217181043153558,
    -0.1733641634406751, 0.23841100529906428, 0.06917796111207744,
    -0.08362286622011769, 0.25644553482467064, 0.18117680318110746,
    -0.3463168933046997, 0.046628737486249586, -0.1678963558016148,
    0.14885701225631096, -0.04297157159820293, 0.1218056034074863,
)


def hedging_drift(
    *,
    mu: float = 0.7803249330559888,
    beta_position: float = 0.2283620815852247,
    rho: float = 0.24758953448987794,
    ar_sigma: float = 0.5436530037277453,
    levels: tuple | None = None,
    d_in: int = 64,
    K_c: int = 19,
    n_c: int = 3,
    mag_conf: float = 2.5,
    mag_content: float = 1.0,
    sigma: float = 0.0,
    seq_len: int = 64,
    n_seqs: int = 4096,
    seed: int = 0,
) -> SyntheticData:
    """Hedging-drift bench — DC confidence persistence (expansion C1, C3 mirror).

    Mirrors the *measured* slow drift + persistence of expressed confidence in
    real R1-Distill traces (``experiments/explorations/synthetic/hedging_drift/
    bench_spec.md``): ACF(1) = 0.316 ≫ nulls, with a long-memory *plateau*
    (ACF ≈ 0.13 through lag 8) that only the C3 **hierarchical AR(1)** mirror
    reproduces (gate-8 PASS on ACF(2)+ACF(4)). Two layers:

    **Layer 1 — hierarchical AR(1)** (``mirror_params_hier.json``; sampling
    matches ``explorations.synthetic.expansion.mirrors.gen_hier_ar1``)::

        c_i = mu + beta·(i/L) + l_j + r_i ,   r_i = rho·r_{i-1} + ar_sigma·eta_i

    with ``l_j`` drawn per sequence from the 210 EMPIRICAL per-trace levels
    (heavy tails preserved) and ``r_0`` at the stationary sd. Matched: the
    trend, the per-trace level distribution, lag-1 persistence. Deliberately
    NOT matched: ACF lags ≥ 2 (the plateau *emerges* from the level variance)
    and the confidence↔correctness coupling.

    **Layer 2 — emission**: the confidence feature's *magnitude* carries
    ``c_i`` (continuous loading)::

        x_i = c_i · m · u_conf  +  Σ_{j ∈ content_i} m_j · u_j  +  σ · ε_i

    ``m, m_j`` folded-normal per token (backtracking conventions) — so a
    per-token reading of ``c_i`` carries irreducible multiplicative noise and
    temporal aggregation (the DC axis under test) is what denoises it.
    ``F = 1 + K_c``; the confidence direction goes in ``hidden_features``
    (→ ``gauc``), content in ``emission_features`` (→ ``eauc``).

    Ground truths exposed (in ``extra``):
        conf_labels:  (n_seqs, seq_len)  the confidence state c_i (probe target)
        level_labels: (n_seqs,)          the per-sequence slow level l_j
    """
    if K_c + 1 > d_in:
        raise ValueError(f"F = 1+K_c ({K_c + 1}) > d_in ({d_in})")
    if n_c > K_c:
        raise ValueError(f"n_c ({n_c}) > K_c ({K_c})")
    levels_arr = np.asarray(
        _HEDGING_LEVELS_HIER if levels is None else levels, dtype=np.float64)
    if levels_arr.ndim != 1 or levels_arr.size == 0:
        raise ValueError("levels must be a non-empty 1-D sequence")

    rng = np.random.default_rng(seed)

    # Layer 1: hierarchical AR(1) confidence stream (matches gen_hier_ar1).
    pos = np.arange(seq_len, dtype=np.float64) / seq_len
    l = rng.choice(levels_arr, size=n_seqs)
    stat_sd = ar_sigma / max(np.sqrt(max(1.0 - rho ** 2, 1e-9)), 1e-9)
    r = np.zeros((n_seqs, seq_len), dtype=np.float64)
    r[:, 0] = stat_sd * rng.standard_normal(n_seqs)
    for i in range(1, seq_len):
        r[:, i] = rho * r[:, i - 1] + ar_sigma * rng.standard_normal(n_seqs)
    c = mu + beta_position * pos[None, :] + l[:, None] + r

    # Orthonormal dictionary: u_conf + K_c content directions.
    raw = rng.standard_normal((d_in, d_in))
    Qd, _ = np.linalg.qr(raw)
    u_conf = Qd[0].astype(np.float32)                     # (d_in,)
    content = Qd[1:1 + K_c].astype(np.float32)            # (K_c, d_in)

    # Layer 2: emission — magnitude-carried confidence + content filler.
    mconf = np.abs(rng.normal(mag_conf, 0.3 * mag_conf, size=(n_seqs, seq_len)))
    x = ((c * mconf)[:, :, None] * u_conf[None, None, :]).astype(np.float32)
    pick = np.argsort(rng.random((n_seqs, seq_len, K_c)), axis=-1)[:, :, :n_c]
    cmag = np.abs(rng.normal(mag_content, 0.3 * mag_content,
                             size=(n_seqs, seq_len, n_c))).astype(np.float32)
    x += (cmag[..., None] * content[pick]).sum(axis=2)
    if sigma > 0:
        x = x + (sigma * rng.standard_normal(x.shape)).astype(np.float32)

    return SyntheticData(
        x=torch.from_numpy(x),
        emission_features=torch.from_numpy(content),      # content set → eauc
        hidden_features=torch.from_numpy(u_conf[None, :]),  # conf dir → gauc
        support=None,
        hidden_support=None,
        seq_len=seq_len,
        d_in=d_in,
        extra={
            "conf_labels": torch.from_numpy(c.astype(np.float32)),
            "level_labels": torch.from_numpy(l.astype(np.float32)),
            "mu": float(mu), "beta_position": float(beta_position),
            "rho": float(rho), "ar_sigma": float(ar_sigma),
        },
    )


# ── Recipe-instruction phase-runs bench (expansion C3→C4, stage 6 #3) ──


# Canonical C4 hier_categorical mirror, pinned as package data. The file is a
# byte-identical copy (sha256 855637b7e0730a…) of the frozen spec artifact
# experiments/explorations/synthetic/recipe_instruction_phase_runs/
# mirror_params.json (275 per-doc propensity vectors + per-symbol empirical
# dwell lists + the alpha-tilted global jump chain) — too large to pin in YAML
# the way the g7 P/pi are.
_RECIPE_MIRROR_FILE = (Path(__file__).resolve().parent / "params"
                       / "recipe_instruction_hier_categorical.json")
_RECIPE_MIRROR_CACHE: dict | None = None


def _recipe_mirror_params() -> dict:
    global _RECIPE_MIRROR_CACHE
    if _RECIPE_MIRROR_CACHE is None:
        _RECIPE_MIRROR_CACHE = json.loads(_RECIPE_MIRROR_FILE.read_text())
    return _RECIPE_MIRROR_CACHE


def recipe_instruction_phase_runs(
    *,
    d_in: int = 64,
    K_c: int = 15,
    n_c: int = 3,
    mag_phase: float = 2.5,
    mag_content: float = 1.0,
    sigma: float = 0.0,
    seq_len: int = 64,
    n_seqs: int = 4096,
    seed: int = 0,
) -> SyntheticData:
    """Recipe-instruction phase-runs bench — interaction/equality (C4 SPEC).

    Mirrors the *measured* functional-phase run structure of instructional web
    text (``experiments/explorations/synthetic/recipe_instruction_phase_runs/
    bench_spec.md``): 5 per-sentence content classes (other / context /
    ingredient-listing / imperative-step / tip-caution) with self-match
    ACF(1) = 0.479 ≫ nulls. Two layers:

    **Layer 1 — hierarchical categorical phase dynamics** (the C4
    ``hier_categorical`` fit, gate-8 PASS on held-out ACF(4)+MI(2); sampling
    matches ``explorations.synthetic.expansion.mirrors.gen_hier_categorical``).
    Per sequence, a phase propensity vector ``pi_j`` is drawn from the 275
    EMPIRICAL deconvolved per-doc vectors (heavy tails preserved); the phase
    ``c_t`` then dwells per the per-symbol empirical dwell lists and, on a
    jump out of ``c``, moves to ``d`` with probability::

        P(d | c, j) = alpha · pi_j(d | d != c) + (1 - alpha) · J(c, d)

    (``alpha = 0.94`` MLE-fit; ``J`` the global no-self-jump chain). Matched:
    dwell, one-step jumps, doc-marginal heterogeneity. Deliberately NOT
    matched: ACF(4) / MI(2) (gate-8-verified on held-out docs) and anything
    beyond order-1-with-hierarchy.

    **Layer 2 — emission** over a fixed orthonormal dictionary of
    ``F = 5 + K_c`` directions, the changepoint/assumption pattern::

        x_t = m · u_{c_t}  +  Σ_{j ∈ content_t} m_j · u_j  +  σ · ε_t

    ``u_{c_t}`` is the phase-signature direction (dominant magnitude
    ``mag_phase > mag_content`` → top-1 atom, survives ``k_pos = 1``);
    ``content_t`` is a random size-``n_c`` subset of the ``K_c`` content
    directions, phase-independent (so ``x_t ⊥ past | c_t`` exactly).

    Direction sets reported separately: the 5 phase directions go in
    ``hidden_features`` (→ ``gauc``), the ``K_c`` content directions in
    ``emission_features`` (→ ``eauc``).

    Ground truths exposed (in ``extra``):
        phase_class_labels: (n_seqs, seq_len)  c_t ∈ {0..4} (DC control probe)
        equality_labels:    (n_seqs, seq_len)  e_t = [c_t = c_{t-1}] — the
                            PRIMARY regime-3 latent. Convention (spec § 3 /
                            briefing): position 0 of each sequence has no
                            predecessor, so ``e_0 = 0``.
    """
    params = _recipe_mirror_params()
    n_phases = int(params["n_symbols"])
    alpha = float(params["alpha"])
    J = np.asarray(params["jump_P"], dtype=np.float64)
    props = np.asarray(params["doc_props"], dtype=np.float64)
    dwell = [np.asarray(d, dtype=np.int64) for d in params["dwell"]]
    if n_phases + K_c > d_in:
        raise ValueError(f"F = n_phases+K_c ({n_phases + K_c}) > d_in ({d_in})")
    if n_c > K_c:
        raise ValueError(f"n_c ({n_c}) > K_c ({K_c})")

    rng = np.random.default_rng(seed)

    # Layer 1: hier_categorical phase stream (matches gen_hier_categorical).
    c = np.empty((n_seqs, seq_len), dtype=np.int64)
    doc_idx = rng.integers(0, len(props), size=n_seqs)
    for i in range(n_seqs):
        pi_j = props[doc_idx[i]]
        cur = int(rng.choice(n_phases, p=pi_j))
        t = 0
        while t < seq_len:
            r = int(rng.choice(dwell[cur]))
            c[i, t:t + r] = cur
            t += r
            tilde = pi_j.copy()
            tilde[cur] = 0.0
            tot = tilde.sum()
            if tot > 0:
                p = alpha * (tilde / tot) + (1.0 - alpha) * J[cur]
            else:   # doc propensity degenerate on cur — global chain only
                p = J[cur].copy()
            p = np.clip(p, 0.0, None)
            p /= p.sum()
            cur = int(rng.choice(n_phases, p=p))
    e = np.zeros((n_seqs, seq_len), dtype=np.float32)
    e[:, 1:] = (c[:, 1:] == c[:, :-1]).astype(np.float32)   # e_0 = 0 convention

    # Orthonormal dictionary: n_phases phase-signature + K_c content dirs.
    raw = rng.standard_normal((d_in, d_in))
    Qd, _ = np.linalg.qr(raw)
    U_p = Qd[:n_phases].astype(np.float32)                 # phase-signature dirs
    U_c = Qd[n_phases:n_phases + K_c].astype(np.float32)   # content dirs

    # Layer 2: emission (the changepoint/assumption pattern).
    mp = np.abs(rng.normal(mag_phase, 0.3 * mag_phase,
                           size=(n_seqs, seq_len))).astype(np.float32)
    x = mp[..., None] * U_p[c]                             # (n_seqs, seq_len, d_in)
    pick = np.argsort(rng.random((n_seqs, seq_len, K_c)), axis=-1)[:, :, :n_c]
    cmag = np.abs(rng.normal(mag_content, 0.3 * mag_content,
                             size=(n_seqs, seq_len, n_c))).astype(np.float32)
    x += (cmag[..., None] * U_c[pick]).sum(axis=2)
    if sigma > 0:
        x = x + (sigma * rng.standard_normal(x.shape)).astype(np.float32)

    return SyntheticData(
        x=torch.from_numpy(x),
        emission_features=torch.from_numpy(U_c),           # content set → eauc
        hidden_features=torch.from_numpy(U_p),             # phase set → gauc
        support=None,
        hidden_support=None,
        seq_len=seq_len,
        d_in=d_in,
        extra={
            "phase_class_labels": torch.from_numpy(c),
            "equality_labels": torch.from_numpy(e),
            "n_phases": int(n_phases),
            "alpha": alpha,
            "match_rate_realized": float(e[:, 1:].mean()),
        },
    )


# ── Refill-source factory (used by ActivationBuffer / WindowBuffer) ────


# Generator name → fn. Reverse index (generator → configs/data.yaml datasource(s)
# → bench) so it's obvious which generator feeds which result:
#   markov            → toy_markov_n20_d40_noisy                     (§ 4 denoising)
#   coupled           → toy_coupled_K10_M20_d256,
#                       toy_coupled_noisy_K10_M20_d256_pB05_np10,
#                       synth_smoke                                   (§ 4 coupling)
#   signed_motion     → toy_signed_motion_M19_d40                    (signed_motion)
# ── Multilane superposition (FreqBench FB-2, theorem-first) ────────────


def multilane_tones(
    *,
    M: int = 101,
    omega: tuple = (0, 1, 2, 4, 8, 16, 24, 32, 40, 50),
    n_lanes: int = 3,
    d_in: int = 24,
    sigma: float = 0.25,
    seq_len: int = 64,
    n_seqs: int = 4096,
    seed: int = 0,
) -> SyntheticData:
    """Multilane superposition — 3 simultaneous circle tones (FB-2).

    ``n_lanes`` independent circle processes, each in its own 2-plane; the
    planes are mutually orthogonal, embedded in ``R^{d_in}`` by one
    Haar-random isometry ``P`` (QR of a Gaussian). Lane ``k`` carries a
    hidden velocity ``Y_k ~ Unif(Ω)`` and phase ``B_k ~ Unif(Z_M)``::

        θ_k(t) = 2π (B_k + Y_k·t mod M) / M
        x_t    = Σ_k [cos θ_k(t) · P_k0 + sin θ_k(t) · P_k1] + σ ε_t

    The frozen card is ``freqbench/cards/FB-2.md``: per-lane ceilings are the
    per-lane periodogram (orthogonal planes ⇒ exact lane separation, P5);
    per-token information is provably zero per lane (P1); additive-over-time
    readouts are mean-degenerate (P2). Whole-window template count
    ``|Ω|^n_lanes · M^n_lanes ≈ 1e9`` kills the memorization route (P6) by
    construction.

    Ground truths exposed (``extra``):
        lane_velocity_labels: (n_seqs, seq_len, n_lanes) int64 — Ω class idx
        lane_velocity:        (n_seqs, seq_len, n_lanes) int64 — Y values
        lane_phase_labels:    (n_seqs, n_lanes)          int64 — hidden B_k
        lane_planes:          (n_lanes, d_in, 2)         — the orthogonal planes
        lane_of_atom:         (n_lanes·M,)               int64 — codebook lane tag
        omega, M, n_lanes, sigma, freq_by_class
    plus ``emission_features`` = the ``n_lanes·M`` per-lane circle atoms
    (the reconstruction codebook — derived, NOT the strict direction count;
    the strict feature directions are the ``2·n_lanes`` plane axes, exposed
    via ``lane_planes``).
    """
    omega = tuple(int(y) for y in omega)
    if any(y < 0 or y >= M for y in omega):
        raise ValueError(f"omega values must be in [0, M={M}); got {omega}")
    if d_in < 2 * n_lanes:
        raise ValueError(f"d_in ({d_in}) must be >= 2·n_lanes ({2 * n_lanes})")

    rng = np.random.default_rng(seed)

    # One Haar-random isometry: n_lanes mutually orthogonal 2-planes.
    A = rng.standard_normal((d_in, 2 * n_lanes))
    P, _ = np.linalg.qr(A)                                   # (d_in, 2·n_lanes)
    P = P.astype(np.float32)

    # Hidden latents, independent per lane.
    lab = rng.integers(0, len(omega), size=(n_seqs, n_lanes)).astype(np.int64)
    Y = np.asarray(omega, dtype=np.int64)[lab]               # (n_seqs, n_lanes)
    B = rng.integers(0, M, size=(n_seqs, n_lanes)).astype(np.int64)

    # Phase trajectories and the superposed activation.
    t = np.arange(seq_len)
    theta = 2 * np.pi * ((B[:, :, None] + Y[:, :, None] * t[None, None, :]) % M) / M
    cos_t = np.cos(theta).astype(np.float32)                 # (n_seqs, n_lanes, seq_len)
    sin_t = np.sin(theta).astype(np.float32)
    x = np.zeros((n_seqs, seq_len, d_in), dtype=np.float32)
    for k in range(n_lanes):
        x += cos_t[:, k, :, None] * P[None, None, :, 0 + 2 * k]
        x += sin_t[:, k, :, None] * P[None, None, :, 1 + 2 * k]
    if sigma > 0:
        x = x + (sigma * rng.standard_normal(x.shape)).astype(np.float32)

    # Reconstruction codebook: the M circle atoms of each lane (unit-norm).
    ang = 2 * np.pi * np.arange(M) / M
    circ = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)  # (M, 2)
    atoms = np.concatenate(
        [circ @ P[:, 2 * k: 2 * k + 2].T for k in range(n_lanes)], axis=0
    )                                                        # (n_lanes·M, d_in)
    lane_of_atom = np.repeat(np.arange(n_lanes), M).astype(np.int64)

    # Per-sequence labels broadcast along the sequence (tiled leading-edge eval).
    lane_lab = np.broadcast_to(lab[:, None, :], (n_seqs, seq_len, n_lanes)).copy()
    lane_vel = np.broadcast_to(Y[:, None, :], (n_seqs, seq_len, n_lanes)).copy()

    extra = {
        "lane_velocity_labels": torch.from_numpy(lane_lab),
        "lane_velocity": torch.from_numpy(lane_vel),
        "lane_phase_labels": torch.from_numpy(B),
        "lane_planes": torch.from_numpy(
            np.stack([P[:, 2 * k: 2 * k + 2] for k in range(n_lanes)], axis=0)
        ),                                                   # (n_lanes, d_in, 2)
        "lane_of_atom": torch.from_numpy(lane_of_atom),
        "omega": list(omega),
        "M": int(M),
        "n_lanes": int(n_lanes),
        "sigma": float(sigma),
        "freq_by_class": [float(y) / M for y in omega],
    }

    return SyntheticData(
        x=torch.from_numpy(x),
        emission_features=torch.from_numpy(atoms),           # n_lanes·M codebook atoms
        hidden_features=None,                                # velocities are not directions
        support=None,
        hidden_support=None,
        seq_len=seq_len,
        d_in=d_in,
        extra=extra,
    )


# ── Colored sources (FreqBench FB-3, theorem-first) ────────────────────


def colored_sources(
    *,
    N: int = 32,
    d_in: int = 32,
    D: int = 2,
    sigma: float = 0.1,
    rho_min: float = 0.1,
    rho_max: float = 0.9,
    seq_len: int = 64,
    n_seqs: int = 4096,
    seed: int = 0,
) -> SyntheticData:
    """Delayed temporally-colored sources — FB-3 (feature-direction recovery).

    ``N`` latent sources on a Haar-random orthonormal basis ``F`` (rows), each
    coordinate an independent AR(1) **at lag D** (``D`` independent residue
    classes mod D), with distinct autocorrelations ``ρ_i``::

        z_{t,i} = ρ_i · z_{t−D,i} + sqrt(1 − ρ_i²) · η_{t,i},   η ~ N(0,1)
        x_t     = Σ_i z_{t,i} f_i + σ ε_t
        (stationary init: z_{t,i} ~ N(0,1) for t < D)

    The frozen card is ``freqbench/cards/FB-3.md``: with ``d_in = N`` the
    one-token marginal is exactly ``N(0, (1+σ²) I)`` — isotropic, carrying
    ZERO information about F — and all lags 0 < ℓ < D have ``C_ℓ = 0``, so
    windows of length ≤ D are exactly iid isotropic Gaussians (CS-1 local
    impossibility: any learner on them outputs F-independent directions).
    The lag-D covariance ``C_D = Fᵀ diag(ρ) F`` makes F recoverable by
    eigendecomposition for windows ≥ D+1 (CS-2), with angular error ~ ε/γ,
    ``γ = min_i≠j |ρ_i − ρ_j|``. Continuous Gaussian data ⇒ no template set ⇒
    the P6 memorization route does not exist.

    Ground truths exposed: ``emission_features`` = the N rows of F (the strict
    Part II direction set — F anchors ``d_sae`` directly); ``extra``:
        rho_schedule: (N,) float64 — per-source autocorrelations (dispatch key)
        lag_D, sigma_obs: scalars
        source_z: (n_seqs, seq_len, N) float32 — the continuous latents
                  (optional ridge diagnostic only; NOT a direction)
    """
    if N > d_in:
        raise ValueError(f"N ({N}) > d_in ({d_in}): rows cannot be orthonormal")
    if D < 1:
        raise ValueError(f"D must be >= 1, got {D}")
    rng = np.random.default_rng(seed)

    # Haar-random orthonormal rows (exact orthonormality is load-bearing:
    # the CS-1 statement needs the marginal to be exactly isotropic at d=N).
    A = rng.standard_normal((d_in, d_in))
    Q, _ = np.linalg.qr(A)
    F = Q[:, :N].T.astype(np.float64)                       # (N, d_in)

    rho = np.linspace(rho_min, rho_max, N)                  # distinct ρ_i
    innov = np.sqrt(np.clip(1.0 - rho * rho, 0.0, None))

    z = np.empty((n_seqs, seq_len, N), dtype=np.float64)
    init_len = min(D, seq_len)
    z[:, :init_len, :] = rng.standard_normal((n_seqs, init_len, N))
    for t in range(D, seq_len):
        eta = rng.standard_normal((n_seqs, N))
        z[:, t, :] = rho * z[:, t - D, :] + innov * eta

    x = z @ F
    if sigma > 0:
        x = x + sigma * rng.standard_normal(x.shape)

    extra = {
        "rho_schedule": torch.from_numpy(rho.copy()),
        "lag_D": int(D),
        "sigma_obs": float(sigma),
        "source_z": torch.from_numpy(z.astype(np.float32)),
    }

    return SyntheticData(
        x=torch.from_numpy(x.astype(np.float32)),
        emission_features=torch.from_numpy(F.astype(np.float32)),  # the N rows of F
        hidden_features=None,
        support=None,
        hidden_support=None,
        seq_len=seq_len,
        d_in=d_in,
        extra=extra,
    )


#   self_exciting     → toy_backtracking_selfexcite_d64              (backtracking)
#   semi_markov_modes → toy_changepoint_modes_d64                    (changepoint)
#   cyclic_tones      → toy_cyclic_circle_M101_d128,
#                       toy_cyclic_random_M101_d128                  (frequency)
#   assumption_consequence → toy_assumption_consequence_d64          (assumption_consequence)
#   hedging_drift     → toy_hedging_drift_d64                        (hedging_drift)
#   recipe_instruction_phase_runs → toy_recipe_instruction_d64       (recipe_instruction_phase_runs)
#   multilane_tones   → toy_multilane_circle_M101_d24                (FB-2 multilane)
#   colored_sources   → toy_colored_sources_N32_D2_d32               (FB-3 colored)
_GENERATORS = {
    "markov":  markov_chain_support,
    "coupled": coupled_hmm,
    "signed_motion": signed_motion,
    "self_exciting": self_exciting,
    "semi_markov_modes": semi_markov_modes,
    "cyclic_tones": cyclic_tones,
    "assumption_consequence": assumption_consequence,
    "hedging_drift": hedging_drift,
    "recipe_instruction_phase_runs": recipe_instruction_phase_runs,
    "multilane_tones": multilane_tones,
    "colored_sources": colored_sources,
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
