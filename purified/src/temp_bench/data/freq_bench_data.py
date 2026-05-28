"""FreqBench synthetic generators — DC / AC / Mixed (v2 port).

Reconstruction of Dmitry's FrequencyBench (2026-05-06). The original
drivers/lib were never committed (see docs/components/freq_bench.md); this
module rebuilds the three generative families from the proposal
description embedded in his writeup + the schema of his committed results
JSON. Constants that were lost (ring size M, velocity sets Ω) are chosen
sensibly here and documented; treat this as a faithful reconstruction, not
a byte-exact replay.

Each generator returns a :class:`FreqBenchData`:

    x:          (N, W, d_in)   activations (orthonormal emission dirs + σ noise)
    y:          (N,)           integer probe label
    A_loc:      float          one-token Bayes ceiling
    A_oracle:   float          symbolic temporal oracle
    n_classes:  int            number of label classes
    velocities: (N,) or None   per-seq velocity (Mixed only; for R_j curve)

The generic synthetic refill path (``temp_bench.data.synthetic.build_refill``)
only touches ``.x``, so registering these by ``module:fn`` path in
``configs/data.yaml`` is enough to train any arch on a FreqBench cell. The
labels + oracle are read by ``temp_bench.evals.freq_bench``.

Shared knobs:
    W (== seq_len): probe/sequence window.
    sigma:          Gaussian emission noise std.
    d_in:           activation dim (≥ M).
    n_seqs:         number of sequences to materialise.
    seed:           RNG seed (also fixes the emission directions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FreqBenchData:
    x: torch.Tensor                 # (N, W, d_in)
    y: torch.Tensor                 # (N,) long
    A_loc: float
    A_oracle: float
    n_classes: int
    bench: str
    velocities: torch.Tensor | None = None   # (N,) — Mixed only


# ── shared helpers ──────────────────────────────────────────────────────


def _emission_dirs(M: int, d_in: int, rng: np.random.Generator) -> np.ndarray:
    """M orthonormal emission directions in R^d_in (deterministic in seed)."""
    if M > d_in:
        raise ValueError(f"M ({M}) > d_in ({d_in})")
    raw = rng.standard_normal((d_in, d_in))
    Q, _ = np.linalg.qr(raw)
    return Q[:M].astype(np.float32)               # (M, d_in)


def _emit(idx: np.ndarray, E: np.ndarray, sigma: float,
          rng: np.random.Generator) -> np.ndarray:
    """idx: (N, W) phase/bit indices → x: (N, W, d_in) = E[idx] + σ·noise."""
    x = E[idx]                                     # (N, W, d_in)
    if sigma > 0:
        x = x + sigma * rng.standard_normal(x.shape).astype(np.float32)
    return x.astype(np.float32)


def _binomial_majority_acc(W: int, p: float) -> float:
    """P(majority of W iid Bernoulli(p) votes equals the truth), ties → 0.5."""
    acc = 0.0
    for k in range(W + 1):
        pk = math.comb(W, k) * (p ** k) * ((1 - p) ** (W - k))
        if k > W / 2:
            acc += pk
        elif k == W / 2:
            acc += 0.5 * pk
    return float(acc)


def _pack(x: np.ndarray, y: np.ndarray, **kw) -> FreqBenchData:
    return FreqBenchData(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y.astype(np.int64)),
        **kw,
    )


# ── DC bench: constant binary state under emission noise (≈ §4 smoothing) ─


def freq_bench_dc(
    *, W: int = 8, p: float = 0.65, sigma: float = 0.1,
    d_in: int = 256, n_seqs: int = 4096, seed: int = 0, **_ignore,
) -> FreqBenchData:
    """Per-sequence latent state s∈{0,1}; each token emits s with prob p.

    A single token's best guess of s is correct w.p. p (one-token Bayes
    ceiling). Aggregating W tokens → majority vote → Binomial oracle.
    Separates per-token archs (stuck at p) from windowed archs (climb).
    """
    rng = np.random.default_rng(seed)
    E = _emission_dirs(2, d_in, rng)                          # bit 0 / bit 1
    s = (rng.random(n_seqs) < 0.5).astype(np.int64)           # balanced state
    correct = rng.random((n_seqs, W)) < p                     # emit s vs 1-s
    bits = np.where(correct, s[:, None], 1 - s[:, None]).astype(np.int64)
    x = _emit(bits, E, sigma, rng)
    return _pack(x, s, A_loc=float(p),
                 A_oracle=_binomial_majority_acc(W, p),
                 n_classes=2, bench="dc")


# ── AC bench: signed velocity on a phase ring ────────────────────────────


def freq_bench_ac(
    *, W: int = 8, sigma: float = 0.1, M: int = 32, omega: int = 1,
    d_in: int = 256, n_seqs: int = 4096, seed: int = 0, **_ignore,
) -> FreqBenchData:
    """Phase walks the ring Z_M at fixed signed velocity v∈{+ω,−ω}.

    phase(t) = (φ0 + v·t) mod M; emission is one-hot(phase)+σ noise. Label
    = sign(v). A single token reveals phase but not direction (A_loc=0.5);
    two adjacent phases determine the sign exactly (A_oracle=1.0). The
    critical test: does the arch encode the *transition* (order), or just
    aggregate phases?
    """
    if M <= W * omega:
        # keep the walk from wrapping within the window (sign stays decodable)
        M = int(W * omega + 2)
    rng = np.random.default_rng(seed)
    E = _emission_dirs(M, d_in, rng)
    phi0 = rng.integers(0, M, size=n_seqs)
    sign = np.where(rng.random(n_seqs) < 0.5, 1, -1)
    v = sign * omega
    t = np.arange(W)[None, :]
    phase = (phi0[:, None] + v[:, None] * t) % M              # (N, W)
    x = _emit(phase, E, sigma, rng)
    y = (sign > 0).astype(np.int64)                           # 1 = forward
    return _pack(x, y, A_loc=0.5, A_oracle=1.0, n_classes=2, bench="ac")


# ── Mixed bench: velocity-class frequency ladder ─────────────────────────


def freq_bench_mixed(
    *, W: int = 8, sigma: float = 0.1, M: int = 127, variant: str = "unsigned",
    n_classes: int = 10, d_in: int = 256, n_seqs: int = 4096, seed: int = 0,
    **_ignore,
) -> FreqBenchData:
    """Per-sequence velocity drawn from a ladder Ω of |Ω|=n_classes speeds.

    Unsigned: Ω = {1, …, n_classes}. Signed: Ω = ±{1, …, n_classes/2}.
    Label = velocity-class index. A_loc = 1/n_classes; A_oracle = 1.0. The
    per-class accuracy traces the architecture's frequency response R_j.
    """
    rng = np.random.default_rng(seed)
    E = _emission_dirs(M, d_in, rng)
    if variant == "signed":
        half = n_classes // 2
        speeds = list(range(1, half + 1))
        omega_set = [-s for s in speeds] + speeds            # length n_classes
    else:
        omega_set = list(range(1, n_classes + 1))
    omega_set = np.array(omega_set, dtype=np.int64)
    n_classes = len(omega_set)

    cls = rng.integers(0, n_classes, size=n_seqs)
    v = omega_set[cls]
    phi0 = rng.integers(0, M, size=n_seqs)
    t = np.arange(W)[None, :]
    phase = (phi0[:, None] + v[:, None] * t) % M
    x = _emit(phase, E, sigma, rng)
    return _pack(x, cls.astype(np.int64), A_loc=1.0 / n_classes, A_oracle=1.0,
                 n_classes=n_classes, bench=f"mixed_{variant}",
                 velocities=torch.from_numpy(v.astype(np.int64)))


# ── public materialiser for the evaluator (held-out probe set) ───────────


_GEN = {
    "dc": freq_bench_dc, "ac": freq_bench_ac,
    "mixed_unsigned": lambda **kw: freq_bench_mixed(variant="unsigned", **kw),
    "mixed_signed": lambda **kw: freq_bench_mixed(variant="signed", **kw),
}


def materialise_probe_set(*, bench: str, params: dict, seed: int) -> FreqBenchData:
    """Generate a labelled probe set for evaluation.

    Uses a seed OFFSET from the training seed so the probe data is disjoint
    from (but the same distribution as) what the SAE trained on.
    """
    fn = _GEN[bench]
    kw = {k: v for k, v in params.items() if k != "seed"}
    return fn(seed=seed + 10_000, **kw)
