"""Ported ``verify_theory`` checks — sprint theory propositions as permanent tests.

FB-C2 phase-2 port (briefings/freqbench-t16-fbc2.md) of the
``origin/dmitry-spectral-sprint2`` ``verify_theory.py`` checks onto the BUILT
generators, in the analytic-test pattern of ``test_freqfrac.py``. Pins three
proofs-registry statements (freqbench ``PORT.md`` § B) to the shipped
parameterizations rather than the sprint's private ones:

- **P2 phase-averaging** (``cyclic_tones``, circle, M=101, σ=0.10): any
  additive-over-time per-token feature statistic has velocity-independent
  class means (gap inside the label-permutation null), while a single order-2
  window statistic — the lag-1 inner product — separates classes with class
  means matching the analytic ``cos(2πy/M)``. Additivity across time, not
  per-token encoder power, is the load-bearing restriction.
- **P5 periodogram = ML + Rayleigh**: the eval's ``_periodogram_pred`` is
  decision-equivalent to brute-force ML over the discrete phase grid; the
  clean-tone gram equals the Dirichlet kernel to machine precision; and the
  Rayleigh limit is behavioral — at elevated noise the sub-Rayleigh cluster
  {0,1,2,4} confuses at W=16 and resolves at W=64 (resolution ∝ 1/W; inside
  a cell the distinction is SNR-only, so at the built σ=0.10 the oracle
  saturates at both W — also asserted).
- **CS-2 lag-D eigen-recovery** (``colored_sources``, N=32, D=2 as built):
  empirical lag covariances match the population forms (Ĉ₁ ≈ 0 below the
  delay, Ĉ_D ≈ Fᵀdiag(ρ)F — the engine of the W ≥ D+1 transition),
  eigenvectors of sym(Ĉ_D) recover F far above the time-shuffled control,
  eigenvalues track ρ; a large-eigengap variant (N=4) recovers near-perfectly
  (angular error ~ ε/γ). The shuffled control sits near 0.58, not 0 — the
  orthonormal-basis geometry floor documented in the FB-3 gating record.

Thresholds are margins around values measured at these exact seeds
(deterministic), not tuned bars.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from temp_bench.data.synthetic import colored_sources, cyclic_tones
from temp_bench.evals.frequency_recovery import _periodogram_pred

M = 101
OMEGA = (0, 1, 2, 4, 8, 16, 24, 32, 40, 50)


@pytest.fixture(scope="module")
def tones_built():
    """The built circle parameterization (σ=0.10, the headline datasource)."""
    return cyclic_tones(M=M, embedding="circle", d_in=128, sigma=0.10,
                        seq_len=64, n_seqs=2000, seed=0)


@pytest.fixture(scope="module")
def tones_noisy():
    """Same generator, SNR knob turned down (σ=1.5) so the oracle actually
    errs — needed to *see* the Rayleigh confusion structure."""
    return cyclic_tones(M=M, embedding="circle", d_in=128, sigma=1.5,
                        seq_len=64, n_seqs=2000, seed=0)


def _ml_pred(x_tiles: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Brute-force ML over (velocity, discrete phase B ∈ Z_M) on the circle
    plane: score(y) = max_B Re(e^{-2πiB/M} · Σ_t c_t e^{-2πiyt/M})."""
    proj = x_tiles @ R
    c = proj[..., 0] + 1j * proj[..., 1]
    t = np.arange(x_tiles.shape[1])
    basis = np.exp(-2j * np.pi * np.asarray(OMEGA, dtype=np.float64)[:, None]
                   * t[None, :] / M)
    z = c @ basis.T                                     # (N, |Ω|) matched filter
    phases = np.exp(-2j * np.pi * np.arange(M) / M)
    return (z[:, :, None] * phases[None, None, :]).real.max(axis=2).argmax(axis=1)


# ───────────────────────────── P2 — phase-averaging ─────────────────────────


def test_p2_additive_statistic_is_velocity_independent(tones_built) -> None:
    """Class means of a random per-token feature battery, pooled additively
    over time, are indistinguishable from the label-permutation null."""
    x = tones_built.x.numpy().astype(np.float64)
    lab = tones_built.extra["velocity_labels"][:, 0].numpy()
    g = torch.Generator().manual_seed(7)
    W = torch.randn(128, 32, generator=g).numpy() / np.sqrt(128)
    b = torch.randn(32, generator=g).numpy()
    S = np.maximum(x @ W + b, 0.0).mean(axis=1)         # (N, 32) additive score

    def max_gap(scores, labels):
        mus = np.stack([scores[labels == c].mean(axis=0) for c in range(10)])
        return float(np.abs(mus[:, None, :] - mus[None, :, :]).max())

    obs = max_gap(S, lab)
    rng = np.random.default_rng(0)
    null = [max_gap(S, rng.permutation(lab)) for _ in range(20)]
    # measured: obs 0.0067 vs null [0.0078, 0.0234] — below even the null MIN.
    assert obs <= max(null), (obs, null)


def test_p2_order2_statistic_separates_with_analytic_means(tones_built) -> None:
    """One lag-1 inner product (order-2, non-additive) separates classes and
    its class means equal cos(2πy/M): ⟨u_{a+y}, u_a⟩ on the unit circle."""
    x = tones_built.x.numpy().astype(np.float64)
    lab = tones_built.extra["velocity_labels"][:, 0].numpy()
    lag1 = (x[:, 1:, :] * x[:, :-1, :]).sum(-1).mean(1)
    mus = np.array([lag1[lab == c].mean() for c in range(10)])
    pred = np.cos(2 * np.pi * np.asarray(OMEGA) / M)
    assert float(np.abs(mus - pred).max()) < 0.02       # measured 0.0026
    assert float(np.abs(mus[:, None] - mus[None, :]).max()) > 1.5  # measured 2.0


# ─────────────────────── P5 — periodogram = ML + Rayleigh ───────────────────


def test_p5_periodogram_is_ml_and_saturates_at_built_sigma(tones_built) -> None:
    x = tones_built.x.numpy().astype(np.float64)
    lab = tones_built.extra["velocity_labels"][:, 0].numpy()
    R = tones_built.extra["circle_plane"].numpy()
    for Wlen in (16, 64):
        xt = x[:, :Wlen, :]
        p = _periodogram_pred(xt, R, list(OMEGA), M)
        assert float((_ml_pred(xt, R) == p).mean()) >= 0.999
        assert float((p == lab).mean()) >= 0.999        # the recorded ceiling


def test_p5_periodogram_is_ml_under_noise(tones_noisy) -> None:
    x = tones_noisy.x.numpy().astype(np.float64)
    R = tones_noisy.extra["circle_plane"].numpy()
    for Wlen in (16, 64):
        xt = x[:, :Wlen, :]
        p = _periodogram_pred(xt, R, list(OMEGA), M)
        assert float((_ml_pred(xt, R) == p).mean()) >= 0.995   # measured 0.9985+


def test_p5_clean_tone_gram_is_dirichlet_kernel() -> None:
    Wlen = 16
    om = np.asarray(OMEGA, dtype=np.float64)
    t = np.arange(Wlen)
    tones = np.exp(2j * np.pi * om[:, None] * t[None, :] / M)
    gram = np.abs(tones @ tones.conj().T) / Wlen
    df = (om[:, None] - om[None, :]) / M
    with np.errstate(divide="ignore", invalid="ignore"):
        dk = np.abs(np.sin(np.pi * Wlen * df) / (Wlen * np.sin(np.pi * df)))
    dk[np.isnan(dk)] = 1.0
    assert float(np.abs(gram - dk).max()) < 1e-9


def test_p5_rayleigh_resolution_in_window_length(tones_noisy) -> None:
    """At σ=1.5: the sub-Rayleigh cluster {0,1,2,4} (pairwise Δf < 1/16) is
    confused at W=16 while the coarse classes {16..50} (Δy ≥ 8 > M/16) stay
    separable; W=64 shrinks the cell (M/64 ≈ 1.6) and the cluster resolves."""
    x = tones_noisy.x.numpy().astype(np.float64)
    lab = tones_noisy.extra["velocity_labels"][:, 0].numpy()
    R = tones_noisy.extra["circle_plane"].numpy()
    sub = np.isin(lab, [0, 1, 2, 3])
    coarse = np.isin(lab, [5, 6, 7, 8, 9])

    p16 = _periodogram_pred(x[:, :16, :], R, list(OMEGA), M)
    acc_sub16 = float((p16[sub] == lab[sub]).mean())       # measured 0.408
    acc_coarse16 = float((p16[coarse] == lab[coarse]).mean())  # measured 0.703
    assert acc_sub16 <= 0.55
    assert acc_coarse16 >= 0.65
    assert acc_coarse16 - acc_sub16 >= 0.15

    p64 = _periodogram_pred(x, R, list(OMEGA), M)
    assert float((p64[sub] == lab[sub]).mean()) >= 0.97    # measured 0.996
    assert float((p64 == lab).mean()) >= 0.98              # measured 0.9975


# ─────────────────────── CS-2 — lag-D eigen-recovery ────────────────────────


def _lag_cov(x: np.ndarray, lag: int) -> np.ndarray:
    a = x[:, lag:, :].reshape(-1, x.shape[-1])
    b = x[:, : x.shape[1] - lag, :].reshape(-1, x.shape[-1])
    return (a.T @ b) / a.shape[0]


def _eigen_recover(x: np.ndarray, F: np.ndarray, lag: int) -> np.ndarray:
    C = _lag_cov(x, lag)
    ev, V = np.linalg.eigh(0.5 * (C + C.T))
    order = np.argsort(np.abs(ev))[::-1]
    return np.abs(F @ V[:, order[: F.shape[0]]]).max(axis=1)


def test_cs2_lag_covariances_match_population_forms() -> None:
    cs = colored_sources(N=32, d_in=32, D=2, sigma=0.1, rho_min=0.1,
                         rho_max=0.9, seq_len=64, n_seqs=512, seed=0)
    x = cs.x.numpy().astype(np.float64)
    F = cs.emission_features.numpy().astype(np.float64)
    rho = cs.extra["rho_schedule"].numpy()
    C0, C1, C2 = _lag_cov(x, 0), _lag_cov(x, 1), _lag_cov(x, 2)
    pop0 = F.T @ F + cs.extra["sigma_obs"] ** 2 * np.eye(32)
    pop2 = F.T @ np.diag(rho) @ F
    # below the delay the covariance vanishes (why W ≤ D windows are blind):
    assert np.linalg.norm(C1) / np.linalg.norm(C2) <= 0.2      # measured 0.079
    assert np.linalg.norm(C2 - pop2) / np.linalg.norm(pop2) <= 0.15  # 0.081
    assert np.linalg.norm(C0 - pop0) / np.linalg.norm(pop0) <= 0.10  # 0.044


def test_cs2_eigenvectors_recover_F_above_shuffled_control() -> None:
    cs = colored_sources(N=32, d_in=32, D=2, sigma=0.1, rho_min=0.1,
                         rho_max=0.9, seq_len=64, n_seqs=512, seed=0)
    x = cs.x.numpy().astype(np.float64)
    F = cs.emission_features.numpy().astype(np.float64)
    rho = cs.extra["rho_schedule"].numpy()
    rec = _eigen_recover(x, F, lag=2)
    # time-shuffle destroys the lag structure (the CS-1 side of the coin);
    # the surviving ~0.58 is the orthonormal-candidate geometry floor.
    rng = np.random.default_rng(1)
    xs = np.stack([xi[rng.permutation(x.shape[1])] for xi in x])
    shuf = _eigen_recover(xs, F, lag=2)
    assert float(rec.mean()) >= 0.80                    # measured 0.877
    assert float(shuf.mean()) <= 0.70                   # measured 0.580
    assert float(rec.mean() - shuf.mean()) >= 0.15
    # eigenvalues of sym(C_D) track the rho schedule.
    C = _lag_cov(x, 2)
    ev = np.linalg.eigvalsh(0.5 * (C + C.T))
    top = np.sort(np.abs(ev))[::-1][:32]
    assert float(np.corrcoef(np.sort(top), np.sort(rho))[0, 1]) >= 0.99


def test_cs2_large_eigengap_recovers_near_perfectly() -> None:
    """N=4, ρ ∈ [0.2, 0.8] → eigengap γ = 0.2: angular error ~ ε/γ collapses."""
    cs = colored_sources(N=4, d_in=8, D=2, sigma=0.1, rho_min=0.2,
                         rho_max=0.8, seq_len=64, n_seqs=256, seed=0)
    x = cs.x.numpy().astype(np.float64)
    F = cs.emission_features.numpy().astype(np.float64)
    rec = _eigen_recover(x, F, lag=2)
    assert float(rec.min()) >= 0.98                     # measured 0.996
