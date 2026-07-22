"""Colored-sources bench (FB-3) — generator + ground-truth + metric tests.

Covers the FreqBench FB-3 card (freqbench/cards/FB-3.md):
- ``colored_sources`` shapes, orthonormal basis, ρ schedule,
- the **CS-1 premises** on the built task: the one-token marginal is
  isotropic (covariance ≈ (1+σ²) I, F-blind) and lag-ℓ covariance ≈ 0 for
  0 < ℓ < D,
- the **CS-2 ceiling** on the built task: eigenvectors of the symmetrized
  empirical lag-D covariance recover F well above the random-dictionary
  chance at the card's exact budget,
- AR algebra: lag-D autocorrelation per source ≈ ρ_i,
- the ``colored_metrics`` add-on: keys, chance floor sanity (random-init
  model ≈ chance ⇒ rec_adj ≈ 0), and a planted-dictionary sanity check
  (decoder = F ⇒ rec_adj = 1),
- dispatch contract (other extra-bearing generators don't carry the key).
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.data.synthetic import colored_sources, multilane_tones


def _small(seed=0, **kw):
    p = dict(N=32, d_in=32, D=2, sigma=0.1, seq_len=64, n_seqs=256, seed=seed)
    p.update(kw)
    return colored_sources(**p)


def test_colored_shapes_and_extra() -> None:
    data = _small()
    assert data.x.shape == (256, 64, 32)
    assert data.emission_features.shape == (32, 32)
    rho = data.extra["rho_schedule"].numpy()
    assert rho.shape == (32,)
    assert abs(rho[0] - 0.1) < 1e-9 and abs(rho[-1] - 0.9) < 1e-9
    assert data.extra["lag_D"] == 2
    assert data.extra["source_z"].shape == (256, 64, 32)


def test_colored_basis_orthonormal() -> None:
    F = _small().emission_features.numpy().astype(np.float64)
    assert np.abs(F @ F.T - np.eye(32)).max() < 1e-6


def test_colored_marginal_isotropic_cs1() -> None:
    """One-token covariance ≈ (1+σ²) I — the marginal carries nothing about F."""
    data = _small(n_seqs=2000, seed=1)
    x = data.x.numpy().reshape(-1, 32).astype(np.float64)
    C0 = (x.T @ x) / len(x)
    target = (1 + 0.1 ** 2)
    assert np.abs(np.diag(C0) - target).max() < 0.05
    off = C0 - np.diag(np.diag(C0))
    assert np.abs(off).max() < 0.05, "off-diagonal marginal structure leaks F"


def test_colored_short_lags_blank() -> None:
    """C_ℓ ≈ 0 for 0 < ℓ < D (windows of length ≤ D are iid isotropic)."""
    data = _small(n_seqs=2000, seed=2, D=2)
    x = data.x.numpy().astype(np.float64)
    C1 = np.einsum("ntd,nte->de", x[:, 1:], x[:, :-1]) / (2000 * 63)
    assert np.abs(C1).max() < 0.05, "lag-1 covariance must vanish at D=2"


def test_colored_lag_D_autocorr_matches_rho() -> None:
    data = _small(n_seqs=1000, seed=3)
    z = data.extra["source_z"].numpy().astype(np.float64)
    rho = data.extra["rho_schedule"].numpy()
    ac = (z[:, 2:, :] * z[:, :-2, :]).mean(axis=(0, 1))
    assert np.abs(ac - rho).max() < 0.06


def test_colored_cs2_oracle_recovers_F() -> None:
    """Eigendecomposition of the symmetrized lag-D covariance recovers F at
    the card budget (the T1 ceiling, checked small-scale here; full scale in
    gating)."""
    data = colored_sources(N=32, d_in=32, D=2, sigma=0.1, seq_len=64,
                           n_seqs=4096, seed=4)
    x = data.x.numpy().astype(np.float64)
    F = data.emission_features.numpy().astype(np.float64)
    CD = np.einsum("ntd,nte->de", x[:, 2:], x[:, :-2]) / (4096 * 62)
    Cs = 0.5 * (CD + CD.T)
    w, V = np.linalg.eigh(Cs)
    order = np.argsort(-np.abs(w))
    Vh = V[:, order].T                                  # (32, 32) recovered rows
    cos2 = (Vh @ F.T) ** 2
    rec = cos2.max(axis=0).mean()
    assert rec > 0.7, f"lag-D eigen-oracle should recover F (rec_sq={rec:.3f})"


def test_colored_metrics_random_at_chance_planted_at_one() -> None:
    from temp_bench.archs.batchtopk_sae import BatchTopKSAE
    from temp_bench.evals.colored_recovery import colored_metrics

    data = _small(seed=5)
    m = BatchTopKSAE(d_in=32, d_sae=32, k_pos=2)
    out = colored_metrics(m, data)
    for k in ("colored_rec_adj", "colored_rec_sq", "colored_chance",
              "colored_rec_q1", "colored_rec_q4"):
        assert k in out
    assert abs(out["colored_rec_adj"]) < 0.25, "random init must sit ≈ chance"
    # plant the true dictionary → perfect recovery
    with torch.no_grad():
        m.W_dec.copy_(data.emission_features)
    out2 = colored_metrics(m, data)
    assert out2["colored_rec_adj"] > 0.98


def test_colored_metrics_window_arch_slices() -> None:
    from temp_bench.archs.txc_batchtopk import TXCBatchTopKPost
    from temp_bench.evals.colored_recovery import colored_metrics

    data = _small(seed=6)
    m = TXCBatchTopKPost(d_in=32, d_sae=32, T=4, k_pos=2)
    out = colored_metrics(m, data)
    assert abs(out["colored_rec_adj"]) < 0.25
    # planting F at ONE decoder tap must be found by the per-position max
    with torch.no_grad():
        m.W_dec.zero_()
        m.W_dec[:, 1, :].copy_(data.emission_features)
    assert colored_metrics(m, data)["colored_rec_adj"] > 0.98


def test_other_generators_do_not_trip_colored_dispatch() -> None:
    ml = multilane_tones(seq_len=8, n_seqs=8, seed=0)
    assert "rho_schedule" not in ml.extra
