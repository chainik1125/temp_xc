"""Hedging-drift bench — generator + metric-wiring tests.

Covers the expansion stage-6 #2 add-on:
- the ``hedging_drift`` generator's shapes + ground-truth labels,
- the hierarchical-AR(1) mirror algebra (levels drawn from the empirical
  list; the confidence stream's mean/trend near the fitted params),
- the long-memory plateau (pooled within-sequence ACF: lag-1 ≈ 0.33, and the
  lag-4 plateau well above an AR(1)-only collapse — the gate-8 property the
  hier mirror exists to hold),
- the continuous-loading emission (u_conf projection tracks c_i·m_i),
- that :func:`hedging_metrics` returns the right keys for token and window
  archs, and that other generators don't trip the dispatch.
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.data.synthetic import (
    _HEDGING_LEVELS_HIER,
    hedging_drift,
    self_exciting,
)


def _pooled_acf(c: np.ndarray, lag: int) -> float:
    """Pooled within-sequence autocorrelation at ``lag`` (the mirror stat)."""
    xs, ys = [], []
    for row in c:
        xs.append(row[:-lag])
        ys.append(row[lag:])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return float(np.corrcoef(x, y)[0, 1])


def test_hedging_shapes_and_extra() -> None:
    data = hedging_drift(K_c=6, n_c=2, d_in=16, seq_len=8, n_seqs=32, seed=0)
    assert data.x.shape == (32, 8, 16)
    assert data.emission_features.shape == (6, 16)      # content set
    assert data.hidden_features.shape == (1, 16)        # conf direction
    c = data.extra["conf_labels"]
    l = data.extra["level_labels"]
    assert c.shape == (32, 8) and l.shape == (32,)
    assert abs(data.extra["rho"] - 0.2476) < 1e-3


def test_hedging_levels_from_empirical_list() -> None:
    data = hedging_drift(seq_len=8, n_seqs=256, seed=1)
    lv = data.extra["level_labels"].numpy().astype(np.float64)
    pool = np.array(_HEDGING_LEVELS_HIER)
    dists = np.abs(lv[:, None] - pool[None, :]).min(axis=1)
    assert dists.max() < 1e-6, "levels must be drawn from the empirical list"


def test_hedging_stream_moments_and_plateau() -> None:
    """Mean/trend near the fitted params; ACF holds the long-memory plateau."""
    data = hedging_drift(seq_len=64, n_seqs=4096, seed=2)
    c = data.extra["conf_labels"].numpy().astype(np.float64)
    mu, beta = data.extra["mu"], data.extra["beta_position"]
    # pooled mean ≈ mu + beta/2 (levels are near-zero-mean by construction)
    assert abs(c.mean() - (mu + beta * 0.5)) < 0.05
    # trend: late-half mean minus early-half mean ≈ beta/2
    trend = c[:, 32:].mean() - c[:, :32].mean()
    assert abs(trend - beta * 0.5) < 0.05
    acf1 = _pooled_acf(c, 1)
    acf4 = _pooled_acf(c, 4)
    # the C3 fit: syn ACF lags 1-8 ≈ 0.33, 0.17, 0.14, 0.12, ... (plateau);
    # an AR(1)-only stream would be ≈ rho^4 ≈ 0.004 at lag 4.
    assert 0.25 < acf1 < 0.42
    assert acf4 > 0.06, f"lag-4 plateau lost (acf4={acf4:.3f})"


def test_hedging_conf_loading_tracks_c() -> None:
    """The u_conf projection is c_i·m_i — strongly rank-correlated with c_i."""
    data = hedging_drift(seq_len=16, n_seqs=256, seed=3)
    c = data.extra["conf_labels"].numpy().reshape(-1)
    u = data.hidden_features[0]
    proj = (data.x @ u).numpy().reshape(-1)
    assert np.corrcoef(proj, c)[0, 1] > 0.85


def test_hedging_metrics_keys_token_and_window() -> None:
    from temp_bench.archs.topk_sae import TopKSAE
    from temp_bench.archs.txc_base import TXCBase
    from temp_bench.evals.hedging_recovery import hedging_metrics

    data = hedging_drift(K_c=6, n_c=2, d_in=16, seq_len=16, n_seqs=512, seed=0)
    keys = {"conf_recovery", "conf_corr", "conf_chance"}
    for arch in (TopKSAE(d_in=16, d_sae=10, k_pos=2),
                 TXCBase(d_in=16, d_sae=10, T=4, k_pos=2)):
        out = hedging_metrics(arch, data, eval_window_L=8, n_windows=64)
        assert keys <= set(out)
        assert out["conf_recovery"] <= 1.0
        assert -1.0 <= out["conf_corr"] <= 1.0


def test_other_generators_do_not_trip_hedging_dispatch() -> None:
    """Gating contract: the dispatch keys off extra['conf_labels']; the other
    extra-bearing generator must not carry it."""
    se = self_exciting(seq_len=8, n_seqs=8, seed=0)
    assert "conf_labels" not in se.extra
