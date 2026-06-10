"""Change-point (semi-Markov modes) bench — generator + metric-wiring tests.

Covers the autoresearch #2 add-on:
- the ``semi_markov_modes`` generator's shapes + ground-truth labels,
- the label algebra (c_t marks mode changes; τ_t counts since the last
  boundary, start-as-renewal),
- the dwell anchor (realized switch rate ≈ p_switch = 1/dwell_mean),
- the DPI premise (the switch rate and E[τ] are mode-independent — uniform Π),
- the two named direction sets (content → emission_features, mode-signature →
  hidden_features; orthonormal),
- that :func:`changepoint_metrics` returns the right keys for token and
  window archs, and that other generators don't trip the dispatch.
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.data.synthetic import self_exciting, semi_markov_modes


def test_semi_markov_modes_shapes_and_extra() -> None:
    data = semi_markov_modes(K_m=4, C=6, spread=2, d_in=16, seq_len=8,
                             n_seqs=32, seed=0)
    assert data.x.shape == (32, 8, 16)
    assert data.emission_features.shape == (6, 16)      # content set
    assert data.hidden_features.shape == (4, 16)        # mode-signature set
    m = data.extra["mode_labels"]
    c = data.extra["changepoint_labels"]
    tau = data.extra["time_since_switch"]
    assert m.shape == c.shape == tau.shape == (32, 8)
    assert int(m.min()) >= 0 and int(m.max()) <= 3
    assert data.extra["K_m"] == 4


def test_semi_markov_modes_label_algebra() -> None:
    data = semi_markov_modes(K_m=8, seq_len=64, n_seqs=64, seed=1)
    m = data.extra["mode_labels"].numpy()
    c = data.extra["changepoint_labels"].numpy()
    tau = data.extra["time_since_switch"].numpy()
    assert (c[:, 0] == 0).all() and (tau[:, 0] == 0).all()
    assert np.array_equal(c[:, 1:], (m[:, 1:] != m[:, :-1]).astype(c.dtype))
    expect = np.where(c[:, 1:] == 1, 0.0, tau[:, :-1] + 1.0)
    assert np.array_equal(tau[:, 1:], expect)


def test_semi_markov_modes_dwell_anchor_and_dpi_premise() -> None:
    """Realized switch rate ≈ 1/dwell_mean, and is the SAME for every mode
    (uniform Π) — the § 8 (i) per-token-at-chance premise."""
    data = semi_markov_modes(K_m=8, dwell_mean=1.73, seq_len=64,
                             n_seqs=4096, seed=2)
    m = data.extra["mode_labels"].numpy()
    c = data.extra["changepoint_labels"].numpy()
    tau = data.extra["time_since_switch"].numpy()
    p = 1.0 / 1.73
    assert abs(c[:, 1:].mean() - p) < 0.02
    rates = [c[:, 1:][m[:, 1:] == k].mean() for k in range(8)]
    assert max(rates) - min(rates) < 0.02
    taus = [tau[m == k].mean() for k in range(8)]
    assert max(taus) - min(taus) < 0.05


def test_semi_markov_modes_direction_sets_orthonormal() -> None:
    data = semi_markov_modes(K_m=8, C=12, d_in=64, seq_len=8, n_seqs=8, seed=3)
    U = torch.cat([data.hidden_features, data.emission_features], dim=0)
    G = U @ U.T
    off = (G - torch.eye(20)).abs().max().item()
    assert off < 1e-4, f"direction sets not orthonormal (max dev {off})"


def test_semi_markov_modes_mode_signature_dominant() -> None:
    """The active mode's signature direction is the largest component, so the
    DC readout survives k_pos=1."""
    data = semi_markov_modes(K_m=8, C=12, d_in=64, seq_len=16, n_seqs=64, seed=4)
    m = data.extra["mode_labels"]
    proj_m = torch.einsum("ntd,kd->ntk", data.x, data.hidden_features)
    assert bool((proj_m.argmax(dim=-1) == m).float().mean() > 0.999)


def test_semi_markov_modes_gated_knobs_raise() -> None:
    import pytest
    with pytest.raises(ValueError, match="geometric"):
        semi_markov_modes(dwell="negative_binomial", seq_len=8, n_seqs=8, seed=0)


def test_changepoint_metrics_keys_token_and_window() -> None:
    from temp_bench.archs.topk_sae import TopKSAE
    from temp_bench.archs.txc_base import TXCBase
    from temp_bench.evals.changepoint_recovery import changepoint_metrics

    data = semi_markov_modes(K_m=4, C=6, spread=2, d_in=16, seq_len=16,
                             n_seqs=512, seed=0)
    keys = {"mode_recovery", "mode_balacc", "mode_chance",
            "tss_recovery", "tss_r2", "tss_chance",
            "cp_recovery", "cp_balacc", "cp_chance"}
    for arch in (TopKSAE(d_in=16, d_sae=10, k_pos=2),
                 TXCBase(d_in=16, d_sae=10, T=4, k_pos=2)):
        out = changepoint_metrics(arch, data, eval_window_L=8, n_windows=64)
        assert keys <= set(out)
        assert -1.0 <= out["tss_recovery"] <= 1.0
        assert 0.0 <= out["mode_balacc"] <= 1.0


def test_other_generators_do_not_trip_changepoint_dispatch() -> None:
    """Gating contract: the dispatch keys off extra['mode_labels']; the other
    extra-bearing generator (self_exciting) must not carry it."""
    se = self_exciting(seq_len=8, n_seqs=8, seed=0)
    assert "mode_labels" not in se.extra
