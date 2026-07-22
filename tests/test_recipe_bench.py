"""Recipe-instruction phase-runs bench — generator + metric-wiring tests.

Covers the expansion stage-6 #3 add-on:
- the ``recipe_instruction_phase_runs`` generator's shapes + ground-truth
  labels,
- the label algebra (``e_t = [c_t = c_{t-1}]``, ``e_0 = 0`` convention),
- the mirror anchor (pooled class marginal near the measured
  [0.29, 0.51, 0.06, 0.07, 0.07]; per-class continuation rates ordered by the
  mirror's per-symbol dwell means — the dwell-heterogeneity structure the
  bench rides on; pooled match rate in the measured band),
- the two named direction sets (content → emission_features, phase-signature
  → hidden_features; orthonormal; phase dominant so the DC control survives
  ``k_pos = 1``),
- that :func:`recipe_metrics` returns the right keys for token and window
  archs, and that other extra-bearing generators don't trip the dispatch.
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.data.synthetic import (
    assumption_consequence,
    recipe_instruction_phase_runs,
    signed_motion,
)

# Measured real-signature anchors (bench_spec.md § 1-2): pooled marginal +
# per-symbol dwell-mean ORDER of the canonical hier_categorical fit.
_REAL_MARGINAL = (0.29, 0.51, 0.06, 0.07, 0.07)
_DWELL_ORDER = (1, 0, 2, 3, 4)          # mirror dwell means 4.0>3.0>2.4>1.7>1.5


def test_recipe_shapes_and_extra() -> None:
    data = recipe_instruction_phase_runs(d_in=32, K_c=8, n_c=2, seq_len=16,
                                         n_seqs=64, seed=0)
    assert data.x.shape == (64, 16, 32)
    assert data.emission_features.shape == (8, 32)       # content set
    assert data.hidden_features.shape == (5, 32)         # phase-signature set
    c = data.extra["phase_class_labels"]
    e = data.extra["equality_labels"]
    assert c.shape == e.shape == (64, 16)
    assert int(c.min()) >= 0 and int(c.max()) <= 4
    assert data.extra["n_phases"] == 5
    assert 0.0 < data.extra["alpha"] < 1.0


def test_recipe_label_algebra() -> None:
    data = recipe_instruction_phase_runs(seq_len=64, n_seqs=128, seed=1)
    c = data.extra["phase_class_labels"].numpy()
    e = data.extra["equality_labels"].numpy()
    assert (e[:, 0] == 0).all()                          # e_0 = 0 convention
    assert np.array_equal(e[:, 1:], (c[:, 1:] == c[:, :-1]).astype(e.dtype))
    assert set(np.unique(e)) <= {0.0, 1.0}


def test_recipe_mirror_anchor_marginal_and_dwell_order() -> None:
    """Realized pooled marginal near the measured signature, and the per-class
    continuation rates P(e=1 | c=k) ordered by the mirror's dwell means —
    the class-dependent persistence the equality latent rides on (and the
    documented DC-leak channel the gating quantifies)."""
    data = recipe_instruction_phase_runs(seq_len=64, n_seqs=4096, seed=2)
    c = data.extra["phase_class_labels"].numpy()
    e = data.extra["equality_labels"].numpy()

    marg = np.bincount(c.ravel(), minlength=5) / c.size
    assert np.abs(marg - np.array(_REAL_MARGINAL)).max() < 0.08
    assert marg.argmax() == 1                            # context-background dominates

    cont = np.array([e[:, 1:][c[:, :-1] == k].mean() for k in range(5)])
    assert tuple(np.argsort(-cont)) == _DWELL_ORDER
    assert 0.70 <= cont[1] <= 0.78                       # heavy-dwell class
    assert 0.28 <= cont[4] <= 0.38                       # light-dwell class

    match = float(e[:, 1:].mean())
    assert 0.58 <= match <= 0.68                         # measured P_match ≈ 0.63-0.67
    assert abs(data.extra["match_rate_realized"] - match) < 1e-9


def test_recipe_direction_sets_orthonormal() -> None:
    data = recipe_instruction_phase_runs(d_in=64, K_c=15, seq_len=8, n_seqs=8,
                                         seed=3)
    U = torch.cat([data.hidden_features, data.emission_features], dim=0)
    G = U @ U.T
    off = (G - torch.eye(20)).abs().max().item()
    assert off < 1e-4, f"direction sets not orthonormal (max dev {off})"


def test_recipe_phase_signature_dominant() -> None:
    """The active phase's signature direction is the largest component, so the
    DC control survives k_pos=1."""
    data = recipe_instruction_phase_runs(d_in=64, K_c=15, seq_len=16,
                                         n_seqs=64, seed=4)
    c = data.extra["phase_class_labels"]
    proj = torch.einsum("ntd,kd->ntk", data.x, data.hidden_features)
    assert bool((proj.argmax(dim=-1) == c).float().mean() > 0.999)


def test_recipe_capacity_validation_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="d_in"):
        recipe_instruction_phase_runs(d_in=16, K_c=15, seq_len=8, n_seqs=8, seed=0)
    with pytest.raises(ValueError, match="n_c"):
        recipe_instruction_phase_runs(n_c=20, K_c=15, seq_len=8, n_seqs=8, seed=0)


def test_recipe_metrics_keys_token_and_window() -> None:
    from temp_bench.archs.topk_sae import TopKSAE
    from temp_bench.archs.txc_base import TXCBase
    from temp_bench.evals.recipe_recovery import recipe_metrics

    data = recipe_instruction_phase_runs(d_in=32, K_c=8, n_c=2, seq_len=16,
                                         n_seqs=512, seed=0)
    keys = {"phase_recovery", "phase_balacc", "phase_chance",
            "equality_recovery", "equality_balacc", "equality_chance",
            "equality_base_rate"}
    for arch in (TopKSAE(d_in=32, d_sae=10, k_pos=2),
                 TXCBase(d_in=32, d_sae=10, T=4, k_pos=2)):
        out = recipe_metrics(arch, data, eval_window_L=8, n_windows=64)
        assert keys <= set(out)
        assert 0.0 <= out["phase_balacc"] <= 1.0
        assert 0.0 <= out["equality_balacc"] <= 1.0
        assert 0.0 <= out["equality_base_rate"] <= 1.0


def test_other_generators_do_not_trip_recipe_dispatch() -> None:
    """Gating contract: the dispatch keys off extra['equality_labels']; the
    other extra-bearing generators must not carry it (signed_motion exposes a
    DIFFERENT 'phase_labels' key — per-sequence, unrelated — which is why the
    recipe bench uses 'phase_class_labels' + dispatches on equality)."""
    ac = assumption_consequence(seq_len=8, n_seqs=8, seed=0)
    assert "equality_labels" not in ac.extra
    sm = signed_motion(seq_len=8, n_seqs=8, seed=0)
    assert "equality_labels" not in sm.extra
    assert "phase_class_labels" not in sm.extra
