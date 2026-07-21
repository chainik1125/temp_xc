"""Assumption→consequence bench — generator + metric-wiring tests.

Covers the expansion stage-6 #1 add-on:
- the ``assumption_consequence`` generator's shapes + ground-truth labels,
- the label algebra (next_state is state shifted by one; last column = -1),
- the mirror anchor (empirical transition matrix ≈ the g7 fit; the directed
  A→C asymmetry reproduced),
- the two named direction sets (content → emission_features, state →
  hidden_features; orthonormal) + state-signature dominance,
- that :func:`assumption_metrics` returns the right keys for token and
  window archs, and that other generators don't trip the dispatch.
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.data.synthetic import (
    _AC_G7_P,
    assumption_consequence,
    self_exciting,
    semi_markov_modes,
)


def test_assumption_shapes_and_extra() -> None:
    data = assumption_consequence(K_c=6, n_c=2, d_in=16, seq_len=8,
                                  n_seqs=32, seed=0)
    assert data.x.shape == (32, 8, 16)
    assert data.emission_features.shape == (6, 16)      # content set
    assert data.hidden_features.shape == (3, 16)        # state set
    s = data.extra["state_labels"]
    nx = data.extra["next_state_labels"]
    assert s.shape == nx.shape == (32, 8)
    assert int(s.min()) >= 0 and int(s.max()) <= 2
    assert data.extra["n_states"] == 3
    assert np.allclose(np.array(data.extra["P"]).sum(axis=1), 1.0)


def test_assumption_next_state_algebra() -> None:
    data = assumption_consequence(seq_len=64, n_seqs=64, seed=1)
    s = data.extra["state_labels"].numpy()
    nx = data.extra["next_state_labels"].numpy()
    assert np.array_equal(nx[:, :-1], s[:, 1:])
    assert (nx[:, -1] == -1).all()


def test_assumption_mirror_anchor_and_directed_asym() -> None:
    """The realized chain reproduces the g7 fit: transition matrix within
    sampling tolerance, and the directed A→C edge above the C base rate."""
    data = assumption_consequence(seq_len=64, n_seqs=4096, seed=2)
    s = data.extra["state_labels"].numpy()
    P = np.array(data.extra["P"])
    cur, nxt = s[:, :-1].reshape(-1), s[:, 1:].reshape(-1)
    emp = np.zeros((3, 3))
    for i in range(3):
        sel = nxt[cur == i]
        for j in range(3):
            emp[i, j] = (sel == j).mean()
    assert np.abs(emp - P).max() < 0.02
    # directed asymmetry: fwd P(C@t+1 | A@t) ≫ time-reversed P(C@t-1 | A@t)
    fwd = emp[1, 2]
    rev = (cur[nxt == 1] == 2).mean()
    assert fwd > rev + 0.10
    assert abs(fwd - _AC_G7_P[1][2]) < 0.03


def test_assumption_direction_sets_orthonormal_and_dominant() -> None:
    data = assumption_consequence(seq_len=16, n_seqs=64, seed=3)
    U = torch.cat([data.hidden_features, data.emission_features], dim=0)
    G = U @ U.T
    off = (G - torch.eye(20)).abs().max().item()
    assert off < 1e-4, f"direction sets not orthonormal (max dev {off})"
    # the active state's signature direction is the top component per token
    s = data.extra["state_labels"]
    proj = torch.einsum("ntd,kd->ntk", data.x, data.hidden_features)
    assert bool((proj.argmax(dim=-1) == s).float().mean() > 0.999)


def test_assumption_metrics_keys_token_and_window() -> None:
    from temp_bench.archs.topk_sae import TopKSAE
    from temp_bench.archs.txc_base import TXCBase
    from temp_bench.evals.assumption_recovery import assumption_metrics

    data = assumption_consequence(K_c=6, n_c=2, d_in=16, seq_len=16,
                                  n_seqs=512, seed=0)
    keys = {"state_recovery", "state_balacc", "state_chance",
            "nextstate_recovery", "nextstate_balacc", "nextstate_chance",
            "nextstate_oracle_balacc"}
    for arch in (TopKSAE(d_in=16, d_sae=10, k_pos=2),
                 TXCBase(d_in=16, d_sae=10, T=4, k_pos=2)):
        out = assumption_metrics(arch, data, eval_window_L=8, n_windows=64)
        assert keys <= set(out)
        assert 0.0 <= out["state_balacc"] <= 1.0
        assert 0.0 <= out["nextstate_balacc"] <= 1.0
        assert out["nextstate_oracle_balacc"] > 1.0 / 3.0


def test_bayes_balanced_rule_g7() -> None:
    """On the g7 chain the balanced-optimal rule is 'the state persists'."""
    from temp_bench.evals.assumption_recovery import bayes_balanced_rule
    data = assumption_consequence(seq_len=8, n_seqs=8, seed=0)
    rule = bayes_balanced_rule(np.array(data.extra["P"]),
                               np.array(data.extra["pi"]))
    assert rule.tolist() == [0, 1, 2]


def test_other_generators_do_not_trip_assumption_dispatch() -> None:
    """Gating contract: the dispatch keys off extra['state_labels']; the other
    extra-bearing generators must not carry it."""
    se = self_exciting(seq_len=8, n_seqs=8, seed=0)
    cp = semi_markov_modes(seq_len=8, n_seqs=8, seed=0)
    assert "state_labels" not in se.extra
    assert "state_labels" not in cp.extra
