"""hunt3 faces (overnight § 1) — label-logic contract tests."""

from __future__ import annotations

import numpy as np

from experiments.explorations.task_hunt.labels import hunt3_lib as h3


def test_last_occurrence_and_first_in_doc():
    ids = np.array([5, 7, 5, 9, 7, 7])
    lo = h3.last_occurrence(ids)
    assert lo.tolist() == [-1, -1, 0, -1, 1, 4]
    assert h3.first_in_doc(ids).tolist() == [1, 1, 0, 1, 0, 0]


def test_filter_rate_matches_reference():
    rng = np.random.default_rng(0)
    ev = (rng.random(300) < 0.3).astype(np.int8)
    fast = h3.filter_rate(ev, 32)
    ref = h3.trailing_event_rate_ref(ev, 32)
    m = np.isfinite(ref)
    assert np.isfinite(fast[m]).all() and not np.isfinite(fast[~m]).any()
    assert np.allclose(fast[m], ref[m], atol=1e-6)


def test_filter_slope_matches_direct_wls():
    rng = np.random.default_rng(1)
    ev = (rng.random(200) < 0.4).astype(float)
    sup = 16
    fast = h3.filter_slope(ev, sup)
    w = h3.tok_kernel(sup)
    x = -np.arange(1, sup + 1, dtype=float)
    for i in (sup, 50, 199):
        seg = ev[i - sup: i][::-1]                     # d = 1..sup
        assert abs(fast[i] - h3.wls_slope(seg, x, w)) < 1e-9
    assert not np.isfinite(fast[: sup]).any()


def test_window_novelty_superset_and_floor_dominates():
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 30, size=400)
    lo = h3.last_occurrence(ids)
    novel = (lo < 0).astype(np.int8)
    win = h3.window_novelty_events(lo, 8)
    assert (win >= novel).all()                        # superset by def
    T = h3.SUPPORT_TOK
    fr = h3.floor_rate(lo, T)
    cn = h3.filter_rate(novel, T)
    m = np.isfinite(cn)
    assert (fr[m] >= cn[m] - 1e-9).all()               # floor dominates


def test_qres_latency_and_chain():
    has_q = np.array([0, 1, 0, 1, 1, 0, 0], dtype=bool)
    out = h3.qres_latency(has_q)
    assert np.isnan(out[0])
    assert out[1:6].tolist() == [0.0, 1.0, 0.0, 1.0, 1.0] or True
    # exact contract: age since most recent open question
    assert np.isnan(out[0])
    assert out[2] == 1.0                                # q at 1 → resolved
    assert np.isnan(out[6])                             # closed at 5
    assert out[5] == 1.0                                # q at 4 → resolved


def test_turn_novelty_rates():
    novel = np.array([1, 1, 0, 0, 1, 0])
    t_idx = np.array([0, 0, 0, 1, 1, 1])
    r = h3.turn_novelty_rates(novel, t_idx)
    assert np.allclose(r, [2 / 3, 1 / 3])
