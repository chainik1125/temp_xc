"""Sanity tests for the candidate-factory label logic —
experiments/explorations/task_hunt/labels/factory_lib.py."""

import numpy as np
import pytest

from experiments.explorations.task_hunt.labels import factory_lib as fl


def test_marker_matching_boundaries():
    hit = ["Wait, that is wrong.", "Hmm, let me reconsider.",
           "Actually, I made a mistake earlier.", "No, that fails.",
           "Let me double-check the algebra.",
           "Hold on — scratch that.", "That can’t be right."]
    miss = ["The waiter brought water.", "Awaiting the result.",
            "This is actual progress backwards.".replace("actual", "factual"),
            "There is no solution with x even.",  # 'no' not sentence-initial
            "We know that x = 3."]
    assert all(fl.marker_spans_in_sentence(t) for t in hit)
    assert not any(fl.marker_spans_in_sentence(t) for t in miss)
    ev = fl.sentence_events_markers(hit + miss)
    assert ev.tolist() == [1] * len(hit) + [0] * len(miss)
    # spans come back in sentence coordinates on the ORIGINAL string
    (a, b), = fl.marker_spans_in_sentence("so Wait here")
    assert "so Wait here"[a:b].lower() == "wait"


def test_question_events_and_token_mask():
    ev = fl.sentence_events_question(["Is x even? ", "x is even.", "Why?"])
    assert ev.tolist() == [1, 0, 1]
    offs = [(0, 4), (4, 9), (9, 12)]
    m = fl.token_mask_from_spans(offs, [(3, 5), (10, 11)])
    assert m.tolist() == [1, 1, 1]
    assert fl.token_mask_from_spans(offs, [(4, 4)]).tolist() == [0, 0, 0]


def test_kernel_rate_causal_normalized_and_guarded():
    e = np.zeros(20)
    e[10] = 1.0
    r = fl.kernel_rate(e, k=8, tau=3.0, min_history=4)
    assert np.isnan(r[:4]).all(), "history guard: first MIN_HISTORY NaN"
    assert r[10] == 0.0, "current sentence's own event is not an input"
    w = fl.exp_kernel_weights(8, 3.0)
    assert r[11] == pytest.approx(w[0] / w.sum())
    assert r[12] == pytest.approx(w[1] / w.sum())
    assert r[19] == 0.0, "event fell out of the K=8 window"
    ones = fl.kernel_rate(np.ones(15), k=8, tau=3.0, min_history=4)
    assert np.allclose(ones[4:], 1.0), "all-events rate is exactly 1 " \
        "for i < K too (available-mass normalization)"
    # NaN (unlabeled) inside the window propagates; outside does not
    e2 = np.ones(20)
    e2[5] = np.nan
    r2 = fl.kernel_rate(e2, k=8, tau=3.0, min_history=4)
    assert np.isnan(r2[6]) and np.isnan(r2[13])
    assert np.isfinite(r2[14]), "lag window 6..13 is clean"


def test_shuffle_events_null():
    e = np.array([1.0, 0, np.nan, 0, 1, np.nan, 0])
    rng = np.random.default_rng(0)
    s = fl.shuffle_events(e, rng)
    assert np.isnan(s[[2, 5]]).all(), "unlabeled positions stay fixed"
    fin = ~np.isnan(e)
    assert sorted(s[fin].tolist()) == sorted(e[fin].tolist())
    s2 = fl.shuffle_events(e, np.random.default_rng(0))
    assert np.array_equal(s, s2, equal_nan=True), "deterministic"


def test_trailing_mean_and_slope_prev():
    v = np.array([1.0, 2, 3, 4, 5, 6])
    m = fl.trailing_mean_prev(v, k=4, min_n=2)
    assert np.isnan(m[0]) and np.isnan(m[1]), "min_n guard"
    assert m[2] == pytest.approx(1.5)
    assert m[5] == pytest.approx((2 + 3 + 4 + 5) / 4), "current excluded"
    s = fl.trailing_slope_prev(v, k=4, min_n=2)
    assert s[5] == pytest.approx(1.0), "unit slope on a linear ramp"
    v2 = v.copy()
    v2[3] = np.nan
    assert np.isnan(fl.trailing_mean_prev(v2, k=4, min_n=2)[4])


def test_trailing_token_rates():
    f = np.array([0, 1, 0, 0, 1, 1, 0])
    r = fl.trailing_rate_prev(f, w=3)
    assert np.isnan(r[:3]).all()
    assert r[3] == pytest.approx(1 / 3), "positions 0..2, current excluded"
    assert r[6] == pytest.approx(2 / 3), "positions 3..5"
    c = fl.trailing_count_incl(f, w=3)
    assert np.isnan(c[:2]).all()
    assert c[2] == pytest.approx(1.0) and c[5] == pytest.approx(2.0), \
        "current token included"


def test_zero_inflated_bins_both_schemes():
    spread = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, np.nan])
    scheme, edges, b = fl.zero_inflated_bins(spread)
    assert scheme == "terciles" and b[-1] == -1
    assert sorted(set(b[:-1].tolist())) == [0, 1, 2]
    sparse = np.array([0.0] * 8 + [0.1, 0.2, 0.3, 0.4])
    scheme2, edges2, b2 = fl.zero_inflated_bins(sparse)
    assert scheme2 == "zero_split"
    assert b2.tolist() == [0] * 8 + [1, 1, 2, 2], \
        "zeros / lower-half positive / upper-half positive"


def test_triage_aucs_extremes_and_verdict():
    rng = np.random.default_rng(0)
    n = 4000
    tok_id = rng.integers(0, 50, size=n)
    train = np.arange(n) < n // 2
    test = ~train
    # label fully determined by token id -> AUC ~= 1 -> FAIL
    is_top = (tok_id % 2).astype(int)
    a = fl.token_id_triage_auc(tok_id, is_top, train, test)
    assert a > 0.95
    assert fl.triage_verdict(a, [0.5])["verdict"] == "FAIL"
    # label independent of token id -> AUC ~= 0.5 -> PASS
    is_top2 = rng.integers(0, 2, size=n)
    a2 = fl.token_id_triage_auc(tok_id, is_top2, train, test)
    assert abs(a2 - 0.5) < 0.05
    assert fl.triage_verdict(a2, [0.5])["verdict"] == "PASS"
    # position: monotone label -> AUC 1; inverse counts via extreme
    pos = np.arange(n, dtype=float)
    is_top3 = (pos >= n // 2).astype(int)
    assert fl.position_triage_auc(pos, is_top3, np.ones(n, bool)) == 1.0
    assert fl.triage_verdict(0.5, [0.05])["verdict"] == "FAIL", \
        "inverse-predictive position also kills"


def test_bundle_core_masks_and_balance():
    rng = np.random.default_rng(1)
    N, S = 40, 64
    lam = rng.random((N, S)).astype(np.float32)
    lam_null = rng.random((N, S)).astype(np.float32)
    valid = np.ones((N, S), dtype=bool)
    valid[:, 0] = False
    mask = np.zeros((N, S), dtype=bool)
    mask[:, 40] = True                      # a masked column
    trace_idx = np.repeat(np.arange(8), 5).astype(np.int32)
    win_start = np.zeros(N, dtype=np.int32)
    tok_id = rng.integers(0, 30, size=(N, S)).astype(np.int32)
    core = fl.bundle_core(lam, lam_null, mask, valid, trace_idx,
                          win_start, {t: 200 for t in range(8)}, tok_id)
    d, p, c = core["man"]
    assert (p >= 32).all(), "manifest pos floor"
    assert 40 not in set(p.tolist()), "masked rows never enter a manifest"
    counts = [int((c == k).sum()) for k in (0, 1, 2)]
    assert len(set(counts)) == 1, "class-balanced"
    assert core["scheme"] == "terciles"
    assert core["triage"]["verdict"] == "PASS", \
        "random labels are unreadable from token id / position"
    nd, npos, nc = core["man_null"]
    assert 40 not in set(npos.tolist())
