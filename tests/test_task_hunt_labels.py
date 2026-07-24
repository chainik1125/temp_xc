"""Sanity tests for the task-hunt label engineering (briefing: 5+ for the
repetition-lag builder). Pure-function tests against
experiments.explorations.task_hunt.labels.lib — the exact code the
committed artifacts are built with."""

import numpy as np
import pytest

from experiments.explorations.task_hunt.labels import lib


# ── repetition lag (the 5 required Δ-builder tests) ─────────────────────

def test_delta1_hand_example():
    ids = [5, 6, 5, 5, 7, 6]
    d = lib.delta_prev_ngram(ids, 1)
    assert d.tolist() == [-1, -1, 2, 1, -1, 4]


def test_delta2_hand_example():
    # bigram (5,6) ends at t=1 and again at t=4 -> Δ=3; (6,5) at 2, (5,5)
    # at 3, (6,?) none else.
    ids = [5, 6, 5, 5, 6, 6]
    d = lib.delta_prev_ngram(ids, 2)
    assert d[0] == -1                       # no bigram ends at t=0
    assert d.tolist()[1:] == [-1, -1, -1, 3, -1]


def test_shuffle_null_invariants():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 50, size=400)
    d_null = lib.shuffled_doc_null(ids, 1, np.random.default_rng(1))
    d_real = lib.delta_prev_ngram(ids, 1)
    # first occurrences (Δ = -1) are exactly the distinct tokens — a
    # permutation invariant; and the null must be a valid Δ array.
    assert (d_null == -1).sum() == (d_real == -1).sum() == len(set(ids.tolist()))
    assert d_null.max() < 400 and (d_null[d_null >= 0] >= 1).all()


def test_bucketize_edges_guard_and_none():
    delta = np.array([-1, 1, 4, 5, 8, 9, 16, 17, 64, 65, 1000])
    b = lib.bucketize_delta(delta)
    assert b.tolist() == [3, 0, 0, 1, 1, 2, 2, -1, -1, 3, 3]


def test_manifest_balanced_grouped_and_deterministic():
    rng = np.random.default_rng(2)
    n = 5000
    cls = rng.integers(-1, 4, size=n).astype(np.int8)
    doc = rng.integers(0, 40, size=n)
    pos = rng.integers(0, 500, size=n)
    d1, p1, c1 = lib.balanced_manifest(cls, doc, pos, cap=100, seed=7)
    d2, p2, c2 = lib.balanced_manifest(cls, doc, pos, cap=100, seed=7)
    assert (d1 == d2).all() and (p1 == p2).all() and (c1 == c2).all()
    counts = np.bincount(c1)
    assert (counts == counts[0]).all() and len(counts) == 4
    assert (p1 >= lib.MIN_MANIFEST_POS).all() and (c1 >= 0).all()
    split = lib.doc_split(40, seed=0)
    assert split.sum() == 8 and set(split.tolist()) <= {0, 1}


# ── backtracking intensity ──────────────────────────────────────────────

def test_lambda_matches_hand_computed_mirror():
    b = [0, 1, 0, 0]
    icpt, cpos, w = -2.0, 0.5, [1.5, 0.7]
    lam, lam_hist = lib.lambda_for_sentences(b, icpt, cpos, w)
    sig = lambda z: 1 / (1 + np.exp(-z))
    # i=0: no history; i=1: b0=0; i=2: 1.5*b1+0.7*b0; i=3: 1.5*b2+0.7*b1
    exp = [sig(-2.0 + 0.0), sig(-2.0 + 0.125), sig(-2.0 + 0.25 + 1.5),
           sig(-2.0 + 0.375 + 0.7)]
    assert np.allclose(lam, exp, atol=1e-6)
    assert np.allclose(lam_hist, [sig(-2.0), sig(-2.0), sig(-0.5), sig(-1.3)],
                       atol=1e-6)
    # the current sentence's own label must not leak into λ̂
    lam_b, _ = lib.lambda_for_sentences([0, 1, 1, 0], icpt, cpos, w)
    assert lam_b[1] == lam[1]


def test_tercile_bins_cover_and_flag_nan():
    v = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, np.nan])
    edges, b = lib.tercile_bins(v)
    assert b[-1] == -1 and set(b[:-1].tolist()) == {0, 1, 2}
    assert edges[0] <= edges[1]


# ── proof-operation runs ────────────────────────────────────────────────

def test_run_features_with_unlabeled_breaks():
    labels = [1, 1, 2, None, 2, 2, 2, 1]
    op, tir, start = lib.run_features(labels)
    assert op.tolist() == [1, 1, 2, -1, 2, 2, 2, 1]
    assert tir.tolist() == [0, 1, 0, -1, 0, 1, 2, 0]
    assert start.tolist() == [1, 0, 1, -1, 1, 0, 0, 1]


# ── confidence slope ────────────────────────────────────────────────────

def test_trailing_slope_known_values_and_gaps():
    s = lib.trailing_slope([0, 1, 2, 3], 3)
    assert np.isnan(s[0]) and np.isnan(s[1])
    assert s[2] == pytest.approx(1.0) and s[3] == pytest.approx(1.0)
    s2 = lib.trailing_slope([0, None, 2, 2], 2)
    assert np.isnan(s2[1]) and np.isnan(s2[2]) and s2[3] == pytest.approx(0.0)


# ── sentence↔token bridge ───────────────────────────────────────────────

def test_sentence_index_midpoint_rule():
    spans = [(0, 10), (12, 20)]
    offsets = [(0, 4), (4, 10), (10, 12), (12, 20), (20, 24)]
    idx, ok = lib.sentence_index_per_token(offsets, spans)
    assert idx.tolist() == [0, 0, 0, 1, 1]
    # gap token (10,12) and beyond-last token (20,24) are flagged
    assert ok.tolist() == [True, True, False, True, False]
