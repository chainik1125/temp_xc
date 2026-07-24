"""Tests for the doc-level (cluster) bootstrap used by the corpus
scale-up campaign (`briefings/corpus-scaleup.md`)."""

import numpy as np

from experiments.explorations.task_hunt.labels import boot_lib as bl
from experiments.explorations.task_hunt.labels import novelty_lib as nl
from experiments.explorations.task_hunt.labels.interleave_lib import rank_auc


def test_counting_auc_matches_rank_auc_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        # heavy ties on purpose: rank_auc gives ties half credit and the
        # level-counting form must agree exactly, not approximately
        scores = rng.integers(0, 5, size=200).astype(float)
        labels = rng.integers(0, 2, size=200)
        lev, K = bl._levels(scores)
        cp = np.bincount(lev[labels == 1], minlength=K).astype(float)
        cn = np.bincount(lev[labels == 0], minlength=K).astype(float)
        assert np.isclose(bl._auc_from_counts(cp, cn),
                          rank_auc(scores, labels), atol=1e-12)


def test_point_estimate_is_the_shipped_triage_number():
    rng = np.random.default_rng(1)
    doc_of = np.repeat(np.arange(30), 20)
    scores = rng.normal(size=600)
    labels = (rng.random(600) < 0.4).astype(int)
    out = bl.bootstrap_auc(scores, labels, doc_of, n_reps=50)
    assert np.isclose(out["point"], rank_auc(scores, labels))
    assert out["n_docs"] == 30 and out["n_rows"] == 600
    assert out["ci_lo"] <= out["mean"] <= out["ci_hi"]


def test_direction_agnostic_fold():
    # a deliberately inverted score: raw AUC well below 0.5, and the
    # frozen bars read the fold
    doc_of = np.repeat(np.arange(20), 10)
    labels = np.tile([0] * 5 + [1] * 5, 20)
    scores = 1.0 - labels + np.random.default_rng(2).normal(0, .01, 200)
    out = bl.bootstrap_auc(scores, labels, doc_of, n_reps=100)
    assert out["point"] < 0.5
    assert np.isclose(out["point_direction_agnostic"], 1 - out["point"])
    assert out["ci_lo_direction_agnostic"] > 0.5


def test_seeded_and_reproducible():
    doc_of = np.repeat(np.arange(25), 8)
    rng = np.random.default_rng(3)
    scores, labels = rng.normal(size=200), rng.integers(0, 2, size=200)
    a = bl.bootstrap_auc(scores, labels, doc_of, n_reps=100, seed=7)
    b = bl.bootstrap_auc(scores, labels, doc_of, n_reps=100, seed=7)
    c = bl.bootstrap_auc(scores, labels, doc_of, n_reps=100, seed=8)
    assert a == b
    assert a["se"] != c["se"]


def test_resampling_unit_is_the_document_not_the_row():
    """With rows perfectly correlated inside documents, the cluster
    bootstrap's spread must not change when each document is repeated
    more times — and must be far wider than a row-level bootstrap."""
    n_docs, rng = 20, np.random.default_rng(4)
    doc_score = rng.normal(size=n_docs)
    doc_label = (rng.random(n_docs) < 0.5).astype(int)

    def cluster_se(rows_per_doc):
        doc_of = np.repeat(np.arange(n_docs), rows_per_doc)
        out = bl.bootstrap_auc(np.repeat(doc_score, rows_per_doc),
                               np.repeat(doc_label, rows_per_doc),
                               doc_of, n_reps=200, seed=0)
        return out["se"]

    # the AUC of a resample depends only on doc multiplicities here, so
    # the whole bootstrap distribution is invariant to rows-per-doc
    assert np.isclose(cluster_se(5), cluster_se(500))

    # the row-level bootstrap that this replaces: same data, resampling
    # rows, SE collapses as rows-per-doc grows -> a spuriously tight CI
    rows_per_doc = 500
    s = np.repeat(doc_score, rows_per_doc)
    y = np.repeat(doc_label, rows_per_doc)
    r = np.random.default_rng(0)
    row_reps = []
    for _ in range(200):
        idx = r.integers(0, s.size, s.size)
        row_reps.append(rank_auc(s[idx], y[idx]))
    assert np.std(row_reps, ddof=1) < 0.2 * cluster_se(rows_per_doc)


def test_tercile_wrapper_selects_the_same_rows_as_the_triage():
    rng = np.random.default_rng(5)
    n = 500
    doc_of = np.repeat(np.arange(25), 20)
    scores = rng.normal(size=n)
    tercile = rng.integers(-1, 3, size=n).astype(np.int8)   # -1/0/1/2
    mask = rng.random(n) < 0.8
    out = bl.bootstrap_tercile_auc(scores, tercile, mask, doc_of, n_reps=20)
    assert np.isclose(out["point"], nl.tercile_auc(scores, tercile, mask))
    assert out["n_rows"] == int((mask & (tercile == 0)).sum()
                                + (mask & (tercile == 2)).sum())


def test_degenerate_selection_reports_nan_not_crash():
    doc_of = np.arange(10)
    scores = np.arange(10, dtype=float)
    labels = np.ones(10, dtype=int)          # one class only
    out = bl.bootstrap_auc(scores, labels, doc_of, n_reps=10)
    assert np.isnan(out["point"]) and np.isnan(out["ci_lo"])
    assert out["n_neg"] == 0
