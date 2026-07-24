"""Sanity tests for the sentence-event intensity label logic
(CANDIDATES.md B3/B4) — the exact code `labels/build_punctint.py`
uses."""

import numpy as np
import pytest

from experiments.explorations.task_hunt.labels import punctint_lib as pl


def test_list_marker_grammar():
    yes = ["- item one", "* star bullet", "• dot bullet", "3. third point",
           "12) twelfth", "a) lettered", "b. lettered dot", "(3) parens",
           "(iv) roman", "  - indented bullet"]
    no = ["Plain sentence.", "3rd place went well", "a cat sat",
          "1234 units sold", "(unrelated parenthetical)", "-not a list-"]
    for s in yes:
        assert pl.is_list_sentence(s), s
    for s in no:
        assert not pl.is_list_sentence(s), s


def test_question_predicate():
    assert pl.is_question_sentence("Why?  ")
    assert not pl.is_question_sentence("Why not.")


def test_sentence_lambda_excludes_current_and_guards():
    ev = np.zeros(30, dtype=np.int8)
    ev[10] = 1
    lam = pl.sentence_lambda(ev)
    assert np.isnan(lam[: pl.SUPPORT_S]).all()
    assert lam[10] == 0.0                       # own event not in own label
    w = pl.kernel_weights(pl.HALF_LIFE_S, pl.SUPPORT_S)
    assert lam[11] == pytest.approx(w[0], abs=1e-7)     # lag-1 weight
    assert lam[12] == pytest.approx(w[1], abs=1e-7)
    assert lam[10 + pl.SUPPORT_S + 1] == 0.0    # beyond support


def test_zero_split_fires_on_inflated_and_not_on_smooth():
    rng = np.random.default_rng(0)
    train = np.ones(9000, dtype=bool)
    inflated = np.where(rng.random(9000) < 0.6, 0.0, rng.random(9000))
    scheme, edges, bins = pl.zero_split_bins(inflated, train)
    assert scheme == "zero_split"
    assert (bins[inflated == 0] == 0).all()
    pos = inflated[inflated > 0]
    med = float(np.median(pos))
    assert (bins[inflated > med] == 2).all()
    smooth = rng.random(9000) + 0.01
    scheme2, _, bins2 = pl.zero_split_bins(smooth, train)
    assert scheme2 == "terciles"
    assert set(np.unique(bins2)) == {0, 1, 2}


def test_zero_split_nan_rows_unlabeled():
    v = np.array([np.nan, 0.0, 0.5, np.nan, 1.0])
    _, _, bins = pl.zero_split_bins(v, np.isfinite(v))
    assert bins[0] == -1 and bins[3] == -1


def test_token_inheritance():
    sent_vals = np.array([np.nan, 0.25, 0.5])
    sent_idx = np.array([0, 0, 1, 1, 1, 2])
    lab = pl.token_labels_from_sentences(sent_vals, sent_idx)
    assert np.isnan(lab[0]) and np.isnan(lab[1])
    assert lab[2:5].tolist() == [0.25, 0.25, 0.25]
    assert lab[5] == 0.5
