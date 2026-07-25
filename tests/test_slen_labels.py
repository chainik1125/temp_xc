"""Sanity tests for the slen recency-ladder label logic
(experiments/explorations/task_hunt/labels/slen_lib.py — B8)."""

import numpy as np
import pytest

from experiments.explorations.task_hunt.labels import slen_lib as sl
from experiments.explorations.task_hunt.labels.novelty_lib import (
    kernel_weights,
    trailing_rate,
)


def test_weighted_trailing_matches_trailing_rate_indexing():
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0], dtype=np.int8)
    ours = sl._weighted_trailing(bits, sl.HALF_LIFE_S, sl.SUPPORT_S)
    theirs = trailing_rate(bits, half_life=sl.HALF_LIFE_S,
                           support=sl.SUPPORT_S)
    m = np.isfinite(theirs)
    assert (np.isfinite(ours) == m).all()
    assert ours[m] == pytest.approx(theirs[m], abs=1e-6)


def test_log_lengths_exact():
    x = sl.sent_log_lengths(["a b c", "one", "  ", "w1 w2 w3 w4"])
    assert x == pytest.approx([np.log(3), 0.0, 0.0, np.log(4)])


def test_latch_warmup_and_values():
    x = np.arange(12, dtype=float)
    lat = sl.trailing_latch(x)
    assert np.isnan(lat[: sl.SUPPORT_S]).all()
    # lat[i] = x[i-1] beyond the unified warm-up
    assert lat[sl.SUPPORT_S:] == pytest.approx(
        x[sl.SUPPORT_S - 1: -1])


def test_level_constant_is_identity_and_disp_zero():
    x = np.full(20, 0.7)
    lev = sl.trailing_level(x)
    disp = sl.trailing_disp(x)
    assert np.isnan(lev[: sl.SUPPORT_S]).all()
    assert np.isnan(disp[: sl.SUPPORT_S]).all()
    assert lev[sl.SUPPORT_S:] == pytest.approx(0.7, abs=1e-6)
    assert disp[sl.SUPPORT_S:] == pytest.approx(0.0, abs=1e-6)


def test_level_and_disp_match_hand_formula():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    w = kernel_weights(sl.HALF_LIFE_S, sl.SUPPORT_S)   # lags 1..8
    lev = sl.trailing_level(x)
    disp = sl.trailing_disp(x)
    for i in (8, 13, 19):
        prev = x[i - np.arange(1, sl.SUPPORT_S + 1)]   # x[i-1], x[i-2], …
        m1 = float((w * prev).sum())
        m2 = float((w * prev * prev).sum())
        assert lev[i] == pytest.approx(m1, abs=1e-6)
        assert disp[i] == pytest.approx(np.sqrt(m2 - m1 ** 2), abs=1e-6)


def test_shift_invariance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=30)
    c = 3.7
    m = ~np.isnan(sl.trailing_level(x))
    assert sl.trailing_level(x + c)[m] == pytest.approx(
        sl.trailing_level(x)[m] + c, abs=1e-6)
    assert sl.trailing_disp(x + c)[m] == pytest.approx(
        sl.trailing_disp(x)[m], abs=1e-6)


def test_kernel_ess_value():
    ess = sl.kernel_ess()
    assert 1.0 < ess < sl.SUPPORT_S
    w = kernel_weights(sl.HALF_LIFE_S, sl.SUPPORT_S)
    assert ess == pytest.approx((w.sum() ** 2) / (w * w).sum())
    assert ess == pytest.approx(5.14, abs=0.05)
