"""Pure-logic tests for the day-2 dialogue faces (diafaces_lib)."""

import numpy as np

from experiments.explorations.task_hunt.labels import diafaces_lib as dfl


def test_trailing_turn_slope_linear_ramp():
    # sizes 2,4,6,8,10,12,... -> slope exactly +2 tokens/turn once
    # 5 previous turns exist, for ANY positive weights.
    sizes = np.arange(2, 22, 2)
    out = dfl.trailing_turn_slope(sizes)
    assert np.isnan(out[:5]).all()
    assert np.allclose(out[5:], 2.0)


def test_trailing_turn_slope_constant_and_falling():
    flat = dfl.trailing_turn_slope(np.full(8, 7.0))
    assert np.allclose(flat[5:], 0.0)
    falling = dfl.trailing_turn_slope(np.arange(30, 0, -3))
    assert np.all(falling[5:] < 0)


def test_slope_excludes_current_turn():
    # a huge current turn must not move its own label
    sizes = np.array([2, 4, 6, 8, 10, 999.0])
    out = dfl.trailing_turn_slope(sizes)
    assert np.isclose(out[5], 2.0)


def test_kernel_weights_recency():
    w = dfl.kernel_weights(5, 2.0)
    assert w[-1] == 1.0 and np.all(np.diff(w) > 0)   # newest heaviest
    assert np.isclose(w[-3], 0.5)                    # HL 2 -> half at +2


def test_turns_since_question():
    has_q = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
    out = dfl.turns_since_question(has_q)
    assert np.isnan(out[0]) and np.isnan(out[1])     # no PREVIOUS q yet
    assert list(out[2:]) == [1.0, 2.0, 3.0, 1.0]


def test_balanced_int_edges_tie_heavy():
    # 50% ones — quantile terciles would empty class 0; integer edges
    # must keep all three classes populated.
    vals = np.array([1] * 50 + [2] * 25 + [3] * 15 + [4] * 10, float)
    a, b = dfl.balanced_int_edges(vals)
    bins = dfl.int_edge_bins(vals, a, b)
    assert all((bins == c).sum() > 0 for c in (0, 1, 2))
    assert (a, b) == (1, 2)


def test_int_edge_bins_nan():
    bins = dfl.int_edge_bins(np.array([np.nan, 1.0, 5.0]), 1, 3)
    assert list(bins) == [-1, 0, 2]
