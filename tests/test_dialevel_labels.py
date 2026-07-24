"""Sanity tests for the dialogue turn-length LEVEL label logic
(CANDIDATES.md B5) — the exact code `labels/build_dialevel.py` uses."""

import numpy as np
import pytest

from experiments.explorations.task_hunt.labels import dialevel_lib as dl


def test_render_spans_exact():
    text, spans = dl.render_dialogue(["Hi there.", "Hello!", "Bye."])
    assert text == "Hi there.\nHello!\nBye."
    for (a, b), t in zip(spans, ["Hi there.", "Hello!", "Bye."]):
        assert text[a:b] == t
    assert text[spans[0][1]] == "\n"


def test_trailing_turn_mean_manual_and_exclusion():
    sizes = [10, 20, 30, 40, 50, 60, 70]
    lev = dl.trailing_turn_mean(sizes, support=5)
    assert np.isnan(lev[:5]).all()
    assert lev[5] == pytest.approx(np.mean([10, 20, 30, 40, 50]))
    assert lev[6] == pytest.approx(np.mean([20, 30, 40, 50, 60]))
    # current turn excluded: changing turn 5's size leaves lev[5] fixed
    sizes2 = list(sizes)
    sizes2[5] = 999
    assert dl.trailing_turn_mean(sizes2, support=5)[5] == lev[5]


def test_boundary_flags():
    text, _ = dl.render_dialogue(["ab", "cd"])
    #  offsets: 'ab'(0,2) '\nc'(2,4) 'd'(4,5)  -> middle token spans \n
    offs = [(0, 2), (2, 4), (4, 5)]
    assert dl.boundary_flags(offs, text).tolist() == [0, 1, 0]


def test_tokens_since_turn_start():
    turn_idx = np.array([0, 0, 0, 1, 1, 2])
    assert dl.tokens_since_turn_start(turn_idx).tolist() == [0, 1, 2, 0, 1, 0]
