"""Label-logic tests for the gen-4 corpus-scout faces
(labels/gen4c_lib.py)."""

import numpy as np

from experiments.explorations.task_hunt.labels import hunt3_lib as h3
from experiments.explorations.task_hunt.labels import gen4c_lib as g4


def _rand(n=400, vocab=50, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, n)
    mask = (rng.random(n) < 0.35).astype(np.int8)
    return ids, mask


def test_last_occurrence_masked_reference():
    ids, mask = _rand(seed=1)
    lom = g4.last_occurrence_masked(ids, mask)
    for i in range(len(ids)):
        if not mask[i]:
            assert lom[i] == -1
            continue
        prev = [j for j in range(i) if mask[j] and ids[j] == ids[i]]
        assert lom[i] == (prev[-1] if prev else -1)


def test_last_occurrence_masked_allones_matches_unmasked():
    ids, _ = _rand(seed=2)
    ones = np.ones(len(ids), dtype=np.int8)
    assert np.array_equal(g4.last_occurrence_masked(ids, ones),
                          h3.last_occurrence(ids))


def test_masked_return_events_hand_example():
    #       pos: 0  1  2  3  4  5
    ids = np.array([7, 7, 8, 7, 8, 7])
    mask = np.array([1, 0, 1, 1, 1, 1], dtype=np.int8)
    lom = g4.last_occurrence_masked(ids, mask)
    ev = g4.masked_return_events(lom, mask, gap=2)
    # pos3: prev masked 7 at 0, gap 3 > 2 -> event; pos5: prev at 3,
    # gap 2 -> no; pos4: prev 8 at 2, gap 2 -> no; pos1 unmasked.
    assert ev.tolist() == [0, 0, 0, 1, 0, 0]


def test_masked_window_novelty_bounds():
    ids, mask = _rand(seed=3)
    lom = g4.last_occurrence_masked(ids, mask)
    wn_small = g4.masked_window_novelty(lom, mask, T=4)
    wn_big = g4.masked_window_novelty(lom, mask, T=256)
    # shrinking the window can only ADD novelty
    assert (wn_small >= wn_big).all()
    # at unmasked positions never an event
    assert not wn_small[mask == 0].any()


def test_section_age_reference():
    b = np.zeros(30, dtype=np.int8)
    b[[0, 7, 20]] = 1
    age = g4.section_age(b)
    ref = []
    last = -1
    for i in range(30):
        if b[i]:
            last = i
        ref.append(np.nan if last < 0 else i - last)
    assert np.allclose(age, ref, equal_nan=True)


def test_section_age_nan_before_first_boundary():
    b = np.zeros(10, dtype=np.int8)
    b[4] = 1
    age = g4.section_age(b)
    assert np.isnan(age[:4]).all() and age[4] == 0 and age[9] == 5


def test_sage_floor_exact_iff_in_window():
    b = np.zeros(200, dtype=np.int8)
    b[[0, 90]] = 1
    T = 16
    fl = g4.sage_floor(b, T)
    age = g4.section_age(b)
    vis = age <= T
    assert np.allclose(fl[vis], np.log2(1 + age[vis]))
    assert np.allclose(fl[~vis], np.log2(1 + (T + 1)))


def test_drev_floor_full_support_matches_filter():
    ids, mask = _rand(n=300, seed=4)
    lom = g4.last_occurrence_masked(ids, mask)
    fl = g4.drev_floor(lom, mask, T=64)
    ev = g4.masked_window_novelty(lom, mask, 64)
    ref = h3.filter_rate(ev, 64)
    assert np.allclose(fl, ref, equal_nan=True)


def test_header_regexes():
    from experiments.explorations.task_hunt.labels.pull_wikitext103 import (
        RE_H2, is_h1)
    assert is_h1(" = Valkyria Chronicles III = \n")
    assert not is_h1(" = = Gameplay = = \n")
    assert RE_H2.match(" = = Gameplay = = \n")
    assert RE_H2.match(" = = = Deep = = = \n")
    assert not is_h1(" plain prose line \n")
    assert not is_h1("")
