"""Wave-3 zero-pull trio label logic (labels/sycpress_lib.py +
labels/wave3_lib.py) — pure logic, no corpus I/O."""

import numpy as np
import pytest

from experiments.explorations.task_hunt.labels import sycpress_lib as sp
from experiments.explorations.task_hunt.labels import wave3_lib as w3


# ── sycpress event matching ─────────────────────────────────────────

def test_pushback_matches_each_frozen_substring():
    for s in sp.PUSHBACK_SUBSTRINGS:
        assert sp.is_pushback_turn(f"well, {s} about this?")


def test_pushback_case_insensitive_and_curly_apostrophe():
    assert sp.is_pushback_turn("ARE YOU SURE about that")
    assert sp.is_pushback_turn("I don’t think that’s right.")
    assert sp.is_pushback_turn("I DON'T THINK THAT'S RIGHT")


def test_pushback_negative():
    assert not sp.is_pushback_turn("thanks, that was helpful")
    assert not sp.is_pushback_turn("please write a poem about rain")


def test_pushback_hits_census():
    hits = sp.pushback_hits("Are you sure? I really dislike it.")
    assert set(hits) == {"are you sure", "i really dislike"}


# ── reask events ────────────────────────────────────────────────────

DEFLECT = "I'm sorry, I cannot help with that request."
ORIG = "please translate this ancient sumerian tablet inscription"
REASK_HIT = "again: translate the ancient sumerian tablet inscription"
UNRELATED = "write a birthday poem for my grandmother please"


def test_reask_fires_on_deflected_overlap():
    msgs = [("user", ORIG), ("assistant", DEFLECT), ("user", REASK_HIT)]
    assert w3.reask_events(msgs).tolist() == [0, 0, 1]


def test_reask_requires_deflection_marker():
    msgs = [("user", ORIG), ("assistant", "Sure! Here it is."),
            ("user", REASK_HIT)]
    assert w3.reask_events(msgs).sum() == 0


def test_reask_requires_overlap():
    msgs = [("user", ORIG), ("assistant", DEFLECT), ("user", UNRELATED)]
    assert w3.reask_events(msgs).sum() == 0


def test_reask_requires_min_content_words():
    msgs = [("user", "hello my friend"), ("assistant", DEFLECT),
            ("user", "hello my friend")]
    # {"hello", "friend"} after stopwords = 2 < REASK_MIN_CONTENT_WORDS
    assert w3.reask_events(msgs).sum() == 0


def test_reask_needs_two_back_user_turn():
    msgs = [("assistant", DEFLECT), ("user", REASK_HIT)]
    assert w3.reask_events(msgs).sum() == 0


def test_jaccard_bounds():
    a = frozenset({"x", "y"})
    assert w3.jaccard(a, a) == 1.0
    assert w3.jaccard(a, frozenset()) == 0.0


# ── event flags ─────────────────────────────────────────────────────

def test_event_flags_first_token_and_mask():
    m_idx = np.array([0, 0, 1, 1, 1, 2, 2])
    ev = np.array([0, 1, 0], dtype=np.int8)
    first = w3.event_first_token_flags(m_idx, ev)
    assert first.tolist() == [0, 0, 1, 0, 0, 0, 0]
    mask = w3.event_token_flags(m_idx, ev)
    assert mask.tolist() == [0, 0, 1, 1, 1, 0, 0]


def test_sage_transplant_age_resets_at_event():
    flags = np.zeros(200, dtype=np.int8)
    flags[[100, 150]] = 1
    face = w3.sage_face(flags, support=64)
    assert np.isnan(face[:64]).all()
    assert face[100] == 0.0
    assert face[149] == pytest.approx(np.log2(1 + 49))
    assert face[150] == 0.0
    floor16 = w3.sage_floor(flags, 16)
    # 33 tokens after the event: true age 33, censored at 17
    assert floor16[133] == pytest.approx(np.log2(1 + 17))
    assert floor16[110] == pytest.approx(np.log2(1 + 10))


# ── msdose construction ─────────────────────────────────────────────

def _toy_stream():
    rng = np.random.default_rng(7)
    flat = rng.integers(10, 500, size=5000).astype(np.int32)
    doc_off = np.array([0, 1200, 2600, 5000], dtype=np.int64)
    return flat, doc_off


def test_msdose_plan_deterministic():
    p1 = w3.msdose_plan(np.random.default_rng(w3.MSDOSE_SEED), 5)
    p2 = w3.msdose_plan(np.random.default_rng(w3.MSDOSE_SEED), 5)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
    assert all(w3.MSDOSE_SPAN_MIN <= x <= w3.MSDOSE_SPAN_MAX
               for lens in p1 for x in lens)


def test_msdose_doc_structure():
    flat, off = _toy_stream()
    delim = np.array([9001, 9002], dtype=np.int32)
    lens = np.array([50, 120, 40])
    ids, bound, dose = w3.msdose_doc(
        np.random.default_rng(0), flat, off, lens, delim)
    assert len(ids) == len(bound) == len(dose)
    assert bound.sum() == 3
    assert dose[0] == 1 and dose[-1] == 3
    assert (np.diff(dose) >= 0).all()
    starts = np.flatnonzero(bound)
    for s in starts:
        assert ids[s] == 9001 and ids[s + 1] == 9002
    assert len(ids) == 3 * len(delim) + lens.sum()


def test_dose_window_count():
    b = np.zeros(30, dtype=np.int8)
    b[[5, 20]] = 1
    c = w3.dose_window_count(b, 8)
    assert c[5] == 0        # current token excluded
    assert c[6] == 1
    assert c[13] == 1       # 5 in [5, 12] view
    assert c[14] == 0       # 5 out of [6, 13] view
    assert c[21] == 1
