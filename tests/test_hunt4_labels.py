"""Label-logic tests for the gen-4 hunt faces (labels/hunt4_lib.py)."""

import numpy as np
import pytest

from experiments.explorations.task_hunt.labels import hunt3_lib as h3
from experiments.explorations.task_hunt.labels import hunt4_lib as h4


def _rand_dialogue(n=400, vocab=60, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, n)
    # alternating turns of random length 3..12, speaker = turn % 2
    spk = np.zeros(n, dtype=np.int8)
    i, t = 0, 0
    while i < n:
        ln = int(rng.integers(3, 13))
        spk[i:i + ln] = t % 2
        i += ln
        t += 1
    return ids, spk


def test_last_occurrence_by_speaker_reference():
    ids, spk = _rand_dialogue(seed=1)
    same, oth = h4.last_occurrence_by_speaker(ids, spk)
    for i in range(len(ids)):
        prev_same = [j for j in range(i) if ids[j] == ids[i]
                     and spk[j] == spk[i]]
        prev_oth = [j for j in range(i) if ids[j] == ids[i]
                    and spk[j] != spk[i]]
        assert same[i] == (prev_same[-1] if prev_same else -1)
        assert oth[i] == (prev_oth[-1] if prev_oth else -1)
    # consistency with the unresolved last_occurrence
    lo = h3.last_occurrence(ids)
    assert np.array_equal(np.maximum(same, oth), lo)


def test_adoption_semantics_hand_example():
    #        A  A  B  B  A  B  A
    ids = np.array([5, 6, 5, 7, 7, 6, 9])
    spk = np.array([0, 0, 1, 1, 0, 1, 0])
    same, oth = h4.last_occurrence_by_speaker(ids, spk)
    ev = h4.adoption_events(same, oth)
    # tok2: B says 5 first-for-B, A said it -> adoption
    # tok4: A says 7 first-for-A, B said it -> adoption
    # tok5: B says 6 first-for-B, A said it -> adoption
    # tok6: 9 is conversation-first -> NOT adoption
    assert ev.tolist() == [0, 0, 1, 0, 1, 1, 0]


def test_long_return_out_of_window_by_construction():
    ids, _ = _rand_dialogue(n=600, vocab=250, seed=2)
    lo = h3.last_occurrence(ids)
    ev = h4.long_return_events(lo, gap=h4.RET_GAP)
    assert ev.sum() > 0, "test dialogue produced no long returns"
    for T in h3.FLOOR_TS:                     # every floor T ≤ RET_GAP
        assert T <= h4.RET_GAP
        wn = h3.window_novelty_events(lo, T)
        assert np.all(wn[ev == 1] == 1), \
            f"long-return event visible inside a T={T} window"


def test_xnov_floor_events_hand_example():
    ids = np.array([5, 1, 2, 3, 4, 5, 5])
    spk = np.array([0, 0, 1, 1, 1, 1, 0])
    same, oth = h4.last_occurrence_by_speaker(ids, spk)
    ev = h4.xnov_floor_events(same, oth, T=3)
    # tok5 (B says 5): A's use at 0 is 5 back > T=3 -> other NOT in
    # window -> 0. tok6 (A says 5): B's use at 5 is 1 back <= 3, A's
    # own use at 0 is 6 back > 3 (out of window) -> floor fires 1
    # (a FALSE positive for true adoption — exactly the cheat's gap).
    assert ev[5] == 0 and ev[6] == 1
    true_ev = h4.adoption_events(same, oth)
    assert true_ev[6] == 0                    # A had said 5 before


def test_cross_return_subset_and_attribution():
    ids, spk = _rand_dialogue(n=700, vocab=300, seed=6)
    same, oth = h4.last_occurrence_by_speaker(ids, spk)
    lo = np.maximum(same, oth)
    tret = h4.long_return_events(lo)
    xret = h4.cross_return_events(same, oth)
    assert xret.sum() > 0
    assert np.all(tret[xret == 1] == 1)       # xret ⊂ tret
    for i in np.flatnonzero(xret):            # most recent use = other's
        assert oth[i] > same[i]
    # hand example: A says 9 at 0; B resumes it far later
    ids2 = np.concatenate([[9], np.arange(100, 100 + 80), [9]])
    spk2 = np.concatenate([[0], np.tile([0, 1], 40), [1]])
    s2, o2 = h4.last_occurrence_by_speaker(ids2, spk2)
    x2 = h4.cross_return_events(s2, o2)
    assert x2[-1] == 1                        # B returns to A's token
    spk3 = spk2.copy()
    spk3[-1] = 0                              # A resumes their OWN token
    s3, o3 = h4.last_occurrence_by_speaker(ids2, spk3)
    assert h4.cross_return_events(s3, o3)[-1] == 0


def test_return_depth_reference_and_mass_guard():
    ids, _ = _rand_dialogue(n=800, vocab=300, seed=7)
    lo = h3.last_occurrence(ids)
    ev = h4.long_return_events(lo).astype(float)
    d = h4.return_depth_face(lo)
    w = h3.tok_kernel(h4.SUPPORT_TOK)
    idx = np.arange(len(ids), dtype=float)
    logg = np.where(ev > 0, np.log2(np.maximum(idx - lo, 2.0)), 0.0)
    checked = 0
    for i in range(h4.SUPPORT_TOK, len(ids)):
        seg = slice(i - h4.SUPPORT_TOK, i)
        e = ev[seg][::-1]
        g = logg[seg][::-1]
        den = (w * e).sum() / w.sum()
        if den >= h4.TRETD_MIN_RATE:
            ref = (w * e * g).sum() / (w * e).sum()
            assert d[i] == pytest.approx(ref, abs=1e-5)
            assert d[i] > np.log2(h4.RET_GAP)     # gaps all exceed 64
            checked += 1
        else:
            assert np.isnan(d[i])
    assert checked > 50


def test_speaker_rates_reference_and_mass():
    ids, spk = _rand_dialogue(n=300, seed=3)
    novel = (h3.last_occurrence(ids) < 0).astype(np.int8)
    k0, k1, mm = h4.speaker_rates(novel, spk, h4.SUPPORT_TOK)
    w = h3.tok_kernel(h4.SUPPORT_TOK)
    for i in (h4.SUPPORT_TOK, 150, 299):
        seg = slice(i - h4.SUPPORT_TOK, i)
        e = novel[seg][::-1].astype(float)
        s = spk[seg][::-1]
        for s_val, k in ((0, k0), (1, k1)):
            m = (s == s_val).astype(float)
            ref = (w * e * m).sum() / max((w * m).sum(), 1e-9)
            assert k[i] == pytest.approx(ref, abs=1e-6)
        ref_mm = min((w * (s == 0)).sum(), (w * (s == 1)).sum()) / w.sum()
        assert mm[i] == pytest.approx(ref_mm, abs=1e-6)
    assert np.all(np.isnan(k0[:h4.SUPPORT_TOK]))


def test_sdom_reference_relabel_invariance_and_mass_guard():
    ids, spk = _rand_dialogue(n=500, seed=4)
    novel = (h3.last_occurrence(ids) < 0).astype(np.int8)
    d = h4.sdom_face(novel, spk)
    # reference at a few positions: current-speaker rate minus other's
    w = h3.tok_kernel(h4.SUPPORT_TOK)
    for i in (100, 250, 499):
        if not np.isfinite(d[i]):
            continue
        seg = slice(i - h4.SUPPORT_TOK, i)
        e = novel[seg][::-1].astype(float)
        s = spk[seg][::-1]
        ks = {v: (w * e * (s == v)).sum() / max((w * (s == v)).sum(), 1e-9)
              for v in (0, 1)}
        ref = ks[spk[i]] - ks[1 - spk[i]]
        assert d[i] == pytest.approx(ref, abs=1e-6)
    # D is RELATIONAL (current minus other): a global speaker relabel
    # renames who is "current" at every token consistently, so D is
    # INVARIANT under it (not sign-flipped).
    d_sw = h4.sdom_face(novel, 1 - spk)
    m = np.isfinite(d) & np.isfinite(d_sw)
    assert m.sum() > 100
    assert np.allclose(d[m], d_sw[m], atol=1e-6)
    # mass guard: a long solo run by speaker 0 NaNs the face
    spk_solo = spk.copy()
    spk_solo[200:400] = 0
    d2 = h4.sdom_face(novel, spk_solo)
    assert np.isnan(d2[380])                  # deep inside the solo run


def test_sdom_floor_equals_face_when_events_coincide():
    # Rewrite any long-gap repeat to a fresh type so that
    # window-novelty at T = 64 coincides EXACTLY with conversation
    # novelty; then sdom_floor(T=64) must equal sdom_face — the two
    # code paths compute the same functional on the same events.
    ids, spk = _rand_dialogue(n=400, vocab=25, seed=5)
    fresh = 10_000
    for _ in range(50):    # rewrites lengthen later gaps — to fixpoint
        lo = h3.last_occurrence(ids)
        gap_long = (lo >= 0) & (np.arange(len(ids)) - lo > h4.SUPPORT_TOK)
        if not gap_long.any():
            break
        ids = ids.copy()
        ids[gap_long] = fresh + np.arange(int(gap_long.sum()))
        fresh += int(gap_long.sum())
    lo = h3.last_occurrence(ids)
    wn = h3.window_novelty_events(lo, h4.SUPPORT_TOK)
    novel = (lo < 0).astype(np.int8)
    assert np.array_equal(wn, novel)
    d_face = h4.sdom_face(novel, spk)
    d_fl, kc, ko = h4.sdom_floor(lo, spk, T=h4.SUPPORT_TOK)
    m = np.isfinite(d_face) & np.isfinite(d_fl)
    assert m.sum() > 100
    assert np.allclose(d_fl[m], d_face[m], atol=1e-6)
    assert np.allclose((kc - ko)[m], d_fl[m], atol=1e-6)
