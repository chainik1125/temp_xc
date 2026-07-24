"""Sanity tests for the vocabulary-novelty label logic (CANDIDATES.md
B2) — the exact code `labels/build_novelty.py` uses."""

import numpy as np
import pytest

from experiments.explorations.task_hunt.labels import novelty_lib as nl


def test_novelty_bits_exact():
    ids = [5, 7, 5, 9, 7, 7, 1]
    assert nl.novelty_bits(ids).tolist() == [1, 1, 0, 1, 0, 0, 1]


def test_kernel_weights_normalized_and_decaying():
    w = nl.kernel_weights(half_life=16, support=64)
    assert w.shape == (64,)
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.all(np.diff(w) < 0)
    assert w[16] / w[0] == pytest.approx(0.5, rel=1e-9)


def test_trailing_rate_matches_manual_and_guards():
    rng = np.random.default_rng(0)
    bits = (rng.random(200) < 0.3).astype(np.int8)
    rate = nl.trailing_rate(bits, half_life=4, support=16)
    assert np.isnan(rate[:16]).all()
    w = nl.kernel_weights(4, 16)
    for t in (16, 57, 199):
        manual = float((w * bits[t - 1:: -1][:16]).sum())
        assert rate[t] == pytest.approx(manual, abs=1e-6)
    ones = np.ones(40, dtype=np.int8)
    r1 = nl.trailing_rate(ones, half_life=4, support=16)
    assert np.allclose(r1[16:], 1.0)


def test_current_token_excluded_from_own_label():
    bits = np.zeros(100, dtype=np.int8)
    rate_a = nl.trailing_rate(bits.copy(), half_life=4, support=16)
    bits_b = bits.copy()
    bits_b[80] = 1                       # flip ONLY position 80
    rate_b = nl.trailing_rate(bits_b, half_life=4, support=16)
    assert rate_a[80] == rate_b[80]      # own bit never leaks into label
    assert rate_b[81] > rate_a[81]       # but the next position sees it


def test_position_bins():
    pos = np.array([0, 63, 64, 127, 128, 500, 1024, 5000])
    assert nl.position_bin(pos).tolist() == [-1, -1, 0, 0, 1, 2, 4, 5]


def test_detrend_removes_pure_position_signal():
    rng = np.random.default_rng(1)
    pos = rng.integers(64, 4096, size=5000)
    pbin = nl.position_bin(pos)
    rate = (0.6 - 0.08 * pbin).astype(np.float32)   # exactly bin-determined
    train = rng.random(5000) < 0.8
    resid, expected = nl.detrend(rate, pbin, train)
    assert np.nanmax(np.abs(resid)) < 1e-5
    assert len(expected) == nl.N_POS_BINS


def test_within_doc_perm_respects_docs_and_seed():
    off = np.array([0, 10, 25, 30])
    p1 = nl.within_doc_perm(off, seed=7)
    p2 = nl.within_doc_perm(off, seed=7)
    p3 = nl.within_doc_perm(off, seed=8)
    assert np.array_equal(p1, p2) and not np.array_equal(p1, p3)
    for a, b in zip(off[:-1], off[1:]):
        assert sorted(p1[a:b]) == list(range(a, b))


def test_novelty_count_invariant_under_permutation():
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 50, size=300)
    n_first = int(nl.novelty_bits(ids).sum())
    perm = rng.permutation(300)
    assert int(nl.novelty_bits(ids[perm]).sum()) == n_first  # = #types


def test_type_mean_triage_detects_planted_leak_and_not_noise():
    rng = np.random.default_rng(3)
    ids = rng.integers(0, 200, size=20000)
    train = rng.random(20000) < 0.8
    leak = (ids % 7).astype(float) + rng.normal(0, 0.1, 20000)  # id-driven
    _, terc = _terciles(leak)
    auc_leak = nl.tercile_auc(nl.type_mean_scores(ids, leak, train),
                              terc, ~train)
    noise = rng.normal(0, 1, 20000)
    _, terc_n = _terciles(noise)
    auc_noise = nl.tercile_auc(nl.type_mean_scores(ids, noise, train),
                               terc_n, ~train)
    assert auc_leak > 0.9
    assert 0.45 < auc_noise < 0.55


def _terciles(v):
    from experiments.explorations.task_hunt.labels.lib import tercile_bins
    return tercile_bins(np.asarray(v, dtype=float))
