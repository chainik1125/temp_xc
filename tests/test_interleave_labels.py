"""Sanity tests for the interleaved-document (anti-conversion) label
logic — experiments/explorations/task_hunt/labels/interleave_lib.py."""

import numpy as np
import pytest

from experiments.explorations.task_hunt.labels.interleave_lib import (
    block_shuffle, content_types, jaccard, pair_docs_by_overlap,
    plan_blocks, random_pairing, source_lexical_auc, token_labels)


def test_plan_blocks_alternates_and_truncates():
    plan = plan_blocks(20, 20, seed=0, lo=1, hi=4)
    assert plan, "non-empty plan"
    srcs = [s for s, _ in plan]
    assert all(a != b for a, b in zip(srcs, srcs[1:])), "strict alternation"
    assert all(1 <= n <= 4 for _, n in plan), "jitter bounds"
    used = [sum(n for s, n in plan if s == k) for k in (0, 1)]
    assert used[0] <= 20 and used[1] <= 20, "never overdraws a source"
    assert plan == plan_blocks(20, 20, seed=0, lo=1, hi=4), "deterministic"
    assert plan != plan_blocks(20, 20, seed=1, lo=1, hi=4), "seed matters"


def test_token_labels_tss_and_source():
    plan = [(1, 2), (0, 1), (1, 3)]
    counts = [4, 3, 5]                      # tokens per block
    src, tss, blk = token_labels(plan, counts)
    assert len(src) == 12
    assert list(src) == [1] * 4 + [0] * 3 + [1] * 5
    assert list(tss) == [-1] * 4 + [0, 1, 2] + [0, 1, 2, 3, 4], \
        "first block guarded -1; tss resets to 0 at every switch"
    assert list(blk) == [0] * 4 + [1] * 3 + [2] * 5


def test_block_shuffle_preserves_tokens_and_merges_runs():
    plan = [(0, 1), (1, 1), (0, 1), (1, 1)]
    counts = [3, 2, 4, 1]
    perm, src_null, tss_null = block_shuffle(plan, counts, seed=3)
    assert sorted(perm.tolist()) == list(range(10)), "a true permutation"
    p2, _, _ = block_shuffle(plan, counts, seed=3)
    assert perm.tolist() == p2.tolist(), "deterministic"
    # labels correspond to the permuted order: source moves with tokens
    src_orig, _, _ = token_labels(plan, counts)
    assert src_null.tolist() == src_orig[perm].tolist()
    # tss: -1 exactly on the first run; resets ONLY at source changes,
    # so adjacent same-source blocks continue counting through the join
    changes = np.flatnonzero(np.diff(src_null.astype(int)) != 0) + 1
    expected = np.empty(10, dtype=int)
    bounds = [0, *changes.tolist(), 10]
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        expected[a:b] = -1 if k == 0 else np.arange(b - a)
    assert tss_null.tolist() == expected.tolist()


def test_pairing_greedy_and_random():
    vocabs = [frozenset(v) for v in
              ({"cat", "dog", "fish"}, {"cat", "dog", "bird"},
               {"quark", "gluon", "meson"}, {"quark", "gluon", "boson"})]
    pairs = pair_docs_by_overlap(vocabs)
    assert {(i, j) for i, j, _ in pairs} == {(0, 1), (2, 3)}, \
        "greedy pairing finds the two high-overlap pairs"
    assert all(ov == pytest.approx(0.5) for _, _, ov in pairs)
    assert jaccard(vocabs[0], vocabs[2]) == 0.0
    rnd = random_pairing(4, seed=0)
    flat = [d for i, j, _ in rnd for d in (i, j)]
    assert sorted(flat) == [0, 1, 2, 3], "random pairing is disjoint"
    assert content_types(["The cat SAT."]) == frozenset({"the", "cat", "sat"})


def test_source_lexical_auc_extremes():
    # estimation ids are held-out halves, disjoint from the blocks
    plan = [(0, 1), (1, 1), (0, 1), (1, 1)]
    # disjoint vocabularies: token identity gives the source away -> 1
    ids_a, ids_b = [1, 2, 3] * 30, [7, 8, 9] * 30
    blocks = [[1, 2, 3], [7, 8, 9], [2, 3, 1], [9, 8, 7]]
    assert source_lexical_auc(ids_a, ids_b, plan, blocks) > 0.95
    # identical unigram distributions: nothing to read -> exactly 0.5
    same = [1, 2, 3] * 30
    blocks_same = [[1, 2, 3], [1, 2, 3], [3, 2, 1], [3, 2, 1]]
    auc = source_lexical_auc(same, list(same), plan, blocks_same)
    assert auc == pytest.approx(0.5), \
        "symmetric held-out estimation must NOT leak the source"
