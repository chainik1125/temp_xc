"""Pure label logic for the interleaved-document (anti-conversion)
corpus — no tokenizers, no I/O. Covered by tests/test_interleave_labels.py.

Design (hunt-support-stats item 3): two lexically-matched fineweb docs
are interleaved in strictly alternating blocks of jittered sentence
counts. Per-token labels: which source is active (`source`, 0/1 within
the pair) and tokens since the last source switch (`tss`, -1 on the
first block — no prior switch, the round-1 undefined convention). The
shuffled-block null permutes whole blocks (seeded): token multisets per
doc are preserved, adjacent same-source blocks merge into one run, and
`tss` is recomputed on the permuted order.
"""

from __future__ import annotations

import numpy as np

STOP_MIN_LEN = 3            # content types = lowercased alpha, len >= 3


def content_types(sentences) -> frozenset:
    """Lexical fingerprint of a doc: lowercased alphabetic word types of
    length >= STOP_MIN_LEN (short/function words carry no source info)."""
    out = set()
    for s in sentences:
        for w in s.split():
            w = "".join(c for c in w.lower() if c.isalpha())
            if len(w) >= STOP_MIN_LEN:
                out.add(w)
    return frozenset(out)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_docs_by_overlap(vocabs) -> list:
    """Greedy max-Jaccard disjoint pairing over all docs. Returns
    [(i, j, overlap)] with i < j, sorted by decreasing overlap; every
    doc appears in exactly one pair (odd doc counts drop the last one).
    Deterministic: ties break on (i, j)."""
    n = len(vocabs)
    scored = sorted(((jaccard(vocabs[i], vocabs[j]), i, j)
                     for i in range(n) for j in range(i + 1, n)),
                    key=lambda t: (-t[0], t[1], t[2]))
    used, pairs = set(), []
    for ov, i, j in scored:
        if i in used or j in used:
            continue
        used.update((i, j))
        pairs.append((i, j, ov))
        if len(used) >= n - (n % 2):
            break
    return pairs


def random_pairing(n: int, seed: int = 0) -> list:
    """Disjoint random pairing (the matching null), deterministic."""
    order = np.random.default_rng(seed).permutation(n)
    return [(int(min(order[k], order[k + 1])),
             int(max(order[k], order[k + 1])), None)
            for k in range(0, n - (n % 2), 2)]


def plan_blocks(n_sents_a: int, n_sents_b: int, seed: int,
                lo: int = 1, hi: int = 4) -> list:
    """Strictly alternating block plan [(source, n_sentences)], block
    length uniform on [lo, hi] (seeded), starting source seeded. The
    plan stops when the active source cannot serve the drawn block in
    full — no partial blocks, so every emitted block is jitter-sized."""
    rng = np.random.default_rng(seed)
    remaining = [n_sents_a, n_sents_b]
    src = int(rng.integers(0, 2))
    plan = []
    while True:
        want = int(rng.integers(lo, hi + 1))
        if remaining[src] < want:
            break
        plan.append((src, want))
        remaining[src] -= want
        src = 1 - src
    return plan


def token_labels(plan, block_token_counts) -> tuple:
    """Per-token (source int8, tss int32, block int32) for a doc, given
    the block plan and each block's token count. tss = tokens since the
    last switch (0 at each switch); the first block has no prior switch
    -> tss = -1 (guard, never train on it)."""
    src_l, tss_l, blk_l = [], [], []
    for b, ((src, _), n_tok) in enumerate(zip(plan, block_token_counts)):
        src_l.append(np.full(n_tok, src, dtype=np.int8))
        tss_l.append(np.full(n_tok, -1, dtype=np.int32) if b == 0
                     else np.arange(n_tok, dtype=np.int32))
        blk_l.append(np.full(n_tok, b, dtype=np.int32))
    if not src_l:
        z = np.zeros(0)
        return z.astype(np.int8), z.astype(np.int32), z.astype(np.int32)
    return np.concatenate(src_l), np.concatenate(tss_l), np.concatenate(blk_l)


def block_shuffle(plan, block_token_counts, seed: int) -> tuple:
    """The shuffled-block null. Returns (perm, source_null, tss_null):
    `perm` is a within-doc token permutation realized by shuffling whole
    blocks (null token order = original_tokens[perm]); labels are
    recomputed on the permuted order — where the shuffle lands two
    same-source blocks adjacent they merge into one run (no switch
    between them), and the first run gets tss = -1."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(plan))
    starts = np.concatenate([[0], np.cumsum(block_token_counts)])[:-1]
    perm = np.concatenate([np.arange(starts[b], starts[b] +
                                     block_token_counts[b])
                           for b in order]) if len(plan) else np.zeros(
                               0, dtype=np.int64)
    src_seq = [plan[b][0] for b in order]
    cnt_seq = [block_token_counts[b] for b in order]
    # merge adjacent same-source blocks into runs, then relabel
    runs = []
    for s, c in zip(src_seq, cnt_seq):
        if runs and runs[-1][0] == s:
            runs[-1][1] += c
        else:
            runs.append([s, c])
    src_null, tss_null, _ = token_labels([(s, 0) for s, _ in runs],
                                         [c for _, c in runs])
    return perm.astype(np.int64), src_null, tss_null


def source_lexical_auc(est_ids_a, est_ids_b, plan, block_ids,
                       alpha: float = 0.5):
    """Label-side per-token triage for ONE pair: how well does the
    current token's IDENTITY alone give away the active source?

    Scores every interleaved token by the unigram log-odds of the two
    sources, with the distributions estimated from HELD-OUT halves of
    the source docs (est_ids_a / est_ids_b — token lists disjoint from
    the interleaved corpus; add-alpha smoothing). Any asymmetric
    in-corpus estimator systematically leaks the answer through the
    count subtraction itself (verified in tests), so the estimation
    data must be disjoint and symmetric. Returns the AUC of source
    prediction over all tokens of the pair."""
    from collections import Counter
    cnt = [Counter(list(est_ids_a)), Counter(list(est_ids_b))]
    tot = [sum(cnt[0].values()), sum(cnt[1].values())]
    scores, labels = [], []
    for (src, _), toks in zip(plan, block_ids):
        for t in toks:
            scores.append(
                np.log((cnt[1][t] + alpha) / (tot[1] + alpha))
                - np.log((cnt[0][t] + alpha) / (tot[0] + alpha)))
            labels.append(src)
    return rank_auc(np.asarray(scores), np.asarray(labels))


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC of `scores` predicting binary `labels` (1 =
    positive), ties get half credit. NaN if one class is absent."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sv = np.sort(scores)
    uniq, inv = np.unique(sv, return_inverse=True)
    mean_rank = np.zeros(uniq.size)
    np.add.at(mean_rank, inv, np.arange(1, sv.size + 1))
    mean_rank /= np.bincount(inv)
    r = mean_rank[np.searchsorted(uniq, scores)]
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def switch_hazard(plan_token_counts_list) -> dict:
    """Empirical switch hazard h(t) = P(block ends at token offset t |
    block reached t), pooled over all non-first blocks of all docs —
    the honest disclosure of how much generative signal `tss` carries
    (uniform sentence-jitter blocks are NOT memoryless)."""
    lengths = np.array([c for counts in plan_token_counts_list
                        for c in counts[1:]], dtype=np.int64)
    if lengths.size == 0:
        return {"n_blocks": 0}
    grid = np.arange(1, int(lengths.max()) + 1)
    at_risk = np.array([(lengths >= t).sum() for t in grid])
    ends = np.array([(lengths == t).sum() for t in grid])
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(at_risk > 0, ends / at_risk, np.nan)
    q = np.quantile(lengths, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"n_blocks": int(lengths.size),
            "block_tokens_q10_25_50_75_90": [float(x) for x in q],
            "hazard_by_offset": {str(int(t)): float(hz)
                                 for t, hz, risk in zip(grid, h, at_risk)
                                 if risk >= 50}}
