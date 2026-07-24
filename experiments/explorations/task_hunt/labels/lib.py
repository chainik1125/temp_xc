"""Pure label-engineering functions for the task hunt (no tokenizers, no I/O).

Everything the four builders (`build_replag`, `build_ward_lambda`,
`build_proofops`, `build_confidence`) compute label-side lives here so the
sanity tests (`tests/test_task_hunt_labels.py`) exercise the exact code the
committed artifacts were built with.

Conventions shared by every artifact:

- positions index the document's token sequence tokenized with
  ``add_special_tokens=False`` (the consumer prepends BOS etc. itself);
- ``-1`` is the universal "undefined / unlabeled" sentinel in integer
  arrays; ``NaN`` in float arrays;
- probe-row manifests are (doc_idx, pos) pairs, class-balanced by
  subsampling to the smallest class (seeded), restricted to
  ``pos >= MIN_MANIFEST_POS`` so a trailing window of any screened T fits
  inside the document;
- splits are BY DOCUMENT (``doc_split``) — never split rows of one doc
  across train/test (the leakage rule every probe in the program follows).
"""

from __future__ import annotations

import numpy as np

# Δ-bucket scheme (briefing: Δ ∈ {≤4, ≤8, ≤16, none}). Disjoint bins
# 1–4 / 5–8 / 9–16 → ids 0/1/2; "none" (id 3) = no previous occurrence
# within NONE_MIN positions, so it stays a clean negative class for every
# screened T ≤ 32; the guard band (16, NONE_MIN] is excluded (-1).
BUCKET_EDGES = (4, 8, 16)
NONE_MIN = 64
MIN_MANIFEST_POS = 32


# ── repetition lag ──────────────────────────────────────────────────────

def delta_prev_ngram(ids, n: int) -> np.ndarray:
    """Distance to the previous occurrence of the n-gram ENDING at each
    position (end-to-end distance); -1 if none (or pos < n-1)."""
    ids = list(ids)
    out = np.full(len(ids), -1, dtype=np.int32)
    last: dict = {}
    for t in range(len(ids)):
        if t < n - 1:
            continue
        key = tuple(ids[t - n + 1: t + 1])
        if key in last:
            out[t] = t - last[key]
        last[key] = t
    return out


def bucketize_delta(delta: np.ndarray) -> np.ndarray:
    """Δ → bucket id: 0 (1–4), 1 (5–8), 2 (9–16), 3 ("none": Δ > NONE_MIN
    or no previous occurrence), -1 (guard band 17–NONE_MIN, excluded)."""
    b = np.full(delta.shape, -1, dtype=np.int8)
    lo = 1
    for i, hi in enumerate(BUCKET_EDGES):
        b[(delta >= lo) & (delta <= hi)] = i
        lo = hi + 1
    b[(delta > NONE_MIN) | (delta < 0)] = 3
    return b


def shuffled_doc_null(ids, n: int, rng: np.random.Generator) -> np.ndarray:
    """Δ recomputed after a within-doc token shuffle: the frequency-only
    null (what Δ structure survives with order destroyed)."""
    perm = rng.permutation(np.asarray(ids))
    return delta_prev_ngram(perm, n)


# ── balanced manifests ──────────────────────────────────────────────────

def doc_split(n_docs: int, frac_test: float = 0.2, seed: int = 0) -> np.ndarray:
    """Per-doc split flags (0 train / 1 test), seeded."""
    rng = np.random.default_rng(seed)
    n_test = int(round(n_docs * frac_test))
    flags = np.zeros(n_docs, dtype=np.int8)
    flags[rng.choice(n_docs, size=n_test, replace=False)] = 1
    return flags


def balanced_manifest(class_of_row: np.ndarray, doc_of_row: np.ndarray,
                      pos_of_row: np.ndarray, cap: int = 20_000,
                      min_pos: int = MIN_MANIFEST_POS, seed: int = 0):
    """Subsample rows so every class (label >= 0) has equal count
    (min class size, capped). Returns (doc, pos, cls) int32/int32/int8
    arrays, deterministic under the seed."""
    rng = np.random.default_rng(seed)
    ok = (class_of_row >= 0) & (pos_of_row >= min_pos)
    classes = np.unique(class_of_row[ok])
    per = [np.flatnonzero(ok & (class_of_row == c)) for c in classes]
    n = min(cap, min(len(p) for p in per)) if per else 0
    take = []
    for p in per:
        take.append(rng.choice(p, size=n, replace=False) if len(p) > n else p)
    idx = np.sort(np.concatenate(take)) if take else np.array([], dtype=int)
    return (doc_of_row[idx].astype(np.int32), pos_of_row[idx].astype(np.int32),
            class_of_row[idx].astype(np.int8))


# ── backtracking intensity (the fitted mirror, applied causally) ────────

def lambda_for_sentences(b, intercept: float, coef_pos: float, kernel_w):
    """Per-sentence intensity from the committed backtracking mirror:
    λ̂_i = σ(intercept + coef_pos·(i/L) + Σ_l w_l · b_{i-l}), with zeros
    for pre-history (the mirror's generation convention). Also returns
    the position-term-free variant λ̂_hist (self-excitation component
    only). ``b`` is the trace's 0/1 backtracking sentence indicator; the
    current sentence's own label is NOT an input to λ̂_i."""
    b = np.asarray(b, dtype=float)
    L = b.size
    K = len(kernel_w)
    hist = np.zeros(L)
    for lag, w in enumerate(kernel_w, start=1):
        if lag < L:
            hist[lag:] += w * b[:-lag]
    pos = np.arange(L) / max(L, 1)
    lam = 1.0 / (1.0 + np.exp(-(intercept + coef_pos * pos + hist)))
    lam_hist = 1.0 / (1.0 + np.exp(-(intercept + hist)))
    return lam.astype(np.float32), lam_hist.astype(np.float32)


def tercile_bins(values: np.ndarray):
    """(edges, bin ids ∈ {0,1,2}) over finite values; NaN → -1."""
    v = values[np.isfinite(values)]
    edges = np.quantile(v, [1 / 3, 2 / 3])
    out = np.full(values.shape, -1, dtype=np.int8)
    m = np.isfinite(values)
    out[m] = np.digitize(values[m], edges).astype(np.int8)
    return edges, out


# ── categorical run structure (proof operations) ────────────────────────

def run_features(labels):
    """Per-sentence run features from a categorical label list that may
    contain None (unlabeled — breaks runs, gets -1 everywhere).

    Returns (op int8, time_in_run int32, is_run_start int8): time_in_run
    is the 0-based index within the current constant-label run;
    is_run_start = 1 iff labeled and (first of doc / after unlabeled /
    label changed)."""
    L = len(labels)
    op = np.full(L, -1, dtype=np.int8)
    tir = np.full(L, -1, dtype=np.int32)
    start = np.full(L, -1, dtype=np.int8)
    prev = None
    run = 0
    for i, lab in enumerate(labels):
        if lab is None:
            prev = None
            continue
        op[i] = lab
        if prev is not None and lab == prev:
            run += 1
            start[i] = 0
        else:
            run = 0
            start[i] = 1
        tir[i] = run
        prev = lab
    return op, tir, start


# ── trailing slope (confidence trend) ───────────────────────────────────

def trailing_slope(vals, k: int) -> np.ndarray:
    """Least-squares slope of vals over the trailing k entries (positions
    i-k+1..i), NaN when the window has any unlabeled entry (None/-1) or
    doesn't fit. vals in {0,1,2,...} or None."""
    L = len(vals)
    x = np.arange(k, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())
    v = np.array([np.nan if (a is None or a < 0) else float(a) for a in vals])
    out = np.full(L, np.nan, dtype=np.float32)
    for i in range(k - 1, L):
        w = v[i - k + 1: i + 1]
        if np.isnan(w).any():
            continue
        out[i] = float((x * (w - w.mean())).sum() / denom)
    return out


# ── sentence↔token bridge ───────────────────────────────────────────────

def sentence_index_per_token(offsets, spans):
    """Map each token (char offset pair) to the sentence span containing
    its char midpoint. spans: sorted (char_start, char_end) pairs.

    Returns (sent_idx int32, in_span bool): tokens in the gap between
    spans inherit the PREVIOUS sentence's index with in_span=False;
    tokens before the first span get sent_idx 0, in_span False."""
    starts = np.array([a for a, _ in spans], dtype=float)
    ends = np.array([b for _, b in spans], dtype=float)
    mids = np.array([(a + b) / 2 for a, b in offsets], dtype=float)
    idx = np.searchsorted(starts, mids, side="right") - 1
    below = idx < 0
    idx = np.clip(idx, 0, len(spans) - 1)
    in_span = (mids >= starts[idx]) & (mids < ends[idx]) & ~below
    return idx.astype(np.int32), in_span
