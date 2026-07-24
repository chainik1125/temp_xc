"""Pure label logic for the vocabulary-novelty candidate (no tokenizers,
no I/O) — CANDIDATES.md B2. Covered by ``tests/test_novelty_labels.py``.

The primary label is the kernel-smoothed trailing novelty rate over
PREVIOUS tokens only (the current token never contributes to its own
label — the anchor lesson applied at token level), position-detrended
because first-occurrence rates decay mechanically along a document
(Heaps' law): the raw rate would otherwise be partly a position label.
"""

from __future__ import annotations

import numpy as np

HALF_LIFE = 16          # kernel half-life in tokens (lag weights 2^{-l/h})
SUPPORT = 64            # kernel truncation; label undefined for pos < SUPPORT
POS_BIN_MIN = 64        # first position bin starts here (log2 bins above)
N_POS_BINS = 6          # 64–127, 128–255, 256–511, 512–1023, 1024–2047, 2048+


def novelty_bits(ids) -> np.ndarray:
    """1 at the first in-document occurrence of the token TYPE, else 0."""
    seen: set = set()
    out = np.zeros(len(ids), dtype=np.int8)
    for t, i in enumerate(ids):
        if i not in seen:
            out[t] = 1
            seen.add(i)
    return out


def kernel_weights(half_life: int = HALF_LIFE,
                   support: int = SUPPORT) -> np.ndarray:
    """Normalized exponential lag weights w_l for lags l = 1..support."""
    w = 2.0 ** (-np.arange(1, support + 1) / half_life)
    return w / w.sum()


def trailing_rate(bits: np.ndarray, half_life: int = HALF_LIFE,
                  support: int = SUPPORT) -> np.ndarray:
    """rate[t] = sum_l w_l * bits[t-l] (lags 1..support); NaN while the
    kernel does not fully fit (t < support). bits[t] never contributes
    to rate[t]."""
    w = kernel_weights(half_life, support)
    conv = np.convolve(bits.astype(np.float64), w)
    rate = np.full(len(bits), np.nan, dtype=np.float32)
    if len(bits) > support:
        rate[support:] = conv[support - 1: len(bits) - 1]
    return rate


def kernel_mass_within(T: int, half_life: int = HALF_LIFE,
                       support: int = SUPPORT) -> float:
    """Fraction of kernel mass a trailing window of length T can see
    (lags 1..T) — the clock bridge number for the screen ladder."""
    w = kernel_weights(half_life, support)
    return float(w[: min(T, support)].sum())


def position_bin(pos: np.ndarray) -> np.ndarray:
    """log2 position bins from POS_BIN_MIN up (clipped to N_POS_BINS);
    -1 below POS_BIN_MIN."""
    pos = np.asarray(pos)
    out = np.full(pos.shape, -1, dtype=np.int8)
    ok = pos >= POS_BIN_MIN
    b = (np.floor(np.log2(pos[ok])) - np.floor(np.log2(POS_BIN_MIN)))
    out[ok] = np.clip(b, 0, N_POS_BINS - 1).astype(np.int8)
    return out


def detrend(rate: np.ndarray, pbin: np.ndarray, train_mask: np.ndarray):
    """Subtract the train-row mean rate of each position bin (the exact,
    deterministic Heaps-trend removal). Empty-bin fallback = global
    train mean. Returns (resid float32, expected list[N_POS_BINS])."""
    fin = np.isfinite(rate) & train_mask & (pbin >= 0)
    glob = float(rate[fin].mean()) if fin.any() else 0.0
    expected = np.full(N_POS_BINS, glob)
    for b in range(N_POS_BINS):
        m = fin & (pbin == b)
        if m.any():
            expected[b] = float(rate[m].mean())
    resid = np.full(rate.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(rate) & (pbin >= 0)
    resid[ok] = rate[ok] - expected[pbin[ok]]
    return resid, [float(e) for e in expected]


def within_doc_perm(doc_off: np.ndarray, seed: int) -> np.ndarray:
    """Global index permutation that shuffles tokens WITHIN each doc
    segment (doc_off = n_docs+1 prefix offsets), seeded per doc."""
    out = np.empty(int(doc_off[-1]), dtype=np.int64)
    for d in range(len(doc_off) - 1):
        a, b = int(doc_off[d]), int(doc_off[d + 1])
        rng = np.random.default_rng(seed + d)
        out[a:b] = a + rng.permutation(b - a)
    return out


def pooled_doc_autocorr(vals: np.ndarray, doc_off: np.ndarray,
                        lag: int, min_len: int = 50) -> float:
    """Lag-`lag` autocorrelation of `vals` pooled over docs (no
    cross-doc pairs; NaNs dropped; per-doc demeaned covariances and
    variances summed unweighted, then ratioed). For a kernel-filtered
    label with truncation SUPPORT, lags > SUPPORT share no input bits,
    so any surviving autocorrelation is signal structure, not filter
    overlap."""
    num, den = [], []
    for d in range(len(doc_off) - 1):
        v = vals[int(doc_off[d]): int(doc_off[d + 1])]
        v = v[np.isfinite(v)]
        if len(v) < lag + min_len:
            continue
        x, y = v[:-lag], v[lag:]
        x = x - x.mean()
        y = y - y.mean()
        num.append(float((x * y).mean()))
        den.append(float(v.std() ** 2))
    return float(np.sum(num) / np.sum(den)) if den else float("nan")


def type_mean_scores(ids: np.ndarray, values: np.ndarray,
                     train_mask: np.ndarray) -> np.ndarray:
    """Per-row score = train-set mean of `values` for the CURRENT
    token's type (unseen types -> global train mean) — the
    current-token-identity leak estimator used by every triage here."""
    fin = train_mask & np.isfinite(values)
    n_types = int(ids.max()) + 1
    sums = np.zeros(n_types)
    cnts = np.zeros(n_types)
    np.add.at(sums, ids[fin], values[fin])
    np.add.at(cnts, ids[fin], 1)
    glob = float(values[fin].mean()) if fin.any() else 0.0
    tmean = np.where(cnts > 0, sums / np.maximum(cnts, 1), glob)
    return tmean[ids]


def tercile_auc(scores: np.ndarray, tercile: np.ndarray,
                mask: np.ndarray) -> float:
    """AUC of `scores` separating tercile 2 from tercile 0 on rows in
    `mask` (tercile 1 and unlabeled rows ignored)."""
    from .interleave_lib import rank_auc
    m = mask & (tercile >= 0) & (tercile != 1) & np.isfinite(scores)
    return rank_auc(scores[m], (tercile[m] == 2).astype(int))
