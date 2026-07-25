"""Pure label logic for the sentence-length recency ladder
(CANDIDATES.md B8 `slen`) — no tokenizers, no I/O. Covered by
``tests/test_slen_labels.py``.

One exact value stream — x_i = ln(word count of sentence i) — carries
three faces that differ ONLY in temporal weighting, all computed from
PREVIOUS sentences (the current sentence never contributes to its own
label — the anchor lesson; warm-up is unified at ``SUPPORT_S`` so the
three faces share eligible rows):

- ``lat``  — the previous sentence's x (a latch: pure recency /
  distance-to-anchor structure; order-free aggregation of a window
  cannot represent it);
- ``lev``  — kernel-weighted trailing mean (HL 2 / support 8
  sentences, the punctint kernel; the P6 level face);
- ``disp`` — kernel-weighted trailing std (the program's first
  second-moment face; see ``kernel_ess`` for how few sentences it
  effectively sees).
"""

from __future__ import annotations

import numpy as np

from .novelty_lib import kernel_weights

HALF_LIFE_S = 2         # kernel half-life in SENTENCES (punctint kernel)
SUPPORT_S = 8           # kernel support in sentences; unified warm-up


def sent_log_lengths(sents) -> np.ndarray:
    """x_i = ln(word count of sentence i) — whitespace words, floor 1 —
    tokenizer-independent by construction (labels differ across
    tokenizers only through the sentence->token bridge)."""
    return np.array([np.log(max(1, len(s.split()))) for s in sents],
                    dtype=np.float64)


def trailing_latch(x, support: int = SUPPORT_S) -> np.ndarray:
    """lat[i] = x[i-1]; NaN for i < support (warm-up unified with the
    kernel faces so all three faces share eligible rows)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if x.size > support:
        out[support:] = x[support - 1: -1]
    return out


def _weighted_trailing(x, half_life: int, support: int) -> np.ndarray:
    """``novelty_lib.trailing_rate`` at float64 — identical indexing
    (out[t] = sum_l w_l * x[t-l], lags 1..support, NaN warm-up while
    the kernel does not fully fit), without the float32 round-off that
    ``disp``'s m2 - m1^2 cancellation would amplify."""
    w = kernel_weights(half_life, support)
    x = np.asarray(x, dtype=np.float64)
    conv = np.convolve(x, w)
    out = np.full(len(x), np.nan)
    if len(x) > support:
        out[support:] = conv[support - 1: len(x) - 1]
    return out


def trailing_level(x, half_life: int = HALF_LIFE_S,
                   support: int = SUPPORT_S) -> np.ndarray:
    """Kernel-weighted mean of the previous ``support`` sentences
    (normalized weights — for constant x the level equals x); NaN
    warm-up while the kernel does not fully fit."""
    return _weighted_trailing(x, half_life, support).astype(np.float32)


def trailing_disp(x, half_life: int = HALF_LIFE_S,
                  support: int = SUPPORT_S) -> np.ndarray:
    """Kernel-weighted std of the previous ``support`` sentences:
    sqrt(E_w[x^2] - E_w[x]^2) at float64, variance clipped at 0
    before the root; NaN warm-up as for the level."""
    x = np.asarray(x, dtype=np.float64)
    m1 = _weighted_trailing(x, half_life, support)
    m2 = _weighted_trailing(x * x, half_life, support)
    var = np.clip(m2 - m1 ** 2, 0.0, None)
    out = np.sqrt(var).astype(np.float32)
    out[~np.isfinite(m1)] = np.nan
    return out


def kernel_ess(half_life: int = HALF_LIFE_S,
               support: int = SUPPORT_S) -> float:
    """Kish effective sample size of the normalized kernel weights —
    the disclosure number for how few sentences ``disp`` really
    sees (≈ 5.1 of 8 at the punctint kernel)."""
    w = kernel_weights(half_life, support)
    return float((w.sum() ** 2) / (w * w).sum())
