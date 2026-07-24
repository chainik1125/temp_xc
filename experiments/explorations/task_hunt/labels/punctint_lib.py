"""Pure label logic for the sentence-event intensity candidates
(CANDIDATES.md B3 list/enumeration density, B4 question rate) — no
tokenizers, no I/O. Covered by ``tests/test_punctint_labels.py``.

Both candidates are the winner family's shape (kernel-smoothed event
intensity) on sentence-level event streams over fineweb: λ̂ at sentence
i is the exponential-kernel rate over the PREVIOUS ``SUPPORT_S``
sentences (the current sentence never contributes to its own label —
the anchor lesson), matching the backtracking-λ̂ 8-sentence-lag shape.
Every token of sentence i inherits λ̂_i; tokens of EVENT sentences are
masked from probe manifests (they read the event ambiently).
"""

from __future__ import annotations

import re

import numpy as np

from .novelty_lib import kernel_weights, trailing_rate  # generic over units

HALF_LIFE_S = 2         # kernel half-life in SENTENCES
SUPPORT_S = 8           # kernel support in sentences (the winner's 8 lags)
ZERO_INFLATION_BAR = 1 / 3   # zero_split scheme fires above this

# Frozen list-marker grammar (committed before the builder runs):
# bullets (ASCII ones need a following space; unicode bullets may butt),
# `1.` / `12)` numbered items, `a.` / `b)` lettered items, `(3)` / `(iv)`
# parenthesized enumerators — all anchored at sentence start.
LIST_RE = re.compile(
    r"^\s*(?:[-*]\s|[•●▪‣◦]\s?|\d{1,2}[.)]\s|[a-z][.)]\s"
    r"|\((?:\d{1,2}|[ivxlc]+)\))")


def is_list_sentence(sent: str) -> bool:
    return bool(LIST_RE.match(sent))


def is_question_sentence(sent: str) -> bool:
    return sent.rstrip().endswith("?")


def sentence_lambda(events: np.ndarray,
                    half_life: int = HALF_LIFE_S,
                    support: int = SUPPORT_S) -> np.ndarray:
    """Per-sentence kernel intensity from previous sentences only
    (NaN while the kernel does not fully fit, i.e. sentence idx <
    support)."""
    return trailing_rate(np.asarray(events, dtype=np.int8),
                         half_life=half_life, support=support)


def kernel_mass_within_sentences(n: int, half_life: int = HALF_LIFE_S,
                                 support: int = SUPPORT_S) -> float:
    w = kernel_weights(half_life, support)
    return float(w[: min(max(int(n), 0), support)].sum())


def zero_split_bins(vals: np.ndarray, train_mask: np.ndarray):
    """3-class scheme for zero-inflated intensities. If the exact-zero
    fraction among finite train rows exceeds ZERO_INFLATION_BAR:
    class 0 = exactly 0, class 1 = (0, median of positive train
    values], class 2 = above — else plain train-edge terciles.
    Returns (scheme, edges, bins int8 with -1 for NaN)."""
    fin = np.isfinite(vals) & train_mask
    out = np.full(vals.shape, -1, dtype=np.int8)
    m = np.isfinite(vals)
    zero_frac = float((vals[fin] == 0).mean()) if fin.any() else 1.0
    if zero_frac > ZERO_INFLATION_BAR:
        pos = vals[fin & (vals > 0)]
        med = float(np.median(pos)) if pos.size else 0.0
        out[m & (vals == 0)] = 0
        out[m & (vals > 0) & (vals <= med)] = 1
        out[m & (vals > med)] = 2
        return "zero_split", [0.0, med], out
    edges = np.quantile(vals[fin], [1 / 3, 2 / 3])
    out[m] = np.digitize(vals[m], edges).astype(np.int8)
    return "terciles", [float(e) for e in edges], out


def token_labels_from_sentences(sent_vals: np.ndarray,
                                sent_idx: np.ndarray) -> np.ndarray:
    """Every token inherits its sentence's value (float faces keep
    NaN; works for int event flags too when passed as float)."""
    return np.asarray(sent_vals)[sent_idx]
