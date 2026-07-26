"""Pure label logic for the day-2 dialogue-native faces (W2 bundle,
`briefings/day2-dialogue-mac-a.md`) — no tokenizers, no I/O. Covered by
``tests/test_diafaces_labels.py``.

Two faces on the dialevel substrate (same token stream, same caches):

- ``ttrend`` — kernel-weighted trailing SLOPE of turn lengths over the
  previous ``SUPPORT_TURNS`` turns (current turn never in its own
  label; NaN while fewer than ``SUPPORT_TURNS`` previous turns exist).
  The Δ-face of what dialevel screened as LEVEL: a slope needs at
  least two levels at different distances, so it is the
  regime-3-shaped candidate on the one substrate with measured
  order-carriage (R11).
- ``dqgap`` — turns since the most recent PREVIOUS turn containing a
  question mark (current turn excluded, the family rule; NaN while no
  previous question turn exists). Distance-to-anchor in its purest
  form; on DailyDialog "?" is DENSE (the per-turn rate ships in the
  stats and the card), unlike the fineweb qgap that parked P7.
"""

from __future__ import annotations

import numpy as np

SUPPORT_TURNS = 5      # trailing support, matched to dialevel's tlevel
KERNEL_HL = 2.0        # half-life (in turns) of the slope kernel


def kernel_weights(support: int = SUPPORT_TURNS,
                   hl: float = KERNEL_HL) -> np.ndarray:
    """Recency weights for the previous `support` turns, oldest first:
    the most recent previous turn (offset 1) has weight 1, each
    additional turn of distance halves the weight every `hl` turns."""
    offsets = np.arange(support, 0, -1, dtype=float)   # oldest -> newest
    return 0.5 ** ((offsets - 1.0) / hl)


def wls_slope(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> float:
    """Weighted least-squares slope of y on x."""
    w = np.asarray(w, float)
    xbar = float((w * x).sum() / w.sum())
    ybar = float((w * y).sum() / w.sum())
    den = float((w * (x - xbar) ** 2).sum())
    return float((w * (x - xbar) * (y - ybar)).sum() / den)


def trailing_turn_slope(turn_sizes, support: int = SUPPORT_TURNS,
                        hl: float = KERNEL_HL) -> np.ndarray:
    """Per-turn trend: kernel-weighted OLS slope (tokens/turn) of the
    PREVIOUS `support` turn sizes against their turn offset
    [-support .. -1], so positive = turns lengthening toward the
    present. Current turn never in its own label; NaN while fewer than
    `support` previous turns exist."""
    sizes = np.asarray(turn_sizes, dtype=float)
    out = np.full(len(sizes), np.nan, dtype=np.float32)
    x = np.arange(-support, 0, dtype=float)
    w = kernel_weights(support, hl)
    for i in range(support, len(sizes)):
        out[i] = wls_slope(sizes[i - support: i], x, w)
    return out


def turns_since_question(has_q) -> np.ndarray:
    """Per-turn gap to the most recent PREVIOUS turn with a question
    (>= 1 when defined; NaN while no previous question turn exists)."""
    has_q = np.asarray(has_q, dtype=bool)
    out = np.full(len(has_q), np.nan, dtype=np.float32)
    last = -1
    for i in range(len(has_q)):
        if last >= 0:
            out[i] = i - last
        if has_q[i]:
            last = i
    return out


def balanced_int_edges(vals: np.ndarray) -> tuple[int, int]:
    """Deterministic 3-class integer edges (a, b) for a small-integer
    face: classes {v <= a}, {a < v <= b}, {v > b}, chosen to maximize
    the minimum class fraction on the finite rows given. Quantile
    terciles are unusable here — ties at gap = 1 can empty a class —
    so the split is an exhaustive search over integer thresholds,
    fully determined by the (train) distribution, no seed."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)].astype(int)
    lo, hi = int(v.min()), int(v.max())
    best, best_min = (lo, lo + 1), -1.0
    for a in range(lo, hi):
        for b in range(a + 1, hi + 1):
            fr = (float((v <= a).mean()), float(((v > a) & (v <= b)).mean()),
                  float((v > b).mean()))
            if min(fr) > best_min:
                best_min, best = min(fr), (a, b)
    return best


def int_edge_bins(vals: np.ndarray, a: int, b: int) -> np.ndarray:
    """int8 bins under `balanced_int_edges` edges; -1 for NaN."""
    out = np.full(np.asarray(vals).shape, -1, dtype=np.int8)
    m = np.isfinite(vals)
    out[m & (vals <= a)] = 0
    out[m & (vals > a) & (vals <= b)] = 1
    out[m & (vals > b)] = 2
    return out
