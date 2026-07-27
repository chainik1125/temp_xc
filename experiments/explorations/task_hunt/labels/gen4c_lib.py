"""Pure label logic for the gen-4 CORPUS SCOUT faces (mac-c lane,
beat review ~12:15 item 3) — no tokenizers, no I/O. Covered by
``tests/test_gen4c_labels.py``.

The scout transplants the return family (hunt4's ``tret``/``tretd``,
imported verbatim) onto two substrates no hunt face has touched —
WikiText-103 encyclopedic narrative and permissively-licensed Python
code — and adds two substrate-native faces:

- ``sage`` (wikitext) — SECTION AGE: log2(1 + tokens since the last
  section-header token). An intensity/recency face (λ̂ family, not a
  novelty face): the marker itself is surface-VISIBLE when it falls
  inside the window, so the claim zone is wherever the per-T floor
  (exact-age-if-in-window, censored otherwise) leaves room. Disclosed
  up front; the floor is the instrument.
- ``drev`` (pycode) — DORMANT-IDENTIFIER REVIVAL: kernel trailing
  rate (support 64, HL 16 — cnov's kernel) of identifier occurrences
  whose previous occurrence of the SAME identifier is > 64 tokens
  back. The identifier chain runs over identifier positions only
  (keywords/strings/comments excluded), so the face reads
  definition→use reach-back, not string repetition. Out-of-window by
  construction at every ladder T ≤ 64 (tret's guarantee, restricted
  to the name graph). Its within-corpus anti-dup partner is raw-token
  ``tret`` — the 0.8 bar decides which earns any screen.
"""

from __future__ import annotations

import numpy as np

from .hunt3_lib import (  # noqa: F401  (house machinery, reused verbatim)
    CNOV_HL,
    FLOOR_TS,
    SUPPORT_TOK,
    filter_rate,
    last_occurrence,
    window_novelty_events,
)
from .hunt4_lib import RET_GAP, long_return_events  # noqa: F401


# ── masked-chain occurrences (identifier graph) ─────────────────────


def last_occurrence_masked(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per token: index of the previous occurrence of the same type
    among MASKED positions only (-1 if none, and -1 at unmasked
    positions). With an all-ones mask this is ``last_occurrence``."""
    n = len(ids)
    out = np.full(n, -1, dtype=np.int64)
    seen: dict[int, int] = {}
    for i in range(n):
        if not mask[i]:
            continue
        v = int(ids[i])
        if v in seen:
            out[i] = seen[v]
        seen[v] = i
    return out


def masked_return_events(last_occ_m: np.ndarray, mask: np.ndarray,
                         gap: int = RET_GAP) -> np.ndarray:
    """1 at masked positions whose previous masked same-type
    occurrence is more than `gap` tokens back (drev's event)."""
    idx = np.arange(len(last_occ_m))
    return (np.asarray(mask, dtype=bool) & (last_occ_m >= 0)
            & (idx - last_occ_m > gap)).astype(np.int8)


def masked_window_novelty(last_occ_m: np.ndarray, mask: np.ndarray,
                          T: int) -> np.ndarray:
    """drev's window-computable ingredient at T: masked positions
    whose previous masked occurrence is OUTSIDE the last-T view
    (novel-in-window on the identifier chain)."""
    idx = np.arange(len(last_occ_m))
    return (np.asarray(mask, dtype=bool)
            & ((last_occ_m < 0) | (idx - last_occ_m > T))).astype(np.int8)


def drev_floor(last_occ_m: np.ndarray, mask: np.ndarray, T: int,
               hl: float = CNOV_HL) -> np.ndarray:
    """drev's visible floor at T: trailing kernel rate (truncated
    support min(T, 64)) of identifier window-novelty events."""
    ev = masked_window_novelty(last_occ_m, mask, T)
    return filter_rate(ev, min(T, SUPPORT_TOK), hl)


# ── section age (wikitext) ──────────────────────────────────────────


def section_age(is_boundary: np.ndarray) -> np.ndarray:
    """Per token: tokens since the most recent boundary token at or
    before it (0 on the boundary itself); NaN before any boundary."""
    b = np.asarray(is_boundary, dtype=bool)
    n = len(b)
    idx = np.arange(n)
    last = np.where(b, idx, -1)
    last = np.maximum.accumulate(last)
    out = (idx - last).astype(np.float32)
    out[last < 0] = np.nan
    return out


def sage_face(is_boundary: np.ndarray,
              support: int = SUPPORT_TOK) -> np.ndarray:
    """log2(1 + section age); NaN below full support (house
    convention: every labeled row carries identical support)."""
    age = section_age(is_boundary)
    out = np.log2(1.0 + age, dtype=np.float32)
    out[:min(support, len(out))] = np.nan
    return out


def sage_floor(is_boundary: np.ndarray, T: int) -> np.ndarray:
    """sage's window-computable cheat at T: the exact age when a
    boundary is visible in the last-T view, censored at T + 1 when it
    is not ("older than my window"), log2-scaled like the face."""
    age = section_age(is_boundary)
    cens = np.minimum(age, float(T + 1))
    return np.log2(1.0 + cens, dtype=np.float32)
