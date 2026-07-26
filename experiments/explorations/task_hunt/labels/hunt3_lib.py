"""Pure label logic for the third-generation hunt faces (overnight
allocation `briefings/actmix-overnight.md` § 1) — no tokenizers, no
I/O. Covered by ``tests/test_hunt3_labels.py``.

Dialogue faces on the REUSED dialevel substrate (same token stream,
same caches — zero new forward passes):

- ``cnov`` — kernel-weighted trailing rate of first-in-CONVERSATION
  token-type events over the previous ``SUPPORT_TOK`` tokens (current
  token never in its own label; defined only at FULL support,
  pos ≥ SUPPORT_TOK — a stated deviation that makes every labeled row
  carry identical kernel support). The txcwin out-of-window
  definition transplanted to the order-carried substrate: whether an
  earlier occurrence exists BEFORE the window start is invisible to
  any T-window by construction. The window-computable cheat is
  first-in-WINDOW novelty — shipped as the per-T visible floor.
- ``nvtrend`` — the Δ-face: kernel-WLS slope of PER-TURN novelty
  rates over the previous ``SUPPORT_TURNS`` turns (exact ttrend
  machinery on a new per-turn series; current turn never in its own
  label). Regime-3-shaped; far less unigram-readable than the level.
- ``tempo`` — trailing turn-TEMPO trend: the ttrend kernel-WLS slope
  applied to 1/turn_size. On a strict-alternation corpus this is
  expected to collapse into (anti-)ttrend; the label-side overlap
  stat decides BEFORE any GPU is spent (the briefing's
  "improve/replace" clause exercised as a documented pre-screen).
- ``qres`` — question→resolution latency per turn: turns since the
  most recent OPEN previous question, resolved by the first ?-free
  turn. Built ONLY for the design pre-measure (degeneracy +
  ?-anchor visibility); any screen is gated on it per the briefing.

Fast paths: the per-T floors are linear filters over the
``last_occurrence`` array; ``filter_rate``/``filter_slope`` are the
vectorized twins of the reference loops and are cross-checked against
them in the tests.
"""

from __future__ import annotations

import numpy as np

from .diafaces_lib import (  # noqa: F401  (re-used house machinery)
    SUPPORT_TURNS,
    kernel_weights,
    trailing_turn_slope,
    wls_slope,
)

SUPPORT_TOK = 64       # trailing token support for cnov (= min defined pos)
CNOV_HL = 16.0         # half-life (tokens) of the cnov kernel
FLOOR_TS = (4, 8, 16, 32, 64)


# ── events ──────────────────────────────────────────────────────────


def last_occurrence(ids: np.ndarray) -> np.ndarray:
    """Per token: index of the PREVIOUS occurrence of the same type in
    this dialogue slice, or -1 if none (first occurrence)."""
    out = np.full(len(ids), -1, dtype=np.int64)
    seen: dict[int, int] = {}
    for i, v in enumerate(ids):
        v = int(v)
        if v in seen:
            out[i] = seen[v]
        seen[v] = i
    return out


def first_in_doc(ids: np.ndarray) -> np.ndarray:
    """1 where this token type has not occurred earlier in the SAME
    dialogue (`ids` is one dialogue's token slice)."""
    return (last_occurrence(ids) < 0).astype(np.int8)


# ── kernels & filters ───────────────────────────────────────────────


def tok_kernel(support: int, hl: float = CNOV_HL) -> np.ndarray:
    """Weights by DISTANCE d = 1..support back from the current token
    (index 0 ↔ d=1, most recent previous token, weight 1)."""
    d = np.arange(1, support + 1, dtype=float)
    return 0.5 ** ((d - 1.0) / hl)


def filter_rate(events: np.ndarray, support: int,
                hl: float = CNOV_HL) -> np.ndarray:
    """Kernel-weighted trailing rate of a 0/1 series, FULL support
    only (NaN for pos < support); current position excluded."""
    ev = np.asarray(events, dtype=float)
    n = len(ev)
    w = tok_kernel(support, hl)
    out = np.full(n, np.nan, dtype=np.float32)
    if n <= support:
        return out
    acc = np.zeros(n - support)
    for d in range(1, support + 1):                     # i - d, i >= support
        acc += w[d - 1] * ev[support - d: n - d]
    out[support:] = (acc / w.sum()).astype(np.float32)
    return out


def filter_slope(events: np.ndarray, support: int,
                 hl: float = CNOV_HL) -> np.ndarray:
    """Kernel-WLS slope of a 0/1 series against token offset
    x = -d (positive slope = events rising toward the present); FULL
    support only. Reduces to the linear filter
    slope(i) = Σ_d a_d · e_{i-d}, a_d = w_d (x_d − x̄_w) / Σ w (x−x̄_w)²."""
    ev = np.asarray(events, dtype=float)
    n = len(ev)
    w = tok_kernel(support, hl)
    x = -np.arange(1, support + 1, dtype=float)
    xbar = float((w * x).sum() / w.sum())
    den = float((w * (x - xbar) ** 2).sum())
    a = w * (x - xbar) / den
    out = np.full(n, np.nan, dtype=np.float32)
    if n <= support:
        return out
    acc = np.zeros(n - support)
    for d in range(1, support + 1):
        acc += a[d - 1] * ev[support - d: n - d]
    out[support:] = acc.astype(np.float32)
    return out


# ── reference loops (tests cross-check the filters against these) ──


def trailing_event_rate_ref(events: np.ndarray, support: int,
                            hl: float = CNOV_HL) -> np.ndarray:
    ev = np.asarray(events, dtype=float)
    n = len(ev)
    w = tok_kernel(support, hl)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(support, n):
        seg = ev[i - support: i][::-1]                  # d = 1..support
        out[i] = float((w * seg).sum() / w.sum())
    return out


# ── per-T visible floors (window-computable cheats) ────────────────


def window_novelty_events(last_occ: np.ndarray, T: int) -> np.ndarray:
    """1 where the previous same-type occurrence is OUTSIDE the last-T
    view at the token's own position — i.e. first-in-window novelty
    for a window ending at each token (the ingredient a length-T
    window can compute; conversation novelty additionally requires
    last_occ = -1, which the window cannot verify)."""
    idx = np.arange(len(last_occ))
    return ((last_occ < 0) | (idx - last_occ > T)).astype(np.int8)


def floor_rate(last_occ: np.ndarray, T: int,
               hl: float = CNOV_HL) -> np.ndarray:
    """cnov's visible floor at window length T: trailing kernel rate
    (support = min(T, SUPPORT_TOK)) of FIRST-IN-WINDOW events, where
    "window" is the last T tokens seen from each summed position —
    conservatively approximated per-event at the event's own position
    (exact for T ≥ support since every summed event lies inside the
    current window)."""
    ev = window_novelty_events(last_occ, T)
    return filter_rate(ev, min(T, SUPPORT_TOK), hl)


def floor_slope(last_occ: np.ndarray, T: int,
                hl: float = CNOV_HL) -> np.ndarray:
    """nvtrend's visible floor at T: kernel-WLS token-slope of
    first-in-window events over the same truncated support."""
    ev = window_novelty_events(last_occ, T)
    return filter_slope(ev, min(T, SUPPORT_TOK), hl)


# ── turn-level series ───────────────────────────────────────────────


def turn_novelty_rates(novel: np.ndarray, turn_idx: np.ndarray) -> np.ndarray:
    """Mean first-in-conversation rate per turn (one value per turn)."""
    n_turns = int(turn_idx.max()) + 1
    s = np.zeros(n_turns)
    c = np.zeros(n_turns)
    np.add.at(s, turn_idx, np.asarray(novel, dtype=float))
    np.add.at(c, turn_idx, 1.0)
    return s / np.maximum(c, 1.0)


def qres_latency(has_q: np.ndarray) -> np.ndarray:
    """Per-turn age of the most recent OPEN previous question: turns
    since that question turn, attached from its successor up to and
    including the resolving (?-free) turn; NaN when no question is
    open. The resolving turn closes the question."""
    has_q = np.asarray(has_q, dtype=bool)
    n = len(has_q)
    out = np.full(n, np.nan, dtype=np.float32)
    q_at = -1
    for i in range(n):
        if q_at >= 0:
            out[i] = i - q_at
            if not has_q[i]:
                q_at = -1          # resolved at i; question closed
        if has_q[i]:
            q_at = i
    return out
