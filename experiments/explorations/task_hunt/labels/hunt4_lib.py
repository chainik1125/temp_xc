"""Pure label logic for the FOURTH-generation hunt faces (gen-4
directive, LOG 2026-07-27 ~11:55 / commit 59ad15f38) — no tokenizers,
no I/O. Covered by ``tests/test_hunt4_labels.py``.

Same recipe that produced cnov: offset-weighted trailing functionals
of sparse per-token-SILENT events on the order-carried dialogue
substrate, with out-of-window / cross-distance structure preferred.
All kernels and filters are hunt3's verbatim (imported, not copied).

- ``xnov`` — cross-speaker ADOPTION rate: kernel trailing rate
  (support 64 tok, HL 16) of tokens whose type was seen earlier in
  the conversation but NEVER by the current speaker (in a strictly
  alternating 2-speaker corpus every prior occurrence is then the
  other speaker's) — lexical entrainment intensity. Speaker-resolved
  memory is required: a T-window can see recent other-speaker use,
  but "never by the current speaker" is unbounded-history. The
  window-computable cheat is shipped as the per-T visible floor
  (other-in-window ∧ same-not-in-window, PLUS the window-novelty
  rate as a second feature — strengthening the kill instrument is
  the conservative direction).
- ``tret`` — TOPIC-RETURN intensity: kernel trailing rate of
  long-return events, gap = idx − last_occ > 64 (= SUPPORT_TOK).
  Out-of-window BY CONSTRUCTION for every floor T ≤ 64: the prior
  occurrence sits outside the window, so in-window evidence cannot
  distinguish a long return from a conversation-first token. tret
  and cnov PARTITION the out-of-window novelty guarantee (resumed
  vs genuinely new); the shared visible floor is window-novelty
  (hunt3's ``floor_rate``), and label-side overlap vs cnov decides
  whether the two faces are distinct enough to earn a screen.
- ``sdom`` — SIGNED speaker novelty-dominance: D = K_cur − K_oth,
  where K_s is the per-speaker kernel novelty rate over the trailing
  64 tokens (ratio of two linear filters; kernel normalization
  cancels). "The current speaker has been introducing more than the
  listener lately." Sign is attached to the CURRENT speaker, so the
  face is a cross-distance comparison of two trailing states. Rows
  where either speaker holds < ``SDOM_MIN_MASS`` of the kernel mass
  are NaN (mass guard).
"""

from __future__ import annotations

import numpy as np

from .hunt3_lib import (  # noqa: F401  (house machinery, reused verbatim)
    CNOV_HL,
    FLOOR_TS,
    SUPPORT_TOK,
    filter_rate,
    last_occurrence,
    tok_kernel,
    window_novelty_events,
)

RET_GAP = SUPPORT_TOK      # long-return threshold: gap > 64 tokens
SDOM_MIN_MASS = 0.15       # min kernel mass per speaker for sdom rows
TRETD_MIN_RATE = 0.02      # min trailing return mass for tretd rows


# ── speaker-resolved events ─────────────────────────────────────────


def last_occurrence_by_speaker(
        ids: np.ndarray, spk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per token: index of the previous occurrence of the same type by
    the SAME speaker, and by the OTHER speaker (-1 if none). `ids` and
    `spk` (0/1) are one dialogue's slices."""
    n = len(ids)
    same = np.full(n, -1, dtype=np.int64)
    oth = np.full(n, -1, dtype=np.int64)
    seen: dict[tuple[int, int], int] = {}
    for i in range(n):
        v, s = int(ids[i]), int(spk[i])
        same[i] = seen.get((v, s), -1)
        oth[i] = seen.get((v, 1 - s), -1)
        seen[(v, s)] = i
    return same, oth


def adoption_events(last_same: np.ndarray, last_oth: np.ndarray) -> np.ndarray:
    """1 where the type was seen before in the conversation but never
    by the current speaker (= first personal use of another speaker's
    word; with 2 speakers, seen-before ∧ not-by-me ⇒ by-the-other)."""
    return ((last_oth >= 0) & (last_same < 0)).astype(np.int8)


def long_return_events(last_occ: np.ndarray,
                       gap: int = RET_GAP) -> np.ndarray:
    """1 where the type occurred before but more than `gap` tokens ago
    — for gap ≥ every floor T, the prior occurrence is out-of-window
    by construction (a window sees only 'novel in window')."""
    idx = np.arange(len(last_occ))
    return ((last_occ >= 0) & (idx - last_occ > gap)).astype(np.int8)


def cross_return_events(last_same: np.ndarray, last_oth: np.ndarray,
                        gap: int = RET_GAP) -> np.ndarray:
    """1 where the token is a LONG return (gap > `gap` tokens) AND the
    most recent prior occurrence was the OTHER speaker's — resuming
    material the other party left long ago. Doubly silent: the window
    can see neither that it is a return nor whose it was."""
    last_occ = np.maximum(last_same, last_oth)
    idx = np.arange(len(last_same))
    long_ret = (last_occ >= 0) & (idx - last_occ > gap)
    return (long_ret & (last_oth > last_same)).astype(np.int8)


def return_depth_face(last_occ: np.ndarray,
                      support: int = SUPPORT_TOK, hl: float = CNOV_HL,
                      gap: int = RET_GAP,
                      min_rate: float = TRETD_MIN_RATE) -> np.ndarray:
    """`tretd` — kernel-weighted mean log2(gap) over trailing
    long-return events ("how DEEP into history the dialogue is
    currently reaching"; a cross-distance VALUE, not a rate — the gap
    of an out-of-window return is uncomputable from any window).
    NaN where trailing return mass < `min_rate` (or below support).
    Ratio of two linear filters; kernel normalization cancels."""
    ev = long_return_events(last_occ, gap).astype(float)
    idx = np.arange(len(last_occ), dtype=float)
    logg = np.where(ev > 0, np.log2(np.maximum(idx - last_occ, 2.0)), 0.0)
    num = filter_rate(ev * logg, support, hl)
    den = filter_rate(ev, support, hl)
    out = (num / np.maximum(den, 1e-9)).astype(np.float32)
    out[~(den >= min_rate)] = np.nan
    return out


def xnov_floor_events(last_same: np.ndarray, last_oth: np.ndarray,
                      T: int) -> np.ndarray:
    """The window-computable adoption cheat at window length T:
    other-speaker occurrence VISIBLE in the last T tokens AND no
    same-speaker occurrence in the last T tokens. False-positives on
    repeats whose same-speaker use is out-of-window; misses adoptions
    whose cross-occurrence is out-of-window — exactly the gap the
    activation probe must beat."""
    idx = np.arange(len(last_same))
    oth_in = (last_oth >= 0) & (idx - last_oth <= T)
    same_out = (last_same < 0) | (idx - last_same > T)
    return (oth_in & same_out).astype(np.int8)


# ── per-speaker kernel rates & the signed dominance face ───────────


def speaker_rates(events: np.ndarray, spk: np.ndarray, support: int,
                  hl: float = CNOV_HL) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
    """Kernel trailing event rate PER SPEAKER over the previous
    `support` tokens: K_s = Σ w·e·1_s / Σ w·1_s (the w.sum()
    normalization of ``filter_rate`` cancels in the ratio). Returns
    (K_spk0, K_spk1, min_mass) at each position; NaN below full
    support. `min_mass` is the smaller speaker's kernel mass share,
    for the caller's mass guard."""
    ev = np.asarray(events, dtype=float)
    s0 = (np.asarray(spk) == 0).astype(float)
    s1 = 1.0 - s0
    num0 = filter_rate(ev * s0, support, hl)
    num1 = filter_rate(ev * s1, support, hl)
    den0 = filter_rate(s0, support, hl)
    den1 = filter_rate(s1, support, hl)
    k0 = num0 / np.maximum(den0, 1e-9)
    k1 = num1 / np.maximum(den1, 1e-9)
    min_mass = np.minimum(den0, den1)
    return k0, k1, min_mass


def sdom_face(novel: np.ndarray, spk: np.ndarray,
              support: int = SUPPORT_TOK, hl: float = CNOV_HL,
              min_mass: float = SDOM_MIN_MASS) -> np.ndarray:
    """Signed dominance D = K_current_speaker − K_other_speaker of
    conversation-novelty rates; NaN where either speaker holds less
    than `min_mass` of the kernel mass (or below full support)."""
    k0, k1, mm = speaker_rates(novel, spk, support, hl)
    spk = np.asarray(spk)
    d = np.where(spk == 0, k0 - k1, k1 - k0).astype(np.float32)
    d[~(mm >= min_mass)] = np.nan          # also catches NaN min_mass
    return d


def sdom_floor(last_occ: np.ndarray, spk: np.ndarray, T: int,
               hl: float = CNOV_HL,
               min_mass: float = SDOM_MIN_MASS) -> tuple[np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray]:
    """sdom's window-computable cheat at T: the SAME signed-dominance
    functional computed from first-in-WINDOW novelty over truncated
    support min(T, 64). Returns (D_floor, K_cur_floor, K_oth_floor) —
    the floor probe gets all three (conservative direction)."""
    ev = window_novelty_events(last_occ, T)
    sup = min(T, SUPPORT_TOK)
    k0, k1, mm = speaker_rates(ev, spk, sup, hl)
    spk = np.asarray(spk)
    d = np.where(spk == 0, k0 - k1, k1 - k0).astype(np.float32)
    kc = np.where(spk == 0, k0, k1).astype(np.float32)
    ko = np.where(spk == 0, k1, k0).astype(np.float32)
    bad = ~(mm >= min_mass)
    for a in (d, kc, ko):
        a[bad] = np.nan
    return d, kc, ko
