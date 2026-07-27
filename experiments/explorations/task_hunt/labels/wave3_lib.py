"""Pure label logic for the wave-3 zero-pull trio pre-measures
(SAFETY_TASK_MENU.md § 4 #1 `sycpress`, #2 `reask`, #4 `msdose`;
directive ae1ce5fb0) — no tokenizers, no I/O. Covered by
``tests/test_wave3_labels.py``.

Every constant here is FROZEN pre-count (refmark rule). The age
faces reuse ``gen4c_lib.sage_face`` / ``sage_floor`` VERBATIM over
event-token flag arrays — the menu's T2 template, already audited —
so the only new logic is (a) the reask event definition, (b) the
event-message → first-token flag mapping, (c) the msdose
construction. The binding menu bars are § 1.2
out-of-window-by-construction (reask's justification is a user turn
two messages back, ≈ 260–290 tok; sycpress markers sit on user
turns probed from assistant tokens) and § 2 clock-stated-first (the
builder measures tokens/message per tokenizer and ships it).
"""

from __future__ import annotations

import numpy as np

from . import refmark_lib as rl
from .gen4c_lib import sage_face, sage_floor  # noqa: F401  (T2, verbatim)

# ── reask (frozen pre-count) ────────────────────────────────────────

REASK_JACCARD = 0.3
REASK_MIN_CONTENT_WORDS = 3

# Frozen minimal stopword list — function words only, no content
# judgment. Small on purpose: a longer list would be a tuning knob.
STOPWORDS = frozenset(
    "a an and are as at be but by can could do does for from had has "
    "have he her his i if in is it its me my no not of on or our she "
    "so that the their them they this to was we were what when which "
    "who will with would you your".split())

_WORD_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789'")


def content_words(text: str) -> frozenset:
    """Lowercase [a-z0-9']+ tokens minus STOPWORDS (apostrophes
    normalized as in sycpress_lib)."""
    from .sycpress_lib import normalize
    words, cur = [], []
    for ch in normalize(text):
        if ch in _WORD_CHARS:
            cur.append(ch)
        elif cur:
            words.append("".join(cur))
            cur = []
    if cur:
        words.append("".join(cur))
    return frozenset(w for w in words if w not in STOPWORDS)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def reask_events(msgs) -> np.ndarray:
    """Per-message 0/1 over a WildChat-style [(role, content)] list:
    1 at user turn u_i iff (i) the immediately preceding message is
    an ASSISTANT turn firing ``refmark_lib.is_marker_turn`` (the
    frozen 12-substring deflection list, verbatim), (ii) a USER turn
    u_{i-2} exists immediately before that assistant turn, and
    (iii) content-word Jaccard(u_i, u_{i-2}) ≥ REASK_JACCARD with
    both sides carrying ≥ REASK_MIN_CONTENT_WORDS content words.
    The event indicator is out-of-window-by-construction: (ii)+(iii)
    reference a turn two messages back."""
    out = np.zeros(len(msgs), dtype=np.int8)
    for i, (role, content) in enumerate(msgs):
        if role == "assistant" or i < 2:
            continue
        r1, c1 = msgs[i - 1]
        r2, c2 = msgs[i - 2]
        if r1 != "assistant" or not rl.is_marker_turn(c1):
            continue
        if r2 == "assistant":
            continue
        w_now, w_orig = content_words(content), content_words(c2)
        if (len(w_now) >= REASK_MIN_CONTENT_WORDS
                and len(w_orig) >= REASK_MIN_CONTENT_WORDS
                and jaccard(w_now, w_orig) >= REASK_JACCARD):
            out[i] = 1
    return out


# ── event-message → token flags (shared by both age faces) ──────────


def event_first_token_flags(m_idx: np.ndarray,
                            events_msg: np.ndarray) -> np.ndarray:
    """Per-token 0/1: 1 at the FIRST token of each event message.
    ``m_idx`` is the per-token message index (lib.
    sentence_index_per_token output)."""
    m = np.asarray(m_idx, dtype=np.int64)
    ev = np.asarray(events_msg, dtype=np.int8)
    is_ev_msg = ev[m] == 1
    first = np.ones(len(m), dtype=bool)
    first[1:] = m[1:] != m[:-1]
    return (is_ev_msg & first).astype(np.int8)


def event_token_flags(m_idx: np.ndarray,
                      events_msg: np.ndarray) -> np.ndarray:
    """Per-token 0/1: 1 at EVERY token of an event message (the
    masking array — event text must not be probe-eligible)."""
    m = np.asarray(m_idx, dtype=np.int64)
    return (np.asarray(events_msg, dtype=np.int8)[m] == 1).astype(np.int8)


# ── msdose construction (frozen pre-count) ──────────────────────────

MSDOSE_SEED = 0
MSDOSE_N_DOCS = 400
MSDOSE_N_EX_LO, MSDOSE_N_EX_HI = 4, 25          # rng.integers bounds
MSDOSE_SPAN_MU, MSDOSE_SPAN_SIGMA = np.log(120.0), 0.6
MSDOSE_SPAN_MIN, MSDOSE_SPAN_MAX = 40, 400
MSDOSE_DELIM_TEXT = "\n###\n"


def msdose_plan(rng: np.random.Generator, n_docs: int = MSDOSE_N_DOCS):
    """Deterministic construction plan: per doc, the exemplar count
    and span lengths (content tokens per exemplar). Length jitter is
    the position-decorrelation design (§ 4 #4 trap (a))."""
    plan = []
    for _ in range(n_docs):
        n_ex = int(rng.integers(MSDOSE_N_EX_LO, MSDOSE_N_EX_HI))
        lens = np.clip(np.round(np.exp(rng.normal(
            MSDOSE_SPAN_MU, MSDOSE_SPAN_SIGMA, size=n_ex))),
            MSDOSE_SPAN_MIN, MSDOSE_SPAN_MAX).astype(np.int64)
        plan.append(lens)
    return plan


def msdose_doc(rng: np.random.Generator, flat: np.ndarray,
               doc_off: np.ndarray, lens: np.ndarray,
               delim_ids: np.ndarray):
    """One constructed many-shot doc from a committed token stream:
    ``lens[k]`` content tokens sampled as a contiguous span from a
    random source doc, each exemplar prefixed by ``delim_ids``.
    Returns (ids, is_boundary, dose) — dose[t] = exemplars whose
    delimiter starts at or before t (the running dose)."""
    ids_parts, bound_parts, dose_parts = [], [], []
    n_src = len(doc_off) - 1
    for k, ln in enumerate(lens):
        ln = int(ln)
        for _ in range(64):  # rejection-sample a long-enough source doc
            d = int(rng.integers(0, n_src))
            seg = flat[doc_off[d]:doc_off[d + 1]]
            if len(seg) >= ln:
                s = int(rng.integers(0, len(seg) - ln + 1))
                span = seg[s:s + ln]
                break
        else:  # every draw too short: truncate the last draw
            span = seg[:ln]
        ids_parts += [delim_ids, span]
        b = np.zeros(len(delim_ids) + len(span), dtype=np.int8)
        b[0] = 1
        bound_parts.append(b)
        dose_parts.append(np.full(len(b), k + 1, dtype=np.int32))
    return (np.concatenate(ids_parts).astype(np.int32),
            np.concatenate(bound_parts),
            np.concatenate(dose_parts))


def dose_window_count(is_boundary: np.ndarray, T: int) -> np.ndarray:
    """msdose's visible floor ingredient at T: boundaries inside the
    trailing T-token view (current token excluded)."""
    b = np.asarray(is_boundary, dtype=np.int64)
    c = np.concatenate([[0], np.cumsum(b)])
    idx = np.arange(len(b))
    lo = np.maximum(idx - T, 0)
    return (c[idx] - c[lo]).astype(np.float32)
