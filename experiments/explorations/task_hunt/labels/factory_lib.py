"""Pure label logic for the candidate factory (trace corpus, QUANTITY MODE).

Shared by the factory builders (`build_sc_lambda`, `build_qrate`,
`build_oprate`, `build_verbosity`, `build_redundancy`) so the sanity
tests (`tests/test_factory_labels.py`) exercise the exact code the
committed bundles were built with. No tokenizers, no I/O.

Frozen choices (stated in each candidate's CARD_DRAFT before any label
was computed; changing them invalidates every shipped bundle):

- **Kernel.** New event streams do NOT reuse the backtracking mirror's
  fitted ``kernel_w`` (those weights encode backtracking's own
  self-excitation profile). The factory kernel is a parameter-free-of-fit
  exponential, w_l = exp(-(l-1)/τ), τ = 3 sentences, K = 8 lags, applied
  as a NORMALIZED causal trailing rate over the previous sentences
  (never the current one). Tercile targets are invariant under any
  monotone transform, so the backtracking mirror's logistic wrapper
  (intercept / slope constants) would not change the classification
  target — dropping it removes borrowed constants, not information.
- **Position floor.** Kernel-only rates, no position term (the
  ward_lambda lesson: the position term is what positional probes read).
- **History guard.** Labels are NaN for sentences with fewer than
  MIN_HISTORY = 4 previous sentences, so the mechanical low-rate band at
  trace starts cannot feed a position confound.
- **Label-side triage thresholds** (kill authority): the bundle FAILS if
  top-vs-bottom-class AUC from the current token's identity alone
  reaches ``TRIAGE_TOK_AUC_KILL`` or from position alone reaches
  ``TRIAGE_POS_AUC_KILL`` (both measured as max(auc, 1-auc) on
  test-split manifest rows).
"""

from __future__ import annotations

import re

import numpy as np

from .interleave_lib import rank_auc

FROZEN_TAU = 3.0
FROZEN_K = 8
MIN_HISTORY = 4
TRIAGE_TOK_AUC_KILL = 0.65
TRIAGE_POS_AUC_KILL = 0.70

# ── frozen self-correction marker list (candidate 1) ────────────────────
# Applied to the sentence text after normalize_text(); the last pattern
# is sentence-initial by construction (^). Frozen BEFORE any frequency
# was computed. "let me verify/double-check/review" overlaps the
# proofops verification-check class — disclosed in both cards.
MARKER_PATTERNS = (
    r"\bwait\b",
    r"\bactually\b",
    r"\bhmm+\b",
    r"\bhold on\b",
    r"\blet me (?:re-?check|reconsider|re-?examine|re-?calculate"
    r"|re-?compute|redo|re-?derive|rethink|re-?evaluate|re-?read"
    r"|revisit|review|verify|double-?check)\b",
    r"\bdouble-?check\b",
    r"\b(?:i|we) made (?:an error|a mistake)\b",
    r"\bmy (?:mistake|error)\b",
    r"\bi was wrong\b",
    r"\b(?:that|this) can'?t be right\b",
    r"\b(?:that'?s|that is|this is) (?:wrong|incorrect|not right"
    r"|not correct)\b",
    r"\bdoesn'?t seem right\b",
    r"\bon second thought\b",
    r"\bscratch that\b",
    r"\boops\b",
    r"^\s*no\b",
)
_MARKER_RE = [re.compile(p) for p in MARKER_PATTERNS]


def normalize_text(text: str) -> str:
    """Length-preserving normalization so match offsets map back to the
    original char coordinates: curly apostrophe -> ascii, lowercase."""
    return text.replace("’", "'").lower()


def marker_spans_in_sentence(sent_text: str) -> list:
    """(start, end) char spans of every frozen-marker match, in SENTENCE
    coordinates."""
    t = normalize_text(sent_text)
    spans = []
    for rx in _MARKER_RE:
        spans.extend(m.span() for m in rx.finditer(t))
    return spans


def sentence_events_markers(sent_texts) -> np.ndarray:
    """0/1 per sentence: contains >= 1 frozen self-correction marker."""
    return np.array([1 if marker_spans_in_sentence(t) else 0
                     for t in sent_texts], dtype=np.int8)


def sentence_events_question(sent_texts) -> np.ndarray:
    """0/1 per sentence: last non-whitespace char is '?'."""
    return np.array([1 if t.rstrip().endswith("?") else 0
                     for t in sent_texts], dtype=np.int8)


def token_mask_from_spans(offsets, char_spans) -> np.ndarray:
    """int8 per token: 1 iff the token's char span overlaps any of the
    given (start, end) char spans (all in the same coordinates)."""
    out = np.zeros(len(offsets), dtype=np.int8)
    if not char_spans:
        return out
    for i, (a, b) in enumerate(offsets):
        for s, e in char_spans:
            if max(a, s) < min(b, e):
                out[i] = 1
                break
    return out


# ── the frozen factory kernel ───────────────────────────────────────────

def exp_kernel_weights(k: int = FROZEN_K, tau: float = FROZEN_TAU):
    """w_l = exp(-(l-1)/tau) for lags l = 1..k."""
    return np.exp(-np.arange(k) / tau)


def kernel_rate(events, k: int = FROZEN_K, tau: float = FROZEN_TAU,
                min_history: int = MIN_HISTORY) -> np.ndarray:
    """Causal exponentially-weighted trailing event rate per sentence:
    rate_i = sum_l w_l e_{i-l} / sum_l w_l over l = 1..min(i, k).
    The current sentence's own event is NEVER an input. NaN if i <
    min_history or any of the min(i, k) trailing events is NaN
    (unlabeled). Normalizing by the AVAILABLE weight mass keeps the
    value a rate for sentences with min_history <= i < k."""
    e = np.asarray(events, dtype=float)
    w = exp_kernel_weights(k, tau)
    out = np.full(e.size, np.nan, dtype=np.float32)
    for i in range(min_history, e.size):
        m = min(i, k)
        hist = e[i - m:i][::-1]                 # lags 1..m
        if np.isnan(hist).any():
            continue
        ww = w[:m]
        out[i] = float((ww * hist).sum() / ww.sum())
    return out


def shuffle_events(events, rng: np.random.Generator) -> np.ndarray:
    """Within-trace event shuffle for the label-side null: permute the
    FINITE entries among their own positions; NaN (unlabeled) positions
    stay fixed. Preserves the per-trace event rate exactly, destroys the
    local clustering the kernel reads."""
    e = np.asarray(events, dtype=float).copy()
    fin = np.flatnonzero(~np.isnan(e))
    e[fin] = e[fin][rng.permutation(fin.size)]
    return e


# ── trailing sentence-level aggregates (candidate 4) ────────────────────

def trailing_mean_prev(vals, k: int = FROZEN_K,
                       min_n: int = MIN_HISTORY) -> np.ndarray:
    """Unweighted mean of the previous min(i, k) entries (current entry
    excluded); NaN if fewer than min_n or any of them is NaN."""
    v = np.asarray(vals, dtype=float)
    out = np.full(v.size, np.nan, dtype=np.float32)
    for i in range(v.size):
        m = min(i, k)
        w = v[i - m:i]
        if m >= min_n and not np.isnan(w).any():
            out[i] = float(w.mean())
    return out


def trailing_slope_prev(vals, k: int = FROZEN_K,
                        min_n: int = MIN_HISTORY) -> np.ndarray:
    """OLS slope of the previous min(i, k) entries against their order
    (oldest = 0); NaN under the same conditions as trailing_mean_prev."""
    v = np.asarray(vals, dtype=float)
    out = np.full(v.size, np.nan, dtype=np.float32)
    for i in range(v.size):
        m = min(i, k)
        w = v[i - m:i]
        if m < max(min_n, 2) or np.isnan(w).any():
            continue
        x = np.arange(m, dtype=float)
        x -= x.mean()
        out[i] = float((x * (w - w.mean())).sum() / (x * x).sum())
    return out


# ── trailing token-level rates (candidate 5 + evidence baselines) ───────

def trailing_rate_prev(flags, w: int) -> np.ndarray:
    """Mean of flags over positions t-w..t-1 (current token EXCLUDED —
    the causal label); NaN for t < w."""
    f = np.asarray(flags, dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(f)])
    out = np.full(f.size, np.nan, dtype=np.float32)
    for t in range(w, f.size):
        out[t] = (cs[t] - cs[t - w]) / w
    return out


def trailing_count_incl(flags, w: int) -> np.ndarray:
    """Count of flags over positions t-w+1..t (current token INCLUDED —
    what a probe window of size w can see); NaN for t < w - 1."""
    f = np.asarray(flags, dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(f)])
    out = np.full(f.size, np.nan, dtype=np.float32)
    for t in range(w - 1, f.size):
        out[t] = cs[t + 1] - cs[t + 1 - w]
    return out


# ── binning ─────────────────────────────────────────────────────────────

def zero_inflated_bins(values: np.ndarray):
    """(scheme, edges, bins int8). Primary = terciles (lib.tercile_bins
    semantics). Terciles are accepted only if every resulting bin holds
    >= 10% of the finite rows; otherwise (mass collapse — sparse-event
    rates piling on an edge) the FROZEN fallback is the zero-inflated
    3-bin: 0 = rate exactly 0, then a median split of the positive
    rates. NaN -> -1 in both schemes."""
    v = values[np.isfinite(values)]
    edges = np.quantile(v, [1 / 3, 2 / 3])
    out = np.full(values.shape, -1, dtype=np.int8)
    m = np.isfinite(values)
    out[m] = np.digitize(values[m], edges).astype(np.int8)
    counts = [(out[m] == c).sum() for c in (0, 1, 2)]
    if min(counts) >= 0.10 * v.size:
        return "terciles", [float(e) for e in edges], out
    pos = v[v > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    out = np.full(values.shape, -1, dtype=np.int8)
    m = np.isfinite(values)
    out[m & (values == 0)] = 0
    out[m & (values > 0) & (values <= med)] = 1
    out[m & (values > med)] = 2
    return "zero_split", [0.0, med], out


# ── label-side triage (the kill authority) ──────────────────────────────

def _extreme(auc: float) -> float:
    return max(auc, 1.0 - auc)


def token_id_triage_auc(tok_id, is_top, train_mask, test_mask) -> float:
    """AUC of predicting top-vs-bottom class membership from the CURRENT
    token's identity alone: per-id mean of is_top fit on train rows
    (unseen ids -> train global mean), rank-AUC on test rows."""
    tok_id = np.asarray(tok_id)
    is_top = np.asarray(is_top, dtype=float)
    ids_tr, y_tr = tok_id[train_mask], is_top[train_mask]
    uniq, inv = np.unique(ids_tr, return_inverse=True)
    sums = np.zeros(uniq.size)
    cnts = np.zeros(uniq.size)
    np.add.at(sums, inv, y_tr)
    np.add.at(cnts, inv, 1.0)
    means = sums / cnts
    glob = float(y_tr.mean()) if y_tr.size else 0.5
    ids_te = tok_id[test_mask]
    pos_in = np.searchsorted(uniq, ids_te)
    pos_in = np.clip(pos_in, 0, max(uniq.size - 1, 0))
    seen = uniq.size > 0
    known = (uniq[pos_in] == ids_te) if seen else np.zeros(ids_te.size, bool)
    scores = np.where(known, means[pos_in] if seen else glob, glob)
    return rank_auc(scores, is_top[test_mask].astype(int))


def position_triage_auc(pos_score, is_top, test_mask) -> float:
    """Rank-AUC of a position score (raw token index or trace fraction)
    predicting top-vs-bottom class membership on test rows."""
    return rank_auc(np.asarray(pos_score, dtype=float)[test_mask],
                    np.asarray(is_top)[test_mask].astype(int))


def triage_verdict(tok_auc: float, pos_aucs) -> dict:
    """The frozen kill rule. FAIL iff token-identity extreme-AUC >=
    TRIAGE_TOK_AUC_KILL or any position extreme-AUC >=
    TRIAGE_POS_AUC_KILL."""
    tok_x = _extreme(tok_auc)
    pos_x = max(_extreme(a) for a in pos_aucs)
    fail = tok_x >= TRIAGE_TOK_AUC_KILL or pos_x >= TRIAGE_POS_AUC_KILL
    return {"tok_auc_extreme": tok_x, "pos_auc_extreme": pos_x,
            "tok_kill_at": TRIAGE_TOK_AUC_KILL,
            "pos_kill_at": TRIAGE_POS_AUC_KILL,
            "verdict": "FAIL" if fail else "PASS"}
