"""Doc-level (cluster) bootstrap CIs for the label-side triage AUCs —
pure numpy, no I/O. Covered by ``tests/test_boot_lib.py``.

Every triage number in this hunt is an AUC over probe rows, and the rows
are NOT independent: thousands of them come from one document, sharing
its vocabulary, its position profile and (for the doc-mean statistic) its
label outright. A row-level bootstrap would therefore quote an interval
an order of magnitude too narrow. The honest unit of resampling is the
DOCUMENT: draw ``n_docs`` documents with replacement, keep every row of
each draw (with multiplicity), recompute the AUC. That is what the
threshold-pinning review needs — a distribution, not a point estimate.

Exactness matters as much as speed here: the resampled AUC is computed
from tie-collapsed score LEVELS by Mann-Whitney counting, not by sorting
the resampled rows —

    AUC = Σ_l  n⁺_l · (N⁻_{<l} + ½ n⁻_l)  /  (n⁺ · n⁻)

— which is algebraically identical to ``interleave_lib.rank_auc`` (ties
get half credit; asserted in the tests) and costs two weighted
``bincount`` passes per replicate instead of an O(n log n) sort. A
million-row statistic is then ~10 ms/rep, so 1,000 reps × the campaign's
statistic slots is minutes, single-process.

Reported per statistic: the point estimate on the full data (identical to
the shipped triage number), the bootstrap mean/SE, and percentile CIs —
BOTH on the raw AUC (what a threshold on a signed statistic consumes)
and on the direction-agnostic fold ``max(a, 1-a)`` applied per replicate
(what the frozen 0.65 bars actually read).
"""

from __future__ import annotations

import numpy as np

from .interleave_lib import rank_auc

N_REPS = 1_000          # campaign floor (briefing: >= 1,000 reps)
SEED = 0
CI_PCT = (2.5, 97.5)    # percentile interval


def _levels(scores: np.ndarray):
    """Tie-collapsed ascending level ids for scores (uniq, inv)."""
    uniq, inv = np.unique(np.asarray(scores, dtype=float),
                          return_inverse=True)
    return inv.astype(np.int64), int(uniq.size)


def _auc_from_counts(cp: np.ndarray, cn: np.ndarray) -> float:
    """Mann-Whitney AUC from per-level positive/negative counts (ties at
    a level get half credit — the rank_auc convention)."""
    n_pos = float(cp.sum())
    n_neg = float(cn.sum())
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")
    below = np.concatenate(([0.0], np.cumsum(cn)[:-1]))
    return float((cp * (below + 0.5 * cn)).sum() / (n_pos * n_neg))


def bootstrap_auc(scores, labels, doc_of, *, n_reps: int = N_REPS,
                  seed: int = SEED, ci_pct=CI_PCT) -> dict:
    """Doc-level bootstrap of the AUC of `scores` predicting binary
    `labels` (1 = positive), clustering rows by `doc_of`.

    Resampling draws ``n_docs`` document ids with replacement (a
    multinomial multiplicity vector) and reweights that document's rows;
    replicates in which one class vanishes are counted, not silently
    dropped from the denominator claim.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    doc_of = np.asarray(doc_of)
    point = rank_auc(scores, labels)

    docs, doc_idx = np.unique(doc_of, return_inverse=True)
    n_docs = int(docs.size)
    lev, K = _levels(scores)
    pos = labels == 1
    neg = labels == 0
    lev_p, doc_p = lev[pos], doc_idx[pos]
    lev_n, doc_n = lev[neg], doc_idx[neg]

    out = {"point": point,
           "point_direction_agnostic": (float(max(point, 1.0 - point))
                                        if np.isfinite(point)
                                        else float("nan")),
           "n_rows": int(scores.size), "n_pos": int(pos.sum()),
           "n_neg": int(neg.sum()), "n_docs": n_docs,
           "n_reps": int(n_reps), "seed": int(seed),
           "ci_pct": list(ci_pct)}
    if n_docs == 0 or not pos.any() or not neg.any() or n_reps <= 0:
        out.update({"mean": float("nan"), "se": float("nan"),
                    "ci_lo": float("nan"), "ci_hi": float("nan"),
                    "ci_lo_direction_agnostic": float("nan"),
                    "ci_hi_direction_agnostic": float("nan"),
                    "n_degenerate_reps": 0})
        return out

    rng = np.random.default_rng(seed)
    p = np.full(n_docs, 1.0 / n_docs)
    reps = np.empty(n_reps, dtype=float)
    for r in range(n_reps):
        m = rng.multinomial(n_docs, p).astype(float)
        cp = np.bincount(lev_p, weights=m[doc_p], minlength=K)
        cn = np.bincount(lev_n, weights=m[doc_n], minlength=K)
        reps[r] = _auc_from_counts(cp, cn)

    ok = np.isfinite(reps)
    da = np.maximum(reps[ok], 1.0 - reps[ok])
    lo, hi = np.percentile(reps[ok], ci_pct)
    lo_da, hi_da = np.percentile(da, ci_pct)
    out.update({
        "mean": float(reps[ok].mean()),
        "se": float(reps[ok].std(ddof=1)) if ok.sum() > 1 else float("nan"),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "ci_lo_direction_agnostic": float(lo_da),
        "ci_hi_direction_agnostic": float(hi_da),
        "n_degenerate_reps": int((~ok).sum()),
    })
    return out


def bootstrap_tercile_auc(scores, tercile, mask, doc_of, *,
                          n_reps: int = N_REPS, seed: int = SEED) -> dict:
    """Bootstrap the exact statistic ``novelty_lib.tercile_auc`` reports:
    class 2 vs class 0 on rows in `mask` (class 1 and unlabeled ignored),
    clustered by document."""
    scores = np.asarray(scores, dtype=float)
    tercile = np.asarray(tercile)
    m = (np.asarray(mask) & (tercile >= 0) & (tercile != 1)
         & np.isfinite(scores))
    return bootstrap_auc(scores[m], (tercile[m] == 2).astype(int),
                         np.asarray(doc_of)[m], n_reps=n_reps, seed=seed)
