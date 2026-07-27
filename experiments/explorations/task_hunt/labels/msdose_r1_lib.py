"""``msdose_r1`` — FRESH PRE-COUNT for the msdose re-entry
(``WAVE3_SECOND_SOURCE.md`` § B; dispatch ``47040da59``).

Committed BEFORE the r1 build/pre-measure runs anything (commit-then-run).
This is a re-entry AMENDMENT, not a widening of the killed plan: the frozen
``wave3_lib`` msdose constants are untouched and remain the pre-count of
record for the KILLED construction; everything not named here inherits them
verbatim (n_docs, n_ex bounds, inner span sigma, clip bounds, delimiter).

The single design change (§ B): the exemplar-length scale is drawn PER
DOCUMENT instead of shared corpus-wide, breaking the near-shared
dose↔position map that made the frozen plan a position probe
(realised rho 0.962, position AUC 1.0).

Also frozen here: the § B position-matched usable-mass CENSUS — the
instrument the re-entry verdict is judged on. It runs first on the
committed frozen corpus (``wave3_msdose_<tok>.npz``) to give the
realised baseline, then on the r1 corpus; the § B simulated numbers
(pooled rho 0.844, 10/66 strata, 397,481 usable tokens) are the
pre-registered prediction the realised numbers must confirm.
"""

from __future__ import annotations

import numpy as np

from . import wave3_lib as w3

# ── the amendment (all other msdose constants inherited from wave3_lib) ──

MSDOSE_R1_SEED = 0                       # construction seed (same convention
#                                          as frozen; the draw SEQUENCE
#                                          differs by design — mu_doc draw)
MSDOSE_R1_DOC_MU_MU = np.log(120.0)      # per-doc scale: mu_doc ~ N(log 120, 0.7)
MSDOSE_R1_DOC_MU_SIGMA = 0.7             # sigma_doc — § B's recommendation
#                                          (0.7, not 1.0: 1.0 saturates)

# ── the § B census instrument (frozen; verdict-bearing) ──────────────────

STRATUM_W = 128                          # position-stratum width (tokens)
STRATUM_MIN_PER_TERCILE = 50             # rows of EACH global dose tercile


def msdose_r1_plan(rng: np.random.Generator,
                   n_docs: int = w3.MSDOSE_N_DOCS):
    """r1 construction plan. Draw order per doc is FROZEN:
    n_ex first (as in the frozen plan), then mu_doc, then the spans."""
    plan = []
    for _ in range(n_docs):
        n_ex = int(rng.integers(w3.MSDOSE_N_EX_LO, w3.MSDOSE_N_EX_HI))
        mu_doc = float(rng.normal(MSDOSE_R1_DOC_MU_MU,
                                  MSDOSE_R1_DOC_MU_SIGMA))
        lens = np.clip(np.round(np.exp(rng.normal(
            mu_doc, w3.MSDOSE_SPAN_SIGMA, size=n_ex))),
            w3.MSDOSE_SPAN_MIN, w3.MSDOSE_SPAN_MAX).astype(np.int64)
        plan.append(lens)
    return plan


def strata_census(dose: np.ndarray, pos_of: np.ndarray, elig: np.ndarray,
                  width: int = STRATUM_W,
                  min_rows: int = STRATUM_MIN_PER_TERCILE) -> dict:
    """Position-matched usable-mass census (§ B, frozen definition).

    Global dose terciles are computed over ALL eligible rows (pooled — the
    census is a construction property, not a probe; the card discloses the
    train-only edges separately). A position stratum (``width``-token bins
    of absolute position) QUALIFIES iff it holds >= ``min_rows`` rows of
    EACH global tercile. Usable tokens = all eligible rows in qualifying
    strata. Reported denominator ``n_strata_any`` counts strata with >= 1
    eligible row (the "31"/"66" of § B's table).
    """
    m = elig & np.isfinite(dose)
    q1, q2 = np.quantile(dose[m], [1 / 3, 2 / 3])
    terc = np.full(len(dose), -1, dtype=np.int8)
    terc[m & (dose <= q1)] = 0
    terc[m & (dose > q1) & (dose <= q2)] = 1
    terc[m & (dose > q2)] = 2
    strat = (pos_of // width).astype(np.int64)
    n_span = int(strat[m].max()) + 1
    qualifying, counts, usable, n_any = [], {}, 0, 0
    for s in range(n_span):
        sm = m & (strat == s)
        n_s = int(sm.sum())
        if n_s == 0:
            continue
        n_any += 1
        c = [int((terc[sm] == t).sum()) for t in (0, 1, 2)]
        if min(c) >= min_rows:
            qualifying.append(int(s))
            counts[str(s)] = c
            usable += n_s
    return {
        "stratum_width": int(width),
        "min_rows_per_tercile": int(min_rows),
        "tercile_edges_pooled": [float(q1), float(q2)],
        "n_strata_any": int(n_any),
        "n_qualifying": len(qualifying),
        "qualifying_strata": qualifying,
        "usable_tokens": int(usable),
        "per_stratum_tercile_counts": counts,
    }
