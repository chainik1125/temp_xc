# Mini-card (frozen pre-screen) — confidence trend (hedging→commitment)

**Candidate 2, task-hunt arm B** (`briefings/task-hunt-b.md`; card
written by `runpod-b` per `briefings/task-hunt-prep.md` — labels:
`../labels/confidence.npz`, built by the committed
`labels/build_confidence.py`). Science frozen by this commit; the
running agent (`runpod-e`) appends its screen-cell table before running.
Screen order per its briefing: only after the replag Stage-1 verdict —
and the clock numbers below may justify going straight to its
candidate 3 instead (that decision is the running agent's).

## Latent + labels (frozen judge, committed record)

Per-sentence confidence state over the same 300 R1-Distill traces — the
frozen record
`synthetic/expansion/records/uncertainty-hedging-drift/labels.json`
(0 hedged / 1 neutral / 2 committed; bulk haiku + sonnet adjudication,
κ = 0.636, ε̂ = 0.113; 300/300 docs). The drift is real in the labels:
mean state by trace third 0.819 → 0.955 → 0.976 (computed in
`../labels/confidence_stats.json`), and ACF(1) 0.316 vs N2 hi 0.030 in
its calibration record.

Targets on the Ward grid (`../labels/confidence.npz`):

- **`slope8_bin`** (terciles of the trailing-8-sentence least-squares
  slope of the state; PRIMARY) and `slope4` (reachability backup) — the
  hedging→commitment TREND;
- **`hedge`** state 3-class (CONTROL, not a target): lexically stamped —
  the synthetic hedging bench verdict was SPLIT (per-token 0.755 vs best
  window 0.775, STORY.md § 1) — predicted regime-1.

## Why non-ambient (order-sensitive regime 2 — not mixing)

The state is ambient; the SLOPE is not: a trailing slope is a linear
functional of the window with CENTERED weights (early sentences weighted
negative, late positive), so a single position carries none of it and a
within-window shuffle destroys it by construction — g_order should carry
essentially all of the window advantage. This predicts a clean
dissociation: state target → shuffle-immune, per-token-sufficient;
slope target → shuffle-killed, window-only. No order-2 mixing is
required (slope is linear given per-position hedge features), so this is
a regime-2/order candidate, not a subtype-rule case.

## Clock bridge (measured — the named risk)

Median 16 tokens/sentence ⇒ slope8's support is ~128 tokens: **beyond
every panel-feasible window**. slope4 ≈ 64 tokens = exactly T = 64.
Screen at T ∈ {16, 32, 64}: only slope4 at T = 64 reaches full
coverage; slope8 at T = 64 sees half its support. Per the briefing, an
honest "window cannot reach the trend's timescale at panel-feasible T"
is a valid kill — the coverage bookkeeping distinguishes it from a real
negative.

## Predicted T-pattern (STORY.md § 7: threshold, reach-limited)

Gap ≈ 0 at T ≤ 16 (sub-sentence), grows T = 32 → 64 as slope-window
coverage grows (slope4 coverage 0.5 → 1.0), no saturation visible
within the screen range. Per-token flat in T.

## Falsifier / KEEP-KILL (frozen)

- **KEEP** iff slope4 (or slope8) shows window−token gap ≥ 0.05 AUC at
  T = 64 with the T-growth shape above AND shuffled-window collapses
  the gap (the order receipt) AND the state control stays regime-1
  (per-token ≈ window).
- **KILL — converted/ambient** if per-token ≈ window at full slope4
  coverage (the model already summarizes trend per token, or the slope
  is readable from the anchor sentence's own hedging level — the named
  confound: slope correlates with current state; the state control +
  identity-style matching on the anchor sentence's hedge class is the
  guard).
- **KILL — timescale unreachable** if no target reaches coverage ≥ 1.0
  at any screened T (recorded as a clock measurement, not a regime
  claim).

---

## Screen cells (appended by runpod-e, frozen by this commit BEFORE the run)

Candidate-1 (replag) verdict is committed (KILL — see `../LOG.md`);
proceeding here per the briefing order since the measured clock bridge
does NOT kill this candidate (slope4 support ≈ 64 tokens = T64).

- **Substrate:** Ward stream rebuilt on the runpod-e volume (stats
  reproduce the committed reference, map_ok 99.97 %); readers = base +
  distill via the committed `cache_depth.py`; screen layer resid_post
  L13 = hs14 (the measured g(ℓ) peak) for both.
- **Rows** (`screen.py build_rows`): eligibility valid ∧ p ≥ 63
  (uniform so every screened T ≤ 64 fits) ∧ hedge(anchor) ≥ 0; split by
  trace (labels' `trace_split`); caps 4000/1500 per class.
- **Targets:** `slope8` = the labels' committed slope8_bin terciles
  (PRIMARY); `slope4` = terciles of the slope4 grid computed over
  eligible rows (edges recorded in results meta; pooled-split binning,
  same convention as the committed slope8 edges); `state` = hedge
  3-class (CONTROL, regime-1 predicted).
- **Matching (the card's guard):** slope targets exact-histogram
  matched across the 3 classes on (anchor hedge state × position
  bucket {[63,80),[80,96),[96,112),[112,128)}) — the anchor's own
  hedging level cannot carry the slope label; state control matched on
  position bucket only. Fallback + row-floor rules as in replag
  (`matched_sample`, MIN_ROWS 300).
- **Probe grid:** frozen problib stack; per-token linear/MLP once;
  T ∈ {16, 32, 64}: window linear, window-mean linear,
  context-shuffled linear (anchor slot fixed, seeded); MLPs at
  T ∈ {32, 64}; permutation nulls (seed 99) at T = 32. Metric:
  acc_test (3-class, balanced; chance 1/3) + per_class.
- **KEEP/KILL:** exactly the frozen rules above (§ Falsifier) with the
  gap read as window − per-token accuracy on slope targets at T = 64
  (either probe pair, MLP allowed per the regime-2 linear prediction —
  slope is linear given per-position features, so the LINEAR pair is
  the primary reader here); ≥ +0.05 accuracy with the T-growth shape +
  shuffle collapse = KEEP.
