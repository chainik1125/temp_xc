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
