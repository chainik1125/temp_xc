# Estimator card — C7 monotone segment-composition extraction

> **Frozen 2026-07-23 (runpod agent, Cycle 7), committed BEFORE any C7
> estimator code, battery run, or real-data measurement.** Never revised —
> dated amendments only. Governs `briefings/expansion-c7.md`: the LAST
> estimator cycle for the reasoning int/eq cell — one candidate, a
> monotonicity pre-check, variance-aware gate margins, and a pre-specified
> close either way. Predecessor: the C6 card ("expansion C6: estimator
> card + r4 amendment FROZEN (pre-build)") and its battery record
> `results/estimator_battery_c6.md` (reviewed & APPROVED).

## The one candidate: `seg_hier_categorical_mono`

**Mechanism — deconvolve FIRST, then shrink in deconvolved space.** The
C6 finding 1: shrinking *observed* compositions before the C4
self-consistency deconvolution is non-monotone — the fixed point
re-amplifies any composition slightly off the doc marginal, so the λ knob
is flat then cliffs. The fix inverts the order so shrinkage is the LAST
operation and nothing downstream re-sharpens:

1. Fit stages exactly as r3 (`_seg_collect` → tilt-MLE → per-segment
   deconvolution with the fitted (a, b)) → the r3-identical deconvolved
   segment propensities **u_seg** (λ = 0 reproduces the r3 estimator
   parameter-for-parameter).
2. Per doc, the **deconvolved doc fixed point u_doc**: the same fixed
   point (same (a, b), same 15 iterations) with the doc marginal as
   target — the C6 λ=1 limit, whose null-cleanliness C6 verified.
3. **Shrink last, in u-space:** segment table i gets
   `u_i(λ) = (1−λ)·u_seg_i + λ·u_doc`, renormalized. Generation consumes
   these tables directly; generated segment contrast interpolates
   between r3 (λ=0) and inert (λ=1) with no re-amplification stage.

**In-loop calibration (C6 candidate-A principle, unchanged).** λ* is
calibrated on `run_permuted_streams(train)` (PERM_NULL_SEED): the
smallest λ whose fit round-trips BOTH preregistered moments (`mi[lag2]`,
`acf[lag4]`) within the null-referenced uniform tolerance
`max(0.20·|perm value|, floors mi 0.003 / acf 0.01)`. Real held-out
moments never enter any objective (the C6 card's non-fitted-moment
argument carries over verbatim). Procedure: coarse scan
λ ∈ {0, 0.25, 0.5, 0.75, 1.0}, then bisection between the largest
failing and smallest passing coarse points to interval width ≤ 0.05,
taking the PASSING (upper) endpoint — conservative. Insertion at each λ
is the mean over **R_cal = 6** generation replicates (seeds 9000+i,
i < 6; supersedes C6's 3 — the variance-aware requirement applied
in-loop). If even λ = 1 fails, the fit flags `uncalibratable`.

## Monotonicity pre-check (BEFORE any gate is scored)

On the committed real streams (r3 record labels; zero API): one fit-stage
pass, then sweep λ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}; m(λ) = generated
ACF(4), mean over **R_pre = 6** replicates (seeds 9100+i — the C6
measurement family extended). With SE(λ) = std(reps, ddof=1)/√R and
SE_pool = mean over grid points, require BOTH:

- **(i) no material increase anywhere:**
  m(λ_{i+1}) − m(λ_i) ≤ max(0.5·SE_pool, 0.002) for every adjacent pair;
- **(ii) the knob actually moves:** m(0) − m(1) ≥ 5·SE_pool.

(The C6 shrink-then-deconvolve λ-scan violates (i) — the check has
teeth.) Spearman ρ(λ, m) recorded, report-only. **If the pre-check fails,
the concept is dead: record it and go directly to the close — no gate
scoring, no mechanism iteration** (briefing § mandate 1b).

## Acceptance battery — C6 gates 1–3 VERBATIM + variance-aware margins

Batteries 1–5 exactly as the C6 card (same datasets, same data seeds:
gate-1 run-permuted committed real streams with the same permutation and
tolerance rule; gate-2 heavy-dwell null, data rng 31, bound
max(0.2·acf_null4, 0.01); gate-3 strong planted truth, data seed 21,
bound 0.20·real acf4; battery-4 weak truth seed 22 and battery-5
five-replicate variance panel report-only). Measurement seeds 9100+i.

**Variance-aware margin rule (adopted at the C6 review; numeric).** Every
gate comparison (measured statistic `s` vs bound `tol`) is decided only
with a ≥ 2·SE margin, escalating replicates at the boundary:

- start R = 6; PASS if s ≤ tol − 2·SE; FAIL if s > tol + 2·SE;
- otherwise escalate R → 12 → 24 (seeds 9100..9123, the same preregistered
  family) and re-decide;
- still inside ±2·SE at R = 24 ⇒ **conservative FAIL** (gates 1–2:
  hallucination not demonstrably subdominant; gate 3: sensitivity not
  demonstrably preserved). A seed flip can never decide a verdict.

## The pre-specified fork (verbatim close; no discretion at run time)

Let λ*_real = the calibrated λ of the fit on the (full) committed real
streams. **Inert threshold: λ*_real > 0.85** (retained u-space contrast
< 15%, below every noise scale measured in C6).

- **Pre-check passes AND gates 1–3 pass AND λ\*_real ≤ 0.85 → run r4**
  (`proof-operation-phase-runs-r4` via the canonical calibrate pipeline;
  mirror `seg_hier_categorical_mono` per the dated C7 amendment; gate-8
  and the recorded insertion control unchanged from the r3/r4
  amendments; labels reused; skeptic on PROCEED, `claude-fable-5`, raw
  persisted pre-parse, never re-rolled).
  - r4 PASS ⇒ the reasoning int/eq card graduates to SPEC
    (registry/BENCHMARKS/LEDGER; **no stage-6 grid this session**).
  - r4 FAIL ⇒ record; the cell **closes NEGATIVE** (below).
- **Pre-check fail, OR any gate fails, OR λ\*_real > 0.85 → the reasoning
  half of the int/eq prize closes NEGATIVE at this corpus resolution:**
  the three-timescale structure is real (model-independently confirmed,
  C5) but not extractable at 287 docs × ~85 sentences; the next lever is
  more/longer traces, not another estimator. No C8 estimator proposals.
  The close is a SUCCESS of the loop and is written as such (LEDGER C7,
  BENCHMARKS § B proof-operation row, research STATUS § 0).
- **Skeptic fires on WHICHEVER branch resolves** (briefing § 4): the SPEC
  branch uses the in-pipeline r4 skeptic; the close branch (including
  r4-FAIL) runs the same expansion rubric over this card + the battery
  and close summary, persisted raw pre-parse under
  `records/proof-operation-phase-runs-c7-close/`.

## Process rails

Strict commit-then-run: the estimator implementation and every
battery/analysis script are committed BEFORE first execution. Commit
citations by subject line (or SHA re-verified post-push). Spend to
`expansion/results/spend.json` under the $10 C7 cap; no fresh labeling;
no program-rule/gate edits; no `temp_bench/core/` edits.

_Frozen-by: claude-fable-5 (runpod agent, Cycle 7), before any C7
implementation._
