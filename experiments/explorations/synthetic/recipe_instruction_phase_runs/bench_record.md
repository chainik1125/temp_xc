# Bench record — recipe_instruction_phase_runs (stage 6, 2026-07-22)

**Status: § 8 STOP — no grid was run.** The equality-variant discriminability
gate (the C4-review addition, preregistered in [`gating.py`](gating.py) before
any architecture run) **failed condition (i)**: raw-LINEAR access to the
primary latent `e_t = [c_t = c_{t-1}]` is far above chance, so the substrate
does not test the regime-3 claim as frozen. Per the gate's rule the grid was
**not** launched; the frozen § 5 predictions remain untested and blind (no
architecture has seen this substrate). Awaiting review re-scope.

Full numbers: [`results/recipe_gating_stats.json`](results/recipe_gating_stats.json)
· figure: [`figs/recipe_gating.png`](figs/recipe_gating.png)
· build: generator `recipe_instruction_phase_runs` + datasource
`toy_recipe_instruction_d64` + evaluator add-on `recipe_recovery` are
registered and tested (the bench is runnable the moment a re-scoped gate
passes).

## What the gate measured (noiseless substrate; balanced-accuracy access
ceilings, threshold-optimized held-out)

| access route | e_t balacc | reading |
|---|---|---|
| chance | 0.500 | — |
| **per-token raw-linear** (x_t) | **0.614** | = the analytic from-`c_t` line (0.609): the **DC leak** |
| **window raw-linear** T=2 / 4 / 8 | **0.720** / 0.696 / 0.694 | genuine *additive* cross-position access |
| pair-additive ceiling (one-hot ⊕ one-hot, analytic) | 0.771 | ceiling for ANY additive readout |
| nonlinear (MLP on raw T=2 tile) | 1.000 | latent present — gate (ii) PASS |
| exact pair rule (in-tile, T ≥ 2) | 1.000 | oracle |

Noisy check (σ = 0.5): same ordering, mildly attenuated (per-token 0.605,
T=2 0.655). DC control `c_t`: per-token balacc 1.000 (oracle reachable —
expected). Mirror sanity PASS (marginal max dev 0.053 < 0.08; match rate
0.625 in [0.58, 0.68]).

## Why the regime-3 claim failed — the class-conditional continuation leak

Changepoint's boundary latent survived this exact gate because its Π was
**rebalanced uniform by design**, making `P(c_t | m_t)` constant — the § 8 (i)
premise held *exactly*. This grounded mirror cannot rebalance without
un-grounding: the per-symbol dwell heterogeneity IS the measured phenomenon,
and it makes the continuation rate class-dependent —
`P(e_t=1 | c_t) = {0: 0.63, 1: 0.74, 2: 0.56, 3: 0.41, 4: 0.33}`. Since `c_t`
is per-token linearly readable *by design* (the DC control), every code that
exposes the control also hands a linear `e_t` readout ≈ 0.61. Windows add
genuinely additive access on top (0.72 → additive ceiling 0.77): with skewed
marginals the additive one-hot-pair fit approximates the diagonal indicator
well above chance — the XOR-blindness argument in the frozen § 4 assumed
class-exchangeable dynamics it doesn't have.

**The frozen prediction structure was mis-scoped at freeze**: § 5 predicted
"per-token blind to `e_t` beyond its chance floor" — already false at the
raw-access level with no training involved. The C4 design-time
discriminability argument reasoned from the changepoint precedent's symmetry;
the empirical gate exists precisely because design-time arguments can miss
this, and it did its job. An ABORT-grade STOP is a success of the process
(prime directive), not a failure of the phenomenon — the signal is real and
the mirror is sound; the *architectural question* was mis-posed.

## What survives, for the review's re-scope decision (NOT acted on here)

The nonlinear-only residual is large: additive ceiling 0.771 vs exact 1.000 —
0.23 of balanced accuracy is accessible only through position-*mixing*
(coincidence / post-squash) routes. Two re-scope options the review could
take, in order of how much they preserve:

1. **Re-normalize the primary axis**: score `equality_recovery` against
   [pair-additive ceiling, 1] instead of [0.5, 1] — the bench then tests
   exactly the regime-3 residual, keeping the grounded substrate intact. The
   § 5 predictions would need re-freezing against the new floor (a new dated
   amendment, review-gated).
2. **Demote the claim**: file the bench as regime 2/3-mixed
   (interaction/equality with a documented additive floor), run the grid, and
   interpret window–token separation only above the measured raw-access lines
   (0.614 per-token / 0.72 window). Weakest but cheapest.

Rebalancing the mirror's dwell to kill the leak (the changepoint move) is
**not** available — it would discard the measured heterogeneity the C4 gate-8
validated (the ACF(4) plateau rides on it).

*Recorded by `runpod`, stage-6 session 2026-07-22 (briefing
`stage6-recipe-then-c5.md` Phase 1). Gate thresholds were preregistered in
`gating.py` before the first run; nothing was tuned after seeing a metric.*

## Self-audit vs the README checklist + validity gates (Phase 3, same session)

Audited against README § "Validity gates" / § "Required output artifact"
without re-running anything:

- **Equality-variant discriminability gate — followed as adopted** (C4
  review wording): (i) both raw-LINEAR readouts checked → NOT at chance →
  regime-2 leak → recorded + STOPPED before any grid, no engineering around
  it; (ii) presence verified (MLP → 1.0). Ground-truth hygiene clean
  (F = 20, 5 + 15, never conflated); real-side controls inherited from the
  C4 calibration record (pinned corpus, N1/N2/N3, ε̂, held-out gate-8).
  Memorization-budget / untrained-encoder / realistic-regime /
  capability-vs-artifact gates are grid-stage gates — N/A, no grid ran.
  Required-artifact fields present except the architecture frontier
  (deliberately absent — withheld by the STOP).
- **Gap (honest):** the gate thresholds in `gating.py` were written before
  the first run *within the session*, but the script and its results were
  committed TOGETHER (`b463c4a0`) — unlike the C5 card amendment (frozen in
  `f8c1deb6` strictly before the calibration commit), there is no
  commit-order evidence of threshold preregistration. Future § 8 gating
  scripts should be committed before their first execution.
- **Gap (minor):** the nonlinear presence check ran only at T=2 (noiseless
  + noisy); T ≥ 4 presence is argued (pair in-tile ⇒ exact), not measured.
  The noisy substrate was a single σ = 0.5 point, not a sweep.
- **Note for any re-scoped run:** the § 5 falsifier's "raw-access line" must
  reference the threshold-optimized CEILING (0.614 per-token / 0.720
  window), not the plain-probe numbers (0.595/0.619) — a plain probe can sit
  at balacc 0.5 under class imbalance while real access exists.

## Gating addendum (2026-07-23, runpod — stage-6 #3b, re-scoped axis)

No new computation; all numbers cite the committed § 8 record
(`results/recipe_gating_stats.json`, commit `b463c4a0`). Under the re-scoped
primary axis the discriminability condition reads: **nonlinear access 1.000 ≫
additive ceiling 0.771** — a 0.229 balanced-accuracy separation that only
position-mixing can close; that separation, not distance from chance, is what
the bench now tests. Condition (ii) of the original gate (latent present via
the nonlinear route — MLP 1.000 on the raw T=2 tile) is exactly the PASS side
the re-scoped axis rides on; condition (i)'s measured raw-linear lines (0.614
per-token / 0.720 window, threshold-optimized) become named floors reported
alongside the residual. The § 8 gate for the re-scoped bench is therefore
**satisfied by the existing record** — no re-run, no new thresholds.

## Review (2026-07-23, mac-local) — APPROVED; re-scope decision

Verified: no grid rows (leaderboard clean), STOP direction analytically
forced (the from-`c_t` line 0.609 ≈ measured 0.614 — thresholds could not
have manufactured it), frozen § 5 untouched, build + 8 tests sound, records
honest. The two self-audit gaps are accepted and the first is now a program
rule (README: gating scripts committed before first execution;
threshold-optimized ceilings as the raw-access lines).

**Decision — re-scope option 1 (re-normalize), queued:** the bench's primary
axis becomes the **regime-3 residual** — `equality_recovery` scored against
`[pair-additive ceiling 0.771, exact 1.0]` — with the DC leak and additive
access reported as named floors, not noise. This preserves the grounded
substrate (rebalancing the dwell would discard the C4-validated phenomenon)
and poses the honest architectural question: which code *linearizes* the
0.23 that only position-mixing can reach. Requires a dated § 5 re-freeze
(new per-arch predictions + the corrected falsifier referencing the
threshold-optimized lines) committed BEFORE any grid — a future briefing;
not executed in this review.
