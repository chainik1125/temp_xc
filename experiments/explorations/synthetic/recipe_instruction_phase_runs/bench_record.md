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
