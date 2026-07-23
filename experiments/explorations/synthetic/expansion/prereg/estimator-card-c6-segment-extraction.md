# Estimator card — C6 calibrated segment-composition extraction

> **Frozen 2026-07-23 (runpod agent, Cycle 6), committed BEFORE any estimator
> code, battery run, or real-data measurement.** Never revised — dated
> amendments only. This card governs the C6 mandate (briefing
> `expansion-c6.md`): fix the `seg_hier_categorical` EXTRACTION ESTIMATOR,
> not the model family, for the reasoning int/eq cell
> (`proof-operation-phase-runs`).

## The measured problem (C5 record, frozen numbers)

`fit_seg_hier_categorical` (r3) over-extracts: fit on run-permuted real
streams — data with **no** segment-scale structure by construction — its
generated output carries hallucinated lag structure:

| moment | perm held-out | null-fit synthetic | hallucination | tol (real-magnitude) |
|---|---|---|---|---|
| mi[lag2] | 0.0326 | 0.0501 | **+0.0176** | 0.0130 → FAIL |
| acf[lag4] | 0.0709 | 0.1094 | **+0.0385** | 0.0255 → FAIL |

Mechanism (C5 diagnosis): the changepoint DP *selects* boundaries where
random fluctuation creates apparent composition contrast; the *same* data
then estimates the within-segment composition (winner's curse); the
self-consistency deconvolution sharpens the spurious concentration further.
The three-timescale structure itself is real (real-vs-permuted acf[lag4]
gap 0.056 ≫ tolerances) and the family closes lag-2–8 — the estimator is
the gap. Target (LEDGER C6): cut hallucination ~2–3× at preserved
sensitivity.

## Candidates (exactly two — depth over breadth)

Both keep the r3 model family, DP segmentation, deconvolution, tilt-MLE,
and generator **unchanged**; each replaces only how segment compositions
are *estimated* from the DP's segments. `fit_seg_hier_categorical` (r3) is
never modified — new fit functions, new MENU keys, same params schema and
generator.

### A. `seg_hier_categorical_cal` — null-calibrated global shrinkage

**Mechanism.** Every observed segment composition is shrunk toward its
doc marginal before deconvolution:

    pi_i(λ) = (1 − λ)·pi_obs_i + λ·pi_doc ,   λ ∈ [0, 1]

with λ **global** (one value per fit). λ = 0 is the r3 estimator; λ = 1
makes every segment table equal the doc marginal — the segment layer is
provably inert (the mirror degenerates to doc-level `hier_categorical`
behavior, which run-permutation preserves). Tilt-MLE and deconvolution
run on the shrunk compositions.

**Calibration principle (the insertion control moved in-loop).** λ* is the
smallest λ on the grid {0.0, 0.1, …, 1.0} such that the estimator, fit at
λ on `run_permuted_streams(train)` (the preregistered C5 null,
`PERM_NULL_SEED`), round-trips BOTH preregistered r3/r4 gate-8 moments
(`mi[lag2]`, `acf[lag4]`) to within the **null-referenced uniform
tolerance**

    |syn_null(λ) − perm_value|  ≤  max(0.20·|perm_value|, floor_m),
    floors: mi 0.003 / acf 0.01

— the frozen C3 uniform relative-tolerance rule evaluated at the *null's
own magnitude*. This is strictly tighter than the recorded insertion
control (whose tolerance sits at the larger real-data magnitude).
`syn_null(λ)` is averaged over 3 generation replicates with fixed seeds
(generation-noise suppression). If **no** λ on the grid passes, the fit
returns λ = 1 with an `uncalibratable: true` flag (honest degeneration;
the real-data gate-8 will then fail exactly as C4's doc-level mirror did).

**Why the gate-8 moments remain non-fitted.** The in-loop constraint sees
those moments ONLY through their values on run-permuted (segment-free)
streams — the run/dwell/doc floors. The held-out real values never enter
any objective; shrinkage moves expressed segment structure monotonically
*toward the no-segment floor* and cannot tune the mirror toward the real
moment. The real-data gate-8 comparison stays a genuine out-of-fit check:
whether the extraction that survives null-calibration still carries the
real streams' lag-2–8 structure is exactly the open question. (The r4
skeptic is directed to scrutinize this argument.)

**Known risk (a priori).** A global λ shrinks every doc and segment
uniformly; if the null-cleanliness bar demands heavy shrinkage, genuine
segments pay the same tax and sensitivity dies (the failed-campaign
failure mode). The battery measures this before any real-data run.

### B. `seg_hier_categorical_deflate` — per-doc length-matched null deflation

**Mechanism ("subtract the run-permuted estimate", composition level).**
Per doc: run the DP on the real stream → segments {(runs_i, pi_obs_i)};
build R = 20 run-permuted replicas of the SAME doc (seeds `1000+r`,
independent of `PERM_NULL_SEED`); run the same DP on each replica and
collect null segments with their concentrations D = KL(pi ‖ pi_doc).
Bin null segments by run count ({4–5, 6–8, 9–12, 13+}; a bin with < 8
null segments merges into its neighbor below, bottom bin merges upward)
and take the **75th percentile** D_q(bin) per bin. Deflate each real
segment to its *excess* concentration:

    excess_i = max(0, D(pi_obs_i) − D_q(bin(runs_i)))
    pi_i     = (1 − s_i)·pi_obs_i + s_i·pi_doc,  s_i solving
               D(pi_i ‖ pi_doc) = excess_i   (bisection; D monotone in s)

excess = 0 ⇒ pi_i = pi_doc exactly. Deconvolution + tilt-MLE on the
deflated compositions. No global knob.

**Calibration principle.** Under run-exchangeability the real doc is
distributionally one of its own permuted replicas, so D(pi_obs_i) is a
draw from the null concentration distribution: per segment,
P(excess > 0) ≤ ~25% and the retained excess is the thin tail above the
75th percentile — the quantile (not the mean) is the a-priori guard
against retaining above-average null fluctuations (the winner's-curse
tail that a mean-subtraction would keep). Under genuine segments,
D(pi_obs) ≫ D_q ⇒ deflation removes only the null share.

**Known risk (a priori).** DP segment-size mismatch between real and
permuted output killed the C5 permutation-matched split-half variant;
conditioning on run-count bins is the mitigation, and whether it suffices
is what the battery decides. Residual tail-retention (the 25% of null
segments above D_q) may still leak.

## Verification battery (ALL before any real-data r4 run)

Committed as `expansion/estimator_battery_c6.py` (writes
`expansion/results/estimator_battery_c6.json` + summary) plus permanent
pytest rails in `tests/test_expansion_harness.py`. Gates 1–3 are
pass/fail per candidate; 4–5 are measured and reported. **Zero API spend
— committed r3 labels and synthetic toys only.**

1. **Null-safety on real material.** Fit each candidate on
   `run_permuted_streams` of the full committed labeled streams (r3
   record `labels.json`); generate at the real lengths; for BOTH moments,
   insertion ≤ the null-referenced tolerance above. (For A this is
   near-tautological on train — it is the constraint — and still binds as
   an end-to-end check; for B it is the primary null test.)
2. **Null-safety on the harness heavy-dwell null** — the committed toy
   the r3 estimator provably fails (`test_seg_mirror_insertion_control`):
   insertion on acf[lag4] ≤ max(0.2·acf_null4, 0.01). The r3 estimator's
   FAIL on this toy stays asserted (regression rail); each candidate must
   PASS it.
3. **Planted-truth sensitivity (strong).** On `_seg_hier_truth()` streams
   (the committed three-timescale toy): round-trip
   |acf_syn[4th lag] − acf_real| ≤ 0.20·acf_real — the same bound the r3
   estimator meets. Calibration must not drown the genuine signal.
4. **Planted-truth sensitivity (weak) — report only.** Same toy with
   concentration 0.9 → 0.60 (off-symbols uniform) and tilt_seg 0.85 →
   0.70: report round-trip errors for r3-raw vs A vs B. Near-threshold
   sensitivity loss is the variance penalty made visible; no a-priori
   bound is honest here.
5. **Variance penalty — report only.** Across 5 replicate data draws of
   the strong toy: mean |round-trip acf[lag4] err| and the retained
   contrast fraction Σ len·D(pi_est) / Σ len·D(pi_obs) for each
   candidate vs r3-raw.

**Selection rule (frozen).** Among candidates passing gates 1–3, select
the one with the lower mean round-trip |acf[lag4] error| across batteries
3+4; tie (< 10% relative difference) → A (simpler, directly constrained).
If exactly one passes, it is selected. If neither passes, C6 records the
estimator family as uncalibratable under this card, no r4 runs, and the
outcome is written up as the briefing's third branch (structure present
but unextractable — C7 direction or close). The battery JSON + selection
are committed BEFORE the r4 run.

## r4 (real traces) — after battery + selection commit

- Record `proof-operation-phase-runs-r4`; dated amendment in
  `proof-operation-phase-runs.md` (same commit as this card). C3 labels +
  validation reused verbatim; signature/gate/mirror/skeptic fresh.
- Gate-8 UNCHANGED: `mi[lag2]` + `acf[lag4]`, uniform ±20% relative
  tolerance at real magnitude, floors 0.003/0.01, ALL must pass.
- Recorded insertion control UNCHANGED (real-magnitude tolerance, both
  moments, fit on permuted train vs permuted held-out) — for A it should
  pass by construction; it remains the recorded, out-of-sample check.
- Skeptic (expansion rubric, `claude-fable-5`) on PROCEED only; raw
  verdicts persisted pre-parse; never re-rolled. Spend under the C6 $10
  cap, metered to `expansion/results/spend.json`.
- Outcomes, all acceptable (briefing): PASS ⇒ the reasoning int/eq card
  graduates to SPEC (stage-6 later, NOT this session); FAIL with the
  structure gone ⇒ NEGATIVE, close the card; FAIL with structure present
  but unextractable ⇒ record + C7 direction, stop.

_Frozen-by: claude-fable-5 (runpod agent, Cycle 6), before any C6
implementation._
