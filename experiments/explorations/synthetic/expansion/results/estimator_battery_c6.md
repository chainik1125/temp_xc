# C6 battery record — calibrated segment-composition extraction

**Verdict: NEITHER candidate selected → NO r4 run** (the frozen card's
"neither passes" branch; briefing outcome 3 — structure present but
unextractable by these candidates). Card:
[`prereg/estimator-card-c6-segment-extraction.md`](../prereg/estimator-card-c6-segment-extraction.md)
(frozen `be8e2b6d`, pre-build). Raw numbers:
[`estimator_battery_c6.json`](estimator_battery_c6.json); λ-scans:
[`estimator_battery_c6_lamscan.json`](estimator_battery_c6_lamscan.json).
Zero API spend (committed r3 labels + synthetic toys only; cumulative
expansion spend unchanged at $10.82).

## Gate results (frozen battery, run 2026-07-23)

| battery | r3 raw | A `_cal` | B `_deflate` |
|---|---|---|---|
| 1 — null-safety, run-permuted real streams (tol mi2 0.0072 / acf4 0.0140) | FAIL (0.0156 / 0.0364) | **PASS — but at λ\*=1.0, zero extraction** (0.0025 / 0.0000) | FAIL (0.0115 / 0.0160) |
| 2 — heavy-dwell null, insertion ≤ 0.0351 | FAIL (0.0455) | FAIL (0.0401) | PASS (0.0128) |
| 3 — strong planted truth, acf4 err ≤ 20% | PASS (1%) | FAIL (25%, λ\*=0.4) | FAIL (35%) |
| 4 — weak planted truth (report) | **+34% overshoot** | 3% | 0.2% |
| 5 — variance penalty (report): mean acf4 err · retained contrast | 0.0033 · 1.00 | 0.0365 · 0.35 | 0.0360 · 0.45 |

Selection rule (frozen): gates 1–3, all → **passing set empty** → no
selection, no r4. The r3 ABORT stands as the reasoning int/eq verdict.

## What the battery established (each mechanism, with its evidence)

1. **Shrink-then-deconvolve is non-monotone — candidate A's knob is
   broken, not its concept.** On run-permuted real streams the generated
   ACF(4) barely responds to λ (0.105 at λ=0 → 0.085 at λ=0.9, all
   hallucinating above tol) then cliffs to 0.058 at λ=1.0: the C4
   self-consistency deconvolution *re-amplifies* any composition that is
   slightly off the doc marginal (the fixed point solves for the u whose
   mixed stationary reproduces the shrunk target — sharpening it back).
   The λ-grid therefore offers only "full extraction" or "none", and on
   real material the null-clean choice is **none** (λ\*=1.0). See
   `real_material` in the λ-scan JSON.
2. **Even the inert λ=1 limit undershoots the permuted streams' own
   mid-lag structure** (syn 0.058 vs perm 0.070 with the in-loop seeds):
   the family's no-segment limit under-generates exactly as C4's
   `hier_categorical` did on real data. The null-clean extraction window
   between undershoot and hallucination is **empty** at the card's
   null-referenced tolerance.
3. **Per-doc quantile deflation leaks through its tails.** On real
   material candidate B collapses 75% of segments outright
   (mean deflation 0.79) and the surviving 25% — the segments above the
   75th-percentile null concentration — still carry +0.0115 MI(2) /
   +0.0160 ACF(4) hallucination. The a-priori tail-retention risk in the
   card is exactly what happened. (It does pass the synthetic heavy-dwell
   null, where run material is homogeneous.)
4. **Both candidates cancel the winner's curse in the weak regime — the
   regime the real streams live in.** Battery 4: raw r3 overshoots a weak
   truth by +34% (its selection bias dominating genuine weak signal)
   while A lands at 3% and B at 0.2%. Pinned as a permanent rail
   (`test_c6_calibrated_estimators_cancel_weak_regime_curse`). The cost
   is battery 5: 35–45% retained contrast and ~0.036 mean error on
   *strong* signal — the frozen gates demanded both regimes and neither
   candidate delivers both.
5. **Boundary verdicts are generation-noise brittle.** A's gate-2 FAIL is
   a seed flip at the tolerance boundary (in-loop seeds: insertion 0.022,
   PASS; measurement seeds: 0.0401 vs bound 0.0351, FAIL) — 3-replicate
   means move by ~0.01–0.02 at this magnitude. Recorded as-is under the
   frozen gate; a future card needs variance-aware margins at the
   boundary.

## Interpretation for the reasoning int/eq cell

The C5 diagnosis stands and sharpens: the three-timescale structure is
real (model-independent real-vs-permuted ACF(4) gap 0.056) but at this
corpus size (287 docs, ~85 sentences/doc, segments of ~10–20 positions)
segment-composition extraction cannot be simultaneously null-clean and
sensitive with these two calibration mechanisms — per-segment null
concentration fluctuations sit at the same scale as the genuine segment
contrast. This is a resolution/power limit of the extraction step, not a
family failure and not evidence against the structure.

## C7 direction (proposed, NOT frozen here)

- **Monotone calibration: deconvolve first, then shrink in deconvolved
  space** — shrink the *deconvolved* segment propensities u toward the
  deconvolved doc fixed point, so the generated composition interpolates
  monotonically between r3 and inert (finding 1 shows the current order,
  shrink-then-deconvolve, is what breaks monotonicity). Continuous λ by
  bisection once monotone.
- **Variance-aware in-loop margins**: replicate-count adaptive near the
  tolerance boundary (finding 5), so seed flips cannot decide verdicts.
- If a monotone estimator still finds λ\*≈inert on real material, the
  honest close is the briefing's NEGATIVE branch: the reasoning half of
  the int/eq prize is unreachable at this corpus resolution — the next
  lever would be more/longer traces, not another estimator.

_Recorded-by: claude-fable-5 (runpod agent, Cycle 6 / expansion-c6
briefing), 2026-07-23. Battery run exactly as frozen; no gate, tolerance,
or selection-rule was altered after seeing any result._
