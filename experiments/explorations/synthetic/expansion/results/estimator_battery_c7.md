# C7 battery record — monotone extraction, and the close of the reasoning int/eq question

**Verdict: CLOSE — the reasoning half of the interaction/equality prize is
NEGATIVE at this corpus resolution.** The three-timescale structure is real
(model-independently confirmed in C5: real-vs-permuted ACF(4) gap 0.056);
it is **not extractable** at 287 docs × ~85 sentences. The next lever is
more/longer traces, not another estimator — per the briefing this was the
LAST estimator cycle, and no C8 estimator proposals follow. **This close is
a success of the loop**: three estimator generations produced a sound
negative with the failure mechanism identified at each step, instead of a
leaking benchmark.

Card frozen pre-build: `prereg/estimator-card-c7-monotone-extraction.md`
(commit "expansion C7: monotone estimator card + r4 mirror substitution
FROZEN (pre-build)"); implementation + battery + skeptic runner committed
pre-run (commit "expansion C7: mono estimator + battery + close-skeptic
runner (pre-run commit)"). Raw numbers:
[`estimator_battery_c7.json`](estimator_battery_c7.json). Skeptic (close
branch, `claude-fable-5`): **survived all 5 rubric items, no kills** —
`records/proof-operation-phase-runs-c7-close/`. C7 spend $0.19
(cumulative $11.01/$25).

## The candidate worked — that is what makes the close credible

`seg_hier_categorical_mono` (deconvolve-first, shrink in deconvolved
space; λ=0 reproduces r3 to machine epsilon, λ=1 collapses every segment
to the deconvolved doc fixed point):

- **Monotonicity pre-check: PASS, emphatically.** Generated ACF(4) vs λ on
  the real streams: 0.134 → 0.117 → 0.111 → 0.102 → 0.091 → 0.084,
  Spearman ρ = −1.00, every adjacent step decreasing, span 3.3× the
  required 5·SE_pool bound. The C6 non-monotonicity (finding 1) is fixed:
  shrinkage as the LAST operation interpolates cleanly.
- **Sensitivity restored where C6 failed it:** gate 3 (strong truth)
  PASSES robustly (err 0.0178 vs bound 0.0220 − 2SE; data-adaptive
  λ = 0.19) where C6-cal failed at 25%; weak-regime winner's-curse
  cancellation preserved (battery 4: raw +38% → mono 4%).

## Why it still closes (fork conditions, both firing independently)

1. **λ\*_real = 0.906 > 0.85 (inert threshold).** The in-loop null
   calibration on the real material — coarse scan + bisection, procedure
   followed exactly (0.875 FAIL / 0.90625 PASS, width 0.031 ≤ 0.05, upper
   endpoint) — lands at ~91% shrinkage: the null-clean extraction level of
   the real streams is ≈ nothing (retained u-space contrast < 10%).
2. **Gate 1 FAIL, robust (> 2·SE at R = 6):** fit on run-permuted real
   streams, the estimator's own calibration demands λ = 1.0 — and even
   that fully-inert limit inserts +0.0181 ACF(4) (tol 0.0140). Sharpest
   single fact of the cycle: mono's λ=1 keeps the raw fit's tilt weights
   (tilt_seg ≈ 1), so every jump targets the deconvolved doc propensity
   rather than mixing with the global chain — and on permuted real
   streams that *family-level* jump law alone over-generates mid-lag
   structure. The hallucination floor of this family on this data sits
   ABOVE the null tolerance before any segment layer is even added.
3. **Gate 2: conservative FAIL at R = 24** (0.0367 vs bound 0.0351,
   persistently inside the ±2·SE zone through the full escalation) — the
   variance-aware rule adopted at the C6 review doing exactly its job: a
   boundary case is decided conservatively, not by a seed flip.

Battery 5 (report): mono retains 41% of segment contrast at 0.034 mean
strong-toy error — the bias-variance trade is real but no longer the
binding constraint; the binding constraint is the family's null floor on
this corpus (point 2).

## The three-generation arc, closed

| cycle | estimator | outcome | what it established |
|---|---|---|---|
| C5 | raw DP + deconvolution | ABORT (insertion control) | structure real; estimator over-extracts (+0.018/+0.039 on null) |
| C6 | shrink-then-deconvolve; per-doc quantile deflation | empty passing set | shrinkage non-monotone (deconvolution re-amplifies); deflation leaks via tails; both cancel weak-regime curse |
| C7 | deconvolve-first, shrink in u-space | monotone ✓, sensitive ✓ — **and honestly calibrates to inert on real data** | the null-clean extraction window on this corpus is empty *even for a well-behaved estimator*; the family's inert limit itself sits above the null tolerance |

**Close statement (LEDGER C7):** the reasoning-trace interaction/equality
cell closes NEGATIVE at this corpus resolution. Reopening requires new
data — more/longer reasoning traces, so that per-segment null composition
fluctuations (which shrink with segment sample size) drop below the
measured contrast — not estimator work.

_Recorded-by: claude-fable-5 (runpod agent, Cycle 7 / expansion-c7
briefing), 2026-07-23. Card and scripts committed before execution; fork
followed verbatim; skeptic verdict persisted raw pre-parse, no kills._

## Review (2026-07-23, mac-local) — APPROVED; the CLOSE stands

Freeze order proven (card → scripts pre-run → close, 13:56 → 14:00 →
14:08); both close-conditions fired independently and the fork was
followed without discretion; the variance-aware rule adopted at the C6
review decided the gate-2 boundary case exactly as intended. The
sharpest fact — the family's fully-inert limit alone over-generates
mid-lag on permuted real streams (+0.018 > 0.014) — makes this close
robust to any future estimator inside this family: the null floor, not
extraction skill, binds. The three-generation arc (C5 over-extraction →
C6 non-monotonicity → C7 honest inert calibration) is the loop working
as designed. **The int/eq prize ends half-claimed: text POSITIVE
(stage-6 #3b) / reasoning NEGATIVE-at-resolution. Reopening requires
more/longer traces — logged as a data lever, not a cycle.** Spend $0.19
verified.
