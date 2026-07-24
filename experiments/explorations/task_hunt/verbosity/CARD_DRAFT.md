# DRAFT mini-card — verbosity LEVEL v̄ (+ slope secondary) (factory candidate 4)

**Status: DRAFT (runpod-b, `briefings/candidate-factory-traces.md`
item 4). Everything in § Frozen was fixed BEFORE any label was
computed. The running agent freezes its own screen card.**

Bundle: `../labels/build_verbosity.py` → `../labels/verbosity.npz` +
`../labels/verbosity_stats.json`. Same Ward grid / manifest / split
conventions as `sc_lambda`; deltas only below.

## The candidate logic (the hedging-LEVEL lesson applied)

Trailing mean sentence LENGTH is the style register the trace is
currently writing in — terse symbol-pushing vs expansive prose.
Levels are aggregation-recoverable (the hedging-LEVEL lesson: window
means carry them; slopes collapse to anchor − mean), so LEVEL is
primary and slope ships only as the disclosed secondary. Regime-2
aggregation framing; shuffle-immunity as the mechanism receipt.
Zero-API: the "event" is a token count, no judge anywhere.

## Frozen (deltas from sc_lambda; before computing anything)

- **Per-sentence quantity**: len_s = number of IN-SPAN tokens of
  sentence s under the Ward tokenizer (the deployment clock, same
  counting as the committed proofops clock bridge).
- **Labels**: `vlevel` (primary) = unweighted mean of the previous
  min(i, 8) sentence lengths, current sentence EXCLUDED, NaN below 4
  previous sentences (`factory_lib.trailing_mean_prev`). `vslope`
  (secondary) = OLS slope over the same trailing lengths
  (`trailing_slope_prev`). Unweighted (not the exponential kernel):
  a LEVEL is a plain trailing mean by definition — the kernel family
  is for event intensities.
- **Independent triage** per label (frozen thresholds); a failing
  label's manifests are dropped and logged; npz ships iff ≥ 1 passes.
- **Masking**: none beyond valid/in-span — no current-token property
  reveals PREVIOUS sentences' lengths. `cur_sent_len` (the current
  sentence's total length — partially future-peeking by construction)
  ships as the disclosed ambient control, the `is_bt` analogue.
- **Null**: within-trace permutation of the sentence lengths
  (seed 104 + trace_idx), both labels recomputed from the permuted
  lengths.
- **Build-sanity**: `vlevel` must show autocorrelation —
  cur_sent_len monotone in the vlevel bin (verbose stretches stay
  verbose), else kill. `vslope` has NO monotonicity gate: trend
  continuation vs mean reversion is an empirical outcome, not a
  sanity requirement (stated here so the asymmetry is not post-hoc).
- **Evidence baseline**: in-window sentence-START count
  (`tok_in_sent = 0` flags) at T ∈ {8, 16, 32} — boundary density is
  the visible proxy for mean length (T / count ≈ mean length of the
  window's sentences).

## Predicted T-pattern + falsifier

At T = 32 a window holds ~2 median sentences — the boundary-count
proxy is coarse (0–4 boundaries), so unlike the marker candidates the
visible-evidence line should stay MODEST across the ladder while the
label's support (8 sentences ≈ 128 tokens) lies mostly out of window.
Prediction: window-MEAN recovery rises with T and clears the evidence
line if the register is carried state; per-token near the triage
floor. Falsifier: recovery tracking the boundary-count line (probe is
counting sentence starts, not reading register), or real ≈
length-shuffle null (trace-ambient verbosity, not the local level).
Screen kill rule: as sc_lambda, per label.
