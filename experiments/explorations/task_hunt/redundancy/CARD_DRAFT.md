# DRAFT mini-card — window redundancy rate ρ̂ (factory candidate 5)

**Status: DRAFT (runpod-b, `briefings/candidate-factory-traces.md`
item 5). Everything in § Frozen was fixed BEFORE any label was
computed. The running agent freezes its own screen card. The briefing
itself flags this candidate as the triage-risky one ("repetition
detection was regime-1"); the frozen thresholds decide, and a kill
here is a free win.**

Bundle: `../labels/build_redundancy.py` → `../labels/redundancy.npz` +
`../labels/redundancy_stats.json`. Same Ward grid / manifest / split
conventions as `sc_lambda`; deltas only below.

## The candidate logic

Round 1 killed per-token repetition DETECTION as converted (the model
marks repeated tokens per-token — regime 1). The RATE over a trailing
window is the aggregation face of those same converted marks: an
exact, order-free window aggregate (mean of per-token flags) — the
accepted regime-2 framing, where a window-MEAN win against
per-token-decoded T-SAE counts. The label needs no judge and no
lexicon: pure token arithmetic.

## Frozen (deltas from sc_lambda; before computing anything)

- **Per-token event**: b_t = 1 iff the bigram ENDING at t occurred
  anywhere earlier in the trace (`lib.delta_prev_ngram(ids, 2) > 0`,
  original-token coordinates; t = 0 has no bigram ⇒ 0). "Earlier in
  the trace" per the briefing — NOT distance-limited.
- **Label**: ρ̂_t = mean of b over the previous **W = 32** tokens,
  current token EXCLUDED (`trailing_rate_prev`), NaN for t < 32.
  W matches the top of the screen T ladder: at T = 32 the label's
  support is fully visible (the pure-aggregation regime), at small T
  only a suffix is.
- **Masking rule (event-token principle, as sc_lambda)**: rows with
  b_t = 1 (`is_rep`) are excluded from every manifest — the current
  token being ITSELF a repeat is the converted regime-1 face. The
  masked fraction is disclosed in the stats (expected large; if the
  remaining rows can't fill balanced manifests, build-sanity kills).
  `is_rep` ships as the ambient control.
- **Null**: within-trace TOKEN shuffle (seed 105 + trace_idx), b and
  ρ̂ recomputed on the shuffled sequence — the frequency-only null
  (the replag `shuffled_doc_null` convention): what redundancy
  structure survives when order is destroyed but the trace's token
  composition is kept.
- **Triage**: standard frozen thresholds. Stated kill expectation:
  POSITION is the likely failure face (ρ̂ grows mechanically with
  history length — the briefing's warning); token identity second
  (frequent tokens repeat). If either crosses its bar, the kill line
  goes to the LOG and no npz ships.
- **Binning / build-sanity**: as sc_lambda (terciles expected — the
  rate is continuous); is_rep-by-bin monotone gate computed pre-mask.

## Predicted T-pattern + falsifier

Evidence baseline = in-window b count (`trailing_count_incl`) at
T ∈ {8, 16, 32}: at T = 32 it contains the label's whole support, so
its AUC should approach 1 — DOCUMENTED, not a bug: the candidate's
screen question is whether ACTIVATION window-MEANs approach that
ceiling (aggregating the model's own converted repeat-marks) while
per-token (novel-token rows only) stays near the triage floor.
Prediction: recovery rises steeply with T toward the fully-visible
ceiling; flat ≈ shuffled ≈ mean (order-free). Falsifier: real ≈
token-shuffle null (frequency composition, not repetition structure),
or per-token rows already high (the regime-1 conversion reappearing
through the mask — kill). Screen kill rule: as sc_lambda.
