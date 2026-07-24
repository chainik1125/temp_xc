# DRAFT mini-card — question-rate intensity λ̂_q (factory candidate 2)

**Status: DRAFT (runpod-b, `briefings/candidate-factory-traces.md`
item 2). Everything in § Frozen was fixed BEFORE any label was
computed. The running agent freezes its own screen card.**

Bundle: `../labels/build_qrate.py` → `../labels/qrate.npz` +
`../labels/qrate_stats.json`; shared frozen logic
(`../labels/factory_lib.py`) under `tests/test_factory_labels.py`.
Same Ward grid, manifest and split conventions as `sc_lambda`
(see that card); only the deltas are stated here.

**Sibling disclosure**: runpod's broad-corpus factory lists a
question-rate candidate on fineweb (its B3/B4). Different corpus,
different regime question — self-posed questions in R1 reasoning
traces (this bundle) vs interrogative density in web prose. Neither
substitutes for the other; cross-cite when either screens.

## The candidate logic

Self-questioning is the interrogative face of deliberation: R1-style
traces pose sub-questions ("Is n even? What if x = 3?") in bursts.
The latent = the same kernel intensity machinery as `sc_lambda` on a
DIFFERENT frozen event stream — exact, zero-API, judge-free. Regime-2
aggregation framing; shuffle-immunity as the mechanism receipt.

## Frozen (deltas from sc_lambda; before computing anything)

- **Event**: sentence's last non-whitespace char is "?"
  (`factory_lib.sentence_events_question`, applied to the exact judged
  char spans of `full_response`).
- **Kernel**: identical frozen family — exponential τ = 3, K = 8,
  causal, normalized, history guard i ≥ 4, kernel-only.
- **Masking rule**: probe rows whose current token's char span
  contains ANY "?" character are excluded from every manifest
  (`is_q_tok`) — mid-sentence "?" included, not only sentence-final.
  "?" tokens in the trailing window remain legitimately visible
  (regime-2 aggregation). `is_q` (current sentence is a question)
  ships as the ambient control.
- **Null**: within-trace event shuffle, seed 102 + trace_idx.
- **Binning / build-sanity / triage kill thresholds**: identical to
  sc_lambda (frozen in `factory_lib`; FAIL ⇒ stats JSON ships as the
  kill receipt, npz does not).
- **Disclosed correlations** (in stats JSON): corr(λ̂_q, λ̂_sc) —
  questioning and self-correction plausibly co-bursty; corr(λ̂_q,
  ward λ̂_hist).

## Predicted T-pattern + falsifier

Same clock arithmetic as sc_lambda (kernel support ~128 tokens; screen
ladder T ∈ {2 … 32} sees 0.1–2 sentences). Prediction: window-MEAN
recovery rises with T; per-token near the token-identity triage floor;
flat ≈ shuffled ≈ mean. `visible_evidence_auc` (in-window "?"-token
count) at T ∈ {8, 16, 32} is the line activation probes must beat at
matched T. Falsifier: probes only tracking visible "?" tokens, or
real ≈ event-shuffle-null recovery (trace-ambient rate, not local
history). Screen kill rule: as sc_lambda.
