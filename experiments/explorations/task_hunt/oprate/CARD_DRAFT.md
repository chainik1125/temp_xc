# DRAFT mini-card — operation-class run-rates λ̂_ver / λ̂_case (factory candidate 3, ×2)

**Status: DRAFT (runpod-b, `briefings/candidate-factory-traces.md`
item 3). Everything in § Frozen was fixed BEFORE any label was
computed. The running agent freezes its own screen card.**

Bundle: `../labels/build_oprate.py` → `../labels/oprate.npz` +
`../labels/oprate_stats.json`. One npz, TWO independently-triaged
labels: `rate_ver` (verification-check, class 3) and `rate_case`
(case-enumeration, class 2), both kernel rates over the committed
proofops 5-class per-sentence labels
(`synthetic/expansion/records/proof-operation-phase-runs/labels.json`,
judged sentence-in-isolation). Same Ward grid / manifest / split
conventions as `sc_lambda`.

## The candidate logic

This is the intensity (regime-2) face of the proofops latent — NOT the
killed-ish time-in-run contrast. `op` itself is per-sentence readable
by construction (regime-1, the committed anchor); the RATE of an
operation class over recent history is an aggregation quantity the
current sentence does not carry. Ops come in runs (boundary rate 0.54),
so the rate self-excites like the marker streams.

## Frozen (deltas from sc_lambda; before computing anything)

- **Events (×2)**: e_ver,i = [op_i = 3], e_case,i = [op_i = 2];
  UNLABELED sentences (judge gap, ~14% of tokens) are NaN events — the
  kernel label is NaN whenever ANY of the previous min(i, 8) sentences
  is unlabeled (no silent imputation).
- **Kernel**: identical frozen family (exponential τ = 3, K = 8,
  causal, normalized, history guard i ≥ 4, kernel-only).
- **Masking rule (the anchor lesson, per label)**: probe rows are
  excluded from a label's manifests iff the CURRENT sentence's op is
  the label's event class OR the current sentence is unlabeled (it
  could be the event class). Tokens of OTHER classes stay: reading
  "algebra now" ambiently is fine, the label is about history. The
  full `op` grid ships so the screen can build stricter masks.
- **Null**: ONE within-trace permutation of the labeled op entries
  among labeled positions (seed 103 + trace_idx), BOTH null event
  streams derived from the same permuted sequence — preserves each
  trace's class composition and the ver/case coupling.
- **Independent triage**: each label passes or dies alone (frozen
  thresholds in `factory_lib`). A failing label's manifests are
  dropped and the kill logged; the npz ships iff ≥ 1 label passes.
- **Binning / build-sanity**: as sc_lambda (zero_split fallback
  expected — both classes are sparse: ~7–8% of sentences).
- **Disclosed correlations** (stats JSON): corr(rate_ver, rate_case)
  (composition constraint — rates compete for the same sentences);
  corr(rate_ver, λ̂_sc) ("let me verify / double-check" sits in BOTH
  the marker list and the verification class — expected positive,
  stated up front).

## Predicted T-pattern + falsifier

Clock: median 16 tokens/sentence ⇒ kernel support ~128 tokens; ops run
in blocks so adjacent-sentence autocorrelation is high. Prediction:
window-MEAN recovery rises with T and (unlike the marker candidates)
leans on MID-window sentences — at T = 32 the window typically spans
2 sentences, mostly non-event, so the visible-evidence line
(in-window event-class-token count) should be LOW where the kernel
mass is old; a probe beating it implies carried rate state.
Falsifier: real ≈ op-shuffle-null recovery (trace-ambient class
composition, not history), or probes never beating the visible-
evidence line at matched T. Screen kill rule: as sc_lambda, per label.
