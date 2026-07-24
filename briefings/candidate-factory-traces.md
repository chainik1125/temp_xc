---
status: active
created: 2026-07-24
for: runpod-b
venue: runpod (32C CPU)
---

# Candidate factory, trace corpus — QUANTITY MODE (Han directive)

**You are `runpod-b`.** Program directive (Han, 2026-07-24 evening):
**quantity over quality — produce as many screen-ready case-study
candidates as possible.** The economics: runpod-d/e already hold the
Ward base+distill caches (17 layers), so any candidate with
exact-computable labels on the Ward token grid costs YOU a label build
and THEM a minutes-scale probe run. Your job: a BATCH of label bundles,
each in the exact format the frozen `problib` screen stack consumes
(same npz + balanced-manifest conventions as
`labels/ward_lambda.npz` / `proofops.npz`), so the GPU pods screen
them unmodified. **Bundles wanted by Saturday morning PT; ship
incrementally (one LOG line per bundle as it lands), not as one drop.**

Discipline that survives quantity mode: builders committed before
outputs; a draft mini-card per candidate (regime framing, predicted
T-pattern, falsifier — the running agent freezes); and the
**label-side triage** per candidate, which is your kill authority: a
candidate whose label is well-predicted from the current token's
identity alone, or from position, FAILS TRIAGE — record one LOG line
and do not ship the bundle (a free kill is a win; it saves GPU
minutes). All zero-API, exact labels only.

## The batch (priority order — all on the Ward/R1-Distill token grid)

1. **Self-correction marker intensity λ̂_sc (top prior — the winner's
   family on a new event stream).** Events = a FROZEN marker list
   ("wait", "actually", "hmm", "no,", "let me re", "I made an error",
   … — freeze the list in the card before computing anything); labels
   = the same exponential-kernel λ̂ machinery as `ward_lambda`
   (kernel-only λ̂_hist primary, per the position-floor lesson).
   Triage must show the label is NOT readable from the current token
   (the marker tokens themselves must be masked/excluded from probe
   rows — state the masking rule in the card).
2. **Question-rate intensity.** Events = "?" sentence endings; same
   machinery, same masking discipline.
3. **Operation-class run-rates ×2.** Kernel-smoothed rate of
   `verification-check` sentences, and of `case-enumeration`
   sentences, from the committed proofops 5-class labels — the
   intensity (regime-2) face of the proofops latent, distinct from the
   killed-ish tir contrast. Current-sentence tokens excluded (they
   read `op` ambiently — the anchor lesson).
4. **Verbosity LEVEL (trailing mean sentence length).** Level primary,
   slope secondary — the hedging-LEVEL lesson (levels are
   aggregation-recoverable; slopes collapse to anchor − mean).
5. **Window redundancy rate.** Fraction of window tokens whose bigram
   occurred earlier in the trace — an exact, order-free aggregation
   latent; triage carefully (repetition detection was regime-1 —
   the RATE over a window may differ; if triage says converted, kill
   free and log).

Per bundle: labels npz + balanced manifests + shuffled-window null +
triage stats JSON + `<name>/CARD_DRAFT.md`. Aggregation-framed
(shuffle-immunity as mechanism receipt) per the program decision —
regime-2 wins against per-token-decoded T-SAE are accepted.

## Acceptance gate — stop for review

≥ 3 shipped bundles (or their honest triage kills); LOG line per
bundle/kill; STATUS rewritten; no reviewer/meeting quotes. Briefing
stays until mac-local review.
