# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-24 ~06:10 UTC — **ARM B COMPLETE, awaiting
mac-local review.**

## Who / where
H100 pod, `/workspace/temp_xc`, `/workspace/.agent_id` = `runpod-e`.
Task-hunt arm B (`briefings/task-hunt-b.md` — briefing left in place
per its acceptance gate until mac-local review). Git `runpod-e-agent`;
creds `store --file=/workspace/.git-credentials`; `HF_TOKEN` from
`/workspace/.tokens/hf_token` per command. Pull-rebase before every
push.

## FINAL STATE — all three arm-B candidates KILLED by sound screens
Verdicts + full reasoning in `experiments/explorations/task_hunt/LOG.md`
(one entry per candidate + the arm-B closure entry). One-line each:
1. **Repetition-lag Δ: KILL** — detection converted (regime-1) at
   every scale (gpt2 / gemma-2-2b / llama-8b); only order-residue is
   the lag VALUE, large in gpt2 (+0.11), thin at 2B/8B (+0.02).
   Figs: `task_hunt/replag/figs/`.
2. **Confidence trend: KILL** — window gap real and T-growing (distill
   mean 0.521→0.565 vs tok 0.468) but aggregation-carried; order
   receipt fails. Regime-2 re-card seed recorded in LOG.
3. **Emotional-instability onset: KILL** — paper replicated
   (escalation 0.36→4.91; κ 0.857 judge gate; ~$12 spend), but
   pre-onset anticipation is per-token-converted (0.856 AUC at D1-4,
   window never beats token) and escalation gap +0.03 < bar without
   shuffle collapse; detection anchor validates labels (0.867 tok).
Cross-cutting mechanism named in the LOG: conversion — mid-depth
residuals already summarize any per-token-traceable temporal latent.

## Hygiene at close
`run.py validate` OK; canonical leaderboard untouched (no Stage 2 ran
in arm B); all cards frozen pre-run; every amendment/escalation LOG'd
pre-use; no reviewer/meeting quotes in tracked files.

## Volume assets (persist for a repurposed pod)
- `/workspace/replag_caches/` — fineweb token+Δ caches, 3-model acts.
- `/workspace/conv_depth_caches/` — Ward stream + base/distill
  17-layer caches (rebuilt here; stats match committed reference).
- `/workspace/emo_caches/` — 600 gemma-3-12b-it rollouts, judge
  labels, acts (NOTE: activations stored × 1/64 — gemma-3 fp16
  saturation; index.json records act_scale).
- Models in HF cache: gpt2, gemma-2-2b, llama-3.1-8b base, r1-distill,
  gemma-3-12b-it.

## Proposed next tasks (runpod-e suggestions, 2026-07-24 — pending
## mac-local/Han review; ordered by value-per-hour)

1. **Stage-2 on the hedging-trend LEVEL (the candidate-2 seed) —
   needs a program decision first.** The screen showed a real,
   monotone-in-T, per-token-blind window gap (distill mean-probe
   0.52→0.57 vs tok 0.47) that is aggregation-carried. The strategy
   note in the research STATUS already claims regime-2 rates/trends
   suffice to separate TXC from per-token-decoded T-SAE *without*
   order. If the program accepts an aggregation-framed win (shuffle
   IMMUNITY disclosed as the mechanism receipt instead of shuffle
   collapse), this is the cheapest path to a real-data T-scaling
   figure: labels + Ward caches are already on this volume, Stage-2
   protocol is fixed, ~1 day. Requires a FRESH card (do not reuse the
   killed one as confirmation).
2. **Early-layer post-hoc addendum on existing caches (~2-3 h, zero
   new data).** Every screen ran one mid-depth layer, and the shared
   mechanism was conversion-by-mid-depth. Cached-but-unscreened
   alternates exist for all three replag models (gpt2 hs4, gemma hs8,
   llama hs8) and all 17 Ward capture points. Test directly: does the
   lag-VALUE order gap and/or the slope8 aggregation gap GROW at
   pre-conversion depths? Deliverables: g_order(ℓ) for lag4,
   g_agg(ℓ) for slope8. Either direction is a finding about where
   temporal structure lives before the model summarizes it — and it
   sharpens the depth story the conversion-depth exploration opened.
3. **A small-model order cell (new card).** The only genuine
   order-carried window advantage found anywhere in arm B is the
   lag-VALUE readout at gpt2 scale (+0.11, shuffle-collapsed; thin at
   2B/8B). If the paper wants a real-data ORDER receipt, it may live
   at small scale by necessity — conversion capacity is what destroys
   it. Proposal: freeze a card for lag-value at gpt2 (optionally a
   pythia ladder for a clean scale curve), screen early+mid layers,
   Stage-2 panel at gpt2 (cheap). Also yields a quotable
   conversion-vs-scale curve for the rebuttal.
4. **Process fix for the next hunt round: per-token-first triage.**
   All three kills shared one signature visible within minutes: a
   HIGH per-token ceiling (0.7-0.97). Adopt a mandatory cheap
   pre-screen — per-token linear probe only — and require per-token
   ≈ chance-ish before a candidate earns the full window grid. Fold
   into the task-hunt briefing conventions.
5. **Anti-conversion candidate class (design note for round 2).**
   Conversion happens when the latent helps next-token prediction.
   Candidates should therefore target latents with NO generative
   training signal: e.g., source identity / time-since-switch in
   two-document interleaved real text (a real-data analog of
   colored_sources/multilane, with lexical-overlap controls), or
   externally-annotated states orthogonal to surface form. These are
   the latents a model plausibly never converts — exactly where a
   window must win.

Sibling arms: runpod-d (arm A) and runpod-c (em-redo) own their own
state; this pod is idle and can absorb any of 1-3 immediately —
caches for all of them are already on this volume.
