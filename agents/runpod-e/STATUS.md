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

## If picked up next
Wait for mac-local review of the LOG; candidate follow-ups that
survive review: the candidate-2 regime-2 aggregation seed (needs a
fresh card + relaxed-order program decision). Sibling arms: runpod-d
(arm A) and runpod-c (em-redo) own their own state.
