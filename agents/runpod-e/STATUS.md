# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-24 (PRE-COMPACT handoff. Round-2 briefing
received and read; NO r2 work started yet — next context begins it.)

## Who / where / env (verified working this session)
H100 pod, `/workspace/temp_xc`, `/workspace/.agent_id` = `runpod-e`.
Git `runpod-e-agent`, creds `store --file=/workspace/.git-credentials`
(token from `/workspace/.tokens/gh_token`); `export HF_TOKEN=$(cat
/workspace/.tokens/hf_token)` per command (gemma gated); anthropic key
at `/workspace/.tokens/anthropic_key` (judge.py reads it directly).
Pull-rebase before EVERY push (shared `arxiv` branch). venv:
`.venv/bin/python` (torch 2.8 cu128, transformers 5.7, datasets 4.8.5,
anthropic 0.97).

## MISSION NOW: `briefings/task-hunt-r2-e.md` (the orchestrator SPLIT
## the joint r2 briefing per pod — read MINE first; runpod-d has
## task-hunt-r2-d.md). Results by SATURDAY morning PT; check-in Sun
## 10:00 PT.
Round 1 (arm B, three sound kills) is REVIEWED & APPROVED — binding
review notes live in `experiments/explorations/task_hunt/LOG.md` and
methods in `experiments/explorations/task_hunt/RECORD.md` (READ BOTH
in the new context before any run; I have not read the post-review
LOG additions or RECORD.md yet). New binding conventions:
per-token-first triage (my screens already satisfy it) + the depth
sweep as the WHY-diagnostic when per-token is high. Briefing deltas vs
my earlier notes: (a) item-2 requires BLIND directional predictions
committed BEFORE any cell (cand-3 depth-sweep precedent — see
runpod-d's committed script+prediction pair for the format); (b)
runpod-b is shipping a DRAFT Stage-2 card (`hunt-support-stats.md`
item 4) — sharpen and freeze MY OWN, don't wait on it; (c) the
record must carry the review's binding code-readout-convention
sentence (find it in LOG/RECORD review notes); (d) anti-conversion
screen is a possible Saturday add-on ONLY if a pod frees up AND
mac-local greenlights.

### Item 1 — hedging-trend LEVEL Stage-2 (the deliverable)
Program DECISION: aggregation-framed win ACCEPTED; shuffle IMMUNITY is
the disclosed mechanism receipt (order-free pooling is the claim).
- FRESH card required (killed `task_hunt/confidence/CARD.md` is NOT
  its confirmation; screen numbers cited as motivation only:
  distill slope8 mean-probe 0.521→0.545→0.565 at T16/32/64 vs
  per-token 0.468 lin / 0.503 MLP; state control regime-1).
- **Reuse runpod-d's candidate-1 Stage-2 pattern** (its λ̂ panel on
  `ward_real_lambda_base_l12`): plugin datasource (append-only
  data.yaml), single scarce anchor, 5 archs (per-token BatchTopK SAE,
  T-SAE, Stacked, TXC-pre, TXC-post) × T ladder × seeds {1,2,42} +
  untrained, matched REALIZED l0_per_token (note runpod-d r2 item:
  post's `k_win // T` squash collapses realized l0 — set nominal k per
  T to hit target realized l0), per-tile readout. STUDY its files
  under `experiments/explorations/task_hunt/lambda_intensity/` + its
  datasource plugin + leaderboard rows BEFORE writing mine.
- My substrate is on THIS volume: Ward stream + base & distill
  17-layer caches at `/workspace/conv_depth_caches/{ward_stream,base,
  distill}` (hs indices = odd only; slope screen used hs15 = L14);
  labels `task_hunt/labels/confidence.npz` (slope8_bin terciles,
  trace_split; my screen-time row machinery in
  `task_hunt/confidence/screen.py` — hedge×position matching).
- Reader for the panel: distill (the phenomenon's generator) — decide
  + freeze in card; layer L14/hs15 unless the λ̂ pattern dictates the
  datasource convention (theirs used base_l12 — check why and mirror
  the reasoning, not necessarily the layer).
- Frozen per-arch predictions BEFORE training; canonical runner ONLY
  (`temp_bench.core.runner.run_experiment`, clean tree, 0 dup keys);
  deliverable = second real-task T-scaling figure + record + LOG.

### Item 2 — early-layer addendum (~2-3 h, zero new data; run while
### panels train)
g_order(ℓ) for replag lag4 on cached alternates (gpt2 hs4, gemma2
hs8, llama hs8 — caches `/workspace/replag_caches/<model>/hs*.npy`,
manifests `task_hunt/labels/replag_*_manifests.npz`, screen machinery
`task_hunt/replag/screen.py` — extend, don't retune) + g_agg(ℓ) for
slope8 across all 17 Ward capture points (both readers; mean-probe +
tok per layer suffices). Freeze a short addendum note/card pre-run.
Question: does temporal signal GROW at pre-conversion depths?

### Parked (do NOT run): gpt2 order cell, anti-conversion class,
### proof-op Stage-2 — recorded in r2 briefing.

## Round-1 state (closed; context only)
All three arm-B candidates KILLED by frozen cards — full verdicts in
LOG.md (replag: converted at every scale; confidence: aggregation not
order; emotional instability: converted near onset — paper replicated,
κ 0.857, ~$12 spend). Figures: `task_hunt/replag/figs/`. Volume
assets: `/workspace/replag_caches`, `/workspace/conv_depth_caches`,
`/workspace/emo_caches` (emo acts stored ×1/64 — gemma-3 fp16
saturation; act_scale in acts/index.json). Models cached: gpt2,
gemma-2-2b, llama-3.1-8b base, r1-distill, gemma-3-12b-it.
`run.py validate` green; leaderboard untouched by arm B in round 1.

## Sibling context (for dedup awareness)
runpod-d (r2): budget-matched TXC-post re-run on ward_real_lambda.
runpod-b: hunt-support-stats (OWNS the variance-aware renderer — I
re-render with it when it merges, or do minimal l0 annotation myself
if it hasn't landed when my cells finish; reconcile in LOG). runpod:
hunt-support-synthetic (mechanism receipts). Figure ownership was
deconflicted in `be255840` — check those briefings if overlap looms.
