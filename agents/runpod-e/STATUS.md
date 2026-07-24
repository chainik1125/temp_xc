# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-24, mid-session (replag screen: llama running).

## Who / where
GPU RunPod pod (H100 80GB), `/workspace/temp_xc`, `/workspace/.agent_id`
= `runpod-e`. Task-hunt arm B (`briefings/task-hunt-b.md`). Git identity
`runpod-e-agent`; creds `store --file=/workspace/.git-credentials`;
`HF_TOKEN` from `/workspace/.tokens/hf_token` (export per command;
gemma is gated). Pull-rebase before EVERY push (5 agents on `arxiv`).

## Candidate 1 (repetition-lag Δ) — state
- CARD frozen + committed BEFORE screens (`task_hunt/replag/CARD.md`).
- Labels: built inline (runpod-b's landed later, different scheme —
  cross-check only; see LOG duplication note). Manifests + stats
  committed under `task_hunt/labels/replag_*`. All 5 sanity tests pass.
  T4 amended pre-screen (two-sided null divergence; direction = real
  BELOW null at Δ≤4 — LOG entry).
- Caches on volume: `/workspace/replag_caches/<model>/{tokens,delta}.npz`
  + hs*.npy (gpt2 hs7, gemma2 hs14, llama hs14 + alternates).
- Screen results so far (`replag/results/screen_*.json`):
  **gpt2 + gemma DONE, llama RUNNING** (bg process; log at scratchpad
  `screen.log`). Pattern both models: detection CONVERTED (per-token
  0.75–0.97 AUC, window gap ≤ 0 at every T — NO ladder ⇒ frozen rule
  heads to KILL unless llama diverges); lag4 shows a real
  order-carried MLP gap (gpt2 T8 +0.114 over tok, shuffle collapses
  it; gemma only +0.019 — order gap SHRINKS with scale).
- Escalation enacted (card-pre-authorized, LOG-noted, committed):
  `REPLAG_ESCALATE_LAG4=1` extends lag4 MLPs to all T. RUN THIS after
  llama's base grid: `.venv/bin/python -m
  experiments.explorations.task_hunt.replag.screen` with that env var
  (resumable — only adds missing cells).
- Then: `render_screen.py` (figs), verdict entry in `task_hunt/LOG.md`
  (KEEP/KILL per card §Falsifier), commit, STATUS rewrite.

## Candidate 2 prep (running in bg, no screening before verdict)
- Ward stream REBUILT on this volume (`/workspace/conv_depth_caches/
  ward_stream`; stats reproduce committed reference, map_ok 99.97%).
  traces.json re-ported from origin/aniket-ward-stage-b per
  ATTRIBUTION.md (gitignored).
- `cache_depth.py base` then `distill` running in bg (17 capture
  points each, ~68 GB per model; log `cache_depth.log`).
- Candidate 2 (confidence trend) card FROZEN by runpod-b
  (`task_hunt/confidence/CARD.md`) — I append screen cells. Clock
  bridge measured: slope4 support ≈ 64 tok ⇒ T=64 reaches full
  coverage — NOT killed. Labels `task_hunt/labels/confidence.npz`
  (Ward grid, manifests balanced; card requires hedge-class matching
  on slope rows — implement at screen time). Screen T ∈ {16,32,64},
  probe stack problib, readers base+distill mid-depth (hs14 = L13).
- Candidate 3 (emotional instability) draft card staged by runpod-b —
  only if candidate 2 dies/finishes early.

## Deadline
Results by 2026-07-26 morning PT. Stage 2 (if any survivor): canonical
runner only, single best cell.
