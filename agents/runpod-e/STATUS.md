# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-24 (seeded by mac-local at pod creation — no
session run yet).

## Who / where
GPU RunPod pod (H100-class), Linux at `/workspace/temp_xc`, identity
`/workspace/.agent_id` = `runpod-e`. Role: **task-hunt arm B**
(`briefings/task-hunt-b.md`, governed by `task-hunt.md`) —
repetition-lag Δ across model scale (gpt2-small / gemma-2-2b base /
Llama-3.1-8B base) + confidence trend backup. Fully volume-independent:
do NOT mount runpod-c's volume; build own caches locally.

## First session
Setup: `.venv`, CUDA torch check, `HF_HOME` on the big disk, creds at
`/workspace/.tokens/` (never hardcode), pull-rebase before every push
(FIVE agents share the branch tonight). Then: mini-card freeze →
Δ-label pull from runpod-b (or inline build from its committed script)
→ Stage-1 screens across the three models → LOG verdicts → Stage 2 on
the best cell. Deadline: results by 2026-07-26 morning PT.
