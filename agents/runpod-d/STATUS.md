# Working state — agent `runpod-d`

**Last rewrite:** 2026-07-24 (seeded by mac-local at pod creation — no
session run yet).

## Who / where
GPU RunPod pod (H100-class), Linux at `/workspace/temp_xc`, identity
`/workspace/.agent_id` = `runpod-d`. Role: **task-hunt arm A**
(`briefings/task-hunt.md`) — trace-derived candidates (backtracking-λ̂
intensity, proof-op runs) + the backtracking shuffle receipt.

## Volume rules (shared 700 GB volume, mounted at /shared)
Conventions in agents/README.md: read anything (Ward caches at
/shared/conv_depth_caches/); write ONLY under
/shared/task_hunt_caches/d/; NEVER touch /shared/temp_xc or
/shared/.agent_id (runpod-c's live tree/identity). Repo + venv +
.agent_id live on LOCAL /workspace. If the volume is missing: rebuild
via conversion_depth/build_ward_stream.py + cache_depth.py (~1 h).

## First session
Setup: `.venv`, CUDA torch check, `HF_HOME` on the big disk, creds at
`/workspace/.tokens/` (never hardcode), pull-rebase before every push
(FIVE agents share the branch tonight). Then execute the briefing:
mini-card freeze → Stage-1 screen (both cached reader models — free) →
LOG verdicts → Stage 2 on the best cell. Labels arrive from runpod-b
(`task_hunt/labels/`). Deadline: results by 2026-07-26 morning PT.
