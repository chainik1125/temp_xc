# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-22 (created by mac-local at spawn-prep; no session
has run yet).

## Who / where
Second RunPod box (Linux, 32 CPU / 128 GB, no GPU needed), repo at
`/workspace/temp_xc`. **You are `runpod-b` iff `/workspace/.agent_id` says
so** — see `agents/README.md` (the original box has no such file). Tokens in
`/workspace/.tokens/` (`gh_token`, `anthropic_key`, `hf_token`). Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`
(from the repo root). Export anthropic_key as ANTHROPIC_API_KEY.
Role: **the FreqBench line** (theorem-first generator + FreqFrac
instrumentation). The PhenomenonBench loop belongs to `runpod`.

## Setup notes (fresh pod)
- Read `RUNPOD_INSTRUCTIONS.md` + `agents/runpod/STATUS.md` "Gotchas" — the
  cgroup/venv/pkill/background-bash lessons all apply to this box too.
- Verify real CPU quota before sizing pools (`nproc` can lie on RunPod;
  check cgroup cpu quota) — size grid pools accordingly (the original box
  runs 28 workers × OMP1 at 32 real CPUs).
- Claude 5-family models reject `temperature` and think by default.

## Current task
**`briefings/freqbench-c1.md`** — the 12-hour FreqBench overnight session
(FreqFrac full pass; seed cards FB-2/FB-3/FB-1 end-to-end incl. gated grids).
Read the briefing in full first, then `freqbench/LOOP.md` (governing
protocol) + `freqbench/PORT.md` § A–B, § G.

## Next actions
Fresh session: CLAUDE.md read order → this file → the briefing. Rewrite this
file before every compact (12-hour session ⇒ several compacts — the rewrite
discipline is load-bearing).
