---
status: active
created: 2026-07-26 ~21:00 London
for: the ACTMIX shared 3×H100 pod (runpod-1 + runpod-2)
read-first: briefings/actmix-shared.md
---

# Bootstrap — ACTMIX shared 3×H100 pod (84 CPU / 564 GB / 2 TB volume)

One pod, TWO agents, per-agent clones (the proven shared-pod
pattern). GPU roster (paired with `scripts/set_agent_env.sh`):
**runpod-1 = GPUs 0,1 (sparse probing — the bigger grid); runpod-2 =
GPU 2 (EM)**. Rebalancing GPUs later is scheduling, not frozen
config — note it in the ledger line if you do.

**One-time setup (either agent runs it, idempotent):**

```bash
mkdir -p /workspace/agents /workspace/hf_cache /workspace/.tokens
# git creds + HF tokens: copy from the mac (values live at
# ~/.tokens/ there) into /workspace/.tokens/, chmod 600 — Han or
# mac-local does the copy; NEVER commit or echo values.
for id in runpod-1 runpod-2; do
  git clone git@github.com:chainik1125/temp_xc.git /workspace/agents/$id/temp_xc
  git -C /workspace/agents/$id/temp_xc checkout arxiv
  echo $id > /workspace/agents/$id/.agent_id
done
cd /workspace/agents/runpod-1/temp_xc && uv sync   # repeat for runpod-2
```

**Every session:** cd into YOUR clone
(`/workspace/agents/<your-id>/temp_xc`), then
`source scripts/set_agent_env.sh <your-id>`. Identity = your
directory; `/workspace/.agent_id` is NOT used on this pod (two
agents — the per-agent `.agent_id` beside each clone is).

**Storage plan (2 TB volume):** activation caches under
`/workspace/hf_cache` (HF) and per-task cache dirs under
`/workspace/caches/<task>/` — the volume is persistent, so caches
built once are shared-by-path; the two agents coordinate via
briefings before deleting anything under `/workspace/caches/`.
Checkpoints worth keeping: mirror to HF before pod teardown
(token path in the shared briefing).

**Discipline:** the pod never pushes without pull-rebase; pins
taken from origin history; ledger lines under a `RUNPOD` section of
`briefings/MODAL_SPEND.md` (pod-hours × rate, est + actuals);
$150/day/person cap; detached/nohup for long runs so an SSH drop
kills nothing (`nohup ... > /workspace/logs/<name>.log 2>&1 &`).
Work briefings: `actmix-runpod-1.md` (probing), `actmix-runpod-2.md`
(EM).
