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

**One-time setup — SEPARATE clone and SEPARATE venv per agent
(explicit requirement).** Use the repo's canonical pod bootstrap
(`scripts/bootstrap_runpod.sh` — see `RUNPOD_INSTRUCTIONS.md`): it
handles tokens (`/workspace/.tokens/`, GitHub-token HTTPS auth), uv
install, the shared HF cache, and per-`REPO_DIR` clone + `uv sync`.
Run it ONCE PER AGENT with `REPO_DIR` pointed at that agent's clone
(tokens/HF-cache steps are shared and idempotent):

```bash
# fetch the script once (any temp clone or curl from GitHub), then:
GH_TOKEN=... HF_TOKEN=... \
  REPO_DIR=/workspace/agents/runpod-1/temp_xc bash bootstrap_runpod.sh
GH_TOKEN=... HF_TOKEN=... \
  REPO_DIR=/workspace/agents/runpod-2/temp_xc bash bootstrap_runpod.sh
echo runpod-1 > /workspace/agents/runpod-1/.agent_id
echo runpod-2 > /workspace/agents/runpod-2/.agent_id
# ALSO copy Han's dataset token (from the mac's ~/.tokens/) to
#   /workspace/.tokens/hf_token_datasets  (chmod 600)
# verify BOTH venvs independently:
/workspace/agents/runpod-1/temp_xc/.venv/bin/python -c "import temp_bench; print('r1 OK')"
/workspace/agents/runpod-2/temp_xc/.venv/bin/python -c "import temp_bench; print('r2 OK')"
```

Each agent's `.venv` lives INSIDE its own clone — never share a
venv or a clone between the two agents (editable installs point at
their own tree; sharing would cross-wire code versions).

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
