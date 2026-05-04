---
title: Agent delegation — single source of truth
author: agent_paper
date: 2026-05-03
status: locked
---

This is the canonical mapping of agents → components → hardware → storage.
**Update this file (not CLAUDE.md or PROTOCOL.md) when delegations change.**

## Active roster

Agents share pods. Each agent is **pinned to one GPU** via
``CUDA_VISIBLE_DEVICES`` to prevent cross-agent contention. Source
``scripts/set_agent_env.sh <name>`` at session start (the smoke test
verifies the pinning).

| Agent | Pod | GPU index | VRAM | Pod mode | Clone path | Components | Briefing | Status |
|---|---|---|---|---|---|---|---|---|
| **agent_paper** | local 5090 | 0 (only GPU) | 32 GB | persistent (local SSD) | `~/temp_xc/` | orchestration, C1, C2, paper drafting | [`agent_paper/briefing.md`](agent_paper/briefing.md) | active |
| **agent_nlp** | 2× H100 | **0** | 80 GB | persistent | `/workspace/temp_xc/` (primary) | C3 + C4 (shared activation cache) | [`agent_nlp/briefing.md`](agent_nlp/briefing.md) | draft-briefing |
| **agent_em** | 2× H100 | **1** | 80 GB | persistent | `/workspace/temp_xc_em/` | C6 | [`agent_em/briefing.md`](agent_em/briefing.md) | draft-briefing |
| **agent_back** | 4× A40 | **1** | 48 GB | **ephemeral** | `/workspace/temp_xc/` (primary) | C7 | [`agent_back/briefing.md`](agent_back/briefing.md) | draft-briefing |
| **agent_steer** | 4× A40 | **0** | 48 GB | **ephemeral** | `/workspace/temp_xc_steer/` | C5 | [`agent_steer/briefing.md`](agent_steer/briefing.md) | draft-briefing |
| **agent_em_h200** (fallback) | H200 | 0 (only GPU) | 141 GB | persistent | `/workspace/temp_xc/` | C6 if R32 blows H100 | n/a | dormant |
| **agent_em_100k** | 1× H100 (240 GB RAM, 1 TB ephemeral /workspace) | 0 (only GPU) | 80 GB | **ephemeral** | `/workspace/temp_xc/` | C6 — literal copy of agent_em at `n_steps=100_000` | [`agent_em_100k/briefing.md`](agent_em_100k/briefing.md) | active |
| **agent_steer_100k** | 1× H100 (240 GB RAM, 1 TB ephemeral /workspace) | 0 (only GPU) | 80 GB | **ephemeral** | `/workspace/temp_xc/` | C5 — literal copy of agent_steer at `n_steps=100_000` | [`agent_steer_100k/briefing.md`](agent_steer_100k/briefing.md) | active |

**One clone per agent on shared pods.** Two agents sharing a single
`.git/` collide on `index.lock` and risk clobbering each other's
uncommitted edits during pull-rebase. Workaround: each agent gets
its own clone. The first agent on a pod uses the canonical
`/workspace/temp_xc/` clone created by `bootstrap_runpod.sh`; the
second agent's clone is created by Han via
`bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh <agent_name>`.
Tokens (`/workspace/.tokens/`) and HF cache (`/workspace/hf_cache/`)
are shared across both clones, so each agent only pays ~5 GB extra
disk for the second working tree (out of 1 TB).

The **4× A40 pod is fully partitioned**: agent_back gets GPUs 0 and 2;
agent_steer gets GPUs 1 and 3 (Han 2026-05-04 PM re-allocation). 2
dedicated GPUs per agent, **no unassigned slots, no borrow pattern**.
Each agent's own python process is pinned to its first primary via
`CUDA_VISIBLE_DEVICES` (set by `scripts/set_agent_env.sh`); to launch
work on its second GPU, it spawns a subprocess with that GPU's index
in the subprocess env. Convenience wrapper:

```bash
# In agent_steer's session (own process pinned to GPU 1):
bash scripts/run_on_gpu.sh 3 -- python -m experiments.c5_steering.run --seeds 1
```

This sets `CUDA_VISIBLE_DEVICES=3` for the subprocess only, runs the
command, exits. The agent's own python process still sees only GPU 1.

**No lockfile manager and no peer-borrow on the A40 pod.** Each agent
stays in their two-GPU lane (PROTOCOL.md § 13). The 2× H100 pod still
uses the older borrow convention (agent_em borrowing agent_nlp's GPU 0
when idle is fine there). The earlier `claim_gpu` lockfile system was
deleted 2026-05-04 — it was correct but agents bypassed it for
`subprocess.Popen` ergonomics.

Pod sharing keeps activation caches and checkpoints on the same volume
(zero cross-pod transfer cost when an agent on the same pod needs an
artifact another agent produced). Cross-pod sharing flows through
HuggingFace — see *Pod modes* below.

## Component coverage

| Component | Subject | Lead arch | Lead agent | Hardware |
|---|---|---|---|---|
| C1 toy TopK sweep | synthetic Markov n=20, d=40 | TXC-base + TXC-pro vs all baselines | agent_paper | local 5090 |
| C2 toy coupled features | synthetic HMM K=10, M=20, d=256 | TXC-base + TXC-pro at multiple T | agent_paper | local 5090 |
| C3 sparse probing | gemma-2-2b-it L13 | TXC-base + TXC-pro vs T-SAE / TopK-SAE / MLC | agent_nlp | 1× H100 |
| C4 qualitative latents | gemma-2-2b-it L13 (shared with C3) | TXC-pro vs T-SAE | agent_nlp | piggybacks on C3 cache |
| C5 RLHF steering | gemma-2-2b-it L13 | TXC-base + TXC-pro vs T-SAE | agent_steer | 1× A40 |
| C6 emergent misalignment | qwen-2.5-14b-instruct + finance-LoRA | TXC-base+brickenauxk vs SAE arditi | agent_em | 1× H100 (H200 fallback) |
| C7 backtracking | Llama-3.1-8B BASE L10 + R1-Distill-Llama-8B (steering target) | TXC-base + TXC-pro + Stacked-SAE + TopK-SAE + TFA + T-SAE + MLC | agent_back | 1× A40 |

## Pod specifications (RunPod)

| Pod | GPU | VRAM | RAM | vCPU | /workspace | Persistence |
|---|---|---|---|---|---|---|
| 2× H100 | 80 GB × 2 | 160 GB | 500 GB | 56 | **1 TB** | **persistent** — survives stop/start, attachable |
| 4× A40 | 48 GB × 4 | 192 GB | 200 GB | 38 | **1 TB** | **ephemeral** — wiped on pod stop |
| H200 (reserve) | 141 GB × 1 | 141 GB | 256 GB (TBD) | 32 (TBD) | 200 GB | persistent |
| local 5090 | 32 GB | 32 GB | ~50 GB | 16 (WSL) | n/a (local SSD) | persistent |

The **persistence asymmetry** is load-bearing for cross-pod
coordination — see *Pod modes* below.

## Pod modes — persistent vs ephemeral

Two pods, two modes:

- **Persistent** (H100, H200, local): `/workspace` survives pod
  stop/start. Treat /workspace as the source of truth. HF is a backup.
- **Ephemeral** (A40): `/workspace` is wiped on pod stop. Treat HF as
  the source of truth. /workspace is a working cache that may
  disappear.

The framework knows which mode it's in via the
``TEMP_BENCH_POD_MODE`` env var (set by ``scripts/set_agent_env.sh``):

| Agent | TEMP_BENCH_POD_MODE |
|---|---|
| agent_nlp, agent_em, agent_em_h200, agent_paper | `persistent` |
| agent_steer, agent_back (+ any helper processes on the A40 pod) | `ephemeral` |

The ephemeral mode triggers two things:
1. **Bootstrap**: ``scripts/sync_from_hf.sh`` pulls the latest
   checkpoints + activation caches from
   ``han1823123123/temp-bench-{models,data}`` into /workspace before
   the agent does any work.
2. **Auto-push**: ``cache.save_checkpoint`` automatically pushes the
   freshly-saved checkpoint to HF after the local write. Cell metrics
   (small) are also pushed at end of cell. Failure to push is fatal —
   we cannot risk losing a multi-hour training run to pod death.

Persistent-mode agents may still upload to HF as a backup but it isn't
mandatory; the next pod start will find /workspace intact.

## Storage layout (per pod)

| Path | Contents | Typical size |
|---|---|---|
| `/workspace/temp_xc/` | git repo (final branch) | 5 GB |
| `/workspace/temp_xc/purified/.venv/` | uv-managed env | 12 GB |
| `/workspace/temp_xc/purified/results/act_cache/<key>/` | activation caches | 14 GB per Gemma layer; up to 60 GB |
| `/workspace/temp_xc/purified/checkpoints/<train_key>/` | trained models | 6 GB per arch-set; up to 30 GB |
| `/workspace/temp_xc/purified/results/runs/<eval_key>/` | per-cell metrics + plots | 5 GB |
| `/workspace/.tokens/` | gh, hf, anthropic (mode 0600) | <1 KB |
| `/workspace/hf_cache/` | HF_HOME — model weights | 30 GB (Gemma) ─ 60 GB (incl. Qwen-14B) |
| Slack (logs, judge transcripts) | misc | 50 GB |
| **Subtotal** | | **~150 GB typical, 250 GB on EM pod** |

Both pods now have 1 TB /workspace, which is comfortably above the
typical budget. Slack capacity is for judge transcripts, large
intermediate Wang artifacts on C6, and multi-seed extensions launched
on the A40 pool GPUs.

## Concurrency budget per pod

| Pod | Concurrent training cells | Concurrent probing cells | Notes |
|---|---|---|---|
| 2× H100 (1 agent / GPU) | 2 (one per GPU) | 16 per GPU (CPU-bound, n_jobs=-1) | 56 vCPU ≈ 28/GPU. |
| 4× A40 (2 agents × 2 GPUs each) | up to 4 (subprocess per GPU) | 9 per GPU | 38 vCPU ≈ 9.5/GPU. agent_back: 0+2, agent_steer: 1+3. |
| H200 (reserve) | 1 (single GPU) | 32 (CPU-bound) | High RAM; for Wang on Qwen-14B. |
| local 5090 | 1 (toy) | 16 | toy training is fast. |

`temp_bench.eval.probing` accepts an `n_jobs` knob (default `-1`) that
maps to `joblib.Parallel`. Cell-level concurrency flows through
`bash scripts/run_on_gpu.sh <idx> -- <cmd>` subprocess launches
(no DDP — see `docs/paper/hardware.md`).

## GPU sharing — see PROTOCOL.md

Pinning + the GPU-sharing convention (no lockfile) live in
**PROTOCOL.md** (§ 12 pinning, § 13 sharing convention). Don't duplicate
them here — `agents/README.md` is the *roster*, not the protocol.

## How to add a new agent

1. Create `purified/agents/<name>/briefing.md` by copying
   `purified/agents/_briefing_template.md`. Han fills in the
   "Identity + mandate" section.
2. Add a row to the *Active roster* table above.
3. Add an entry to `purified/scripts/set_agent_env.sh` mapping
   `<name>` to its primary GPU index + pod mode.
4. Set `AGENT_NAME=<name>` (set_agent_env.sh handles this).
5. **Han** runs `scripts/bootstrap_runpod.sh` (RunPod) or
   `scripts/bootstrap_local.sh` (local) **on the fresh pod, before
   spawning the agent**. The script is interactive (prompts for
   tokens) — agents cannot run it. After bootstrap, `/workspace/.tokens/`
   is populated and the venv exists at `/workspace/temp_xc/purified/.venv/`.
6. **For shared pods** (2× H100 with two agents, 4× A40 with two
   agents): Han also runs
   `bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh <second_agent>`
   to create a separate clone for the second agent. This avoids
   `.git/` lock contention. Idempotent.

## Handoff protocol (cross-agent reassignment)

If an agent is stuck or its pod dies:

1. Update the briefing's "Current state" section with last `eval_key`
   written, last `train_key` saved, any failed cells (stderr excerpt).
   This is the chronological audit trail.
2. Mark the row in *Active roster* as `paused`.
3. Either bring the agent back up (re-attach pod, run smoke test) OR
   reassign the component:
   - Update the new agent's briefing with the component takeover
     (mention the source agent + last `eval_key` taken over).
   - Update the roster.

The cache contract guarantees no work is repeated: the new agent's
first action is `runner.run_cell(...)` for any pending cell, and
cached cells are skipped automatically by `train_key` / `eval_key`
matching.
