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

| Agent | Pod | GPU index | VRAM | Pod mode | Components | Briefing | Status |
|---|---|---|---|---|---|---|---|
| **agent_paper** | local 5090 | 0 (only GPU) | 32 GB | persistent (local SSD) | orchestration, C1, C2, paper drafting | [`agent_paper/briefing.md`](agent_paper/briefing.md) | active |
| **agent_nlp** | 2× H100 | **0** | 80 GB | persistent | C3 + C4 (shared activation cache) | TBD | not provisioned |
| **agent_em** | 2× H100 | **1** | 80 GB | persistent | C6 | TBD | not provisioned |
| **agent_steer** | 4× A40 | **0** | 48 GB | **ephemeral** | C5 | TBD | not provisioned |
| **agent_back** | 4× A40 | **1** | 48 GB | **ephemeral** | C7 | TBD | not provisioned |
| **agent_em_h200** (fallback) | H200 | 0 (only GPU) | 141 GB | persistent | C6 if R32 blows H100 | n/a | dormant |

The **4× A40 pod has 2 named agents + 2 spare GPU slots** (GPUs 2 and 3).
Spare slots are not owned by a named agent. The lead agent on either
component may launch a second process on a spare GPU to run a cell in
parallel — for example, agent_steer could run seed=42 on GPU 0 and
seed=1 on GPU 2 simultaneously by launching two processes with
different ``CUDA_VISIBLE_DEVICES``. We add a named agent to a spare
slot only if a concrete need emerges that the existing roster can't
cover.

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
| C7 backtracking | gemma-2-2b BASE L10 | TXC-base + TXC-pro vs SAE / TFA / T-SAE / MLC | agent_back | 1× A40 |

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
typical budget. Slack capacity is for: judge transcripts, large
intermediate Wang artifacts on C6, multi-seed extensions if Agent QA
or Agent SYNTH gets activated.

## Concurrency budget per pod

| Pod | Concurrent training cells | Concurrent probing cells | Notes |
|---|---|---|---|
| 2× H100 (1 agent / GPU) | 2 (one per GPU) | 16 per GPU (CPU-bound, n_jobs=-1) | 56 vCPU = 28/GPU. Probing fits 16+ tasks easily. |
| 3× A40 (1 agent / GPU) | 3 (one per GPU) | 9 per GPU | 27 vCPU = 9/GPU. |
| H200 (reserve) | 1 (single GPU) | 32 (CPU-bound) | High RAM; for Wang on Qwen-14B. |
| local 5090 | 1 (toy) | 16 | toy training is fast; concurrency mostly via threading inside the script. |

`temp_bench.eval.probing` will accept `n_jobs` knob (default `-1`)
that maps to `joblib.Parallel`. Cell-level concurrency flows through a
`runner.run_concurrent_cells` helper that respects the per-pod GPU
count.

## GPU isolation contract (load-bearing)

When multiple agents share a pod, each one **must** pin
``CUDA_VISIBLE_DEVICES`` to its assigned index before any CUDA work.
This is the only mechanism that prevents agent A from accidentally
allocating tensors on agent B's GPU. PyTorch sees only what the env
var allows; physical index N appears as ``cuda:0`` to that agent.

**At session start**, every agent runs:

```bash
cd /workspace/temp_xc/purified
source scripts/set_agent_env.sh <agent_name>     # pins CUDA_VISIBLE_DEVICES
bash scripts/agent_smoke_test.sh                 # verifies pinning + framework
```

The smoke test calls ``temp_bench.runner.preflight()`` which:
- aborts if ``CUDA_VISIBLE_DEVICES`` is unset on a multi-GPU pod
- aborts if ``torch.cuda.device_count() > 1`` after pinning (would mean
  the env var didn't take — likely a sourcing error or a pre-existing
  Python process already holding GPUs)
- prints which physical GPU is visible

Violations (running training without pinning) corrupt the cache
contract: a checkpoint trained on the wrong GPU has the same
``train_key`` as one trained on the right GPU, but the underlying
hardware behavior may differ (cuBLAS heuristics, fp16 ULP variance).
Different agents writing the same ``train_key`` is undefined.

## How to add a new agent

1. Create `purified/agents/<name>/briefing.md` (copy
   `agent_paper/briefing.md` as a template).
2. Add a row to the *Active roster* table above.
3. Set `AGENT_NAME=<name>` in the agent's environment (it appears in
   leaderboard rows via `runner.run_cell`).
4. Wire the agent's pod into the bootstrap by reusing
   `scripts/bootstrap_runpod.sh`.
5. Open a coordination thread by writing the first dated log entry in
   `purified/agents/<name>/log.md`.

## Handoff protocol

If an agent is stuck or its pod dies:

1. Log the state of the world in `purified/agents/<stuck_agent>/log.md`,
   including last `eval_key` written, last `train_key` saved, and any
   failed cells (with stderr excerpt).
2. Mark the row in *Active roster* as `paused`.
3. Either bring the agent back up (re-attach pod, run smoke test) OR
   reassign the component to another agent: append a row to the new
   agent's `log.md` documenting the takeover, and update the roster.

The cache contract guarantees no work is repeated across handoffs:
the new agent's first action is `runner.run_cell(...)` for any pending
cell, and cached cells are skipped automatically.
