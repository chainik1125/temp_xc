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

| Agent | Pod | GPU index | VRAM | Components | Briefing | Status |
|---|---|---|---|---|---|---|
| **agent_paper** | local 5090 | 0 (only GPU) | 32 GB | orchestration, C1, C2, paper drafting | [`agent_paper/briefing.md`](agent_paper/briefing.md) | active |
| **agent_nlp** | 2× H100 | **0** | 80 GB | C3 + C4 (shared activation cache) | TBD | not provisioned |
| **agent_em** | 2× H100 | **1** | 80 GB | C6 | TBD | not provisioned |
| **agent_steer** | 3× A40 | **0** | 48 GB | C5 | TBD | not provisioned |
| **agent_back** | 3× A40 | **1** | 48 GB | C7 | TBD | not provisioned |
| **agent_synth** (open) | 3× A40 | **2** | 48 GB | reserve — synth helper, multi-seed extension, A/B tests | TBD | open |
| **agent_apple** | TBD | TBD | TBD | TBD | [`agent_apple/`](agent_apple/) | placeholder dir, no briefing yet |
| **agent_em_h200** (fallback) | H200 | 0 (only GPU) | 141 GB | C6 if R32 blows H100 | n/a | dormant |

Two H100s host two distinct agents (agent_nlp on GPU 0, agent_em on
GPU 1). Three A40s host three agents (one per GPU). Sharing pods
keeps activation caches and checkpoints on the same network volume
(zero cross-pod transfer cost for end-of-paper unification).

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

| Pod | GPU | VRAM | RAM | vCPU | Network volume |
|---|---|---|---|---|---|
| 2× H100 | 80 GB × 2 | 80 + 80 | 500 GB | 56 | 200 GB |
| 3× A40 | 48 GB × 3 | 48 × 3 | 150 GB | 27 | 200 GB |
| H200 (reserve) | 141 GB × 1 | 141 | 256 GB (TBD) | 32 (TBD) | 100 GB |
| local 5090 | 32 GB | 32 | ~50 GB | 16 (WSL) | n/a — local SSD |

## Storage layout (network volume per pod)

All paper state lives on the **network volume** (RunPod's `/workspace`).
This persists across pod stop/start AND can be detached and attached to
a different pod. Volume disk (the ephemeral local SSD) is only for
`/tmp` — never paper state.

| Path | Contents | Approx size |
|---|---|---|
| `/workspace/temp_xc/` | git repo (final branch) | 5 GB |
| `/workspace/temp_xc/purified/.venv/` | uv-managed env | 12 GB |
| `/workspace/temp_xc/purified/results/act_cache/` | activation caches keyed by `act_cache_key` | 14 GB / Gemma layer; up to 60 GB across components |
| `/workspace/temp_xc/purified/checkpoints/` | trained models keyed by `train_key` | 6 GB / arch-set; up to 30 GB across seeds |
| `/workspace/temp_xc/purified/results/runs/` | per-cell metrics + plots | 5 GB |
| `/workspace/.tokens/` | gh, hf, anthropic | <1 KB |
| `/workspace/hf_cache/` | HF_HOME — model weights | 30 GB (Gemma-2-2b + Qwen-14B) |
| Slack (logs, judge transcripts, etc.) | misc | 50 GB |
| **Total per pod (typical)** | | **~150 GB** |
| **Provision** | | **200 GB** |

H100 pod handling Qwen-14B (C6) needs an extra 30 GB for Wang
intermediate artefacts and additional 30 GB HF cache for the LoRA
adapter — provision **250 GB** for that pod.

## Why network volume, not volume disk

Network volume:
- Persists across pod stop/start. The 72-hour budget tolerates pod
  reschedules; volume disk does not.
- Can detach + attach to a different pod. If we need to migrate C6
  from H100 → H200 mid-run, the trained checkpoints come with the
  volume.
- Slightly slower IO, but our workloads are GPU-bound, not disk-bound.

Volume disk:
- 30–50% faster IO. Only useful for the `.venv` (which we copy from
  the network volume on first boot via `uv sync`).
- Wiped on pod stop. Catastrophic for `train_key` cache; pointless to
  keep paper state here.

**Decision: ALL paper state on network volume.** `.venv` is rebuilt
in-place by `uv sync` on each pod start (idempotent, ~2 minutes).

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
