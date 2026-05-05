---
title: Agent delegation — single source of truth
author: agent_paper
date: 2026-05-05
status: locked
---

This is the canonical mapping of agents → components → hardware → storage.
**Update this file (not CLAUDE.md or PROTOCOL.md) when delegations change.**

## Active roster (current as of 2026-05-05 PM)

Several agents have pivoted missions during the sprint as new pods
came online and bottlenecks emerged. The "Mission" column reflects
**current** assignment, not original brief. See per-agent briefings
for the detailed pivot history.

Agents share pods. Each agent is **pinned to one GPU** via
``CUDA_VISIBLE_DEVICES`` to prevent cross-agent contention. Source
``scripts/set_agent_env.sh <name>`` at session start (the smoke test
verifies the pinning).

| Agent | Pod | GPU index | Clone path | Pod mode | Mission (current) | Status |
|---|---|---|---|---|---|---|
| **agent_paper** | local 5090 (32 GB) | 0 | `~/temp_xc/` | persistent (local) | orchestration, MW deployment coordination, paper drafting | active |
| **agent_nlp** | 2× H100 (80 GB / 500 GB / 56 vCPU) | 0 | `/workspace/temp_xc/` (primary) | persistent | C3 + C4 canonical sweep (topk_sae finishing, ~last 6 of 24 cells) | active — wrapping canonical |
| **agent_em** | 2× H100 (shares pod w/ agent_nlp) | 1 | `/workspace/temp_xc_em/` | persistent | C6 canonical 8/8 DONE → **pivoting to C6 MW** (txc_base_mw + Bricken × {42,1} × {14B,7B}; 4 cells; bricken_resample_every=5000 per § 14) | active — pivoting |
| **agent_em_100k** | 1× H100 (80 GB / 240 GB / 1 TB ephemeral) | 0 | `/workspace/temp_xc/` | **ephemeral** | original: C6 100K replica (abandoned 2026-05-05 PM after 5.9× per-step slowdown). **Pivoted to C3 MW helper** for agent_nlp (txc_base_mw + txc_pro_mw × {42,1,2}; 6 trainings + 12 evals) | active — pivoted |
| **agent_back** | 4× A40 (48 GB / 200 GB / 38 vCPU) | 0 + 2 (dedicated pair) | `/workspace/temp_xc/` (primary) | **ephemeral** | C7 canonical v4 sweep in flight (TXC-base + TXC-pro + Stacked-SAE + TopK-SAE + TFA + T-SAE + MLC × seed=42 only) | active — mid-flight |
| **agent_steer** | 4× A40 (shares pod w/ agent_back) | 1 + 3 (dedicated pair) | `/workspace/temp_xc_steer/` | **ephemeral** | C5 canonical v1.1.0 sweep DONE (9/9, hypothesis refuted post concept-lift fix). Currently advising agent_filler on C5 MW launch | idle / advising |
| **agent_steer_100k** | 1× H100 (80 GB / 240 GB / 1 TB ephemeral) | 0 | `/workspace/temp_xc/` | **ephemeral** | original: C5 100K replica (abandoned). **Pivoted (1)** to C5 MW (slow per-step on H100; 1 cell landed). **Pivoted (2) 2026-05-05 PM** to C7 MW helper for agent_back (txc_base_mw + txc_pro_mw × seed=42; 2 cells) | active — pivoted twice |
| **agent_filler** | 8× A40 (48 GB × 8 / 401 GB / 76 vCPU / 1 TB ephemeral) | 0 (primary; 0..5 via subprocess) | `/workspace/temp_xc/` | **ephemeral** | C5 MW parallel sweep (txc_base_mw + txc_pro_mw × {42,1,2}; 6 cells in parallel across GPUs 0..5; GPUs 6+7 spare) | active — sweep launched |
| **agent_em_h200** (dormant) | H200 (141 GB) | 0 | `/workspace/temp_xc/` | persistent | reserved fallback for C6 if H100 OOMs (never spun up) | dormant |

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

## Component coverage (canonical + multi-window)

The multi-window TXC archs (`txc_base_mw`, `txc_pro_mw`) were landed
2026-05-05 (decisions.md § 14) to fix the per-step training-FLOPs
disadvantage of the canonical TXC archs (~25× fewer tokens/step than
per-token SAEs at the canonical batch=1024). Each paper-bearing
component now has both a canonical sweep (the historical baseline)
and a multi-window deployment (the canonical headline going forward,
once cells land).

| Component | Subject | Canonical arches + agent | Multi-window deployment + agent | Status |
|---|---|---|---|---|
| C1 toy TopK sweep | synthetic Markov n=20, d=40 | TXC-base + TXC-pro vs all baselines (agent_paper) | not deployed (toy archs only; no paper-headline benefit) | deferred |
| C2 toy coupled features | synthetic HMM K=10, M=20, d=256 | TXC-base + TXC-pro at multiple T (agent_paper) | not deployed | deferred |
| C3 sparse probing | gemma-2-2b-it L13 | TXC-base + TXC-pro vs T-SAE / TopK-SAE / MLC (agent_nlp) | **txc_base_mw + txc_pro_mw × {42,1,2}** (agent_em_100k repurposed as helper) | canonical wrapping; MW in flight |
| C4 qualitative latents | gemma-2-2b-it L13 (shared with C3) | TXC-pro vs T-SAE (agent_nlp) | inherits MW checkpoints from C3 (agent_nlp re-evals) | follows C3 MW |
| C5 RLHF steering | gemma-2-2b-it L13 | TXC-base + TXC-pro vs T-SAE (agent_steer) | **txc_base_mw + txc_pro_mw × {42,1,2}** parallel on 8× A40 (agent_filler) + 1 cell from agent_steer_100k | canonical DONE; MW in flight |
| C6 emergent misalignment | qwen-2.5-14b-instruct + finance-LoRA + qwen-7b-medical | TXC-base + brickenauxk vs SAE-arditi (agent_em — DONE 8/8) | **txc_base_mw + Bricken × {42,1} × {14B,7B}** (agent_em pivoted post-canonical, bricken_resample_every=5000 per § 14) | canonical DONE; MW pivoting |
| C7 backtracking | Llama-3.1-8B BASE L10 + R1-Distill-Llama-8B (steering target) | TXC-base + TXC-pro + Stacked-SAE + TopK-SAE + TFA + T-SAE + MLC × seed=42 (agent_back) | **txc_base_mw + txc_pro_mw × seed=42** (agent_steer_100k pivoted to helper) | canonical in flight; MW pivoting |

## Pod specifications (RunPod)

| Pod | GPU | VRAM | RAM | vCPU | /workspace | Persistence |
|---|---|---|---|---|---|---|
| 2× H100 (agent_nlp + agent_em) | 80 GB × 2 | 160 GB | 500 GB | 56 | **1 TB** | **persistent** — survives stop/start, attachable |
| 4× A40 (agent_back + agent_steer) | 48 GB × 4 | 192 GB | 200 GB | 38 | **1 TB** | **ephemeral** — wiped on pod stop |
| 1× H100 (agent_em_100k) | 80 GB | 80 GB | 240 GB | TBD | 1 TB | **ephemeral** |
| 1× H100 (agent_steer_100k) | 80 GB | 80 GB | 240 GB | TBD | 1 TB | **ephemeral** |
| 8× A40 (agent_filler, NEW 2026-05-05) | 48 GB × 8 | 384 GB | **401 GB** | **76** | 1 TB | **ephemeral** |
| H200 (reserve) | 141 GB × 1 | 141 GB | 256 GB (TBD) | 32 (TBD) | 200 GB | persistent |
| local 5090 (agent_paper) | 32 GB | 32 GB | ~50 GB | 16 (WSL) | n/a (local SSD) | persistent |

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
| agent_steer, agent_back, agent_em_100k, agent_steer_100k, agent_filler (+ any helper processes on the A40 pod) | `ephemeral` |

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
| 2× H100 (agent_nlp + agent_em) | 2 (one per GPU) | 16 per GPU (CPU-bound, n_jobs=-1) | 56 vCPU ≈ 28/GPU. |
| 4× A40 (agent_back + agent_steer) | up to 4 (subprocess per GPU) | 9 per GPU | 38 vCPU ≈ 9.5/GPU. agent_back: 0+2, agent_steer: 1+3. |
| 1× H100 (agent_em_100k) | 1 (single GPU) | 16 (CPU-bound) | 240 GB RAM — preload caches abundantly. |
| 1× H100 (agent_steer_100k) | 1 (single GPU) | 16 (CPU-bound) | Flagged CPU-bandwidth bottleneck for high-throughput cells. |
| 8× A40 (agent_filler) | up to 8 (subprocess per GPU) | 9 per GPU | 76 vCPU ≈ 9.5/GPU. Single agent, 8 parallel cells. |
| H200 (reserve) | 1 (single GPU) | 32 (CPU-bound) | High RAM; for Wang on Qwen-14B. |
| local 5090 (agent_paper) | 1 (toy) | 16 | toy training is fast. |

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
