---
title: Hardware specs and parallelism strategy
author: agent_paper
date: 2026-05-03
status: locked
---

## Pod specifications

| Pod | GPU | VRAM total | System RAM | vCPU | /workspace | Persistence |
|---|---|---|---|---|---|---|
| 2× H100 | 80 GB × 2 | 160 GB | 500 GB | 56 | **1 TB** | persistent |
| 4× A40 | 48 GB × 4 | 192 GB | 200 GB | 38 | **1 TB** | **ephemeral** |
| H200 (reserve) | 141 GB × 1 | 141 GB | 256 GB | 32 | 200 GB | persistent |
| Local 5090 | 32 GB × 1 | 32 GB | ~50 GB | 16 (WSL) | local SSD | persistent |

vCPU budget is non-trivial: we have **56 cores on H100 and 38 cores on
A40**, both more than enough to run sklearn-based probing in parallel
across 16 SAEBench tasks.

The 4× A40 pod's `/workspace` is **ephemeral** (wiped on pod stop), so
we use the framework's auto-push-to-HF mechanism on save. See
*Pod modes* below.

## Three layers of parallelism

The paper has three orthogonal axes that can run concurrently:

### 1. Cell-level (across (arch, seed, k_feat) tuples)

One process per GPU. On the 2× H100 pod, two training cells run
concurrently — same component, different (arch, seed). The cache
contract guarantees that two agents writing to `leaderboard.jsonl` from
different processes append cleanly (flock-protected) without
collisions.

`runner.run_concurrent_cells(...)` (TODO — to add) wraps `run_cell`
with a `concurrent.futures.ProcessPoolExecutor` sized to GPU count.

### 2. Task-level (within probing)

Sparse probing on 16 SAEBench tasks × 5 k_feat values × 3 seeds = 240
sklearn-LR fits per arch. Embarrassingly parallel, CPU-bound.

`temp_bench.eval.probing` exposes `ProbingConfig(n_jobs=-1)` (default
`-1` → use all cores). Internally:

```python
from joblib import Parallel, delayed
results = Parallel(n_jobs=cfg.n_jobs)(
    delayed(probe_one_task)(z, y, k_feat) for task in tasks for k_feat in k_feats
)
```

On H100 pod (56 vCPU): 240 probings / 56 ≈ 4–5 batches × ~2 sec each ≈
**≤ 30 sec total** (vs ~8 min sequential).

### 3. Encode-level (within a forward pass)

After Phase 7's algorithmic speedup, we compute S ∈ {10, 20, 32} from
the same encode (aggregate_s does the per-S window restriction at zero
extra GPU cost). The probing protocol must reuse this trick. Source:
`origin/han-phase7-unification:experiments/phase7_unification/run_probing_phase7.py`.

## Activation cache strategy

Gemma-2-2b activations at L13, fp16, 24K × 128 tokens × 2304 d_in =
**14.1 GB per layer**. This fits comfortably in 500 GB H100 RAM and
148 GB A40 RAM. Strategy:

1. Build the cache once on the H100 pod (Agent NLP, ~3 H100-hours).
2. Push to `han1823123123/temp-bench-data` under
   `act_cache/<act_cache_key>/`.
3. A40 pods (Agent STEER, Agent BACK) `huggingface-cli download` the
   prebuilt cache instead of rebuilding from scratch.
4. Cache key is the contract: any agent loading
   `act_cache/<act_cache_key>/` knows it's bit-identical to whatever
   produced that key.

For probing: keep a separate **probe cache** keyed by
`(probe_dataset, S, n_train, n_test, tokenizer)` — this contains the
tail-S activations of the probing tasks (not FineWeb training data).
~6 GB per (subject_model, layer, S). Re-built once per (model, layer)
pair.

## Single-GPU vs multi-GPU per agent

**Decision: every agent uses exactly one GPU.** No DDP, no FSDP, no
multi-GPU training in this paper.

Rationale (specific to our workload):

- **Cells are embarrassingly parallel.** Each (arch, seed, k_pos)
  training is independent. With 4 A40 GPUs and 12 typical cells per
  component, four single-GPU agents finish in 3 batches of 4
  concurrent cells. A 2-GPU DDP setup with 2 agents finishes in 6
  batches of 2 cells each, each ~1.7× faster (sub-linear DDP scaling
  due to PCIe gradient sync — A40s are not NVLinked) — same
  wall-clock, more code complexity.
- **Models fit on a single A40.** TXC-pro at $d_{\text{sae}}=18432$,
  $T_{\text{max}}=10$ is ~3.4 GB fp32; full training state (model + optim
  + activation buffer) fits comfortably in 48 GB. The only model that
  doesn't fit on a single A40 is Qwen-14B (28 GB fp16 weights + LoRA),
  but that's on the H100 pod (Agent EM, GPU 1, 80 GB).
- **Different agents on different cells gives better wall-clock + more
  flexibility.** A multi-GPU agent training one cell at a time can't
  also run an A/B test on another component; a 4-agent setup can.
- **DDP integration cost.** Adding `torch.distributed`,
  `torchrun`-launching, and process-group teardown is ~100 LoC of
  new framework code that buys nothing for our cells.

The framework's `runner.run_cell` therefore expects exactly one
visible GPU (verified by `preflight()`). DDP/FSDP is forbidden inside
a single process; **multi-GPU work is multi-process** — see *Multi-GPU
access* below for the protocol.

## Multi-GPU access (sharing convention)

When an agent wants to use more than one GPU on its pod, it launches
multiple subprocesses, each with `CUDA_VISIBLE_DEVICES=<single_idx>`.
The agent's own python process stays pinned to its primary; only the
spawned subprocess sees a different GPU.

GPU sharing is a **convention**, not a lockfile-enforced contract
(the earlier `temp_bench.utils.gpu_locks` system was removed
2026-05-04 — see PROTOCOL.md § 13). For the 4× A40 pod:

| GPU | Owner | Borrowable? |
|---|---|---|
| 0 | agent_steer (primary) | only when agent_steer is idle |
| 1 | agent_back (primary) | only when agent_back is idle |
| 2 | unowned | yes |
| 3 | unowned | yes |

**Before borrowing a peer's GPU**: read peer's briefing's
"Current state" + run `nvidia-smi`. **Update YOUR briefing** with
the borrow + ETA before kicking off long work.

Worked example — agent_steer runs 3 seeds in parallel using the
convenience wrapper:

```bash
# agent_steer's own process is pinned to GPU 0. Each launch below
# spawns a subprocess pinned to a different GPU.
bash scripts/run_on_gpu.sh 0 -- python -m experiments.c5_steering.run --seeds 42 &
bash scripts/run_on_gpu.sh 2 -- python -m experiments.c5_steering.run --seeds 1 &
bash scripts/run_on_gpu.sh 3 -- python -m experiments.c5_steering.run --seeds 2 &
wait
```

The wrapper sets `CUDA_VISIBLE_DEVICES=<idx>` for the subprocess,
sanity-checks `nvidia-smi` (warns if the GPU appears occupied),
and execs the command. No lockfile dance.

**Failure mode**: if you and a peer accidentally launch on the same
GPU simultaneously, both crash with CUDA OOM. Recoverable in ~5 min
— each cell is independent and deterministic via `train_key`.

## Pod modes — persistent vs ephemeral

The 2× H100 has 1 TB persistent /workspace; the 4× A40 has 1 TB
ephemeral /workspace. The framework knows the mode via
`TEMP_BENCH_POD_MODE` (set by `scripts/set_agent_env.sh`):

| Pod | Mode | What happens on `save_checkpoint` |
|---|---|---|
| 2× H100 | `persistent` | Saved locally; HF push optional (recommended at session end) |
| 4× A40 | `ephemeral` | Saved locally **then auto-pushed to HF**. Push failure is fatal. |
| H200 reserve | `persistent` | Same as H100. |
| Local 5090 | `persistent` | Same as H100. |

On ephemeral pods, **session start runs `scripts/sync_from_hf.sh`** to
pull every known checkpoint + activation cache from
`han1823123123/temp-bench-{models,data}` into /workspace. This is
idempotent — files already present are not redownloaded.

The cache contract makes this safe across pods: if H100 pod A produced
`train_key=abc123def4567890`, A40 pod B's `runner.run_cell(...)`
computes the same key, sees `checkpoints/abc123def4567890/` exists
(after sync_from_hf), and skips training.

## Storage layout: network volume vs volume disk

All paper state lives in `/workspace`. Volume disk (the local SSD,
e.g. `/root/`) is only for `/tmp` — never paper state.

| What | Where |
|---|---|
| `temp_xc/` repo | `/workspace/temp_xc/` |
| `purified/.venv/` | `/workspace/temp_xc/purified/.venv` (rebuilt by `uv sync` on first boot — idempotent) |
| Activation caches | `/workspace/temp_xc/purified/results/act_cache/<act_cache_key>/` |
| Trained checkpoints | `/workspace/temp_xc/purified/checkpoints/<train_key>/` |
| Per-cell artifacts | `/workspace/temp_xc/purified/results/runs/<eval_key>/` |
| HF cache | `/workspace/hf_cache/` |
| Tokens (gh, hf, anthropic) | `/workspace/.tokens/` mode 0600 |
| `/tmp/*` | local SSD (ephemeral, fine to lose) |

**Per-pod provisioning** (RunPod's `/workspace` allocation):

| Pod | /workspace size | Notes |
|---|---|---|
| 2× H100 (persistent) | **1 TB** | Agent NLP (C3+C4) + Agent EM (C6 with Qwen-14B). 250 GB typical use. |
| 4× A40 (ephemeral) | **1 TB** | Up to 4 agents (steer/back/synth/qa). 200 GB typical use. |
| H200 (reserve, persistent) | 250 GB | Only if R32 organism blows H100. |
| Local 5090 | local SSD | Agent PAPER orchestration + C1/C2. |

## Why this matters

Three concrete pain points the framework guards against:

1. **A40 pod restart wipes /workspace.** Without auto-push, an
   8-hour TXC-pro training that crashes 10 minutes before the pod
   restarts is gone. The auto-push in `cache.save_checkpoint`
   eliminates this — every checkpoint reaches HF before the next
   `run_cell` returns.
2. **Pod migration mid-run.** If we need to move C6 from H100 → H200
   mid-Wang procedure, the H100's persistent /workspace can be detached
   and re-attached on the new pod. Cached `train_key` checkpoints come
   with the volume; the new pod resumes from the last completed cell.
3. **Cross-pod cache sharing.** Agent STEER's A40 pod doesn't need to
   rebuild the C5 activation cache from scratch — `sync_from_hf.sh`
   pulls from `han1823123123/temp-bench-data` (HF dataset repo) which
   Agent NLP uploaded after building the cache on its H100. HF is the
   cross-pod transport.
