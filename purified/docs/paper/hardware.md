---
title: Hardware specs and parallelism strategy
author: agent_paper
date: 2026-05-03
status: locked
---

## Pod specifications

| Pod | GPU | VRAM total | System RAM | vCPU | Notes |
|---|---|---|---|---|---|
| 2× H100 | 80 GB × 2 | 160 GB | 500 GB | 56 | NLP + EM workloads |
| 3× A40 | 48 GB × 3 | 144 GB | 150 GB | 27 | steering + backtracking |
| H200 (reserve) | 141 GB × 1 | 141 GB | 256 GB | 32 | C6 fallback if R32 organism blows H100 |
| Local 5090 | 32 GB × 1 | 32 GB | ~50 GB | 16 (WSL) | toy + orchestration |

vCPU budget is non-trivial: we have **56 cores on H100 and 27 cores on
A40**, both more than enough to run sklearn-based probing in parallel
across 16 SAEBench tasks.

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

## Storage layout: network volume vs volume disk

**ALL paper state on network volume.** Paper state is paper output —
losing it means losing the run. Volume disk is wiped on pod stop.

| What | Where |
|---|---|
| `temp_xc/` repo | network volume `/workspace/temp_xc/` |
| `purified/.venv/` | network volume (rebuilt in place via `uv sync` on first boot) |
| Activation caches | network volume `/workspace/temp_xc/purified/results/act_cache/<key>/` |
| Trained checkpoints | network volume `/workspace/temp_xc/purified/checkpoints/<train_key>/` |
| Per-cell artifacts | network volume `/workspace/temp_xc/purified/results/runs/<eval_key>/` |
| HF cache | network volume `/workspace/hf_cache/` |
| Tokens (gh, hf, anthropic) | network volume `/workspace/.tokens/` mode 0600 |
| `/tmp/*` | volume disk (ephemeral, fine to lose) |

**Per-pod provisioning:**

| Pod | Network volume size |
|---|---|
| 2× H100 (Agent NLP — C3 + C4 + extras) | **200 GB** |
| 2× H100 (Agent EM — Qwen-14B + Wang) | **250 GB** (extra 50 GB for Qwen weights + Wang intermediates) |
| 3× A40 (Agent STEER — C5) | **200 GB** |
| 3× A40 (Agent BACK — C7) | **200 GB** |
| H200 (reserve, EM fallback) | **250 GB** |

## Why this matters

Three concrete pain points the framework guards against:

1. **Pod restart drops the venv.** Volume disk would force a 12 GB
   `uv sync` re-download every restart. Network volume keeps the
   `.venv` warm.
2. **Pod migration mid-run.** If we need to move C6 from H100 → H200
   mid-Wang procedure, we detach the network volume from one pod and
   attach to the other. Cached `train_key` checkpoints come with the
   volume; the new pod resumes from the last completed cell.
3. **Cross-pod cache sharing.** Agent STEER's A40 pod doesn't need to
   rebuild the C5 activation cache from scratch — it pulls from
   `han1823123123/temp-bench-data` (HF dataset repo) which Agent NLP
   uploaded. The HF backup is the cross-pod transport.
