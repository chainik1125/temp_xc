---
author: Aniket Deshpande
date: 2026-04-18
tags:
  - reference
  - bench-harness
---

## Disk sizing for the long-training launch

What to ask RunPod for when spinning up the pod.

### Persistent volume: **250 GB**

| artifact | size | notes |
|----------|-----:|-------|
| Gemma cached activations (8 tasks × 128×d_model each, fp32) | ~40 GB | `results/saebench/saebench_artifacts/` — reused across every eval cell |
| Pre-training activation cache (resid_L10..L14, fp32) | ~30 GB | `data/cached_activations/gemma-2-2b/fineweb/` — 5 layers × 6000 seqs × 128 tokens × 2304 dim |
| Trained checkpoints (10 cells × ~1.5-6 GB) | ~30 GB | `results/saebench/ckpts/` |
| Intermediate sweep dirs (per-cell training outputs) | ~20 GB | `results/saebench/sweeps/` — gitignored, rotated during eval |
| Per-example predictions + texts JSONL (all archs, both shuffle variants) | ~45 GB | `results/saebench/predictions/` — confusion-matrix fodder |
| Aggregate JSONLs + SAEBench raw outputs | ~1 GB | `results/saebench/results/` |
| Training logs | ~100 MB | `results/saebench/logs/` |
| Plots | ~10 MB | `results/saebench/plots/` |
| W&B local staging | ~5 GB | `/workspace/temp_xc/wandb/` |
| HF cache (Gemma-2-2B shards + tokenizer) | ~10 GB | `/workspace/.cache/huggingface/` — persistent via HF_HOME |
| Repo + venv | ~8 GB | |
| **Working total** | **~200 GB** | |
| Margin | ~50 GB | Unexpected artifact growth, mid-run failures that double-write, etc |

### Root disk: **40 GB**

The container root disk is tiny by default (20 GB). Even after sending
HF_HOME, WANDB_DIR, TMPDIR, TORCH_EXTENSIONS_DIR to `/workspace`, a few
things still bleed onto root:

| bleeder | typical size |
|---------|-------------:|
| `/root/.cache/pip`, `/root/.cache/uv` | ~2-3 GB |
| `/root/.cache/wandb` (tiny logs that wandb insists on writing here) | ~500 MB |
| `/tmp/torch_extensions/`, `/tmp/wandb-*` | ~1-2 GB |
| `/root/.gitconfig`, `/root/.runpod/` | kB |
| System + Python install (PyTorch, sae_lens, etc at venv install time if non-standard) | ~5-10 GB |
| Margin for mid-run bloat | ~10-20 GB |

20 GB is NOT enough — we hit `no space left on device` multiple times
in the previous launch (see B7, B20 in `eval_infra_lessons.md`). Ask
for **40 GB root**.

If the pod template only offers 20 GB root and 200+ GB volume, the
harness is likely still usable with these mitigations:

```bash
# Send every cache to /workspace explicitly before anything else runs.
# This lives at the top of scripts/runpod_activate.sh:
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export WANDB_CACHE_DIR=/workspace/.cache/wandb
export WANDB_DIR=/workspace/temp_xc/wandb
export TMPDIR=/workspace/tmp
export TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions
export PIP_CACHE_DIR=/workspace/.cache/pip
mkdir -p /workspace/.cache/huggingface /workspace/.cache/wandb \
         /workspace/tmp /workspace/.cache/torch_extensions \
         /workspace/.cache/pip
```

### GPU: **H100 80 GB SXM** (any vendor)

Must be 80 GB — at T=20 full_window, the TempXC probing step peaks at
~60 GB in use (see B8, B9). 40 GB cards will OOM on the final TempXC
cells. No fallback path exists for smaller GPUs.

### Recommended RunPod template

1. Template: "PyTorch 2.4 + CUDA 12.1" (stock, Ubuntu 22.04)
2. GPU: 1× H100 SXM 80 GB
3. Container disk: 40 GB
4. Volume: 250 GB (mounted at `/workspace`)
5. Ports: 22 (SSH), 8888 (Jupyter if desired)
6. Secrets: HF_TOKEN, WANDB_API_KEY, ANTHROPIC_API_KEY

### Monthly cost estimate (rough)

- H100 SXM 80 GB: ~$2.50-3.50/hr
- 250 GB persistent volume: ~$0.10/GB/month = $25/month
- Full 20-24h launch: **~$50-85 in GPU + prorated volume**

### Post-run cleanup

Once results are pushed via `scripts/runpod_push_results.sh`, the pod
artifacts not needed for future reruns:

```bash
# Safe to delete after results are committed:
rm -rf results/saebench/sweeps/             # per-cell sweep intermediates
rm -rf results/saebench/saebench_artifacts/ # Gemma cache — regenerable
rm -rf /workspace/temp_xc/wandb/            # synced to cloud already

# KEEP:
#   results/saebench/ckpts/             — trained SAE/TempXC/MLC checkpoints
#   data/cached_activations/            — multi-layer training cache
#   results/saebench/results/           — aggregate JSONLs
#   results/saebench/predictions/       — per-example preds for confusion matrices
```
