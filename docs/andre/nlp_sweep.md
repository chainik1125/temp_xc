---
author: Andre
date: 2026-04-07
tags:
  - results
  - in-progress
---

## NLP Sweep — Temporal Crosscoders on Gemma 2 2B

Extends the toy-model sweep ([[v2_tx_v_sae]]) to real language model activations. All code lives in `temporal_crosscoders/NLP/`. To replicate: edit `NLP/config.py`, run `./NLP/run_sweep.sh`, then `python NLP/viz.py --fit-rho`.

## Motivation

The toy-model experiments showed TXCDR advantage grows with temporal correlation (rho) and moderate k. The natural question: **does this advantage transfer to real NLP activations?** Language models process sequences, so their internal representations have temporal structure by construction — if features "stick" across tokens (e.g., topic features, syntactic features), TXCDR should outperform StackedSAE on real data too.

## Experimental Setup

**Model**: Gemma 2 2B (`google/gemma-2-2b`), d_model = 2304, 26 layers.

**Dictionary**: Expansion factor 8 → d_sae = 18,432 latents.

**Data**: 40,000 sequences from FineWeb (`sample-10BT`), each 128 tokens. Activations are cached to disk as memory-mapped `.npy` files (~40GB per layer).

**Layers extracted** (`config.py:32-57`):

| Key | Layer | Component | Description |
|-----|-------|-----------|-------------|
| `mid_res` | 13 | residual | Mid-network residual stream |
| `final_res` | 25 | residual | Final layer residual stream |
| `mid_attn` | 13 | attention | Mid-network attention output |
| `final_attn` | 25 | attention | Final layer attention output |

**Sweep grid** (`config.py:64-67`):

- layers: `[mid_res, final_res, mid_attn, final_attn]`
- k: `[25, 50, 100, 200]`
- T: `[5, 10, 25]`
- architecture: `[stacked_sae, txcdr]`

Total: 4 layers × 4 k × 3 T × 2 architectures = **96 runs**.

**Training**: 50k steps, lr=5e-5, batch_size=32, Adam(0.9, 0.999), grad_clip=1.0.

**Loss function**: MSE reconstruction with TopK sparsity (same as toy experiments — no L1 penalty). StackedSAE uses k active latents per position (window L0 = k×T). TXCDR uses k×T shared latents (matching total L0).

**Architectures**: Identical to `models.py` in the parent folder — `StackedSAE` and `TemporalCrosscoder` take `(B, T, d)` input and return `(loss, x_hat, activations)`.

## Pipeline

### Step 1: Cache activations

```bash
python NLP/cache_activations.py  # ~1-2 hours on A100
```

Uses HuggingFace transformers with `register_forward_hook` to extract activations from specified layers during a single forward pass. Saves each layer as a memory-mapped `.npy` of shape `(40000, 128, 2304)`.

### Step 2: Run sweep

```bash
python NLP/sweep.py --save-checkpoints  # sequential
python NLP/sweep.py --tmux              # generate tmux parallel script
python NLP/sweep.py --job-index 5       # run single job (for manual parallelism)
```

Each job loads cached activations (memory-mapped, constant RAM), trains a model, logs to wandb and JSON, and saves a checkpoint.

### Step 3: Visualize

```bash
python NLP/viz.py --fit-rho  # generates plots + HMM rho fitting
```

Produces:

- Delta loss heatmaps (TXCDR − SAE) per layer
- FVU comparison across layers
- Loss vs k curves per (layer, T)
- Convergence curves per layer
- HMM rho fitting: lag-1 autocorrelation of learned feature support
- Summary table

### Step 4: Autointerp

```bash
python NLP/autointerp.py  # requires ANTHROPIC_API_KEY
```

Selects the best StackedSAE and best TXCDR by lowest final loss, extracts the top-50 most-active features, finds their highest-activating text windows, and sends them to Claude Haiku for 1-2 sentence interpretations.

## Key Differences from Toy Model Experiments

| Aspect | Toy model | NLP |
|--------|-----------|-----|
| Data source | Synthetic Markov+linear | Gemma 2 2B on FineWeb |
| d_model | 256 | 2304 |
| d_sae | 128 | 18,432 (8× expansion) |
| Temporal structure | Controlled via rho | Natural (unknown rho) |
| Ground truth | Known features | None (HMM rho fitting as proxy) |
| Evaluation | AUC, feature recovery | Loss, FVU, rho distribution |
| Autointerp | N/A | Claude Haiku |

## HMM rho Fitting

Since we don't have ground-truth features in the NLP setting, we estimate the effective temporal correlation of *learned* features. For each latent, we compute the binary support (active/inactive) over consecutive non-overlapping windows, then measure the lag-1 autocorrelation. This gives a distribution of rho values per model.

**Hypothesis**: If TXCDR learns more temporally coherent features, its rho distribution should be shifted rightward compared to StackedSAE.

## Code Map

| File | Purpose |
|------|---------|
| `NLP/config.py` | All hyperparameters, sweep grid, layer specs, paths |
| `NLP/cache_activations.py` | Cache Gemma 2 2B activations to disk |
| `NLP/data.py` | Memory-mapped data loading, sliding window iterator |
| `NLP/train.py` | Training loops (MSE + TopK) for both architectures |
| `NLP/sweep.py` | Sweep orchestrator (sequential, tmux, job-index modes) |
| `NLP/viz.py` | Visualization + HMM rho fitting |
| `NLP/autointerp.py` | Feature interpretation via Claude Haiku |
| `NLP/run_sweep.sh` | Full pipeline launcher |

## Results

*Pending: will be filled in after sweep completes.*
