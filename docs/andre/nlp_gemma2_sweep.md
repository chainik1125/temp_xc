---
author: andre
date: 2026-04-07
tags:
  - results
  - in-progress
---

## NLP Pipeline: Temporal Crosscoders on Gemma 2 2B

Scaling the toy-model temporal crosscoder comparison (see [[tx_v_sae]]) to
real language model activations. Uses Gemma 2 2B with FineWeb data.

## Setup

- **Base model**: `google/gemma-2-2b` (d_model = 2304, 26 layers)
- **Dictionary**: 18,432 latents (d_sae = 2304 * 8, expansion factor 8)
- **Loss**: MSE reconstruction with TopK sparsity (matching toy experiments)
- **Data**: 40,000 chains from FineWeb (sample-10BT), 128 tokens each

### Layers

| Key | Layer | Component | Description |
|-----|-------|-----------|-------------|
| `mid_res` | 13 | residual stream | Middle-layer representations |
| `final_res` | 25 | residual stream | Final-layer representations |
| `mid_attn` | 13 | attention output | Middle-layer attention patterns |
| `final_attn` | 25 | attention output | Final-layer attention patterns |

### Sweep Grid

- **k** (active latents per position): 25, 50, 100, 200
- **T** (window length): 5, 10, 25
- **Architecture**: StackedSAE, TemporalCrosscoder
- Total: 4 layers x 4 k x 3 T x 2 arch = **96 runs**

## How to Reproduce

```bash
cd temporal_crosscoders/NLP

# Step 1: Cache activations (requires GPU, ~2-4h for 40k chains)
python cache_activations.py

# Step 2: Run sweep (edit config.py for grid, reports to wandb)
python sweep.py

# Step 3: Visualize results + HMM rho fitting
python viz.py --fit-rho

# Step 4: Autointerp on best models
python autointerp.py

# Or run the full pipeline:
./run_sweep.sh
```

## Key Questions

1. Does the TXCDR advantage from the toy model extend to real NLP activations?
2. How does the effective rho (temporal autocorrelation) of real features compare
   to the toy-model Markov chain values (0.0, 0.6, 0.9)?
3. Do attention-layer activations show more temporal structure than residual stream?
4. What qualitative differences emerge in autointerp between TXCDR and StackedSAE?

## Results

*Pending sweep completion. Run `python viz.py` to populate.*

### Loss Comparison

*To be filled after sweep.*

### HMM rho Fit

*To be filled after `python viz.py --fit-rho`.*

### Autointerp Highlights

*To be filled after `python autointerp.py`.*

## Files

- `temporal_crosscoders/NLP/config.py` — all hyperparameters
- `temporal_crosscoders/NLP/cache_activations.py` — activation extraction
- `temporal_crosscoders/NLP/sweep.py` — sweep runner (tmux-friendly)
- `temporal_crosscoders/NLP/train.py` — training loops
- `temporal_crosscoders/NLP/viz.py` — visualization + HMM rho analysis
- `temporal_crosscoders/NLP/autointerp.py` — Claude Haiku interpretability
- `temporal_crosscoders/NLP/push_to_hf.py` — upload checkpoints to HuggingFace
