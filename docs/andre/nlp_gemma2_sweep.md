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
- **Dictionary**: 18,432 latents (d_sae = 2304 x 8, expansion factor 8)
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
- **D_SAE**: 18,432 (2304 x 8)
- Total: 4 layers x 4 k x 3 T x 2 arch = **96 runs**

## How to Reproduce

```bash
cd temporal_crosscoders/NLP

# Step 1: Cache activations (requires GPU, ~2-4h for 40k chains)
python cache_activations.py

# Step 2: Run sweep (edit config.py for grid, reports to wandb)
python sweep.py                     # sequential
python sweep.py --tmux > launch.sh  # tmux parallel
bash launch.sh

# Step 3: Visualize results + HMM rho fitting + entropy + contrast
python viz.py --all

# Step 4: Autointerp on best models (requires ANTHROPIC_API_KEY)
python autointerp.py

# Step 5: Push checkpoints to HuggingFace (manual)
python push_to_hf.py --repo-id YOUR_HF_USER/temporal-crosscoders-nlp --dry-run
```

## Key Questions

1. Does the TXCDR advantage from the toy model extend to real NLP activations?
2. How does the effective rho (temporal autocorrelation) of real features compare
   to the toy-model Markov chain values (0.0, 0.6, 0.9)?
3. Do attention-layer activations show more temporal structure than residual stream?
4. What qualitative differences emerge in autointerp between TXCDR and StackedSAE?
5. **[NEW]** Does StackedSAE exhibit higher feature activation entropy than TXCDR?
   (Prediction: yes, because features are fit independently per position.)
6. **[NEW]** Within a 25-token sentence, how do activation patterns differ
   between the two architectures? (Expect TXCDR to show more temporal coherence.)

## Results

*Pending sweep completion. Run `python viz.py --all` to populate.*

### Loss Comparison

*To be filled after sweep.*

### Entropy Analysis

- Training logs now include per-step **entropy** (bits) of feature activation
  distributions for both architectures.
- The `viz.py` script generates `entropy_comparison.png` showing StackedSAE vs
  TXCDR entropy side-by-side across all (layer, k, T) combos.
- **Prediction**: StackedSAE should have higher entropy (more uniform, position-
  independent firing) while TXCDR should learn more structured, temporally
  correlated activations (lower entropy).

### Feature Activation Contrast

- `viz.py --contrast` generates a per-sentence activation heatmap comparing how
  the top-30 features fire across 25 token positions for both architectures.
- Includes per-position entropy plot to quantify temporal structure differences.
- Example sentence from chain 42 of the cached activations.

### HMM rho Fit

*To be filled after `python viz.py --fit-rho`.*

### Autointerp Highlights

*To be filled after `python autointerp.py`.*

## Files

- `temporal_crosscoders/NLP/config.py` — all hyperparameters (modifiable grid)
- `temporal_crosscoders/NLP/cache_activations.py` — activation extraction
- `temporal_crosscoders/NLP/sweep.py` — sweep runner (tmux-friendly, --job-index)
- `temporal_crosscoders/NLP/train.py` — training loops (logs loss, L0, FVU, entropy)
- `temporal_crosscoders/NLP/viz.py` — visualization + HMM rho + entropy + contrast
- `temporal_crosscoders/NLP/autointerp.py` — Claude Haiku interpretability
- `temporal_crosscoders/NLP/push_to_hf.py` — upload checkpoints to HuggingFace

## Checkpoints

Saved locally in `temporal_crosscoders/NLP/checkpoints/` (gitignored).
Push to HuggingFace with `push_to_hf.py` when ready.
