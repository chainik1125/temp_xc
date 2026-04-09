# NLP Temporal Crosscoders (Gemma 2 2B)

Trains StackedSAE and TemporalCrosscoder models on cached Gemma 2 2B-IT activations from FineWeb.

## Setup

```bash
cd /home/cs29824/andre/temp_xc
uv sync
```

## Pipeline

### 1. Cache activations

```bash
cd /home/cs29824/andre/temp_xc/temporal_crosscoders/NLP

# Cache all 4 layers (mid_res, final_res, mid_attn, final_attn)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python cache_activations.py

# Or cache specific layers
python cache_activations.py --layers mid_res final_res
```

This streams FineWeb, tokenizes to `SEQ_LENGTH=32`, runs Gemma 2 2B-IT forward passes with hooks, and saves `(NUM_CHAINS, SEQ_LENGTH, d_act)` mmap files to `cached_activations/`.

### 2. Run sweep (training)

Single GPU:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python sweep.py
```

Two GPUs (split layers across GPUs):
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python sweep.py --layer mid_res final_res

# Terminal 2
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python sweep.py --layer mid_attn final_attn
```

Other sweep options:
```bash
python sweep.py --dry-run                          # print plan without training
python sweep.py --steps 1000                       # quick test run
python sweep.py --layer mid_res --k 50 --T 5      # single combo
python sweep.py --no-checkpoint                    # skip saving checkpoints
```

### 3. Visualize results

```bash
python viz.py
```

### 4. Autointerp (optional)

Requires `ANTHROPIC_API_KEY` set.

```bash
python autointerp.py
python autointerp.py --top-features 20
```

### 5. Sentence-level feature visualization (`sentence.py`)

Visualizes top-N feature activations across the tokens of a single sequence,
side-by-side for StackedSAE vs TXCDR. Loads autointerp explanations as a
right-side legend if they exist.

```bash
# Random chain, default exclusive selection (one feature per token position)
python sentence.py

# Specific chain
python sentence.py --chain 42

# Specific model + checkpoint
python sentence.py --layer mid_res --k 100 --T 5 --chain 42

# Top-N by magnitude instead of one-per-position
python sentence.py --select magnitude --n-features 32

# Custom output dir + colormaps
python sentence.py --chain 42 \
  --output-dir viz_outputs/sentences/ \
  --cmap-sae viridis --cmap-tx magma
```

Selection modes:

- `exclusive` (default) — for each of the `seq_len` token positions, pick the
  feature whose total activation mass is most concentrated at that position.
  Greedy assignment, no feature reused. Should produce a clear diagonal pattern.
- `magnitude` — pick the top-N features per model by total activation magnitude
  across the sequence (use `--n-features N`).

Output: `sentence_{layer}_k{k}_T{T}_chain{N}_{select}.png` and matching `_stats.txt`.

## Current sweep grid

| Parameter | Values |
|-----------|--------|
| Layers | mid_res, final_res, mid_attn, final_attn |
| k (top-k) | 50, 100 |
| T (window) | 2, 5 |
| Architecture | stacked_sae, txcdr |

Total: 4 layers x 2 k x 2 T x 2 arch = **32 jobs** (16 per GPU if split by layer).

## Key config (config.py)

- `SEQ_LENGTH = 32` — tokens per sequence
- `NUM_CHAINS = 24,000` — sequences cached from FineWeb
- `D_MODEL = 2304` — Gemma 2 2B hidden dim
- `D_SAE = 18,432` — dictionary size (8x expansion)
- `TRAIN_STEPS = 10,000`
- `LEARNING_RATE = 3e-4`
- `BATCH_SIZE = 256` (via `batch_size_for_T`)

## Metrics to watch

- **FVU** (fraction of variance unexplained): <0.05 is good, <0.01 is excellent
- **L0** (window-level sparsity): = k * T active features per window
- **loss**: raw MSE, scale depends on activation magnitudes
