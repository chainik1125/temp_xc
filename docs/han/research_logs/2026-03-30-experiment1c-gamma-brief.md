---
author: Han
date: 2026-03-30
tags:
  - design
---

## Brief: Experiment 1c — γ sweep (stochastic emissions)

This document contains everything needed to implement and run Experiment 1c. It is intended as a handover to a new agent.

## Goal

Run a TopK sweep (same as Experiment 1) but on data with **stochastic emissions** (γ < 1), where observations are noisy indicators of the underlying hidden state. Compare against Experiment 1 (γ=1, deterministic emissions) to answer:

1. Which models degrade gracefully when per-token observations are noisy?
2. Can temporal models (TXCDRv2, TFA-pos) **denoise** — recover the hidden state better than the noisy observations?

## Data generation

Use Aniket's pipeline in `src/data_generation/`. The generating process per feature:

1. **Hidden state** z_t ∈ {A, B} follows a Markov chain via the reset process with parameters λ (mixing rate) and q (stationary ON probability)
2. **Emission** s_t | z_t ~ Bernoulli(p_{z_t}), where p_A and p_B are emission probabilities
3. **Magnitude** m_t ~ |N(0, 1)| (half-normal)
4. **Activation** a_t = s_t · m_t; observation x_t = Σ_i a_{i,t} · f_i

The observed autocorrelation decomposes as: Corr(s_t, s_{t+τ}) = (1-λ)^|τ| · γ

where γ = (1-q)·q·(p_B - p_A)² / [μ(1-μ)] is the **emission amplitude**.

**Fixed parameters:**
- λ = 0.3 (ρ = 0.7, moderate temporal persistence)
- μ = 0.5 (target marginal sparsity, same as Experiment 1)
- p_A = 0.0 (never fire when hidden state is OFF)
- 20 features, hidden_dim = 40, dict_width = 40, seq_len = 64, seed = 42

**γ = 0.25 configuration:**
- With p_A=0 and μ = q·p_B = 0.5, and γ = (1-q)/q:
- γ=0.25 → q=0.8, p_B=0.625
- Meaning: features fire 62.5% of the time when hidden ON, never when hidden OFF

Example data generation:

```python
from src.data_generation.configs import (
    DataGenerationConfig, EmissionConfig, TransitionConfig,
    FeatureConfig, SequenceConfig,
)
from src.data_generation.dataset import generate_dataset

cfg = DataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.3, p=0.8),  # q=0.8
    emission=EmissionConfig(p_A=0.0, p_B=0.625),
    features=FeatureConfig(k=20, d=40),
    sequence=SequenceConfig(T=64, n_sequences=2500),
    seed=42,
)
result = generate_dataset(cfg)
# result["x"]: (2500, 64, 40) — what models see
# result["support"]: (2500, 20, 64) — observed binary firings (noisy)
# result["hidden_states"]: (2500, 20, 64) — true hidden state (smooth)
# result["features"]: (20, 40) — ground truth feature directions
```

Note: shape convention differs from our experiment framework. `result["x"]` is (n_seq, T, d) which matches. But `support` and `hidden_states` are (n_seq, k, T) — features × time, needs transposing for our eval code.

## Models to run (6 total)

| Model | Spec class | gen_key | Training |
|-------|-----------|---------|----------|
| TFA-pos | `TFAModelSpec(use_pos_encoding=True)` | seq | 30K steps, batch 64, lr 1e-3 |
| TFA-pos-shuf | `TFAModelSpec(use_pos_encoding=True)` | seq_shuffled | same |
| Stacked T=2 | `StackedSAEModelSpec(T=2)` | window_2 | 30K steps, batch 2048, lr 3e-4 |
| Stacked T=5 | `StackedSAEModelSpec(T=5)` | window_5 | same |
| TXCDRv2 T=2 | `TXCDRv2ModelSpec(T=2)` | window_2 | same |
| TXCDRv2 T=5 | `TXCDRv2ModelSpec(T=5)` | window_5 | same |

## k values

Same 12 as Experiment 1: [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]

Note: TXCDRv2 T=5 will hit k×T > d_sae=40 at k≥9, so skip those.

## Evaluation metrics

### 1. NMSE (same as Experiment 1)

NMSE = Σ||x - x̂||² / Σ||x||²

### 2. AUC (same as Experiment 1)

For windowed models, use **decoder-averaged AUC**: average decoder matrices across positions, then compute cosine similarity with true feature directions. This is implemented in `src/v2_temporal_schemeC/experiment/eval_unified.py`.

### 3. Global feature recovery (NEW)

For each true feature i, find the best-matching latent j (highest decoder cosine similarity — same matching as AUC). Then compute:

- **Local correlation**: `corr(z_j_activations, s_i)` — does latent j track the noisy observation?
- **Global correlation**: `corr(z_j_activations, h_i)` — does latent j track the true hidden state?

If global > local, the model is **denoising** — recovering information about the hidden state that isn't directly visible in single-token observations.

To compute this: run eval data through the model to get latent activations z. For SAE/TFA, z is per-token. For TXCDRv2/Stacked, z is per-window — correlate with the hidden state at each position in the window.

## Implementation approach

The cleanest approach is to **not use the unified experiment framework** for data generation (since it uses `DataConfig` which doesn't support Aniket's `EmissionConfig`). Instead:

1. Generate data using Aniket's pipeline directly
2. Compute scaling factor and build eval data manually
3. Create train/eval generators from the generated data
4. Use the model specs from `src/v2_temporal_schemeC/experiment/model_specs.py` for model creation and training
5. Use `evaluate_model` for NMSE and AUC
6. Add custom code for global/local correlation metric

See `src/v2_temporal_schemeC/run_gamma_sweep.py` for a partial implementation (missing the global correlation metric and only has γ=1 results due to being killed).

## Output

### Results file
Save to `src/v2_temporal_schemeC/results/experiment1c_gamma/results.json`

### Plots
Save to `src/v2_temporal_schemeC/results/experiment1c_gamma/`:
- `exp1c_nmse_vs_k.png` — NMSE vs k, overlay γ=1 (from Experiment 1) and γ=0.25
- `exp1c_auc_vs_k.png` — same for AUC
- `exp1c_global_vs_local.png` — global correlation vs local correlation for each model

### Research log
Write findings to `docs/han/research_logs/2026-03-30-experiment1c-gamma.md`

## Key files

| File | Purpose |
|------|---------|
| `src/data_generation/` | Aniket's data generation pipeline |
| `src/v2_temporal_schemeC/experiment/model_specs.py` | Model specs (SAEModelSpec, TFAModelSpec, TXCDRv2ModelSpec, StackedSAEModelSpec) |
| `src/v2_temporal_schemeC/experiment/eval_unified.py` | `evaluate_model()` with decoder-averaged AUC |
| `src/v2_temporal_schemeC/results/reproduction/` | Experiment 1 (γ=1) results for comparison |
| `src/v2_temporal_schemeC/run_gamma_sweep.py` | Partial implementation (reference, don't use directly) |
| `docs/han/research_logs/2026-03-30-experiment1-topk-sweep.md` | Experiment 1 writeup for reference |

## Running experiments

```bash
# Activate environment
conda activate torchgpu
# Or use directly:
TQDM_DISABLE=1 PYTHONPATH=/home/elysium/temp_xc \
  /home/elysium/miniforge3/envs/torchgpu/bin/python -u <script>
```

GPU: RTX 5090 (32GB). Each model uses ~1.8GB. Can run 2-3 in parallel safely. Use `save_figure()` from `src/utils/plot.py` for dual-resolution output (full + thumbnail).

## Expected findings

- **TXCDRv2** should show least NMSE/AUC degradation from γ=1→0.25 because the shared latent aggregates T noisy observations
- **TFA-pos** should be more robust than Stacked SAE (temporal attention can denoise)
- **Stacked SAE** should degrade most (each position processed independently, no denoising)
- **Global correlation should exceed local correlation** for TXCDRv2 and TFA-pos, confirming they recover the hidden state not just the noisy observation
