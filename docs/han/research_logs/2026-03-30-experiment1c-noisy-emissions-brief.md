---
author: Han
date: 2026-03-30
tags:
  - design
---

## Brief: Experiment 1c — TopK sweep with noisy emissions

This document contains everything a new agent needs to implement and run Experiment 1c. Read it fully before starting.

## Context: the project and collaborators

This is a multi-person research project on temporal SAE architectures. Three people work on branches of the same repo:

- **Han** (`han` branch): main experiment pipeline, TFA/TFA-pos analysis, correlation sweeps, this experiment. All code in `src/v2_temporal_schemeC/`.
- **Andre** (`andre` branch): Stacked SAE vs TXCDRv2 sweeps at larger scale (128 features, d=256, π=0.05). Code in `temporal_crosscoders/`. His key contribution: the TXCDRv2 design (k×T active latents for fair comparison with Stacked SAE) and the decoder-averaged AUC approach.
- **Aniket** (`aniket` branch): data generation pipeline with HMM stochastic emissions. Code in `src/data_generation/`. His key contribution: decoupling observed autocorrelation into λ (hidden state decay) and γ (emission amplitude), enabling this experiment.

**This experiment uses Aniket's data generation pipeline** (`src/data_generation/`) which has been pulled into the `han` branch. The pipeline requires `src/utils/logging.py` and `src/shared/temporal_support.py` (both already present).

## Goal

Run a TopK sweep identical to Experiment 1 but with **noisy emissions** (γ=0.25 instead of γ=1). In Experiment 1, observations perfectly reveal the hidden state. Here, features fire only 62.5% of the time when the hidden state is ON, making per-token observations noisy. The questions:

1. Which models degrade gracefully when per-token observations are noisy?
2. Can temporal models (TXCDRv2, TFA-pos) **denoise** — recover the hidden state better than the noisy observations?

## Data generation

### Background: the HMM process

For each feature independently at each position t:

1. **Hidden state** z_t ∈ {A, B} follows a Markov chain. Parametrized via the **reset process**: at each step, with probability λ the chain resets to state B with probability q (and A with 1-q); with probability 1-λ it stays put.
   - **λ** (mixing rate): 0 = frozen, 1 = i.i.d. Our ρ = 1-λ.
   - **q** (stationary ON probability): long-run fraction in state B. Our π.

2. **Emission**: s_t | z_t ~ Bernoulli(p_{z_t})
   - **p_A**: probability of firing when hidden state is A. Default 0.
   - **p_B**: probability of firing when hidden state is B. Default 1.
   - With defaults: s_t = z_t (deterministic, our Experiment 1 setup).

3. **Magnitude**: m_t ~ |N(0, 1)|

4. **Observation**: x_t = Σ_i (s_{i,t} · m_{i,t}) · f_i

The observed autocorrelation: Corr(s_t, s_{t+τ}) = (1-λ)^|τ| · γ, where γ = (1-q)·q·(p_B-p_A)² / [μ(1-μ)].

### Configuration for this experiment

**Fixed**: λ=0.3 (ρ=0.7), μ=0.5, p_A=0.0, 20 features, d=40, dict_width=40, T=64, seed=42.

**γ=0.25**: With p_A=0 and μ=q·p_B=0.5, γ=(1-q)/q. So q=0.8, p_B=0.625.

Meaning: features fire 62.5% of the time when hidden ON, never when OFF. Per-token observation is noisy but temporal averaging across a window can denoise.

### Code example

```python
from src.data_generation.configs import (
    DataGenerationConfig, EmissionConfig, TransitionConfig,
    FeatureConfig, SequenceConfig,
)
from src.data_generation.dataset import generate_dataset

cfg = DataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.3, p=0.8),
    emission=EmissionConfig(p_A=0.0, p_B=0.625),
    features=FeatureConfig(k=20, d=40),
    sequence=SequenceConfig(T=64, n_sequences=2500),
    seed=42,
)
result = generate_dataset(cfg)
# result["x"]: (n_seq, T, d) — what models see
# result["support"]: (n_seq, k, T) — observed binary firings (noisy)
# result["hidden_states"]: (n_seq, k, T) — true hidden state (smooth)
# result["features"]: (k, d) — ground truth feature directions
```

**Shape caveat**: `support` and `hidden_states` are (n_seq, k, T) — features × time. Our eval code expects (n_seq, T, k). Transpose as needed.

## Models (6 total)

| Model | Spec class | Data format | Training config |
|-------|-----------|-------------|-----------------|
| TFA-pos | `TFAModelSpec(use_pos_encoding=True)` | sequences (B, T, d) | 30K steps, batch 64, lr 1e-3 |
| TFA-pos-shuf | `TFAModelSpec(use_pos_encoding=True)` | shuffled sequences | same |
| Stacked T=2 | `StackedSAEModelSpec(T=2)` | windows (B, 2, d) | 30K steps, batch 2048, lr 3e-4 |
| Stacked T=5 | `StackedSAEModelSpec(T=5)` | windows (B, 5, d) | same |
| TXCDRv2 T=2 | `TXCDRv2ModelSpec(T=2)` | windows (B, 2, d) | same |
| TXCDRv2 T=5 | `TXCDRv2ModelSpec(T=5)` | windows (B, 5, d) | same |

**TXCDRv2** uses k×T active latents (not k), matching Stacked SAE's total L0. This is Andre's v2 design for fair sparsity comparison. TXCDRv2 T=5 hits k×T > d_sae=40 at k≥9, so skip those k values.

## k values

Same 12 as Experiment 1: [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]

## Evaluation metrics

### 1. NMSE

NMSE = Σ||x - x̂||² / Σ||x||². Same as Experiment 1. Measures reconstruction quality of the noisy observations.

### 2. Decoder-averaged AUC

For windowed models: average decoder matrices across positions, then compute cosine similarity with true feature directions. Implemented in `_compute_auc()` in `src/v2_temporal_schemeC/experiment/eval_unified.py`. Measures whether the model's dictionary recovers the ground truth feature directions.

### 3. Global feature recovery (NEW metric)

This is the novel metric for this experiment. For each true feature i:

1. Find the best-matching latent j (highest |cos(decoder_j, f_i)| — same matching used for AUC).
2. Run eval data through the model to get latent activations.
3. Compute two correlations:
   - **Local correlation**: Pearson corr(z_j activations, s_i) — does latent j track the noisy observation?
   - **Global correlation**: Pearson corr(z_j activations, h_i) — does latent j track the true hidden state?

If global > local, the model is **denoising**: its latent representations contain more information about the underlying state than the raw observations do.

For per-token models (TFA-pos): z_j and s_i/h_i are both per-token, correlate directly.
For windowed models (TXCDRv2, Stacked): z_j is per-window. Correlate with h_i at each position in the window, then average.

## Architecture and code design

### Unified experiment framework (`src/v2_temporal_schemeC/experiment/`)

This is the shared infrastructure. Key components:

- **`model_specs.py`**: `ModelSpec` classes for each architecture. Each has `create()`, `train()`, `eval_forward()`, `decoder_directions()`. Use these for model creation and training — don't instantiate models directly.
- **`eval_unified.py`**: `evaluate_model(spec, model, eval_data, ...)` returns an `EvalResult` with NMSE, L0, AUC. Uses decoder-averaging for windowed models.
- **`sweeps.py`**: `run_topk_sweep()` and `run_l1_sweep()` with optional `cache_dir` for model caching.
- **`data_pipeline.py`**: `DataConfig` and `build_data_pipeline()`. Currently only supports independent Markov chains, NOT Aniket's HMM emissions. For this experiment, generate data with Aniket's pipeline and build generators manually.
- **`results_io.py`**: JSON save/load with numpy-safe encoding.

### Model caching

`sweeps.py` supports `cache_dir` parameter. After training, model state_dicts are saved as `.pt` files. On re-run, cached models load instantly. This avoids retraining when only evaluation changes. Cached models are in `src/v2_temporal_schemeC/model_cache/`.

For this experiment, implement caching manually if not using `run_topk_sweep()` — save `model.state_dict()` after training, check for cached checkpoint before training.

### Key model architectures

- **Shared SAE** (`relu_sae.py`): Standard per-token SAE. `forward(x) → (x_hat, z)` where x is (B, d).
- **TFA** (`tfa/saeTemporal.py`): Causal attention over sequence. `forward(x) → (x_recons, results_dict)` where x is (B, T, d). `use_pos_encoding=True` adds sinusoidal PE to attention Q/K.
- **Stacked SAE** (`stacked_sae.py`): T independent SAEs per position. `forward(x) → (loss, x_hat, z)` where x is (B, T, d).
- **TXCDRv2** (`temporal_crosscoder.py` via `TXCDRv2ModelSpec`): Shared-latent crosscoder with k×T active latents. Same `TemporalCrosscoder` class, but the spec passes `k_effective = k * T` to the constructor.

### Plotting

Use `save_figure(fig, path)` from `src/utils/plot.py` — saves both full-res PNG and a `.thumb.png` thumbnail. **Never read full-res PNGs** (they blow up context). Always read `.thumb.png` for inspection.

### Color convention for plots

Same method = same color, T variants differ by marker/linestyle:
- T=2: solid line, circle marker
- T=5: dashed line, triangle marker
- Shuffled: dashed line, same color as temporal

## Existing results for comparison

Experiment 1 (γ=1) results are in `src/v2_temporal_schemeC/results/reproduction/`. Each model has a JSON with `topk` (list of per-k results) and `l1` (list of per-L1 results). Load these to overlay γ=1 curves on the γ=0.25 plots.

## Output

### Results
`src/v2_temporal_schemeC/results/experiment1c_noisy/results.json`

### Plots
`src/v2_temporal_schemeC/results/experiment1c_noisy/`:
- `nmse_vs_k.png` — NMSE vs k for all 6 models, with γ=1 Experiment 1 curves overlaid
- `auc_vs_k.png` — AUC vs k, same overlay
- `global_vs_local_recovery.png` — global correlation vs local correlation per model/k

### Research log
`docs/han/research_logs/2026-03-30-experiment1c-noisy-emissions.md`

## Running

```bash
TQDM_DISABLE=1 PYTHONPATH=/home/elysium/temp_xc \
  /home/elysium/miniforge3/envs/torchgpu/bin/python -u <script>
```

GPU: RTX 5090 (32GB). Each model uses ~1.8GB. Run 2-3 in parallel safely. Always disable tqdm (`TQDM_DISABLE=1`).

## Expected findings

- **TXCDRv2** should show least NMSE/AUC degradation from γ=1→0.25 — the shared latent aggregates T noisy observations, effectively denoising
- **TFA-pos** should be more robust than Stacked SAE — temporal attention can attend to multiple positions and aggregate
- **Stacked SAE** should degrade most — each position processed independently, no temporal averaging to denoise
- **Global correlation > local correlation** for TXCDRv2 and TFA-pos, confirming they recover the hidden state not just the noisy observation
- **Stacked SAE** should have global ≈ local — no denoising capability
