---
author: Han
date: 2026-03-30
tags:
  - design
---

## Brief: Experiment 1d — TopK sweep at limiting correlations (ρ=0 and ρ=1)

This document contains everything a new agent needs to implement and run Experiment 1d.

## Goal

Run the Experiment 1 TopK sweep twice, at the two extreme values of temporal correlation:

- **ρ=0.0**: i.i.d. features. No temporal structure exists. Any advantage is purely architectural.
- **ρ=1.0**: frozen features. Each feature is either ON for the entire sequence or OFF, determined at initialization. Maximum possible temporal signal — every context token carries the same information.

Compare against Experiment 1 (mixed ρ ∈ {0.0, 0.3, 0.5, 0.7, 0.9}) and the correlation sweep (Experiment 1b, uniform ρ) to complete the picture of how temporal persistence affects each architecture.

## Context: the project

Three collaborators on branches of the same repo:

- **Han** (`han` branch): main experiments, TFA/TFA-pos, this experiment. Code in `src/v2_temporal_schemeC/`.
- **Andre** (`andre` branch): Stacked SAE vs TXCDRv2 at scale. Code in `temporal_crosscoders/`.
- **Aniket** (`aniket` branch): HMM data generation pipeline. Code in `src/data_generation/`.

This experiment uses Han's experiment framework only. Aniket's pipeline is not needed (deterministic emissions, γ=1).

## Data generation

Use the existing unified framework in `src/v2_temporal_schemeC/experiment/data_pipeline.py`. The `DataConfig` class supports per-feature ρ values. For this experiment, set **all 20 features to the same ρ**.

### ρ=0.0 configuration

```python
from src.v2_temporal_schemeC.experiment import DataConfig

cfg_iid = DataConfig(
    num_features=20,
    hidden_dim=40,
    seq_len=64,
    pi=[0.5] * 20,
    rho=[0.0] * 20,  # all features i.i.d.
    dict_width=40,
    seed=42,
    eval_n_seq=2000,
)
```

At ρ=0.0, each feature's support is an independent Bernoulli(0.5) sequence. Consecutive tokens share features only by chance (~5 shared out of 10 active, since π=0.5). There is zero temporal information for any model to exploit.

### ρ=1.0 configuration

```python
cfg_frozen = DataConfig(
    num_features=20,
    hidden_dim=40,
    seq_len=64,
    pi=[0.5] * 20,
    rho=[1.0] * 20,  # all features frozen
    dict_width=40,
    seed=42,
    eval_n_seq=2000,
)
```

**Important subtlety about ρ=1.0:** The Markov chain transition probabilities become p(OFF→ON) = π(1-ρ) = 0 and p(ON→OFF) = (1-π)(1-ρ) = 0. The chain never transitions — each feature is frozen in its initial state for the entire sequence. With π=0.5, each feature is independently ON or OFF with probability 0.5, but that state is constant across all 64 positions.

This means:
- Every token in a sequence has the **exact same active features** (same x_t up to magnitude noise if using variable magnitudes, or identical x_t with unit magnitudes)
- E[L0] = 10 per token (same as Experiment 1)
- Content matching is trivially perfect — every context token is identical to the current token
- The only variation across positions is magnitude noise (if any)

With our current setup (unit magnitudes, binary support), all tokens in a sequence are **identical**: x_t = x_{t'} for all t, t'. This is a degenerate case where even a per-token Shared SAE achieves perfect reconstruction by memorizing the single activation pattern. To make ρ=1.0 non-trivial, we should verify whether our toy model uses unit magnitudes or variable magnitudes.

Check `src/v2_temporal_schemeC/markov_data_generation.py` — if magnitudes are unit (binary support × 1.0), then ρ=1.0 makes all tokens identical and every model achieves NMSE≈0. If magnitudes are variable (e.g., sampled from |N(1, 0.15²)|), then tokens within a sequence share the same active features but with different magnitudes, making reconstruction non-trivial.

**If magnitudes are unit:** Consider using variable magnitudes for this experiment only (like Andre's setup with FEAT_MEAN=1.0, FEAT_STD=0.15), or accept that ρ=1.0 is trivially solved and focus the analysis on **AUC** (feature recovery) rather than NMSE.

## Models (5 total)

| Model | Spec class | gen_key | Training |
|-------|-----------|---------|----------|
| TFA-pos | `TFAModelSpec(use_pos_encoding=True)` | seq | 30K steps, batch 64, lr 1e-3 |
| Stacked T=2 | `StackedSAEModelSpec(T=2)` | window_2 | 30K steps, batch 2048, lr 3e-4 |
| Stacked T=5 | `StackedSAEModelSpec(T=5)` | window_5 | same |
| TXCDRv2 T=2 | `TXCDRv2ModelSpec(T=2)` | window_2 | same |
| TXCDRv2 T=5 | `TXCDRv2ModelSpec(T=5)` | window_5 | same |

**TXCDRv2** uses k×T active latents (not k), matching Stacked SAE total L0. This is Andre's v2 design. TXCDRv2 T=5 hits k×T > d_sae=40 at k≥9, so skip those k values.

## k values

Same 12 as Experiment 1: [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]

## Evaluation metrics

### 1. NMSE
NMSE = Σ||x - x̂||² / Σ||x||²

### 2. Decoder-averaged AUC
For windowed models: average decoder matrices across positions before computing cosine similarity with true feature directions. Implemented in `_compute_auc()` in `src/v2_temporal_schemeC/experiment/eval_unified.py`.

## Implementation

Use the unified framework directly. The correlation sweep script (`src/v2_temporal_schemeC/run_corr_sweep_windowed.py`) already supports arbitrary ρ values — it can be adapted or the sweep runner can be called with the appropriate `DataConfig`.

Alternatively, use `run_topk_sweep()` from `src/v2_temporal_schemeC/experiment/sweeps.py`:

```python
from src.v2_temporal_schemeC.experiment import (
    DataConfig, run_topk_sweep,
    TFAModelSpec, TXCDRv2ModelSpec, StackedSAEModelSpec, ModelEntry,
)

cfg = DataConfig(
    num_features=20, hidden_dim=40, seq_len=64,
    pi=[0.5]*20, rho=[0.0]*20,  # or [1.0]*20
    dict_width=40, seed=42, eval_n_seq=2000,
)

models = [
    ModelEntry("TFA-pos", TFAModelSpec(use_pos_encoding=True), "seq",
               training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
    ModelEntry("Stacked-T2", StackedSAEModelSpec(T=2), "window_2",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
    ModelEntry("Stacked-T5", StackedSAEModelSpec(T=5), "window_5",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
    ModelEntry("TXCDRv2-T2", TXCDRv2ModelSpec(T=2), "window_2",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
    ModelEntry("TXCDRv2-T5", TXCDRv2ModelSpec(T=5), "window_5",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
]

results = run_topk_sweep(
    models=models,
    k_values=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20],
    data_config=cfg,
    device=device,
    cache_dir="src/v2_temporal_schemeC/model_cache",
)
```

The sweep runner supports `cache_dir` for saving trained model checkpoints (see `src/v2_temporal_schemeC/experiment/sweeps.py`).

## Architecture and code

See the Experiment 1c brief (`2026-03-30-experiment1c-noisy-emissions-brief.md`) for full details on:
- Unified experiment framework (`src/v2_temporal_schemeC/experiment/`)
- Model spec classes and their interfaces
- Model caching infrastructure
- Plotting conventions (same color = same method, T=2 solid, T=5 dashed)
- `save_figure()` for dual-resolution output

## Output

### Results
`src/v2_temporal_schemeC/results/experiment1d_limiting_rho/`
- `rho0_results.json` — ρ=0.0 sweep
- `rho1_results.json` — ρ=1.0 sweep

### Plots
- `exp1d_rho0_nmse.png`, `exp1d_rho0_auc.png` — ρ=0.0 curves
- `exp1d_rho1_nmse.png`, `exp1d_rho1_auc.png` — ρ=1.0 curves
- `exp1d_comparison.png` — overlay ρ=0, ρ=0.7 (from Experiment 1), ρ=1.0 for selected models

### Research log
`docs/han/research_logs/2026-03-30-experiment1d-limiting-rho.md`

## Existing results for comparison

- **Experiment 1** (mixed ρ): `src/v2_temporal_schemeC/results/reproduction/` — all 12 k values for 11 models
- **Correlation sweep** (uniform ρ at 5 values): `src/v2_temporal_schemeC/results/correlation_sweep/` — ρ ∈ {0.0, 0.3, 0.5, 0.7, 0.9} at k=3 and k=10 for 9 models

The correlation sweep already has ρ=0.0 data but only at k=3 and k=10 (2 points). This experiment provides the full 12-point k sweep.

## Running

```bash
TQDM_DISABLE=1 PYTHONPATH=/home/elysium/temp_xc \
  /home/elysium/miniforge3/envs/torchgpu/bin/python -u <script>
```

GPU: RTX 5090 (32GB). Run 2-3 models in parallel safely (~1.8GB each).

## Expected findings

**ρ=0.0 (i.i.d.):**
- All models should perform similarly on NMSE — no temporal signal to exploit
- TXCDRv2 might have slightly worse NMSE than Stacked SAE (shared-latent constraint without temporal benefit)
- AUC: TXCDRv2 should still have better decoder-averaged AUC than Stacked SAE (shared latent enforces cross-position consistency even without temporal signal)
**ρ=1.0 (frozen):**
- NMSE: likely near-zero for all models if magnitudes are unit (all tokens identical). With variable magnitudes, TXCDRv2 may have an advantage (can average across positions)
- AUC: interesting test — do models learn better feature directions when features are perfectly persistent? The data is extremely regular (same features every position), which might help or hurt depending on how the optimizer handles the redundancy
