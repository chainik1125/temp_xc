---
author: aniket
date: 2026-02-21
tags:
  - proposal
  - in-progress
---

## Crosscoder Comparison Experiment

Compare three SAE architectures on synthetic two-position data with controllable cross-position correlation.

## Motivation

Crosscoders have a structural advantage when features are shared across positions, since they use a single latent space with per-position encoders/decoders. This experiment measures that advantage as a function of cross-position correlation (rho).

## Architectures

- **Naive SAE**: Single TopK SAE applied to flattened `(batch*n_pos, d)` input. Ignores positional structure entirely.
- **Stacked SAE**: Two independent TopK SAEs, one per position. No information sharing between positions.
- **Crosscoder**: Custom `nn.Module` with per-position encoder weights `W_enc[t,d,s]` and decoder weights `W_dec[s,t,d]`. Shared latent space enables cross-position feature recovery.

## Data Generation

- Two-position toy model with shared linear embedding (no cross-position mixing)
- Features: `num_features=50`, `hidden_dim=100`, `firing_prob=0.22`
- Cross-position correlation via block matrix `[[I, rho*I], [rho*I, I]]` through Gaussian copula
- Sweep `rho` in `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`

## Training

- SAE Lens `TopKTrainingSAE` for naive/stacked (with dead feature resampling, LR scheduling)
- Custom Adam loop for crosscoder (same hyperparameters: `lr=3e-4`, `betas=(0.9, 0.999)`)
- `training_steps=15000`, `batch_size=4096`, `d_sae=100`

## Evaluation Metrics

- **MSE**: Reconstruction error
- **FVU**: Fraction of variance unexplained (`MSE / Var`)
- **L0**: Average number of active latents
- **Mean Max Cosine Similarity**: Per true feature, find best-matching decoder direction
- **Dead Latent Fraction**: Fraction of latents that never fire

## Sweep Parameters

| Parameter | Values |
|-----------|--------|
| `rho` | 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 |
| `arch` | naive_sae, stacked_sae, crosscoder |
| `top_k` | 5, 8, 11, 14, 17, 20 |
| `seeds` | 42, 43, 44 |

Total: 324 runs (6 rho x 3 arch x 6 top_k x 3 seeds).

## Expected Results

- At `rho=0`: All architectures should perform similarly (no cross-position structure to exploit).
- At high `rho`: Crosscoder should show increasing advantage in `mean_max_cos_sim` because it can share latents across positions.

## Code Location

All code is in `src/v2_crosscoder_comparison/`. Entry point: `run_experiments.py`.
