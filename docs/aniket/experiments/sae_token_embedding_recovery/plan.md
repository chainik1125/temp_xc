---
author: aniket
date: 2026-02-21
tags:
  - proposal
  - in-progress
---

## SAE Token Embedding Feature Recovery

Experiment plan for reproducing the results from `notebooks/sae_token_embeddings.ipynb`: **Can SAEs recover true features when trained on frozen token embeddings?**

## Motivation

In real transformer models, SAEs are trained on token embeddings that are fixed combinations of underlying true features. This experiment tests whether an SAE can decompose those frozen combinations back into the true features, or whether it just memorizes the token embeddings themselves.

The notebook finds that SAEs learn the token embeddings rather than the underlying true features. This is an important negative result for understanding SAE limitations.

## Setup

### Toy Model

- 50 mutually orthogonal true features in `R^100` (via `orthogonalize()`)
- Each feature has firing probability `P_i = 11/50 = 0.22`
- Firing magnitudes sampled from `N(1.0, 0.15)`

### Token Embeddings

- Generate 25 "token embeddings" by sampling feature activations once and freezing them
- Each token embedding is a fixed linear combination of the 50 true features
- Training data: uniformly sample from these 25 frozen embeddings (with replacement)

### SAE Configurations

| Config | `d_sae` | `l1_coefficient` | `l1_warm_up_steps` | SAE Type |
|--------|---------|-------------------|--------------------|----------|
| A | 50 | 1.0 | 5000 | `StandardTrainingSAE` (L1) |
| B | 25 | 1.0 | 5000 | `StandardTrainingSAE` (L1) |

**Note**: The notebook uses `StandardTrainingSAE` with L1 penalty, not TopK. This is a deliberate choice for this experiment.

### Training

- `training_tokens = 15_000_000`
- `lr = 3e-4`
- `batch_size = 1024`
- `adam_beta1 = 0.9`, `adam_beta2 = 0.999`
- Trained via SAE Lens `SAETrainer`

## Evaluation

### Metrics

For each trained SAE, compute:

1. **Decoder-feature cosine similarity**: `cos_sim(W_dec^T, embed.weight)` — heatmap showing whether SAE latents align with true features
2. **Decoder-token cosine similarity**: `cos_sim(W_dec^T, model(token_feats)^T)` — heatmap showing whether SAE latents align with token embeddings instead

### Expected Results

- **50-latent SAE**: Does not recover true features. Decoder vectors have low cosine similarity with true feature directions but high similarity with raw token embeddings.
- **25-latent SAE**: Also does not recover true features. More clearly learns token embeddings (near-identity cosine similarity matrix with token embeddings, ~0.85-0.95).

## Reproduction Steps

1. Set random seed for reproducibility
2. Create toy model: `ToyModel(num_feats=50, hidden_dim=100)`
3. Generate 25 frozen token embeddings via `get_training_batch(25, ...)`
4. Define `generate_batch()` that uniformly samples from these 25 frozen embeddings
5. Train 50-latent SAE, plot decoder-feature and decoder-token heatmaps
6. Train 25-latent SAE, plot decoder-feature and decoder-token heatmaps
7. Compare: confirm SAE learns token embeddings, not true features

## Code Location

- Original notebook: `notebooks/sae_token_embeddings.ipynb`
- Experiment script: `src/v1_token_embedding_recovery/sae_token_embeddings.py`

## Key Takeaway

When an SAE only sees a small number of frozen feature combinations (token embeddings), it learns to reconstruct those combinations directly rather than decomposing them into the true underlying features. This has implications for SAE interpretability on real models where the embedding layer maps tokens to fixed vectors.
