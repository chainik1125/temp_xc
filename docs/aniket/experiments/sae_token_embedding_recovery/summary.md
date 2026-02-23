---
author: aniket
date: 2026-02-21
tags:
  - results
  - complete
---

## SAE Token Embedding Feature Recovery: Results

Reproduction and extension of `notebooks/sae_token_embeddings.ipynb`. Code: `src/v1_token_embedding_recovery/sae_token_embeddings.py`.

## Question

When an SAE is trained on a small set of frozen token embeddings (fixed linear combinations of true features), does it recover the true features or memorize the token embeddings?

## Setup

- 50 orthogonal true features in `R^100`
- 25 frozen token embeddings (each a sparse combination of ~11 features)
- Firing magnitudes `~ N(1.0, 0.15)`
- Training: 15M tokens, `lr=3e-4`, `batch_size=1024`
- Seed: 42

## Experiments

We trained SAEs across two sparsity mechanisms and multiple TopK values:

| SAE | Type | `d_sae` | Sparsity | Key parameter |
| --- | ---- | ------- | -------- | ------------- |
| L1 50-latent | `StandardTrainingSAE` | 50 | L1 penalty | `l1_coefficient=1.0` |
| L1 25-latent | `StandardTrainingSAE` | 25 | L1 penalty | `l1_coefficient=1.0` |
| TopK k=1 (50-latent) | `TopKTrainingSAE` | 50 | TopK | `k=1` |
| TopK k=1 (25-latent) | `TopKTrainingSAE` | 25 | TopK | `k=1` |
| TopK k=2 (50-latent) | `TopKTrainingSAE` | 50 | TopK | `k=2` |
| TopK k=2 (25-latent) | `TopKTrainingSAE` | 25 | TopK | `k=2` |
| TopK k=11 (50-latent) | `TopKTrainingSAE` | 50 | TopK | `k=11` |
| TopK k=11 (25-latent) | `TopKTrainingSAE` | 25 | TopK | `k=11` |
| TopK k=22 (50-latent) | `TopKTrainingSAE` | 50 | TopK | `k=22` |
| TopK k=22 (25-latent) | `TopKTrainingSAE` | 25 | TopK | `k=22` |

## Results

### Core finding: SAEs learn token embeddings, not true features

All SAEs fail to recover the 50 true features. Instead they learn the 25 token embeddings. This confirms the original notebook result and holds across both L1 and TopK sparsity.

### L1 SAEs (original notebook reproduction)

#### 50-latent L1 SAE

![L1 50-latent: decoder vs true features](../../../../results/sae_token_embeddings/l1_50_latent_enc_dec_vs_features.png)

![L1 50-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/l1_50_latent_dec_vs_token_embeddings.png)

#### 25-latent L1 SAE

![L1 25-latent: decoder vs true features](../../../../results/sae_token_embeddings/l1_25_latent_enc_dec_vs_features.png)

![L1 25-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/l1_25_latent_dec_vs_token_embeddings.png)

The L1 25-latent SAE shows a near-permutation matrix against token embeddings (~0.85-0.95 cosine similarity), cleanly memorizing one token per latent. The 50-latent version also memorizes tokens but with redundant latents.

### TopK k=11 SAEs (extension)

#### 50-latent TopK k=11 SAE

![TopK k=11 50-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k11_50_latent_enc_dec_vs_features.png)

![TopK k=11 50-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k11_50_latent_dec_vs_token_embeddings.png)

#### 25-latent TopK k=11 SAE

![TopK k=11 25-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k11_25_latent_enc_dec_vs_features.png)

![TopK k=11 25-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k11_25_latent_dec_vs_token_embeddings.png)

TopK with k=11 shows **weaker** token memorization than L1. The permutation structure is blurrier and peak cosine similarities are lower. This is because k=11 forces exactly 11 latents active per input, making one-latent-per-token memorization impossible. The SAE must spread each token's representation across 11 latents.

### TopK k=1 SAEs

With k=1, only one latent fires per input. This is the ideal setting for one-latent-per-token memorization: each of the 25 tokens can map to a single dedicated latent.

#### 50-latent TopK k=1 SAE

![TopK k=1 50-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k1_50_latent_enc_dec_vs_features.png)

![TopK k=1 50-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k1_50_latent_dec_vs_token_embeddings.png)

#### 25-latent TopK k=1 SAE

![TopK k=1 25-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k1_25_latent_enc_dec_vs_features.png)

![TopK k=1 25-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k1_25_latent_dec_vs_token_embeddings.png)

The k=1 25-latent SAE shows a cleaner permutation structure than k=11, with one dominant red entry per row. However, k=1 is noisier than L1: the heatmap has significant negative cosine similarities (blue cells) scattered throughout the background. This is a geometric consequence of packing 25 decoder vectors in 100D space while forcing each to carry the full reconstruction weight of its assigned token — the vectors arrange themselves to be maximally separated, producing anti-alignments (negative cos sim) with non-matched tokens. L1 avoids this by adaptively choosing its own sparsity level, yielding cleaner near-zero off-diagonal entries. The k=1 50-latent SAE also shows stronger token alignment than k=11 did, though messier because 50 latents compete for 25 tokens and some end up redundant or dead. Neither k=1 variant recovers the true features.

### TopK k=2 and k=22 SAEs (higher-K investigation)

Dmitry asked whether a higher K might improve the decoder-to-token-embedding map: "Did you try to see if the map improves if you use a higher K? It seems plausible that we need the initially higher K for the model to learn the K=11 setting." To test this, we ran TopK with k=2 (slightly above k=1) and k=22 (near the maximum of 25 latents).

#### 25-latent TopK k=2 SAE

![TopK k=2 25-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k2_25_latent_enc_dec_vs_features.png)

![TopK k=2 25-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k2_25_latent_dec_vs_token_embeddings.png)

#### 50-latent TopK k=2 SAE

![TopK k=2 50-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k2_50_latent_enc_dec_vs_features.png)

![TopK k=2 50-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k2_50_latent_dec_vs_token_embeddings.png)

#### 25-latent TopK k=22 SAE

![TopK k=22 25-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k22_25_latent_enc_dec_vs_features.png)

![TopK k=22 25-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k22_25_latent_dec_vs_token_embeddings.png)

#### 50-latent TopK k=22 SAE

![TopK k=22 50-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k22_50_latent_enc_dec_vs_features.png)

![TopK k=22 50-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k22_50_latent_dec_vs_token_embeddings.png)

The answer to Dmitry's question is clear: **higher K does not help**. Token memorization degrades monotonically with increasing K:

- **k=2**: partial permutation structure, similar to k=1 but with a uniformly warm (positive) background instead of k=1's sharper peaks-and-blue-valleys pattern
- **k=11**: blurrier, weaker structure
- **k=22**: completely diffuse, no permutation structure whatsoever — every latent is weakly aligned with every token

This is because **sparsity pressure is the mechanism that forces specialization**. When only 1-2 latents fire per input, each active latent must carry most of the reconstruction weight, so the optimal strategy is to align its decoder with a specific token embedding. When 22/25 latents fire, any token can be reconstructed as a diffuse combination of many latents, so no individual latent needs to specialize.

### Why k=11 is particularly problematic

k=11 has two compounding issues:

1. It cannot do one-latent-per-token memorization (forcing 11 active latents per input, which must share across the 25 tokens)
2. Since 11 matches the true feature count per token embedding, partial feature recovery and token memorization are competing optimization strategies — neither wins effectively

## Key Takeaways

1. **SAEs trained on frozen token embeddings memorize tokens, not true features.** This holds regardless of sparsity mechanism (L1 or TopK), dictionary size (25 or 50 latents), and TopK value (k=1 through k=22).

2. **The sparsity mechanism affects *how* the SAE fails, not *whether* it fails.** L1 naturally converges to sparse one-hot encodings that cleanly memorize one token per latent. TopK with high k forces distributed representations that make the memorization messier.

3. **Higher K monotonically degrades token memorization.** Contrary to the hypothesis that higher K might help (Dmitry's inquiry), the progression k=2 → k=11 → k=22 shows steadily worse permutation structure, from partial alignment to completely diffuse. Sparsity pressure is what forces decoder specialization.

4. **TopK k=1 is a worse approximation to L1 than it first appears.** While k=1 produces visible dominant alignments, it also introduces significant negative cosine similarities (anti-alignments) in the background that L1 does not produce. L1's adaptive sparsity yields cleaner memorization.

5. **Implication for real models**: Token embedding layers present fixed feature combinations to SAEs. SAEs will learn these combinations rather than decomposing them, regardless of architecture. This motivates training SAEs on later layers where representations are more context-dependent.
