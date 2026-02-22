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

We trained four SAE variants:

| SAE | Type | `d_sae` | Sparsity | Key parameter |
|-----|------|---------|----------|---------------|
| L1 50-latent | `StandardTrainingSAE` | 50 | L1 penalty | `l1_coefficient=1.0` |
| L1 25-latent | `StandardTrainingSAE` | 25 | L1 penalty | `l1_coefficient=1.0` |
| TopK k=11 (50-latent) | `TopKTrainingSAE` | 50 | TopK | `k=11` |
| TopK k=11 (25-latent) | `TopKTrainingSAE` | 25 | TopK | `k=11` |
| TopK k=1 (50-latent) | `TopKTrainingSAE` | 50 | TopK | `k=1` |
| TopK k=1 (25-latent) | `TopKTrainingSAE` | 25 | TopK | `k=1` |

## Results

### Core finding: SAEs learn token embeddings, not true features

All SAEs fail to recover the 50 true features. Instead they learn the 25 token embeddings. This confirms the original notebook result and holds across both L1 and TopK sparsity.

### L1 SAEs (original notebook reproduction)

**50-latent L1 SAE**

![L1 50-latent: decoder vs true features](../../../../results/sae_token_embeddings/l1_50_latent_enc_dec_vs_features.png)

![L1 50-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/l1_50_latent_dec_vs_token_embeddings.png)

**25-latent L1 SAE**

![L1 25-latent: decoder vs true features](../../../../results/sae_token_embeddings/l1_25_latent_enc_dec_vs_features.png)

![L1 25-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/l1_25_latent_dec_vs_token_embeddings.png)

The L1 25-latent SAE shows a near-permutation matrix against token embeddings (~0.85-0.95 cosine similarity), cleanly memorizing one token per latent. The 50-latent version also memorizes tokens but with redundant latents.

### TopK k=11 SAEs (extension)

**50-latent TopK k=11 SAE**

![TopK k=11 50-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k11_50_latent_enc_dec_vs_features.png)

![TopK k=11 50-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k11_50_latent_dec_vs_token_embeddings.png)

**25-latent TopK k=11 SAE**

![TopK k=11 25-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k11_25_latent_enc_dec_vs_features.png)

![TopK k=11 25-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k11_25_latent_dec_vs_token_embeddings.png)

TopK with k=11 shows **weaker** token memorization than L1. The permutation structure is blurrier and peak cosine similarities are lower. This is because k=11 forces exactly 11 latents active per input, making one-latent-per-token memorization impossible. The SAE must spread each token's representation across 11 latents.

### TopK k=1 SAEs (testing the hypothesis)

With k=1, only one latent fires per input. This is the ideal setting for one-latent-per-token memorization: each of the 25 tokens can map to a single dedicated latent.

**50-latent TopK k=1 SAE**

![TopK k=1 50-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k1_50_latent_enc_dec_vs_features.png)

![TopK k=1 50-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k1_50_latent_dec_vs_token_embeddings.png)

**25-latent TopK k=1 SAE**

![TopK k=1 25-latent: decoder vs true features](../../../../results/sae_token_embeddings/topk_k1_25_latent_enc_dec_vs_features.png)

![TopK k=1 25-latent: decoder vs token embeddings](../../../../results/sae_token_embeddings/topk_k1_25_latent_dec_vs_token_embeddings.png)

The k=1 25-latent SAE shows a dramatically cleaner permutation structure than k=11, with one strong red entry per row — very similar to L1. The optimal strategy with k=1 is to assign each of the 25 tokens to one dedicated latent whose decoder vector matches that token embedding. The k=1 50-latent SAE also shows stronger token alignment than k=11 did, though messier because 50 latents compete for 25 tokens and some end up redundant or dead. Neither k=1 variant recovers the true features — the fundamental limitation is the same regardless of sparsity mechanism.

## Key Takeaways

1. **SAEs trained on frozen token embeddings memorize tokens, not true features.** This holds regardless of sparsity mechanism (L1 or TopK) and dictionary size (25 or 50 latents).

2. **The sparsity mechanism affects *how* the SAE fails, not *whether* it fails.** L1 naturally converges to sparse one-hot encodings that cleanly memorize one token per latent. TopK with high k forces distributed representations that make the memorization messier.

3. **TopK k=1 recovers clean token memorization.** When only one latent fires per input, TopK behaves like L1 and produces a clean permutation matrix against token embeddings. This confirms that the blurriness at k=11 is a direct consequence of the forced sparsity level, not a fundamental difference in the optimization landscape.

4. **Implication for real models**: Token embedding layers present fixed feature combinations to SAEs. SAEs will learn these combinations rather than decomposing them, regardless of architecture. This motivates training SAEs on later layers where representations are more context-dependent.
