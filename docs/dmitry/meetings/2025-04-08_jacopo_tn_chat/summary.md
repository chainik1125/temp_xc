---
author: Dmitry Manning-Coe
date: 2025-04-08
tags:
  - design
  - in-progress
---

## TN-SAE Proposals vs Jacopo Conversation Insights

Summary of a brainstorming session with Jacopo Gliozzi on tensor-network architectures for temporal sparse autoencoders. Synthesizes the 4 proposals in [[temporal_sae_tensor_network_writeup (1)]] with ideas from the conversation.

---

## Pros/Cons of Current Proposals (informed by the conversation)

### Proposal 1: Pre-Sandwich Temporal Layer

| Pros | Cons |
|------|------|
| Simplest to bolt on -- just prepend temporal mixing to existing TopKSAE | Feature attribution is murky: what part of a learned feature is local vs inherited from preprocessing? |
| Modular: reuses existing SAE unchanged | Jacopo's product-state insight applies: if T_theta is invertible, no actual compression occurs -- just scrambles activations |
| Low barrier to experimentation | May just "explain away" the predictable part (denoiser) rather than discover genuinely new global features |
| | Does not produce a clean local-vs-global feature decomposition |

**Conversation verdict**: Not discussed directly, but Jacopo's insistence that you need genuine compression (not just scrambling) suggests this is the weakest approach. The temporal layer acts in activation space, which is the wrong level of abstraction -- the conversation converged on wanting TN structure in the latent space.

### Proposal 2: Full Pairwise Feature Mixing Sandwich

| Pros | Cons |
|------|------|
| Maximum expressiveness -- can capture cross-feature, cross-time interactions | Interpretability destruction: after mixing, a feature is a linear combination of all features at all times |
| MPO parameterization is the canonical TN operator setting | Overpowered for the aliased regime (which has independent features) -- cross-feature mixing isn't needed |
| Subsumes Proposal 3 as a special case | Jacopo's concern about "no bond dimension" in naive pairwise scheme -- you're not really compressing |
| | Parameter explosion: (L*m)^2 even with TN structure |

**Conversation verdict**: This is essentially what Dmitry initially presented to Jacopo (the pairwise contraction of local codes). Jacopo's response was to push toward something more principled -- either the direct MPS approach or the hierarchical contraction. The "no bond dimension" critique (timestamp ~1:45) is directly about this architecture.

### Proposal 3: Per-Feature Temporal Layer

| Pros | Cons |
|------|------|
| **Already implemented** (`per_feature_temporal.py`) | Cannot capture cross-feature temporal coupling |
| Cleanest interpretability: feature k stays feature k | Dense K^{(k)} kernel is (T,T) per feature -- scales as T^2 * d_sae |
| Directly matched to leaky-reset benchmark (independent per-feature Markov chains) | No built-in multi-scale structure -- doesn't exploit locality of correlations |
| Initializes as independent SAE (K=0) -- safe default | May behave as pure denoiser rather than discovering structure |

**Conversation verdict**: Not directly discussed, but it's the natural "cleanest first step" that the conversation implicitly points toward. Jacopo would likely approve of the interpretability preservation but push for a more TN-native parameterization of the kernel (e.g., exponential decay rather than dense matrix).

### Proposal 4: One-Legged MPS Autoencoder

| Pros | Cons |
|------|------|
| Most principled probabilistic model -- clean separation of local evidence and global prior | The key confusion from the conversation: how do you pipe activation info into the MPS? |
| **BaumWelchFactorialAE already implements** the q=2 per-feature case | Discrete latent alphabet (q symbols) is restrictive for continuous-magnitude features |
| Variant A (marginal decode) preserves clean interpretability | Forward-backward cost O(L * q^2 * chi^2) |
| Very parameter-efficient (55K params in the concrete example) | Jacopo's product-state critique: encoder maps each x_t independently to phi_t, so input to MPS is a product state |
| | Variant B's bond-state decoder has gauge freedom issues |

**Conversation verdict**: This is closest to where the conversation converged. The "W-contraction" idea (timestamp ~1:30) is essentially Dmitry trying to solve the "how to pipe activations into MPS" problem. Jacopo validates the overall direction but identifies the key challenge: the existing BW factorial model is a per-feature specialization that sidesteps the harder question of how to build a single correlated MPS over all features.

---

## Alternative Proposals from the Conversation

### Alt A: Iterative Local Contraction (Simplified MERA)

**Source**: Jacopo's suggestion (~1:37-1:38) that iterating pairwise contractions captures correlations at increasing range, plus the MERA discussion.

**Architecture**:

1. Encode locally: a_t = TopK(ReLU(W_enc @ x_t + b_enc))
2. Layer l: for each pair (t, t+1), apply learned mixing tensor C^{(l)} that updates a_t using a_{t+1} (with residual)
3. After N_layers iterations (each doubling effective range), decode: x_hat_t = W_dec @ v_t^{(N)} + b_dec
4. N_layers controls correlation range (each layer ~doubles it)

| Pros | Cons |
|------|------|
| Explicitly multi-scale: layer l captures 2^l range | Bilinear mixing terms complicate gradient flow |
| Tunable depth = trade-off control | Each layer has own parameters (linear growth) |
| Residual connections = starts as independent SAE | Not as well-studied as MPS/MPO |
| Natural for power-law correlations | May need careful initialization |
| Per-feature variant preserves interpretability | |

**Best regime**: Moderate rho (0.3-0.7) with features at multiple persistence scales.

### Alt B: W-Contraction MPS (from conversation ~1:30-1:33)

**Source**: Dmitry's key idea from the conversation -- learnable W matrices map local codes u_t into MPS physical legs.

**Architecture**:

1. Encode: u_t = ReLU(W_enc @ (x_t - b_dec) + b_enc)
2. Project to MPS physical dimension: p_t = W_phys @ u_t (shape: I)
3. Contract with MPS cores: B_t = sum_i p_{t,i} * A^{(t)}_{alpha1, i, alpha2}
4. Left-right sweep for environments, extract per-site states mu_t
5. Decode: x_hat_t = W_dec_bond @ vec(mu_t) + b_dec

| Pros | Cons |
|------|------|
| Solves the central conversation problem: how to pipe activations into MPS | Novel -- no reference implementation |
| W_phys serves as learned "local Hilbert space" mapping | Bond-state decoding has gauge freedom |
| Translation-invariant version is very compact (~5K params for reasonable sizes) | Gradient flow through left-right sweep may have vanishing/exploding issues |
| Bond dimension directly controls correlation structure | "Soft" contraction (continuous p_t vs discrete i_t) may wash out discrete structure |

**Best regime**: High rho (0.7-0.9) with long windows where MPS prior strongly denoises persistent features.

### Alt C: Binary Log-Space MPS (Moire-Inspired)

**Source**: Jacopo's moire physics analogy (~1:53-1:58) where N sites are encoded as log2(N) MPS sites via binary indexing.

**Architecture**:

1. Encode locally: u_t = encode(x_t), T = 2^n
2. Relabel time indices as binary strings (b_1,...,b_n)
3. Represent as MPS with n = log2(T) sites -- site 1 captures half-chain (global) correlations, last site captures nearest-neighbor
4. Each MPS site has mixing tensors at that scale
5. Decode from hierarchical representation

| Pros | Cons |
|------|------|
| **Logarithmic** parameter scaling with T | Requires T = power of 2 |
| Explicitly multi-scale with built-in interpretation (site j = correlations at scale 2^j) | Binary encoding is conceptually non-trivial |
| Could scale to T=1024 with only 10 levels | What does "3rd binary digit correlation" mean for language? |
| Theoretically motivated for power-law | Overkill for current T=64 benchmark |

**Best regime**: Very long sequences (T >= 128) with power-law correlation structure. The "dream" architecture for scaling.

---

## Rankings

### By ease of implementation

1. **Proposal 3** (already implemented)
2. Proposal 1 (just prepend a layer)
3. Alt A: Iterative contraction (standard PyTorch conv-like ops)
4. Proposal 2 (MPO bookkeeping)
5. Alt B: W-Contraction MPS (novel, needs differentiable sweep)
6. Proposal 4A (needs q-state forward-backward)
7. Proposal 4B (4A + environment extraction)
8. Alt C: Binary log-space MPS (most novel)

### By interpretability preservation

1. **Proposal 3** (feature k stays feature k)
2. Proposal 4A (clean marginal decode)
3. Alt A per-feature variant
4. Alt B (W_phys projection is clean, bond decode less so)
5. Alt C (multi-scale hierarchy is interpretable in principle)
6. Proposal 1 (mixed activations)
7. Proposal 4B (bond/gauge issues)
8. Proposal 2 (destroys feature identity)

### By expected performance in aliased leaky-reset regime

1. **Proposal 4A / BW Factorial** (closest to the DGP -- factorial HMM)
2. Alt B: W-Contraction MPS (MPS prior captures temporal coherence)
3. Proposal 3 (kernel can learn the exponential persistence profile)
4. Alt A: Iterative contraction
5. Proposal 2 (overpowered -- cross-feature mixing not needed)
6. Proposal 1 (indirect temporal coupling)
7. Alt C (overkill for T=64)

### By scalability to long sequences

1. **Alt C** (O(log T) parameters)
2. Alt A (O(log T) layers for full range)
3. Proposal 4A/B (O(T * q * chi^2) -- linear in T)
4. Alt B (same as 4)
5. Proposal 1 (depends on T_theta)
6. Proposal 3 (O(T^2) kernel -- bad)
7. Proposal 2 (O(T * m * chi^2) with TN structure)

---

## Key Conceptual Takeaways from the Conversation

1. **The fundamental tension**: You need genuine compression (not scrambling) while maintaining interpretable latents. Jacopo's product-state argument (timestamp ~1:21) is the clearest articulation: if the architecture just rearranges the same information, you gain nothing. The TN must actually capture correlations that reduce the effective dimensionality.

2. **"How do I pipe activations into the MPS?"** is the core unsolved question. The writeup's Proposal 4 handles it via the exponential map phi_t(i_t) = exp(l_{t,i}), which Jacopo found somewhat arbitrary. The W-contraction idea (Alt B) is a cleaner answer.

3. **Hierarchy matters**: Both Jacopo's MERA suggestion and the moire-inspired log-space encoding point toward the same insight -- correlations at different scales need different treatment. A flat (T,T) kernel or a single-level MPS can't efficiently capture multi-scale structure.

4. **Start simple, then escalate**: Both participants agreed on the pipeline: (a) write down weight matrices, (b) test on synthetic HMM, (c) understand hyperparameter sweeps, (d) then push to language models.
