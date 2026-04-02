---
author: Dmitry
date: 2026-03-26
tags:
  - results
  - in-progress
---

## Temporal crosscoders encode hidden state but not features

### Summary

We train a temporal crosscoder (TXCDR) on synthetic data from a hidden Markov model and ask: does it recover the generative features? Standard metrics say no — feature recovery drops to 0% at high temporal correlation ($\rho = 1$). But a linear probe from the TXCDR's latent representation recovers the true hidden state with 99.5% AUC. The model *knows* the answer; it just encodes it in a non-local basis that doesn't align with the ground truth features.

This is a clean demonstration that temporal crosscoders use **non-local features** — directions in latent space that mix information across sequence positions. The more temporal structure in the data, the more the TXCDR exploits it through non-local encoding, and the further its features drift from the true generative decomposition.

### Background

#### The synthetic task

We generate data from $m = 50$ independent binary Markov chains with stationary probability $\pi = 0.1$. Each feature $k$ has a hidden state $s_{t,k} \in \{0, 1\}$ that persists across time with autocorrelation $\rho$:

$$P(s_{t,k} = 1 \mid s_{t-1,k} = 1) = \rho(1 - \pi) + \pi$$

Observations are linear embeddings $x_t = \sum_k s_{t,k} \cdot m_{t,k} \cdot f_k$ where $f_k$ are orthogonal unit vectors in $\mathbb{R}^{100}$ and $m_{t,k} \sim |\mathcal{N}(1, 0.15^2)|$ are random magnitudes.

The parameter $\rho$ controls temporal persistence: $\rho = 0$ gives iid support at each position, $\rho = 1$ gives constant support across the entire window. Crucially, the *marginal* distribution at each position is the same for all $\rho$ — on average 5 features are active, regardless of temporal correlation.

#### The temporal crosscoder

The TXCDR encodes a window of $T = 5$ positions into a **single shared latent** $z \in \mathbb{R}^m$:

$$z = \text{TopK}\!\left(\text{ReLU}\!\left(\sum_{t} W_{\text{enc}}^{(t)} x_t + b_{\text{enc}}\right), k \cdot T\right)$$

It then decodes back to per-position reconstructions using per-position decoder weights:

$$\hat{x}_t = W_{\text{dec}}^{(t)} z + b_{\text{dec}}^{(t)}$$

The shared latent is the key design choice: all temporal information is compressed into one $m$-dimensional vector with $k \cdot T = 25$ nonzero entries.

#### Standard evaluation metrics

- **MMCS** (mean max cosine similarity): for each ground truth feature $f_k$, find the decoder column with highest cosine similarity. Average over features.
- **Feature recovery (R@0.9)**: fraction of ground truth features matched by a decoder column with cosine $\geq 0.9$.

Both metrics measure **decoder direction alignment** — whether the model has found the right feature directions.

### The $\rho$ sweep: feature recovery collapses

We trained the TXCDR at six $\rho$ values. As temporal correlation increases, reconstruction loss improves but feature recovery degrades:

| $\rho$ | Loss | MMCS | L0 | FeatRec |
|--------|------|------|----|---------|
| 0.0 | 0.716 | 0.637 | 25.0 | 0.20 |
| 0.2 | 0.651 | 0.940 | 25.0 | 0.82 |
| 0.5 | 0.492 | 0.879 | 25.0 | 0.64 |
| 0.7 | 0.354 | 0.795 | 25.0 | 0.46 |
| 0.9 | 0.146 | 0.756 | 25.0 | 0.32 |
| 1.0 | 0.085 | 0.559 | 25.0 | 0.00 |

The loss is *monotonically decreasing* while feature recovery is *monotonically decreasing* (after $\rho = 0.2$). The model gets better at reconstruction by getting worse at decomposition. This can only happen if it encodes cross-position information into its features — **non-local features** that are efficient for compression but don't correspond to the ground truth generative features.

For comparison, the position-independent SAE maintains stable feature recovery ($\geq 0.90$) for $\rho \in [0.2, 0.9]$, and the per-feature temporal model (PerFeat) achieves near-perfect recovery ($1.0$) across the same range.

### The question

The sweep tells us the TXCDR's decoder directions don't match the true features. But this leaves open two possibilities:

1. **Information loss**: the TXCDR's latent representation genuinely loses information about which features are active. It optimizes for reconstruction without tracking the generative decomposition.
2. **Non-local encoding**: the TXCDR's latent encodes the full hidden state, but in a rotated basis that doesn't align with individual features. A linear probe could recover the hidden state even though the decoder directions are wrong.

These have different implications: (1) means the architecture is fundamentally limited; (2) means the architecture is capable but uses the wrong inductive bias.

### The probe experiment

We train a linear probe from the TXCDR's latent representation to the ground truth binary support $s_{t,k}$.

#### Generating matched data

We generate eval data with matched ground truth support by calling the Markov generator and embedder with the same seed:

```python
support = generate_markov_support(n_features, T, pi, rho, n_sequences, generator)
x = toy_model.embed(support, magnitude_mean, magnitude_std, generator)
```

This gives us 2,000 windows of paired $(x, s)$ where $s \in \{0, 1\}^{n_\text{features} \times T}$.

#### Extracting representations

For the TXCDR, we extract two representations:

- **Post-TopK latent** $z$: the sparse shared code $(B, m)$ with $k \cdot T = 25$ nonzero entries. This is what the decoder sees.
- **Pre-TopK activation**: the dense ReLU output $(B, m)$ before sparsification. All $m = 50$ dimensions are potentially nonzero.

#### Probe architecture

For each window position $t$, we train an independent linear probe:

$$\hat{s}_{t,k} = \sigma\!\left(\sum_j W_{kj}^{(t)} z_j + b_k^{(t)}\right)$$

- **Input**: TXCDR latent $z \in \mathbb{R}^{50}$ (same vector at every position, since the latent is shared)
- **Target**: ground truth support $s_{:,t} \in \{0,1\}^{50}$ at position $t$
- **Training**: 2,000 steps of Adam (lr = 1e-3) with BCE loss on a 70/30 train/test split
- **Metric**: macro AUC across features, averaged over all $T$ positions

The probe is deliberately simple — a single linear layer. High probe AUC means the hidden state is **linearly accessible** in the latent representation.

### Results

#### TXCDR only (post-TopK and pre-TopK probes)

| $\rho$ | Loss | MMCS | FeatRec | Probe AUC | Pre-TopK AUC |
|--------|------|------|---------|-----------|--------------|
| 0.0 | 0.723 | 0.652 | 0.20 | 0.681 | 0.681 |
| 0.5 | 0.500 | 0.898 | 0.62 | 0.863 | 0.861 |
| 1.0 | 0.090 | 0.524 | 0.00 | **0.995** | 0.996 |

#### All models compared

| $\rho$ | Model | Loss | MMCS | FeatRec | Probe AUC |
|--------|-------|------|------|---------|-----------|
| 0.0 | SAE | 0.128 | 0.976 | 0.98 | 0.876 |
| 0.0 | TXCDR | 0.723 | 0.652 | 0.20 | 0.681 |
| 0.0 | PerFeat | 0.119 | 0.904 | 0.82 | 0.915 |
| 0.5 | SAE | 0.121 | 0.911 | 1.00 | 0.909 |
| 0.5 | TXCDR | 0.500 | 0.898 | 0.62 | 0.863 |
| 0.5 | PerFeat | 0.115 | 0.910 | 1.00 | 0.921 |
| 1.0 | SAE | 0.558 | 0.691 | 0.02 | 0.871 |
| 1.0 | TXCDR | 0.090 | 0.524 | 0.00 | **0.995** |
| 1.0 | PerFeat | 0.290 | 0.892 | 0.76 | 0.900 |

### Interpretation

#### The headline result

**At $\rho = 1$, the TXCDR achieves 0% feature recovery but 99.5% probe AUC.** The model perfectly encodes the hidden Markov state in its latent representation — a linear function of $z$ can recover which features are on and off with near-perfect accuracy. It just doesn't align this information with the ground truth feature directions. The features it uses are non-local: they jointly encode the state of multiple generative features into single latent dimensions.

#### TXCDR: probe AUC and MMCS move in opposite directions

As $\rho$ increases from 0.5 to 1.0, probe AUC rises from 0.86 to 0.995 while MMCS drops from 0.90 to 0.52. The model gets *better* at hidden state inference while getting *further* from the true feature basis. More temporal structure means more incentive to use non-local features — they compress the redundancy more efficiently.

#### Pre-TopK $\approx$ post-TopK everywhere

The gap between probing the dense pre-activation and the sparse post-TopK latent is negligible ($< 0.01$ at all $\rho$). TopK sparsification does not destroy hidden state information. The encoder learns to concentrate state information into the dimensions that survive TopK. This rules out the hypothesis that TopK is the bottleneck — the non-local encoding is a property of the learned representation, not an artifact of sparsification.

#### SAE and PerFeat: stable probe AUC, contrasting feature recovery

The SAE and PerFeat probe AUCs are remarkably stable across all $\rho$ (0.87--0.92), unlike the TXCDR which swings from 0.68 to 0.995. Their representations consistently encode the hidden state regardless of temporal correlation.

But their feature recovery diverges sharply at $\rho = 1$: the SAE collapses (FeatRec 0.02) while PerFeat degrades gracefully (FeatRec 0.76). Both maintain ~0.87--0.90 probe AUC at $\rho = 1$, so the hidden state information is present in both — the SAE's decoder directions simply drift away from the true features, while PerFeat's architectural constraint (per-feature temporal kernel) keeps them aligned.

#### Why probe AUCs differ at $\rho = 0$

At $\rho = 0$ the data is iid — there is no temporal structure. Yet the three models show different probe AUCs (0.68, 0.88, 0.92). This is not about temporal inference; it is about **how much information the probe can access** in each model's latent format:

| Model | Nonzero dims per position | Position-specific? | Probe AUC |
|-------|--------------------------|-------------------|-----------|
| SAE | 5 (TopK) | Yes | 0.876 |
| PerFeat | ~20 (post-mix) | Yes | 0.915 |
| TXCDR | 25 (shared) | No — encodes all positions | 0.681 |

**SAE** (0.876): the probe sees $k = 5$ nonzero values per position. These directly encode which features are active at that position, but 5 nonzero entries out of 50 dimensions gives the probe very little to work with — it must predict 50 binary states from 5 values and 45 zeros.

**PerFeat** (0.915): the temporal mixing spreads activation to $\sim 20$ nonzero dimensions per position. Even at $\rho = 0$ where there is no temporal structure to exploit, the 4$\times$ more nonzero entries give the linear probe more signal to separate active from inactive features.

**TXCDR** (0.681): the shared latent has 25 nonzero dimensions, but these encode *all 5 positions mixed together*. The per-position probe must figure out which subset of those 25 dimensions carries information about its specific position — a harder linear separation problem than reading position-specific activations directly.

#### $\rho = 1$: easy for TXCDR, hard for SAE

**$\rho = 1$ is trivially easy for the TXCDR's shared latent.** All positions share the same support, so $z$ only needs to encode a single binary vector of $\sim 5$ active features. The 25 active dimensions have massive headroom. Probe AUC = 0.995.

**$\rho = 1$ breaks the SAE's feature recovery but not its probe AUC.** The SAE's latents still encode the hidden state (probe AUC 0.87) but its decoder directions drift away from the true features (FeatRec 0.02). This is puzzling for a position-local model — $\rho$ should not affect it at all, since the per-position marginal distribution is identical at every $\rho$. This may be a training dynamics issue (reduced effective diversity in batches when all positions within a window share the same support) and is worth investigating separately.

### PerFeat is the best all-rounder

Across all $\rho$ values, PerFeat maintains both high feature recovery ($\geq 0.76$) and high probe AUC ($\geq 0.90$). It is the only model that simultaneously:

- Finds the right features (decoder alignment)
- Knows when they are active (hidden state recovery)
- Degrades gracefully at extreme $\rho$

This is a direct consequence of its architectural constraint: per-feature temporal kernels can move information *within* a feature across positions, but cannot create non-local features that mix information across features. The constraint sacrifices some reconstruction efficiency but preserves the alignment between learned features and the generative decomposition.

### Why this matters

The TXCDR's failure mode is **not** a lack of information — it's a misalignment between the model's learned basis and the ground truth generative basis. The shared-latent bottleneck creates pressure to encode information efficiently, and non-local features are more efficient than local ones when there's temporal redundancy.

This is a problem for interpretability. The goal of sparse autoencoders is to find **the** generative features, not just any compressed representation. A model that perfectly encodes the hidden state in an uninterpretable basis is not useful for mechanistic interpretability, even though it achieves excellent reconstruction.

### Connection to the Baum-Welch SAE proposal

The [[baum_welch_sae_idea|Baum-Welch SAE]] is designed to make hidden state recovery and feature recovery *the same thing*. Its per-feature HMM posteriors $\pi_{t,k} = q(z_{t,k} = 1 \mid \text{pre}_{1:T,k})$ are explicitly aligned with the feature decomposition — each posterior directly estimates whether a specific ground truth feature is active. No probe is needed because the intermediate representation IS the hidden state estimate.

The probe results strengthen the case for this architecture: since the TXCDR *does* encode hidden state information (just in the wrong basis), an architecture that enforces the right basis while maintaining temporal inference should get the best of both worlds.
