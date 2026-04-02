---
author: Dmitry
date: 2026-03-26
tags:
  - proposal
  - in-progress
---

## Baum-Welch SAE: inferring latent Markov states alongside sparse features

### Core idea

The SAE discovers the feature decomposition (the "what") and determines the dimensionality of the temporal inference problem. Baum-Welch then solves the temporal problem (the "when") by running per-feature hidden state inference on the SAE's local evidence. The two are trained jointly end-to-end.

This is distinct from:

- **Proposals 1-3**: add a temporal correction layer on top of an SAE. The correction is a linear filter (smoothing), not state inference.
- **Proposal 4 (MPS)**: replaces the SAE with a tensor network prior over a single global latent sequence. Captures cross-feature correlations but the latent symbols aren't directly per-feature hidden states.

The Baum-Welch SAE keeps the SAE's per-feature decomposition and adds per-feature temporal state inference on top. The SAE gives us $m$ independent channels; each channel gets a 2-state HMM.

### Architecture

```text
x_t  (B, T, d)
  │
  ▼
SAE Encoder:
  pre_{t,k} = (W_enc @ x_t + b_enc)_k              (B, T, m)   continuous local evidence
  │
  ▼
Per-feature forward-backward (differentiable, O(mT)):
  For each feature k independently:
    Emission model:  p(pre_{t,k} | z_{t,k}=0),  p(pre_{t,k} | z_{t,k}=1)    learnable
    Transitions:     (α_k, β_k)                                               learnable
    Forward-backward → π_{t,k} = q(z_{t,k}=1 | pre_{1:T,k})                  posterior
  │
  ▼                                                  (B, T, m)   posteriors
Temporal decoding:
  x̂_t = W_dec @ (π_t ⊙ μ) + b_dec                 (B, T, d)
```

**Critical detail**: feed the *pre-TopK activations* into Baum-Welch, not the post-TopK ones. TopK creates point masses at zero which give a degenerate emission model. The pre-activations are continuous and carry graded local evidence for every feature at every position.

### Why the SAE determines the dimensionality

Without the SAE encoder, you'd need to define what a "feature" is before running per-feature HMM inference. The SAE discovers feature directions from data, and those directions define the channels that Baum-Welch operates on:

- Number of features $m$ = number of independent 2-state HMMs
- Encoder pre-activations = emissions for each HMM
- Decoder columns = feature directions for reconstruction

So the optimization has a clean division of labor:

- **SAE encoder**: "what are the features and how do I detect them locally?"
- **Baum-Welch**: "given those local detections, what's the true temporal state?"
- **SAE decoder**: "given the true states, reconstruct the observation"

### Learnable parameters

All trained jointly via backprop through the differentiable forward-backward:

| Component | Parameters | Role |
|-----------|-----------|------|
| SAE encoder | $W_{\text{enc}}, b_{\text{enc}}$ | Feature directions + local detection |
| Transitions | $\alpha_k, \beta_k$ per feature | Persistence/activation rates |
| Emissions | Per-feature $p(\text{pre} \mid z=0)$ and $p(\text{pre} \mid z=1)$ | How activations relate to hidden state |
| Magnitudes | $\mu_k$ per feature | Reconstruction scale when feature is ON |
| SAE decoder | $W_{\text{dec}}, b_{\text{dec}}$ | Feature directions for reconstruction |

### Emission model choices

The forward-backward needs $p(\text{pre}_{t,k} \mid z_{t,k}=0)$ and $p(\text{pre}_{t,k} \mid z_{t,k}=1)$.

- **Gaussian (simplest)**: $z=0 \Rightarrow \mathcal{N}(0, \sigma_0^2)$, $z=1 \Rightarrow \mathcal{N}(\mu_k, \sigma_1^2)$. Clean, differentiable. May not match actual pre-activation distributions well.
- **Learned neural emission**: small MLP per state. More flexible but less interpretable.
- **Log-odds parameterization**: define $\ell_{t,k} = \log p(\text{pre} \mid z=1) / p(\text{pre} \mid z=0)$ as a learnable function of $\text{pre}_{t,k}$. Avoids specifying the full emission model — only the likelihood ratio matters for forward-backward.

### Why this is better than the temporal kernel (Proposal 3)

Proposal 3 adds a learnable linear correction $\tilde{a} = a + Ka$. This is a generic temporal filter with no notion of "state." It smooths and denoises but does not explicitly infer whether a feature is ON or OFF — it just blends nearby activation values.

The Baum-Welch SAE does *state inference*: "this feature has been active for the last 5 positions, so even though the local evidence at this position is weak, the posterior probability that $z_{t,k} = 1$ is high." This is qualitatively different — state inference rather than signal smoothing.

Concretely, the temporal kernel can't distinguish between:

- A feature that is truly ON but has a weak local activation (should boost)
- A feature that is truly OFF but has a spurious local activation (should suppress)

The Baum-Welch approach can, because it maintains a running belief about the hidden state that is updated by each new observation.

### Relationship to Proposal 4 (MPS autoencoder)

Both architectures perform Bayesian inference over a structured latent temporal process via forward-backward. The difference is the prior:

| | Baum-Welch SAE | MPS autoencoder (Prop. 4) |
|--|----------------|--------------------------|
| Latent structure | $m$ independent 2-state chains | Single chain, alphabet size $q$, bond dim $\chi$ |
| Per-feature states | Directly readable: $\pi_{t,k}$ | Not directly readable — compressed into global symbol |
| Cross-feature correlations | None (independence assumption) | Captured via bond dimension |
| Interpretability | $\alpha_k, \beta_k$ are persistence/activation rates | TN parameters are opaque |
| Scalability | $O(mT)$ per window | $O(q^2 \chi^2 T)$ per window |
| Equivalence | — | Reduces to Baum-Welch SAE when $\chi=1$, $q=2$ |

The MPS is more expressive but the Baum-Welch SAE is more interpretable and exactly matched to the generative model in our synthetic data (independent per-feature Markov chains).

### Connection to measuring latent state recovery

A key advantage: the posterior marginals $\pi_{t,k} = q(z_{t,k} = 1 \mid \text{pre}_{1:T,k})$ are an explicit intermediate representation. We can directly evaluate how well the model recovers the true hidden state $z_{t,k}$ — the posterior IS the model's estimate. No linear probe needed.

This contrasts with:

- **Proposals 1-3**: latent activations are shaped by reconstruction and may or may not encode hidden state information accessibly. Requires a linear probe to extract $z$.
- **Proposal 4**: posterior marginals exist but are over the MPS's global latent alphabet, not per-feature hidden states. Whether they decompose into per-feature states depends on what the model learns.

### Why this might be worse

- **Inductive bias too strong**: the architecture assumes independent per-feature 2-state Markov chains. This is exactly the synthetic data's generative model. On real transformer activations, features may not have binary hidden states, may not be Markov, and may not be independent. Risk of overfitting to the synthetic setting.
- **Sequential bottleneck**: forward-backward is sequential in $T$ (can't parallelize along the time axis). For $T=5$ this is trivial. For longer sequences it could bottleneck training.
- **Emission model sensitivity**: the quality of temporal inference depends heavily on the emission model. A bad emission model means the local evidence is miscalibrated, and forward-backward propagates garbage.
- **No cross-feature temporal structure**: the independence assumption means the model can't learn that "feature A turning on predicts feature B turning on 2 steps later." The MPS (Proposal 4) can capture this; the Baum-Welch SAE cannot.

### Implementation notes

Maps cleanly to the existing `TemporalAE` interface in `temporal_bench`:

- Subclass `TemporalAE` (requires `window_size > 1`)
- Encoder: standard SAE encoder, return pre-TopK activations
- Forward-backward: batch over features, vectorize with `torch.cumsum` on log-space forward messages or explicit loop over $T$ (small)
- Decoder: posterior-weighted reconstruction
- Loss: MSE reconstruction, optionally add sparsity penalty on posteriors (L1 on $\pi$)
