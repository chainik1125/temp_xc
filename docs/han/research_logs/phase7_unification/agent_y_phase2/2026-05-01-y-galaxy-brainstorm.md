---
author: Han
date: 2026-05-01
tags:
  - design
  - in-progress
---

## Phase 7 Y — GALAXY brainstorm: beyond the sum

> Han's prompt: "what if instead of a TXC where we boringly sum together
> the encoder result at each position to get the final latent, we do
> something… galactic"

### The current "boring sum"

Current TXC encoder for window `x = [x_0, x_1, ..., x_{T-1}]`:

```
z = TopK( Σ_t W_enc[t] @ x_t  +  b_enc )      # shape (d_sae,)
```

Each position contributes a per-position projection; they sum into a
single window-level feature vector. TopK keeps the K most-activated
features; everything else is zero.

The decoder is correspondingly:

```
x_hat[t] = Σ_j z[j] * W_dec[j, t, :]  +  b_dec[t]
```

Each active feature contributes a per-position decoded direction.

**The "boring sum"** has two problems:

1. **Information bottleneck**: the entire T-window collapses into a
   single d_sae-dim feature vector before TopK. Multi-position
   interactions are summed away before the sparsification decision.
2. **Position-blind features**: at TopK time, the selector doesn't
   know which positions activated which features. A feature that
   fires strongly at t=0 alone is indistinguishable from one that
   fires weakly at all T positions.

### Galaxy 1 — Cross-position attention before TopK

```
{e_t = W_enc[t] @ x_t}_t                       # per-position embeds
e_attn = MultiHeadAttention({e_0, ..., e_{T-1}})  # cross-position
z = TopK( pool(e_attn) + b_enc )
```

Lets each position's embedding attend to others before pooling. The
attention weights LEARN which positions are relevant for which
features. Captures multi-position dependencies (e.g. "feature A is
on only when context B precedes it").

**Pros**: maximum expressivity; can learn arbitrary position-mixing.
**Cons**: more parameters; harder to interpret which positions drive
which features.

### Galaxy 2 — Multiplicative gating (conjunctive features)

```
z_pos[t] = W_enc[t] @ x_t  +  b_enc[t]          # per-position features
z = TopK( z_pos[0] ⊙ z_pos[1] ⊙ ... ⊙ z_pos[T-1] )  # elementwise product
```

A feature is on only if it's strongly activated at EVERY position. This
captures multi-token concepts that REQUIRE the full window context.

**Pros**: very sparse; conjunctive features are interpretable
("this feature requires the concept to span the full window").
**Cons**: sparsity might be too aggressive at large T; gradients can
vanish.

**Hybrid**: sum for "agnostic" features (feature is on if ANY position
has it strong) + product for "conjunctive" features (feature is on
only if ALL positions agree). Two latent groups, sparser product
group, denser sum group.

### Galaxy 3 — Per-feature position-attention

```
For each feature j, learn α_j[t] ∈ R   (temperature → 0: hard select)
z[j] = Σ_t softmax(α_j)[t] * W_enc[t][j] @ x_t
```

Each feature has its OWN per-position weighting. Some features attend
to only one position (per-token features); others span multiple
positions. The α distribution per feature reveals the feature's
"position support".

**Pros**: structurally interprets each feature's window-span;
generalises both per-token and per-window paradigms.
**Cons**: more parameters (one α per feature × T positions);
optimization can be tricky.

### Galaxy 4 — Hierarchical multi-scale latent

```
z_window = TopK_window( Σ_t W_enc[t] x_t )       # window-level features (multi-token)
z_pos[t] = TopK_pos( W_enc_per_pos[t] x_t )      # per-position features (per-token)
```

Decoder:

```
x_hat[t] = Σ_j z_window[j] W_dec_window[j, t, :] +
           Σ_j z_pos[t][j] W_dec_per_pos[t][j, :] +
           b_dec[t]
```

EXPLICIT decomposition of latent into window-level + per-position
features. Window-level features capture multi-token structure;
per-position features capture per-token vocabulary.

**Steering implication**: window steering → modify z_window; per-token
steering → modify z_pos[t]. Natural coexistence of T-SAE-style and
TXC-style steering in one architecture.

**Connection to existing matryoshka**: agentic_txc_02 (W's cell E)
has multiple scales but it's hierarchical-by-scale, not per-position
vs per-window. The proposed split is orthogonal.

### Galaxy 5 — Causal/recurrent encoder

```
h_0 = W_enc[0] @ x_0
h_t = update(h_{t-1}, W_enc[t] @ x_t)            # gate-like update
z = TopK(h_{T-1})
```

Encoder processes positions sequentially. Hidden state accumulates
multi-position context. Like an RNN-encoder on per-position embeds.

**Pros**: matches token streaming; latent represents "feature state
at end of window".
**Cons**: sequential = slower training; less parallelism.

### Galaxy 6 — Differentiable position-aware TopK

```
For each feature j, score = max_t( W_enc[t][j] @ x_t )    # peak per position
z[j] = score * indicator(j ∈ TopK(scores))                # only top-K peak features
```

Instead of summing across positions, take the MAX activation across
positions per feature. Then TopK across features.

This is a "feature wins where it's strongest". Position information
is in the position-of-max but not the latent value.

**Pros**: concentrated activation; position-aware.
**Cons**: max is non-differentiable; hard or soft top-pos needed.

### Galaxy 7 — Attention-from-decoder-back

```
z = TopK( Σ_t W_enc[t] x_t )                    # standard sum + TopK
attn[t, j] = softmax(W_attn[j] @ x_t)            # per-feature, per-position attention
x_hat[t] = Σ_j z[j] * attn[t, j] * W_dec[j, :]   # decoder uses attention to write only at relevant positions
```

Per-feature decoder attention determines WHICH positions to write to.
A "harmful_content" feature might write only to the right edge; a
"discourse style" feature might write across all positions.

**Pros**: learnable write-back protocol per feature; replaces our
hand-designed right-edge / per-position protocols with a learned one.
**Cons**: attention parameters per feature; introduces non-bilinearity
to the decoder.

**Steering implication**: write-back is no longer hand-coded — it's
learned. The "right protocol" varies per feature, which matches our
empirical finding (some concepts work better with right-edge, others
with per-position).

### Galaxy 8 — Sparse latent dynamics across positions

```
For each position t:
    z_t = TopK( W_enc[t] @ x_t  +  M @ z_{t-1} )    # M = transition matrix
```

Latent at position t depends on x_t AND z_{t-1}. Captures
within-window feature evolution. Sparse latent state propagates.

**Pros**: explicit feature-state dynamics; captures temporal feature
chains (e.g. "argument introduces, then escalates").
**Cons**: M has d_sae × d_sae params; risks overfitting.

### Galaxy 9 — Mixture-of-experts encoder

```
For each window:
    expert_assignments = softmax(gating(x_window))   # which expert per token
    z_t = TopK( Σ_e expert_assignments[t, e] * W_enc_e[t] @ x_t )
```

Multiple per-position encoders (experts). Each token routed to an
expert. Different experts specialize in different feature types
(e.g. one for nouns, one for verbs).

**Pros**: scales without growing dense parameter count;
specialization.
**Cons**: routing instability; complex training.

### Galaxy 10 — VAE-style: posterior over features

```
encode → (μ, σ²) over d_sae    # posterior parameters
z ~ N(μ, σ²)                    # sample latent
TopK(z) for sparsity
decode → x_hat
ELBO = recon_loss - KL(N(μ, σ²) || N(0, 1))
```

Stochastic latent. Captures feature uncertainty. Posterior collapse
risk.

**Pros**: principled sparsity via prior; uncertainty quantification.
**Cons**: ELBO optimization is finicky; TopK-sampling interaction
unclear.

### Recommendation ranking (my prior)

Ordered by likely impact × tractability:

1. **Galaxy 4 (hierarchical multi-scale)** — explicit window/per-position
   decomposition. Small modification of existing arch. High interpretability.
   Multi-seed verify already-existing matryoshka cells could be a
   stepping stone.

2. **Galaxy 3 (per-feature position-attention)** — interpretability
   bonus from α per feature. Moderate parameter increase.

3. **Galaxy 7 (decoder-attention write-back)** — replaces hand-designed
   protocols with learned ones. Directly addresses the protocol-zoo
   we've explored empirically.

4. **Galaxy 1 (cross-position attention)** — most expressive but
   least interpretable. Risk of attention learning trivial mixings.

5. **Galaxy 2 (multiplicative gating)** — sparse + conjunctive features.
   Risk of sparsity collapse at high T.

6. Others (Galaxy 5/6/8/9/10) — niche or higher risk.

### Connection to current results

Our paper's headline result is:

> T=2 + H8 multi-distance + shifts=(T,) + per-position write-back
> beats T-SAE k=20 by Δ=+0.872 at coh ≥ 1.75.

The "boring sum" works well at T=2. As T grows, per-position
write-back's averaging dilutes the signal (T=5 H8 PP has σ_seeds=0
but lower peak). Galactic encoders that DON'T sum could maintain
signal strength at larger T:

- **Galaxy 4** (multi-scale) makes window-level vs per-position
  features explicit, which could help T=5+ where there's more
  multi-position structure to capture.
- **Galaxy 7** (decoder attention) lets the encoder pick the right
  write-back per feature, eliminating the right-edge / per-position
  / V3 / V5 / V6 / etc. protocol zoo.

### Suggested first experiment: Galaxy 4 (hierarchical multi-scale)

Implementation sketch:
- Define `TXCMultiScale` with two latent groups: `z_window` (TopK across
  d_sae_window features) and `z_pos[t]` (TopK across d_sae_per_pos
  features per position).
- Train with reconstruction loss = sum of per-position |x - x_hat|².
- Evaluate at matched k_pos with K_window + T·K_pos = total budget.

For matched budget at T=2, k_pos=20: K_window + 2·K_pos = 40. Tradeoff
study: K_window=20 K_pos=10, K_window=10 K_pos=15, etc.

**Hypothesis**: explicit window/per-position decomposition lifts coh ≥
1.75 win further by letting the encoder dedicate features to the
multi-token vs per-token roles.

### Risk & cost

- **Galaxy training time**: each new arch ~30 min × multi-seed × 5+
  configs = 5–10 hr GPU.
- **Failure modes**: posterior collapse, dead features, gradient
  instability — debug + fix iteration.
- **Interpretability bonus**: even a "no improvement" result is
  scientifically valuable if it shows that "boring sum" is in fact
  near-optimal for matched-sparsity steering.

### Decision

Defer for after Phase 2 multi-seed verify completes. Galaxy is the
right NEXT-PHASE direction (Phase 8?) — too ambitious for the NeurIPS
deadline if existing GIGABRAIN reframe already gives a paper-grade
WIN.

Recommend: lock the Phase 7 paper headline (TXC dominates at coh ≥
1.75 by +0.872), then start Galaxy as Phase 8 architectural research.
