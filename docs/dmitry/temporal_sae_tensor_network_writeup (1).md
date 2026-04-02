# Temporal Sparse Autoencoders / Temporal Crosscoders via Tensor Networks

## Overall aim

Standard Sparse Autoencoders (SAEs) are typically trained independently at each sequence position. This makes them well-suited to discovering **local features**, but it likely handicaps them when the true latent structure is **temporally extended** across many positions.

The core aim of this project is to build and test architectures which can discover such **cross-sequence features** while retaining as much of the interpretability of standard SAEs as possible.

The motivating intuition comes from tensor networks, especially Matrix Product States (MPS), which efficiently represent structured correlations in one-dimensional systems. The sequence dimension of language suggests an analogous setting: if important latent variables persist or evolve in a low-complexity way across token positions, then an explicitly temporal latent architecture may outperform a positionwise SAE or an unconstrained cross-sequence crosscoder.

Our goal is therefore to answer:

- When do temporally-aware autoencoder architectures outperform independent SAEs?
- When do they discover qualitatively new features rather than merely improving denoising?
- What is the tradeoff between reconstruction/sparsity gains and interpretability costs?
- Which tensor-network-inspired parameterizations are actually useful in practice?

## Conceptual picture

There are at least two different ontologies one might adopt.

### Local-feature ontology

A standard SAE assumes that the natural basic object is a **local feature at one position**. Cross-position structure is then something secondary, which might be reflected indirectly in correlations between local features.

### Temporal-feature ontology

An alternative view is that some true features are inherently **distributed across sequence positions**. Examples might include:

- span-level states,
- quote / bracket / delimiter tracking,
- variable or entity persistence,
- persistent discourse modes,
- repeated or bursty motifs,
- latent features which are only weakly expressed locally but become clear when tracked over time.

In this ontology, the relevant object is not just “feature \(k\) at token \(t\),” but rather a **trajectory** or **global latent pattern** over many positions.

## Key conceptual issues

### 1. What counts as a “true feature”?

There are at least two plausible notions of truth in the temporal setting.

#### Local ground-truth features
These are the instantaneous generators contributing to the observed activation at a given sequence position.

#### Persistent latent features
These are the temporally evolving hidden variables whose state persists across positions and gives rise to the local generators.

A temporally-aware model may do poorly at recovering the first while excelling at recovering the second, or vice versa. We therefore need to evaluate against both notions.

### 2. Is temporal structure already written into the residual stream?

One concern is that by a sufficiently late layer, the transformer may already have integrated relevant temporal context into each token representation. If so, a local SAE or cross-layer crosscoder might already be enough.

The counterargument is that at many intermediate layers, important variables may still be represented **distributively across positions**, and a sequence-aware autoencoder may identify them earlier and more cleanly.

### 3. Interpretability vs expressive power

A temporal architecture may improve the reconstruction / sparsity tradeoff at fixed budget, but only by making the latent representation harder to interpret. This is especially relevant for:

- full feature-mixing sequence layers,
- monolithic tensor network bottlenecks,
- gauge redundancy in MPS-like parameterizations.

A major design goal is therefore to preserve a clean notion of “feature identity.”

### 4. Are we just building the toy distribution into the architecture?

Some architectures are naturally matched to simple temporal toy models such as HMMs or leaky-reset processes. This is useful for theory and synthetic experiments, but it risks a kind of circularity if the learned architecture is too close to the data-generating model.

This is one reason to test several architectures, from weakly structured to strongly structured.

## Regime where a temporal crosscoder should plausibly win

We expect a temporal model to outperform an independent SAE or naive cross-sequence baseline when:

1. **The true latent process is persistent across positions.**  
   A latent that turns on at one position is likely to remain on or recur over a span.

2. **Single-position evidence is ambiguous.**  
   Multiple latent generators may look similar locally, but differ in temporal behavior.

3. **The process has low temporal complexity.**  
   The cross-position dependence can be summarized by a small state or low-rank structure.

4. **The features of interest are genuinely temporally extended.**  
   The “right” feature is not just an instantaneous local generator.

5. **The model’s temporal inductive bias matches the data well enough.**  
   For example, smoothing-like or low-state propagation mechanisms should help when the latent process is bursty or persistent.

We especially expect a gain in synthetic settings where local superposition makes two features hard to separate from single-token evidence alone, while different temporal persistence parameters make them separable once cross-position information is used.

## Synthetic experiment: leaky-reset features

We propose a synthetic environment with latent persistent features controlled by a parameter \(\lambda\), which sets the temporal correlation length.

### Persistent latent features

For each persistent latent feature \(k\), define a binary hidden process
\[
z_{t,k} \in \{0,1\}
\]
with leaky-reset dynamics:
\[
\Pr(z_{t,k}=1 \mid z_{t-1,k}=1) = \lambda_k + (1-\lambda_k)p_k,
\]
\[
\Pr(z_{t,k}=1 \mid z_{t-1,k}=0) = (1-\lambda_k)p_k.
\]

Here:
- \(p_k\) controls marginal firing probability,
- \(\lambda_k\) controls temporal persistence.

The key knob is \(\lambda\):
- \(\lambda = 0\): no temporal persistence; effectively iid across positions,
- \(\lambda \to 1\): strong persistence over long spans.

### Local generators

At each position, local generators are produced from the persistent latents. The simplest case is to identify the local generator with the current persistent state:
\[
h_{t,k} = z_{t,k}.
\]

A richer case introduces a local mixing layer from persistent states to local generators:
\[
h_t = M z_t,
\]
where \(h_t\) are the local generators and \(z_t\) are the persistent latents. This allows evaluation against both “instantaneous local generators” and “persistent latent features.”

### Observations

Given decoder directions \(u_k \in \mathbb{R}^d\), define
\[
x_t = \sum_{k=1}^K u_k h_{t,k} + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal N(0,\sigma^2 I).
\]

A particularly important regime is one with **local superposition**:
- some decoder directions are similar,
- so local evidence alone is ambiguous,
- but temporal persistence breaks the degeneracy.

## Metrics

We want to track:

### 1. Reconstruction loss
Standard mean squared reconstruction error:
\[
\mathcal L_{\text{recon}} = \frac{1}{L}\sum_t \|x_t - \hat x_t\|_2^2.
\]

### 2. Sparsity
Measure average activation density of the learned latent code, e.g.
\[
\text{sparsity} = \frac{1}{Lm}\sum_{t,k}\mathbf{1}[a_{t,k}\neq 0]
\]
or the corresponding \(L_1\)-style penalty if using continuous activations.

### 3. Feature recovery: local generators
In the style of David Chanin, compute cosine similarity between learned decoder features and the true local generator directions, and build the confusion matrix.

This tells us how well each model rediscovers the “instantaneous local” ground truth.

### 4. Feature recovery: persistent latents
Also compute cosine similarity against the true persistent latent directions / induced latent features.

This measures whether a temporal architecture recovers the **persistent latent ontology**, even if it does not align perfectly with the local generators.

This dual evaluation is important because temporally-aware models may recover a more persistent, globally meaningful basis rather than the purely local one.

---

## Proposal 1: Pre-sandwich temporal layer

### Basic idea

Insert a temporal coupling layer **before** the local sparse code. The model first lets information flow across positions in the raw activation sequence, and only then applies a local sparse encoder.

### Form

Given \(x_t \in \mathbb R^d\), define a temporally mixed representation
\[
\tilde x = T_\theta(x_{1:L}),
\]
where \(T_\theta\) is a structured cross-position map, ideally with a low-rank / tensor-network inductive bias.

Then apply a standard SAE:
\[
a_t = f(W_{\mathrm{enc}}\tilde x_t + b_{\mathrm{enc}}),
\qquad
\hat x_t = W_{\mathrm{dec}} a_t + b_{\mathrm{dec}}.
\]

### Intuition

This lets the encoder build local features from a representation that already includes sequence context. It is close in spirit to “contextual preprocessing followed by local feature extraction.”

### Concerns

- It may be harder to attribute what part of a learned feature is local versus inherited from temporal preprocessing.
- The temporal layer acts in activation space, which may make feature identity less clean.
- Depending on the temporal map, it may resemble “explaining away the predictable part” rather than constructing global features.

### When it might help

This may help most when the activation itself needs denoising or context integration before sparse coding, especially if raw local evidence is weak.

---

## Proposal 2: Sandwich layer with full pairwise feature mixing

### Basic idea

Run independent local SAEs first, then insert a sequence-coupling layer in latent space before decoding.

### Form

First compute local sparse codes:
\[
a_t = f(W_{\mathrm{enc}} x_t + b_{\mathrm{enc}}),
\qquad a_t \in \mathbb R^m.
\]

Then stack them into
\[
a = \mathrm{vec}(a_1,\dots,a_L)\in\mathbb R^{Lm}.
\]

Apply a structured operator:
\[
\tilde a = a + M_\theta a,
\]
where \(M_\theta\in\mathbb R^{Lm\times Lm}\) is parameterized as an MPO / tensor-train operator.

Decode:
\[
\hat x_t = W_{\mathrm{dec}} \tilde a_t + b_{\mathrm{dec}}.
\]

### Intuition

The local SAE proposes candidate features independently at each position. The sandwich layer then lets those local candidates communicate across the sequence and revise one another before reconstruction.

This is the most direct “message passing between local features” formulation.

### Concerns

- Full pairwise feature mixing may be powerful but hard to interpret.
- A learned feature after the sandwich is no longer purely local.
- The architecture may improve reconstruction mainly via latent editing rather than recovering cleaner underlying features.

### When it might help

This should help when multiple local features interact across positions, and when temporal structure is not just self-persistence of one feature but also feature-to-feature influence over time.

---

## Proposal 3: Special case — one feature to one feature tempo1ral layer

### Basic idea

Restrict the sandwich layer so that each feature only talks to itself across time:
\[
\tilde a_{t,k} = a_{t,k} + \sum_{s=1}^L K^{(k)}_{t,s} a_{s,k}.
\]

So there is no cross-feature mixing, only temporal propagation within a feature channel.

### Intuition

This is the cleanest and most interpretable temporal correction layer. Each local feature is “smoothed” or “updated” using the same feature at other positions.

It directly matches the intuition of persistent or bursty features:
- real runs should reinforce themselves,
- isolated noise spikes should be suppressed.

### Concerns

- It may be too restrictive if the true temporal structure involves interactions between different features.
- It may behave mainly like a denoiser rather than discovering genuinely new feature structure.

### When it might help

This is the most natural architecture for the leaky-reset experiment, since each persistent latent feature is itself a temporally correlated on/off process.

It is also the most interpretable temporal baseline and the easiest first thing to test.

---

## Proposal 4: One-legged MPS autoencoder

### Basic idea

Instead of using a tensor network as a layer that edits local features, use an MPS directly as a **global latent prior / latent state** over the whole sequence.

This is the closest analogue to the chain-of-atoms picture: one physical leg per site, plus bond legs connecting neighboring positions.

### Form

Let each site have a local latent symbol
\[
i_t \in \{1,\dots,q\}.
\]

The encoder produces local evidence:
\[
\phi_t(i_t) = \phi_\eta(x_t)_{i_t}.
\]

The global latent compatibility is given by an MPS:
\[
\Psi_\theta(i_{1:L})
=
\sum_{\alpha_0,\dots,\alpha_L}
A^{(1)}_{\alpha_0,i_1,\alpha_1}
A^{(2)}_{\alpha_1,i_2,\alpha_2}
\cdots
A^{(L)}_{\alpha_{L-1},i_L,\alpha_L}.
\]

Define the posterior over latent sequences:
\[
q(i_{1:L}\mid x_{1:L})
\propto
\Big[\prod_{t=1}^L \phi_t(i_t)\Big] \Psi_\theta(i_{1:L}).
\]

Obtain local posterior marginals:
\[
m_t(i) = q(i_t=i \mid x_{1:L}).
\]

Decode from these marginals:
\[
\tilde a_t = \sum_i m_t(i) v_i,
\qquad
\hat x_t = W_{\mathrm{dec}} \tilde a_t + b_{\mathrm{dec}}.
\]

### Intuition

The tensor network is no longer a temporal correction layer; it is the actual structured latent object over the sequence. Local features arise as posterior marginals of this global latent state.

This is conceptually the cleanest tensor-network analogue of a temporal autoencoder.

### Concerns

- Training and inference are more complex.
- Feature identity may become less obvious if the local latent alphabet is not chosen carefully.
- Gauge issues become more central, since the TN is now part of the actual latent representation rather than just a correction layer.

### When it might help

This should help most when the correct ontology is genuinely a **global structured latent state**, not just local features with temporal smoothing.

It is the most faithful to the original many-body / MPS analogy.

---

## Experimental plan

For each \(\lambda\) and each architecture:

1. Train at matched model budgets.
2. Track reconstruction loss.
3. Track sparsity.
4. Compute confusion matrices / cosine-sim-based feature recovery against:
   - local ground-truth generators,
   - persistent latent features.
5. Compare how the feature ontology shifts with increasing temporal persistence.

### Hypotheses

- At low \(\lambda\), independent SAEs should perform competitively.
- As \(\lambda\) increases, temporally-aware models should increasingly outperform independent SAEs.
- The one-feature-to-one-feature temporal layer should be the cleanest early win on leaky-reset data.
- Full pairwise temporal layers may gain more on complex mixtures, but at a greater interpretability cost.
- The one-legged MPS autoencoder may best recover persistent latent features when the true latent ontology is genuinely global.

## Main question

The core scientific question is not only whether temporal architectures reconstruct better, but whether they recover **different and more meaningful features** — especially in regimes where the true underlying causes are persistent latents rather than purely local generators.
