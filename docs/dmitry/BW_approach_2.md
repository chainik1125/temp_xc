---
author: Dmitry
date: 2026-03-26
tags:
  - proposal
  - in-progress
---

# Baum-Welch Approach 2: learn a BW-friendly latent space

## Goal

The goal is not to take an already-learned sparse code and then ask Baum-Welch (BW) to clean it up after the fact.

The goal is to train an encoder-decoder system with a **BW bottleneck in the middle**, so that backprop pressure pushes the learned latent space toward one in which:

1. the encoder produces observations that are easy for a Markov model to explain,
2. the BW posterior trajectory is temporally coherent,
3. decoding from that posterior gives good reconstruction.

In other words, the latent space should be learned **for BW**.

## Central conceptual point

The latent **space** should be atemporal.

That means there is a fixed learned dictionary or state set:
\[
\mathcal{V} = \{v_1, \dots, v_m\}, \qquad v_k \in \mathbb{R}^d,
\]
which has the same meaning at every time step and in every window.

What varies over time is not the space itself, but the **trajectory through that space**:
\[
z_1, z_2, \dots, z_T
\]
or, in a sparse factorial version,
\[
s_{t,k} \in \{0,1\}.
\]

This is the main difference from a sloppy "time-indexed latent space" picture. We do **not** want the basis to change with time. We want a fixed latent basis and a time-indexed hidden-state trajectory inside that basis.

## High-level architecture

For an activation window
\[
x_{1:T}, \qquad x_t \in \mathbb{R}^d,
\]
the architecture is:

1. Use an XC-like encoder to discover a shared latent basis and produce per-token observations in that basis.
2. Run an inner BW step that finds the best HMM dynamics for those observations.
3. Decode only from the BW posterior, and train through that bottleneck.

Schematically:

\[
x_{1:T}
\;\to\;
\text{encoder observations in shared latent space}
\;\to\;
\text{BW / forward-backward}
\;\to\;
\text{posterior latent trajectory}
\;\to\;
\text{decoder}
\;\to\;
\hat x_{1:T}.
\]

## Proposed version: sparse factorial BW space

The most natural version if we want something SAE-like is a **factorial hidden-state model**:

- there are \(m\) shared latent features,
- feature \(k\) has a fixed decoder direction \(v_k\),
- at each time \(t\), feature \(k\) is either OFF or ON,
- each feature evolves as its own 2-state Markov chain.

This gives a hidden state
\[
s_{t,k} \in \{0,1\}
\]
for each feature \(k\) and time \(t\).

The full latent space is atemporal because feature \(k\) means the same thing everywhere. Only its occupancy changes with \(t\).

## Step 1: XC-like discovery encoder

We want an encoder that is more expressive than an independent SAE, but that still outputs **per-token observations** in a shared latent basis.

Let
\[
u_{t,k} = E_\phi(x_{1:T})_{t,k}.
\]

Important:

- The encoder may look at the whole window \(x_{1:T}\).
- But the output coordinate \(k\) must refer to the same global latent feature at every time step.
- So the encoder is XC-like in the sense that it uses window context to produce better latent observations, but it does **not** collapse the entire window into one shared latent vector.

This \(u_{t,k}\) is **not** yet the final latent code. It is an observation or evidence signal that BW will operate on.

## Step 2: observation model for BW

The BW layer should not act directly on a raw sparse coefficient without interpretation. Instead, the encoder output should be treated as an observation emitted by a hidden support state.

For each feature \(k\), introduce a binary Markov chain:
\[
P(s_{t,k}=1 \mid s_{t-1,k}=1) = \alpha_k,
\qquad
P(s_{t,k}=1 \mid s_{t-1,k}=0) = \beta_k.
\]

Then define an emission model for the observation \(u_{t,k}\), for example:
\[
u_{t,k} \mid s_{t,k}=0 \sim \mathcal{N}(\mu_{k,0}, \sigma_{k,0}^2),
\qquad
u_{t,k} \mid s_{t,k}=1 \sim \mathcal{N}(\mu_{k,1}, \sigma_{k,1}^2).
\]

Other emission models are possible, but the key idea is:

- the encoder emits a scalar observation in channel \(k\),
- BW interprets that scalar as noisy evidence for whether feature \(k\) is ON.

For the current latent space, BW then computes posterior marginals:
\[
\gamma_{t,k} = P(s_{t,k}=1 \mid u_{1:T,k}),
\]
and pairwise marginals:
\[
\xi_{t-1,t,k} = P(s_{t-1,k}, s_{t,k} \mid u_{1:T,k}).
\]

These \(\gamma_{t,k}\) are the temporally inferred support variables.

## Step 3: amplitude head

A binary support posterior alone may be too restrictive for reconstruction. So it is useful to separate:

- **support**: does feature \(k\) fire at time \(t\)?
- **amplitude**: how strongly should it contribute if it fires?

Introduce a second encoder head:
\[
r_{t,k} = R_\phi(x_{1:T})_{t,k},
\]
with \(r_{t,k} \ge 0\), for example via `softplus`.

Then define the effective decoded coefficient as
\[
\tilde a_{t,k} = \gamma_{t,k} \, r_{t,k}.
\]

This is important conceptually:

- BW acts on support,
- the amplitude head handles continuous strength,
- reconstruction uses both.

Without this split, a binary HMM is being asked to explain both support and amplitude at once, which is conceptually muddy.

## Step 4: decode from the BW posterior

Decode only from the BW-processed latent:
\[
\hat x_t = \sum_{k=1}^m \tilde a_{t,k} v_k + b
= \sum_{k=1}^m \gamma_{t,k} r_{t,k} v_k + b.
\]

The decoder dictionary
\[
V = [v_1, \dots, v_m]
\]
is shared across time, so this is the learned atemporal BW space.

Crucial design rule:

The decoder should **not** get a direct bypass from the raw encoder observations \(u_{t,k}\). If it does, the model can ignore BW and reconstruct directly from the pre-BW representation.

The only thing that should reach the decoder is the BW posterior, or a support-amplitude version derived from it.

## Optimization picture

This is the main point of Approach 2.

In a normal SAE, the encoder learns a latent space because it is penalized through reconstruction and sparsity:
\[
x \to a \to \hat x.
\]

Here, the objective is instead:
\[
x
\to
u
\to
\text{BW}(u)
\to
\gamma
\to
\hat x.
\]

So the outer loss is
\[
\mathcal{L}_{\mathrm{outer}}
=
\|x - \hat x\|^2
+ \lambda_{\mathrm{sparse}} \, \Omega(\gamma)
+ \lambda_{\mathrm{reg}} \, \Omega_{\mathrm{other}}.
\]

Because \(\hat x\) depends on \(\gamma\), and \(\gamma\) depends on the encoder outputs \(u\), gradients push the encoder to produce a representation in which:

- the latent channels correspond to stable features,
- the HMM can infer a coherent trajectory,
- the posterior trajectory preserves what the decoder needs.

If the latent space is bad for BW, the posterior will blur or misassign the trajectory, and reconstruction will be poor. If the latent space is good for BW, the posterior denoises and stabilizes the sequence, and reconstruction improves.

This is the mechanism by which the model learns a **BW-friendly latent space**.

## Inner BW step: "find the best BW for that space"

There are two ways to interpret the intermediate BW step.

### Option A: fully differentiable forward-backward

Treat the HMM parameters
\[
\theta_{\mathrm{HMM}} = \{\alpha_k, \beta_k, \mu_{k,0}, \mu_{k,1}, \sigma_{k,0}, \sigma_{k,1}\}
\]
as ordinary trainable parameters and backprop through forward-backward.

This is the simplest to implement inside a neural training loop.

### Option B: alternating optimization with an inner BW fit

A more faithful version is:

1. freeze encoder/decoder,
2. fit the best HMM parameters for the current latent observations \(u\) using Baum-Welch / EM,
3. compute posterior marginals \(\gamma\),
4. freeze or partially freeze the HMM,
5. backprop reconstruction loss through \(\gamma\) into the encoder/decoder,
6. repeat.

This makes the architecture closer to the bilevel idea:
\[
\theta_{\mathrm{HMM}}^*(u)
=
\arg\max_\theta p(u \mid \theta),
\]
and then
\[
\mathcal{L}_{\mathrm{outer}}
=
\|x - D_\psi(\gamma^*(u))\|^2.
\]

Conceptually, this is exactly "find the best BW for the currently discovered space, then evaluate how good that space is by reconstruction through the BW posterior."

## Why this differs from the first Baum-Welch SAE idea

The earlier idea was easy to read as:

- train an SAE-like code,
- interpret that code as an HMM observation,
- run BW on it.

Approach 2 is different:

- the latent space is discovered **under the constraint** that BW must be able to act on it,
- the decoder only sees the BW posterior,
- so the encoder is forced to make the latent channels BW-compatible.

This is a stronger and more principled optimization story.

## Why this is XC-like

This approach is XC-like in the sense that:

- the encoder can use the whole token window,
- the learned latent space is shared across positions,
- and the model is discovering a latent basis useful for reconstructing activation windows.

But it is **not** the same as the current temporal crosscoder implementation that sums the whole window into one latent vector. BW needs a sequence of observations across the window, so we still need per-token observations in the shared space.

The correct interpretation is:

- XC-like discovery of a shared latent basis,
- BW inference over trajectories in that basis.

## Advantages

- It directly optimizes for a latent space in which BW is useful.
- It preserves the idea of learned sparse features / dictionary atoms.
- It gives explicit posterior support trajectories \(\gamma_{t,k}\).
- It is well matched to synthetic data with independent Markov support structure.
- It creates a clean scientific comparison between temporal smoothing and temporal state inference.

## Main risks and weaknesses

- The learned latent basis may still be non-identifiable up to rotations, duplication, or feature splitting.
- The independent per-feature HMM assumption misses cross-feature temporal correlations.
- Alternating BW and backprop may be unstable or slow.
- If amplitudes are too expressive, the model may hide most of the work in the amplitude head rather than in the support trajectory.
- If the observation model is poorly chosen, the encoder may learn a strange representation that is formally HMM-compatible but semantically unhelpful.
- If the benchmark data are themselves independent Markov support processes, this model has a very matched inductive bias, so strong performance will be informative but not fully general.

## Recommended first implementation

The cleanest first version is:

1. Shared decoder dictionary \(v_k\).
2. Window-aware encoder producing per-token scalar observations \(u_{t,k}\).
3. Separate nonnegative amplitude head \(r_{t,k}\).
4. Independent 2-state HMM per feature.
5. Forward-backward producing \(\gamma_{t,k}\).
6. Decode from \(\tilde a_{t,k} = \gamma_{t,k} r_{t,k}\).
7. Train with alternating optimization:
   - one or a few BW/EM updates for HMM parameters,
   - then one or a few SGD updates for encoder/decoder through the BW bottleneck.

This is the most direct version of the idea:

- learn an atemporal latent space,
- find the best BW dynamics in that space,
- decode from the inferred posterior trajectory,
- backprop so the latent space becomes the one BW wants to act on.

## Short summary

Approach 2 treats BW not as a post-processing cleanup step, but as the **central bottleneck** that shapes representation learning.

The encoder discovers a shared, atemporal latent space.
BW infers trajectories through that space.
The decoder reconstructs only from those BW posteriors.

Because the reconstruction loss is applied **after** BW, backprop incentivizes the model to converge to the latent space in which BW inference is maximally useful.
