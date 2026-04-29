---
author: Dmitry
date: 2026-04-28
tags:
  - design
  - in-progress
---

## Brief — backtracking case study

### Why this case study

Most of our SAE-architecture comparisons so far have been concept-steering (T-SAE protocol on HH-RLHF concepts). That tests whether an SAE can *find a feature for an arbitrary concept and steer it*. Backtracking is a different kind of stress test: a *behavior* the paper has only managed to elicit via raw activation-space steering vectors, with no SAE-feature explanation. If a single SAE feature (or a tight linear combination of a few features) can recapture the paper's effect, that is direct evidence the behavior factorises through interpretable latents.

### What "backtracking" is, operationally

In Ward et al.'s metric, a token is a "backtracking marker" if it belongs to `B = {wait, hmm}` (case-insensitive). Backtracking *events* are first such tokens in a sentence. The relevant activations sit ~13–8 tokens *before* the marker — i.e. at the start of the sentence preceding the reconsideration. Layer 10 of the residual stream is where the paper's DoM direction is most effective, at all magnitudes.

### What we change vs. the paper

The paper computes `v = MeanAct(D₊) − MeanAct(D)` in the raw 4096-d residual stream. We compute the same DoM but in **SAE feature space**, ranking features by `Δ_j`. The "backtracking direction" then becomes a small set of feature decoder columns we can steer via either (a) additive injection of `α · unit_norm(W_dec[:,j])` (AxBench) or (b) absolute clamping `z'[j] = strength` with reconstruction-error preservation (paper-clamp). We compare against the raw-DoM baseline.

### Headline question

For the top-K (K=1, 5) SAE features ranked by feature-space DoM, does the magnitude sweep of keyword fraction reach the same peak rate as the raw-DoM baseline at comparable magnitudes? If yes, the backtracking direction *is* well-approximated by interpretable SAE features. If no, the paper's finding that the direction is "densely present" and "one of several heuristics" generalises to feature space (no single-feature explanation).

### Subject model

DeepSeek-R1-Distill-Llama-8B. We use a public Llama-Scope SAE trained on *base* Llama-3.1-8B at L10 — the paper's cosine ≈ 0.74 between base and reasoning directions justifies this transfer.

### Out of scope (followups)

- Reproduce the headline base-vs-distilled "repurposing" finding: derive feature direction from base-model activations, apply to distilled model, confirm base-model is not steerable.
- Logit-lens score per top feature (Eq. 2 in the paper) to confirm none of the top features are mere "Wait/But" unembedding boosters.
- Sweep across SAE architectures: train one of our TXC / SubseqH8 / Matryoshka variants on Llama L10 fineweb activations and rerun Stage 3 + 4.
- Offset sweep to reproduce Fig 2's heatmap in feature space.
