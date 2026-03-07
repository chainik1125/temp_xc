---
title: "Project Brief"
author: Han Xuanyuan
date: 2026-02-16
type: document
---

## Abstract

Sparse autoencoders attempt to decompose a models activations into meaningful atomic “features”. Although this has been moderately successful at a given layer, SAEs have two fundamental shortcomings. They cannot explain `global' features present throughout different layers in the network, and they cannot explain features that persist across sequence positions. The second limitation is especially crucial for reasoning models. In this case, there is growing evidence that reasoning capabilities act as independent features that influence the model over many subsequent tokens (https://arxiv.org/pdf/2510.07364).

A natural proposal is hence to generalize SAEs that decompose activations at a single token into “temporal SAEs” that find features across sequence positions. Although there are a number of different architectural choices that could implement this, a natural choice is to consider SAEs based on Tensor Networks (https://arxiv.org/pdf/1306.2164, https://arxiv.org/abs/2011.12127). Tensor Network SAEs provide a way to control the extent to which features are present on a given token, and the extent to which features are delocalized throughout the network.

In this project we would demonstrate the practical utility of Temporal SAEs in the following steps:

Construct a toy model setting based on https://arxiv.org/pdf/2508.16560. Measure the Pareto fronteir for various Temporal SAE architectures (naive stacked SAEs, temporal crosscoders, and Tensor Network SAEs) for different choices of correlations between tokens.

Measure between-token feature correlations in relevant alignment scenarios, and select a minimal proof of concept on a single-GPU model. If possible, construct a dataset with stable correlation behaviours between tokens.

Train a Temporal SAE on base and reasoning models and construct the Pareto fronteir of computational cost vs. reconstruction. Conduct a detailed study of the most important features and compare this to a baseline of stacked SAEs.

If we are able to identify qualitatively different features from temporal SAEs, and better Pareto efficiency in reasoning vs. base models this would constitute a valuable new tool for understanding models and open up a bridge to established techniques in other fields. This would be a strong paper.

## Project motivation

Sparse autoencoders decompose a language model's internal activations into sparse, interpretable features. A standard SAE takes a single activation vector $\mathbf{x}_t \in \mathbb{R}^d$ from one token position at one layer and maps it to a sparse hidden state:

$$\mathbf{a} = \sigma(\mathbf{W}_{\text{enc}}(\mathbf{x}_t - \mathbf{b}_{\text{dec}}) + \mathbf{b}_{\text{enc}})$$
$$\hat{\mathbf{x}}_t = \mathbf{W}_{\text{dec}}\mathbf{a} + \mathbf{b}_{\text{dec}}$$

This per-token decomposition has two fundamental shortcomings:

1. **No cross-position features.** SAEs cannot represent features that persist across sequence positions. This is especially problematic for reasoning models, where reasoning strategies influence generation over many subsequent tokens (see [Reasoning features paper](https://arxiv.org/pdf/2510.07364)).
2. **No cross-layer features.** SAEs cannot explain global features present throughout different layers of the network.

Recent work has attempted to address the first limitation. The **Temporal SAE** paper (Bhalla et al., 2025) adds a contrastive loss encouraging a subset of SAE features to activate consistently across adjacent tokens. This successfully recovers smoother semantic features, but the approach is limited: it uses a soft loss rather than architecturally native cross-token structure, and it requires manually partitioning features into "high-level" and "low-level" groups.

Meanwhile, the **Sparse but Wrong** paper (Chanin et al., 2025) demonstrates that SAE L0 (the average number of active features per token) must be set correctly — if L0 is too low, SAEs mix correlated features together, producing polysemantic latents that appear correct on reconstruction metrics but are fundamentally wrong. This work also provides a toy model framework for evaluating SAEs with known ground-truth features.

## What we are building

We are developing **architecturally native temporal SAEs** — SAE variants where cross-token feature structure is built into the architecture itself, not enforced via auxiliary losses. We are investigating three candidate architectures:

- **Naive stacked SAEs**: Independent per-token SAEs whose outputs are correlated post-hoc (baseline).
- **Temporal crosscoders**: SAEs with shared encoder/decoder structure across positions, so features can explicitly depend on activations at multiple timesteps.
- **Tensor Network SAEs**: SAEs based on tensor network decompositions (e.g., Matrix Product States), which provide a principled way to control feature locality — from single-token features to features delocalized across the full sequence.

The project proceeds in three phases:

1. **Toy model validation.** Construct synthetic data with known cross-token feature correlations (building on the Chanin et al. framework). Measure the Pareto frontier of reconstruction quality vs. computational cost for each architecture.
2. **Empirical measurement.** Measure between-token feature correlations in real alignment-relevant scenarios. Select a minimal proof-of-concept on a single-GPU model.
3. **LLM training and analysis.** Train temporal SAEs on base and reasoning models. Compare the features discovered against stacked SAE baselines. Determine whether temporal SAEs find qualitatively different features and whether they show better Pareto efficiency on reasoning models.

## Key references

You should be familiar with the following:

- **Sparse but Wrong** (Chanin et al., 2025) — (papers/sparse_but_wrong.md). Demonstrates that incorrect L0 corrupts SAE features. Provides the toy model framework we build on.
- **Temporal SAEs** (Bhalla et al., 2025) — (papers/temporal_sae.md). Contrastive loss approach to temporal consistency. Our soft-loss baseline.
- **Tensor Networks intro** — [arXiv:1306.2164](https://arxiv.org/pdf/1306.2164). Background on tensor network decompositions.
- **Tensor Networks for ML** — [arXiv:2011.12127](https://arxiv.org/abs/2011.12127). Applications of tensor networks to machine learning.
- **Reasoning features** — (papers/reasoning_features.md). Evidence that reasoning capabilities act as persistent features across tokens.

Shared documents: the architectural design document is at `docs/shared/temporal_xc_architectures.md`. The project manifesto is at `docs/shared/manifesto.md`. 

## Source code layout

All code lives in `src/`. Each experiment or body of work gets its own subdirectory, but code sharing between experiments is encouraged via the `src/shared/` and `src/utils/` modules.

| Directory | Description |
|---|---|
| `src/shared/` | Shared modules reused across experiments (data generation, SAE training, evaluation, plotting, configs) |
| `src/utils/` | Small standalone utilities (device selection, cosine similarity, seeding) |
| `src/v0_toy_model/` | Reproduction of Chanin et al. "Sparse but Wrong" — the baseline toy model |
| `src/v2_temporal_schemeC/` | Temporal SAE experiments using Scheme C (Markov chain) data |
| `src/TemporalFeatureAnalysis/` | **Reference code** from the paper "Priors in Time: Missing Inductive Biases for Language Model Interpretability". This is *not* our code — it is the paper's source included for reference. Do not modify it. |

When starting a new experiment, create a new subdirectory (e.g. `src/v2_temporal_xc/`) and import shared utilities from `src/shared/` rather than duplicating code.

## Current progress

Check `docs/han/research_logs/` for my most recent research logs. These are updated regularly and contain the latest experimental results, design decisions, and open questions. **Always read the most recent research log before starting work** to avoid duplicating effort or working from stale assumptions.