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

Reusable backend lives in `src/` (organised by role, not version). Phase-specific experiment scripts live at repo root under `experiments/phase{2,3,4}_*/`. Filesystem enforces the separation — experiments can import from `src.*`, but `src/` never imports from `experiments/`.

| Directory | Description |
|---|---|
| `src/architectures/` | SAE variant models (TopK, ReLU, Stacked, TXCDR, TFA) with a shared `ArchSpec` interface. Single source of truth — both toy and NLP pipelines import from here. |
| `src/data/toy/` | Synthetic data generators (Markov chain, OR-gated coupled features, base features) |
| `src/data/nlp/` | Real-LM cached activations — loader, cache builder, layer specs, subject-model registry |
| `src/training/` | Training loops and config dataclasses (train_tfa, train_crosscoder, train_stacked_sae, train_sae) |
| `src/eval/` | Evaluation runner + metrics (NMSE/L0/AUC, temporal MI, activation spans, feature recovery) |
| `src/plotting/` | Shared plot utilities (heatmaps, UMAP, save_figure) |
| `src/pipeline/` | Reusable orchestration (architecture sweep runner, autointerp, toy sweeps, results I/O) |
| `src/utils/` | Small cross-cutting helpers (device, seed, cos_sim, orthogonalize) |
| `experiments/phase2_toy/` | Toy TFA-vs-SAE benchmarks, convergence/multi-seed, correlation sweeps |
| `experiments/phase3_coupled/` | Coupled-features experiments (OR-gated, noisy emission, linear probes) |
| `experiments/phase4_nlp/` | Gemma NLP comparison: cache activations, NLP sweep, decoder alignment, autointerp, UMAP |
| `references/TemporalFeatureAnalysis/` | **Vendored** paper code — reference only, not imported at runtime. |

When starting a new experiment, create a new script under `experiments/phaseN_*/` and import backend modules from `src.*`. When the backend needs a new capability (e.g. a new architecture), add it under `src/architectures/` following the `ArchSpec` interface — it is then automatically available to any experiment.

## Current progress

Research logs live in `docs/han/research_logs/phase{2,3,4}_*/`, grouped by research phase. Each phase's key finding is summarized below; open the log linked for the full methodology and numbers. **Always read the newest log in the active phase before starting work.**

### Phase 2 — toy experiments: TFA vs SAE at varying sparsity (Mar–Apr 2026)

Ran experiments 1a–1d, 2, 3 on the Scheme C toy model. Headline results:

- **TFA wins massively when `k < true L0`.** At matched sparsity budget, TFA reconstructs better than stacked SAE whenever the per-token budget is below the true number of active features. Robust across seeds.
- **Tied at `k = true L0`.** Neither architecture has an advantage when sparsity exactly matches.
- **TFA advantage widens with scale.** At 30 features, TFA dominates across a wider `k` range than at toy scale.
- **TFA convergence is slow but keeps improving.** ~7.5k steps to match SAE at `k=2`; still improving at 30k+.

Interpretation: the predictable-component head gives TFA "free" features that are perfectly reconstructable from context, letting the sparse novel head use its budget on genuinely new information.

- [`phase2_toy_experiments/2026-03-30-synthesis.md`](../research_logs/phase2_toy_experiments/2026-03-30-synthesis.md) — cross-experiment summary
- [`phase2_toy_experiments/2026-03-30-experiment1-topk-sweep.md`](../research_logs/phase2_toy_experiments/2026-03-30-experiment1-topk-sweep.md) — the headline k-sweep

### Phase 3 — coupled features in the toy model (Apr 2026)

Extended the toy model to coupled feature groups (OR-gated + stochastic emission). Tested whether temporal SAEs can recover coupling structure that standard SAEs miss.

Finding: TFA recovers coupled features more cleanly than stacked SAE when the coupling is temporal (OR-gate across positions), but the advantage is narrower than the matched-k result — most of the gap is explained by TFA's pred head, not the coupling per se.

- [`phase3_coupled_features/2026-04-07-experiment1c3-coupled-features.md`](../research_logs/phase3_coupled_features/2026-04-07-experiment1c3-coupled-features.md)
- [`phase3_coupled_features/2026-04-10-experiment1c3-noisy-coupled.md`](../research_logs/phase3_coupled_features/2026-04-10-experiment1c3-noisy-coupled.md) — stochastic emission variant

### Phase 4 — NLP comparison on Gemma-2-2B-IT (Apr 2026, active)

Ported Aniket's NLP pipeline (`src/bench/`) and ran a three-way comparison — Stacked SAE, TXCDR (temporal crosscoder), TFA-pos — on `resid_L25` of Gemma-2-2B-IT, `d_sae = 18432`, FineWeb.

Findings (as written up on `han`):

1. **Nearly disjoint feature dictionaries.** Cross-architecture decoder cosine median = 0.10–0.23, barely above random baseline 0.09. Each architecture fills a different region of direction-space.
2. **TFA's dictionary bimodally splits.** 42.6% pred-only / 57.4% novel-only features; essentially 0% mixed use. Pred-only features have long spans (mean 4.5 tokens, p99 = 51); novel-only are transient (mean 1.1).
3. **Each architecture specializes.** Autointerp on top-unique features per category read as distinct semantic types: Stacked = concrete lexical ("motor"), TXCDR = grammatical/multilingual (function words, Cyrillic), TFA-pred = context-determined structural tokens ("second digit of HH:MM"), TFA-novel = sequence-boundary markers (partly caching artifact).
4. **Joint UMAP** shows four spatially-separated territories confirming the above.

- [`phase4_nlp_comparison/2026-04-17-nlp-comparison-index.md`](../research_logs/phase4_nlp_comparison/2026-04-17-nlp-comparison-index.md) — read first
- [`phase4_nlp_comparison/2026-04-17-nlp-feature-comparison-phase1.md`](../research_logs/phase4_nlp_comparison/2026-04-17-nlp-feature-comparison-phase1.md) — structural metrics
- [`phase4_nlp_comparison/2026-04-17-nlp-feature-comparison-phase2.md`](../research_logs/phase4_nlp_comparison/2026-04-17-nlp-feature-comparison-phase2.md) — autointerp
- [`phase4_nlp_comparison/2026-04-17-nlp-feature-comparison-phase3.md`](../research_logs/phase4_nlp_comparison/2026-04-17-nlp-feature-comparison-phase3.md) — joint UMAP
- [`phase4_nlp_comparison/2026-04-17-high-span-feature-comparison.md`](../research_logs/phase4_nlp_comparison/2026-04-17-high-span-feature-comparison.md) — temporal-subset follow-up

### Open questions and known caveats

1. **Parallel work on the `han-runpod` branch contradicts the "TFA-pred is interpretable" finding.** Using a stronger *feature distinctness* metric (unique top-10 exemplar sets across features), they find 194 / 200 top span-weighted TFA-pred features share *literally identical* top-10 exemplars at L25 (one biochem passage), and 158 / 200 at L13 (one insurance boilerplate). Fisher p ≪ 10⁻⁷². If this replicates on our checkpoints it means the `han` "TFA pred exposes a new category" claim is an artifact of our top-N-by-mass ranking; the real claim should be "TXCDR wins cleanly on feature distinctness." Not yet reconciled on `han`. See `origin/han-runpod:docs/han/research_logs/2026-04-18-txcdr-vs-tfa.md`.
2. **Single seed, single layer, single k** for all NLP results. Needs at least one more of {layer, seed, k} to make the architectural claims robust.
3. **TFA training is fragile at real-LM scale.** Required stability fixes (decoder row renorm, NaN/inf batch skip, lr=3e-4, smaller batch). Still diverges at higher k values. Training budget is not matched across archs.
4. **TXCDR has severe dead-feature problem** at d_sae=18,432 — only 12% alive. All comparisons filter to alive features, but the effective capacity is much smaller than nominal.
5. **Sparsity-budget asymmetry makes TXCDR ↔ TFA-pred comparison unclean.** TFA-pred is dense (ReLU only, no TopK); TXCDR is sparse (TopK at k·T=500). Direct comparison requires either a sparsified TFA-pred variant or a matched-density protocol.
6. **TFA novel on our caching scheme is dominated by sequence-boundary features.** Right-padded 128-token caching concentrates TFA novel mass on first-content tokens. Re-run with a packed-sequence cache to get genuinely transient intra-text features.
7. **No downstream utility demonstration.** Nothing yet tested on circuit discovery, feature steering, or a probing benchmark. Required for a "temporal SAEs are useful" claim.

### Suggested next directions (not committed to)

- Reconcile with `han-runpod`: port their feature-distinctness metric and span-weighted ranking, re-rank our TFA-pred features, update Phase 4 writeup honestly.
- Layer replication: run Phase 4 at `resid_L13` on Gemma.
- Packed-sequence cache + rerun to defuse the TFA-novel boundary artifact.
- Scale to DeepSeek-R1-Distill-Llama-8B (reasoning traces) — infrastructure exists, caching was interrupted.