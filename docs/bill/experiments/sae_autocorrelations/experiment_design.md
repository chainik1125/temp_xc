---
author: bill
date: 2026-02-25
tags:
  - proposal
  - in-progress
---

## Motivation

In [[ideal_results]], we argued that features with **low activation magnitude** and **high temporal correlation** are precisely the features that standard SAEs will struggle to recover. A temporal crosscoder could exploit persistence across token positions to capture these features more efficiently.

Before investing in toy model construction or crosscoder architectures, we need to empirically validate the premise: **do slow-moving features actually exist in real LLMs?**

This experiment runs a pretrained SAE on natural text and measures whether features exhibit significant temporal autocorrelation, and whether any of those features also have low activation magnitude (the regime where SAEs struggle).

## Setup

- **Model**: GPT-2 Small via TransformerLens
- **SAE**: SAELens `gpt2-small-res-jb`, layer 8 (`blocks.8.hook_resid_pre`), 24,576 features
- **Data**: OpenWebText via HuggingFace streaming (in-distribution for these SAEs)
- **Scope**: 512 sequences of length 1024

We use a pretrained SAE to avoid training overhead. GPT-2 Small + SAELens is the most mature and well-validated combination available. Layer 8 (middle layer) is chosen as a reasonable default — features here are neither too low-level (token identity) nor too abstract.

## What We Measure

For each of the 24,576 SAE features, across all sequences:

1. **Temporal autocorrelation at lags 1-10**: How much does a feature's activation at position $t$ predict its activation at position $t + k$?
2. **Mean activation magnitude** (when active): How strong is the feature when it fires?
3. **Activation frequency**: What fraction of token positions activate this feature?

The core question is whether there exist features in the **low magnitude, high autocorrelation** quadrant of the scatter plot.

## Autocorrelation Computation

For a feature with activation sequence $x_1, \ldots, x_T$ (including zeros), the autocorrelation at lag $k$ is:

$$\rho(k) = \frac{\sum_{t=1}^{T-k} (x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_{t=1}^{T} (x_t - \bar{x})^2}$$

We compute this on the raw signal (including zeros at inactive positions). This captures both support persistence (does the feature stay active?) and magnitude persistence (does the activation strength stay similar?). See [[gating_correlation_math]] for the decomposition of these two channels.

Features with fewer than 20 non-zero positions in a sequence are skipped for that sequence to avoid noisy estimates.

## Significance Testing

Two complementary approaches:

### Shuffled Baseline

Randomly permute token positions within each sequence, recompute autocorrelations. This destroys temporal structure while preserving marginal statistics. Overlaying the shuffled distribution on the real distribution visually reveals whether temporal structure exists (rightward tail in the real distribution).

### Ljung-Box Test

Per-feature test from `statsmodels` that tests the joint null hypothesis: "all autocorrelations at lags 1 through $K$ are zero." The test statistic is:

$$Q = T(T+2) \sum_{k=1}^{K} \frac{\hat{\rho}(k)^2}{T - k}$$

which is $\chi^2(K)$-distributed under the null. Returns a p-value per feature. A histogram of p-values across features should be approximately uniform under the null; a spike near 0 indicates widespread temporal structure.

## Success Criteria

This experiment validates the toy model premise if:

1. A non-trivial fraction of features show statistically significant temporal autocorrelation (Ljung-Box p < 0.01)
2. The autocorrelation distribution has a clear rightward tail compared to the shuffled baseline
3. Some features with high autocorrelation also have low activation magnitude — the "slow-moving, hard-to-recover" regime

## Code Structure

Utilities live in `src/bill/temporal_autocorrelation/`, with a Jupyter notebook for interactive exploration.

- `config.py` — `ExperimentConfig` dataclass holding all parameters (model name, SAE release, hook point, sequence counts, batch size, lag settings, output directory)
- `data.py` — OpenWebText streaming, tokenization, and sequence assembly. Returns a `[N, seq_length]` token tensor
- `activations.py` — Model and SAE loading via TransformerLens/SAELens. Batched feature extraction: runs the model, caches only the target hook point, encodes through the SAE, returns `[B, T, D]` feature activations. Handles GPU memory via CPU offload and cache clearing between batches
- `autocorrelation.py` — Vectorized temporal autocorrelation computation. Takes a `[T, D]` array for one sequence and computes autocorrelation at lags 1-10 for all D features simultaneously. Returns `[D, max_lag]` with NaN where variance is near-zero. This is the performance-critical module
- `statistics.py` — Incremental magnitude/frequency accumulation, shuffled baseline computation, and per-feature Ljung-Box testing
- `visualization.py` — Plotting functions: magnitude vs. autocorrelation scatter, autocorrelation histogram with shuffled overlay, lag decay curves, token-level activation heatmaps, and Ljung-Box p-value histogram

Tests in `tests/test_autocorrelation.py` validate the autocorrelation math against synthetic AR(1) signals and edge cases (all-zeros, constant signals).

The main experiment notebook at `notebooks/01_temporal_autocorrelation.ipynb` orchestrates everything: load model/SAE/data, extract features in batches with incremental statistic accumulation (to avoid materializing the full `[512, 1024, 24576]` ~51GB tensor), assemble final statistics, and produce all plots.

## Connection to Next Steps

- If validated, proceed with the toy model from [[toy_model_ideas]] with confidence that the setup captures real phenomena
- Features identified as slow-moving candidates can inform the design of temporal correlation structures in the toy model (e.g., appropriate autocorrelation strengths, sparsity levels)
- The empirical autocorrelation decay curves (lags 1-10) can calibrate the AR/Markov parameters in the toy model
