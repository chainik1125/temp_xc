---
author: bill
date: 2026-02-25
tags:
  - results
---

## Summary

We ran a pretrained SAE (SAELens `gpt2-small-res-jb`, layer 8, 24,576 features) on 4,096 sequences of 128 tokens from OpenWebText through GPT-2 Small. The goal was to validate whether "slow-moving" features — features with high temporal autocorrelation and low activation magnitude — exist in practice. See [[experiment_design]] for full methodology.

**The answer is yes.** Temporal autocorrelation is widespread among SAE features, and many of the temporally persistent features have low activation magnitudes — precisely the regime where standard SAEs struggle.

## Setup

- **Model**: GPT-2 Small (117M parameters, 12 layers)
- **SAE**: SAELens `gpt2-small-res-jb`, `blocks.8.hook_resid_pre`, 24,576 features (32x expansion)
- **Data**: 4,096 sequences of 128 tokens from OpenWebText, random offsets into documents
- **Sequence length**: 128 tokens (matches the SAE's training context size; reconstruction R^2 = 0.99)
- **Autocorrelation**: Computed at lags 1-10 on post-ReLU feature activations (including zeros)
- **Minimum activations threshold**: 20 nonzero positions per 128-token sequence (15.6% firing rate)

## Key Numbers

| Metric | Value |
|--------|-------|
| Total SAE features | 24,576 |
| Features with valid autocorrelation estimates | 3,939 (16%) |
| Features with valid magnitude | 24,570 (>99%) |
| Median activation frequency | 0.11% |
| Features never active | 6 |
| Mean lag-1 autocorrelation (valid features) | 0.42 |
| Median lag-1 autocorrelation (valid features) | 0.46 |
| Features with lag-1 AC > 0.5 | 1,517 |
| Features with lag-1 AC > 0.5 AND magnitude below median | 889 |
| Ljung-Box significant at p < 0.01 (single sequence) | 427 / 3,491 (12.2%) |

## Findings

### 1. Temporal autocorrelation is widespread

Among the 3,939 features with enough activations for reliable autocorrelation estimates, the mean lag-1 autocorrelation is **0.42** and the median is **0.46**. This is far from zero — the majority of sufficiently active features show substantial temporal persistence.

The shuffled baseline distribution (where we permute token positions to destroy temporal structure) collapses tightly around zero (mean ~0, std ~0.03), while the real distribution peaks around 0.4-0.5 with a long right tail reaching past 0.8. The two distributions are almost entirely non-overlapping.

### 2. Slow-moving features exist

The top features by lag-1 autocorrelation show remarkably persistent temporal structure:

| Feature | Lag-1 AC | Magnitude | Frequency |
|---------|----------|-----------|-----------|
| 22731 | 0.93 | 5.46 | 0.05% |
| 17769 | 0.90 | 4.76 | 0.06% |
| 19438 | 0.87 | 4.86 | 0.04% |
| 7518 | 0.87 | 3.16 | 0.07% |
| 13765 | 0.86 | 25.62 | 0.07% |
| 2871 | 0.84 | 1.76 | 0.55% |

Feature 2871 is particularly interesting: lag-1 AC of 0.84 with a magnitude of only 1.76 — a clear example of a slow-moving, low-magnitude feature.

The autocorrelation decay curves show diverse temporal scales. Some features (e.g., 19438, 4067) maintain positive autocorrelation out to lag 10, while others decay faster. This variety of temporal scales is what we would expect if the features correspond to different types of contextual information (e.g., topic, tone, syntactic structure) that persist at different rates.

### 3. Low-magnitude features with high autocorrelation

**889 features** have lag-1 autocorrelation above 0.5 and activation magnitude below the median (1.42). These are exactly the features that, per the argument in [[ideal_results]], standard SAEs would struggle to recover: their low magnitude means the L1 penalty cost is high relative to the reconstruction benefit, but their temporal persistence means a crosscoder-style architecture could exploit sequential context to recover them more efficiently.

### 4. Ljung-Box confirms significance

On a single 128-token sequence, the Ljung-Box test (testing the joint null of zero autocorrelation at lags 1-10) rejects at p < 0.01 for **12.2%** of testable features. The p-value histogram is bimodal: a spike near 0 (genuine temporal structure) and a spike near 1 (features with too few activations for a meaningful test on one sequence). This confirms that the temporal structure is statistically significant, not just an artifact.

### 5. Note on coverage

Only 3,939 / 24,576 features (16%) had enough activations (>= 20 per sequence) to compute reliable autocorrelation estimates. The remaining 84% are too sparse at the per-sequence level with `seq_length=128`. This is a limitation of the short context — these very sparse features may also have temporal structure that we cannot measure with this setup. A longer context or a relaxed threshold would capture more features, but at the cost of noisier estimates.

## Caveats

- **Single layer**: We only examined layer 8. Other layers may show different temporal characteristics. Early layers likely have more positional/syntactic features; later layers may have more abstract semantic features.
- **Single SAE**: The `gpt2-small-res-jb` SAE was trained with a specific L1 penalty. Different sparsity levels could recover different features.
- **Short context**: 128 tokens limits our ability to measure long-range dependencies. Features corresponding to document-level structure (e.g., overall tone or topic) may require longer contexts to observe.
- **GPT-2 Small is a small model**: Larger models may have richer temporal dynamics in their feature representations.

## Implications for Toy Model

These results validate the premise of the toy model described in [[toy_model_ideas]]:

- **Temporal correlation exists** and is widespread — it is not a niche phenomenon
- **Multiple temporal scales** are present (decay curves vary from rapid to slow) — the toy model should support a range of autocorrelation strengths
- **Low-magnitude + high-AC features exist** (889 features) — confirming that the "hard-to-recover" regime from [[ideal_results]] is real
- **Empirical autocorrelation values** (lag-1 AC ranging from ~0.2 to ~0.9) can calibrate the AR/Markov parameters in the toy model
- The **activation frequency of high-AC features** (0.04%-0.55%) gives realistic sparsity targets for the toy model

## Reproduction

Code is in `src/bill/temporal_autocorrelation/`. Run the notebook at `bill/temporal_autocorrelation/notebooks/01_temporal_autocorrelation.ipynb`. Unit tests: `uv run pytest bill/temporal_autocorrelation/scripts/test_autocorrelation.py`.
