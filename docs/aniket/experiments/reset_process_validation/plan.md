---
author: aniket
date: 2026-03-01
tags:
  - proposal
  - todo
---

## Motivation

We need to validate the reset process data generation before using it in downstream
temporal crosscoder experiments. The reset process is a two-state Markov chain
parameterized by a single mixing parameter lambda that controls temporal correlation while
preserving stationary sparsity. Before building the full pipeline, we verify that:

1. The implementation produces the expected temporal structure visually
2. Empirical autocorrelation matches the theoretical curve (1-lambda)^tau
3. The combined activation (support x magnitude) looks reasonable
4. Marginal sparsity is correct for all lambda values

## Background

The reset process transition matrix is:

    T(lambda) = (1 - lambda) * I + lambda * R_S

where R_S has every row equal to the stationary distribution [1-p, p]. This gives:

- alpha = 1 - lambda(1-p)  (stay-on probability)
- beta = lambda * p         (turn-on probability)
- Stationary probability: p (by construction, for all lambda)
- Autocorrelation: Corr(s_t, s_{t+tau}) = (1 - lambda)^|tau|

lambda=0 is perfect memory; lambda=1 is i.i.d. Bernoulli(p).

## Experiment Parameters

- k = 10 features, d = 64 ambient dimension, T = 128 sequence length
- p = 0.05 sparsity
- Magnitude distribution: |N(0, 1)| (half-normal)
- lambda sweep: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
- n_sequences = 1000 for empirical statistics
- max_lag = 30 for autocorrelation
- Seed: 42

## Sub-experiments

### Exp 1: Support Heatmaps

For each lambda, generate ONE sequence. Plot the on/off pattern for feature 0 across
all T=128 positions as a binary heatmap. All lambda values stacked vertically in one
figure.

**Purpose**: visual intuition for how lambda controls block length (long contiguous
bursts at lambda ~ 0, flickering at lambda ~ 1).

### Exp 2: Empirical vs Theoretical Autocorrelation

For each lambda, generate n_sequences=1000 sequences. Compute empirical autocorrelation
of the support variable for feature 0 at lags tau = 0, 1, ..., 30. Overlay the
theoretical curve (1-lambda)^tau. One subplot per lambda value.

**Purpose**: validate that the implementation matches theory.

### Exp 3: Activation Heatmaps

For each lambda, generate ONE sequence. Plot the full activation matrix
a_{i,t} = s_{i,t} * m_{i,t} for all k=10 features across T=128 positions. Use a
sequential colormap (e.g. 'hot') so zeros are clearly distinct from nonzero magnitudes.
All lambda values stacked vertically.

**Purpose**: see how magnitude x support looks end-to-end.

### Exp 4: Stationary Probability Check

For each lambda, generate n_sequences=1000 sequences. At each position t, compute
the empirical firing probability averaged over all features and all sequences. Plot
all lambda curves on one axis with a horizontal dashed line at p=0.05.

**Purpose**: verify the marginal sparsity is correct regardless of lambda.

## Code Structure

### New shared module

- `src/shared/temporal_support.py` -- `generate_support_reset(k, T, p, lam, rng)` that
  runs k independent two-state Markov chains. Uses torch. Returns binary tensor (k, T).

### Experiment module

- `src/v4_reset_process_validation/__init__.py`
- `src/v4_reset_process_validation/run_validation.py` -- main script with all four
  sub-experiments, using module-level constants (v3 pattern)

### Reused shared modules

- `src/utils/seed.py` -- `set_seed(42)` for reproducibility
- `src/utils/logging.py` -- `log()` for progress messages
- `src/shared/plotting.py` -- `save_figure()` for PNG output

### Output

All plots saved to `results/reset_process_validation/`:

- `support_heatmaps.png` -- Exp 1
- `autocorrelation.png` -- Exp 2
- `activation_heatmaps.png` -- Exp 3
- `stationary_probability.png` -- Exp 4

## Expected Results

- **Exp 1**: lambda=0 shows a single constant state (all on or all off for the full
  sequence); lambda=1 shows i.i.d. flickering; intermediate values show blocks
- **Exp 2**: empirical dots should closely track the (1-lambda)^tau curves
- **Exp 3**: similar temporal structure to Exp 1 but with varying magnitudes on active
  positions
- **Exp 4**: all lambda curves should hover around p=0.05 at every position t
