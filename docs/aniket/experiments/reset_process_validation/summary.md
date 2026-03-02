---
author: aniket
date: 2026-03-02
tags:
  - results
  - complete
---

## Overview

This experiment validates our implementation of Dmitry's reset process — a two-state
Markov chain for generating temporally correlated binary support sequences. The reset
process is parameterized by a single mixing parameter $\lambda$:

$$T(\lambda) = (1 - \lambda) I + \lambda R_S$$

where $R_S$ is the reset matrix (every row is the stationary distribution $[1-p,\; p]$).
Key properties:

- **Stationary probability**: $p$ (by construction, for all $\lambda$)
- **Autocorrelation**: $\text{Corr}(s_t, s_{t+\tau}) = (1 - \lambda)^{|\tau|}$
- $\lambda = 0$: perfect memory (state never changes)
- $\lambda = 1$: i.i.d. $\text{Bernoulli}(p)$ (memoryless)

We ran four sub-experiments to verify correctness. All plots are in
`results/reset_process_validation/`.

## Parameters

| Parameter | Value |
|-----------|-------|
| Features ($k$) | 30 |
| Sequence length ($T$) | 256 |
| Sparsity ($p$) | 0.05 |
| Magnitudes | $\|N(0,1)\|$ (half-normal) |
| $\lambda$ values | 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 |
| Sequences for statistics | 1000 |
| Max autocorrelation lag | 30 |
| Seed | 42 |

## Results

### Exp 1: Support Heatmaps

**File**: `support_heatmaps.png`

Shows the binary on/off pattern for feature 0 across all $T = 256$ positions, with one
row per $\lambda$ value. The plot is mostly white because $p = 0.05$ means the feature
is active only ~5% of the time.

**Observations**:

- $\lambda = 0.0, 0.1$: entirely white — feature 0 started OFF (95% chance) and with
  near-zero turn-on probability ($\beta = \lambda p \approx 0$), it never activates.
  This is correct: $\lambda = 0$ is the degenerate "frozen initial state" case.
- $\lambda = 0.3$: a single short burst appears.
- $\lambda = 0.5$: two isolated bursts with some width, showing moderate temporal
  persistence.
- $\lambda = 0.7$--$1.0$: progressively more frequent activations with shorter burst
  lengths, trending toward i.i.d. flickering at $\lambda = 1.0$.

The transition from sustained bursts (low $\lambda$) to scattered points (high
$\lambda$) confirms the temporal structure is working as expected, with the caveat that
at very low $\lambda$ the visualization is dominated by the initial-state lottery.

### Exp 2: Empirical vs Theoretical Autocorrelation

**File**: `autocorrelation.png`

For each $\lambda$, we computed empirical autocorrelation of the support variable for
feature 0 across 1000 sequences, and overlaid the theoretical curve
$(1 - \lambda)^{\tau}$.

**Observations**:

- **Excellent agreement** between empirical (blue dots) and theoretical (red line)
  across all $\lambda$ values and all lags $\tau = 0 \ldots 30$.
- $\lambda = 0.0$: flat autocorrelation at 1.0 for all lags (perfect memory).
- $\lambda = 0.1$: slow exponential decay, reaching ~0.05 around lag 30.
- $\lambda = 0.3$--$0.5$: moderate decay, autocorrelation drops below 0.1 by lag
  10--15.
- $\lambda = 0.7$--$0.9$: fast decay, near-zero by lag 5.
- $\lambda = 1.0$: zero autocorrelation at all lags $> 0$ (i.i.d.).

**Conclusion**: the implementation matches the theoretical autocorrelation exactly.
This is the strongest quantitative validation.

### Exp 3: Activation Heatmaps

**File**: `activation_heatmaps.png`

Shows the full activation matrix $a_{i,t} = s_{i,t} \cdot m_{i,t}$ for all $k = 30$
features across $T = 256$ positions, with a 'hot' colormap (black = 0,
red/yellow = nonzero).

**Observations**:

- $\lambda = 0.0$: mostly black with one or two features active for the entire
  sequence (bottom rows show sustained red bands). These are the features that happened
  to start ON and remain locked on with constant support. Most features started OFF
  and stay dark.
- $\lambda = 0.1$: clear temporal bursts visible — features activate in contiguous
  blocks spanning 10--30 positions, with varying magnitudes within each block.
- $\lambda = 0.3$: shorter bursts, more features cycling on/off.
- $\lambda = 0.5$--$0.7$: activations become more scattered, approaching a
  salt-and-pepper pattern.
- $\lambda = 0.9$--$1.0$: nearly i.i.d. — isolated single-position activations spread
  uniformly, no visible temporal structure.

The global $v_{\max}$ normalization ensures magnitude scale is comparable across
$\lambda$ values.

### Exp 4: Stationary Probability Check

**File**: `stationary_probability.png`

For each $\lambda$, we computed the empirical firing probability at each position $t$,
averaged over all 30 features and 1000 sequences, and compared to the theoretical
stationary probability $p = 0.05$.

**Observations**:

- All $\lambda$ curves hover tightly around $p = 0.05$ at every position $t$, with very
  small sampling fluctuations (within ~0.005 of the true value).
- The $\lambda = 0.0$ line is the flattest — because the initial $\text{Bernoulli}(0.05)$
  draw is locked in forever, averaging over 30,000 samples ($1000 \times 30$) gives a
  very stable estimate.
- No systematic drift or edge effects at $t = 0$ or $t = T$, confirming that
  initializing from the stationary distribution avoids burn-in artifacts.

**Conclusion**: the marginal sparsity is correct and position-independent for all
$\lambda$ values.

## Summary

All four sub-experiments confirm the reset process implementation is correct:

1. Temporal structure (burst length) is controlled by $\lambda$ as expected
2. Empirical autocorrelation matches $(1 - \lambda)^{\tau}$ theory exactly
3. Combined activations (support $\times$ magnitude) show the expected patterns
4. Marginal sparsity equals $p = 0.05$ regardless of $\lambda$, with no positional bias

The implementation in `src/shared/temporal_support.py` is validated and ready for use
in downstream temporal crosscoder data generation.

## Code

- **Shared module**: [[temporal_support.py]] at `src/shared/temporal_support.py`
- **Experiment script**: `src/v4_reset_process_validation/run_validation.py`
- **Experiment plan**: [[plan]]
