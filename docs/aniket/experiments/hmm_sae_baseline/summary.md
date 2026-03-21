---
author: aniket
date: 2026-03-19
tags:
  - results
  - complete
---

## Overview

This experiment validates the HMM data generation pipeline and establishes SAE training
baselines before the full crosscoder comparison. We extended the deterministic emission
model ($s_t = z_t$) to a proper Hidden Markov Model where each hidden state emits
$s = 1$ with probability $p_A$ or $p_B$, then trained a TopK SAE on data generated at
several correlation levels.

The key new quantity is the **autocorrelation amplitude prefactor**:

$$\gamma = \frac{\pi_A \pi_B (p_B - p_A)^2}{\mu(1 - \mu)}$$

which controls how much temporal information the observations carry about the hidden
state. The HMM decouples marginal sparsity ($\mu$), autocorrelation decay rate
($\lambda$), and autocorrelation amplitude ($\gamma$).

## Parameters

| Parameter | Value |
|-----------|-------|
| Features ($k$) | 10 |
| Ambient dimension ($d$) | 64 |
| Sequence length ($T$) | 128 |
| Sequences ($n_{\text{seq}}$) | 200 |
| Magnitudes | $\|N(0,1)\|$ (half-normal) |
| SAE latents | 64 |
| SAE TopK | 5 |
| SAE epochs | 50 |
| SAE learning rate | $10^{-4}$ |
| Seed | 42 |

## Results

### Sub-experiment A: Amplitude Sweep

Fixed $q = 0.5$, $\mu = 0.05$, swept $(p_A, p_B)$ pairs at each $\lambda$.

**Validation**: empirical marginal sparsity matched theoretical $\mu = 0.05$ within
0.2% for all configurations. Empirical autocorrelation matched the theoretical
$\rho^{\tau} \cdot \gamma$ curves. The gamma=0 case ($p_A = p_B = 0.05$) produces
identically zero autocorrelation regardless of $\lambda$, confirming the emission noise
fully masks the hidden temporal structure.

**SAE performance**: Feature recovery AUC (threshold-sweep, matching Andre's
`feature_recovery_score` implementation) ranged from 0.78 to 0.81 across all
configurations. The `mean_max_cos_sim` metric tracks closely but AUC provides a
richer picture via the survival curve. No features reached the 0.9 recovery threshold
(`frac_recovered_90 = 0` in most runs), indicating uniform partial recovery rather
than bimodal all-or-nothing recovery.

Performance is essentially flat across $\lambda$ values — the standard SAE sees each
position independently, so temporal correlation does not affect its reconstruction
ability. The slight upward trend with higher gamma (0.80-0.81 vs 0.78) likely
reflects that higher $|p_B - p_A|$ makes the emission distribution more bimodal,
marginally easier for the SAE to decompose.

Reconstruction loss was uniformly low ($\sim 3 \times 10^{-5}$), confirming the SAE
converges well in this toy setting.

### Sub-experiment B: MC Sanity Check

$q = 0.05$, $p_A = 0$, $p_B = 1$ ($\gamma = 1$), same lambda sweep.

**Autocorrelation validation**: empirical autocorrelation matched the theoretical
$(1 - \lambda)^{\tau}$ curves exactly, confirming the HMM reduces to the MC case when
$p_A = 0, p_B = 1$.

**Edge case at $\lambda = 0$**: the frozen-state case shows lower AUC (0.73) and
near-zero reconstruction loss. This is expected — at $\lambda = 0$, only ~5% of
features are ever active (those that started in state B), producing a very different
data distribution. For $\lambda \geq 0.1$, AUC is in the 0.77-0.80 range, consistent
with sub-experiment A.

## Sanity Check Summary

| Check | Status |
|-------|--------|
| Empirical $\mu$ matches theory within 1% | Pass |
| Empirical autocorrelation matches theory | Pass |
| $p_A = p_B$ gives zero autocorrelation | Pass |
| $p_A = 0, p_B = 1$ recovers MC case | Pass |
| SAE trains without NaN/Inf | Pass |
| Feature AUC > 0.75 for moderate $\lambda$ | Pass |
| All plots saved and readable | Pass |

## Interpretation

The amplitude prefactor $\gamma$ does not meaningfully affect standard SAE training
because the SAE processes each position independently — it cannot exploit temporal
correlations. This is exactly the baseline we need: the standard SAE's performance is
invariant to temporal structure, establishing the floor against which a crosscoder
(which *can* use temporal context) will be compared.

The fact that $\gamma$ has minimal impact on the SAE confirms that the HMM's temporal
parameters ($\lambda$, $\gamma$) are orthogonal to the reconstruction task as seen by
a single-position model.

## Next Steps

1. Train a temporal crosscoder on the same data grid and compare feature AUC against
   these baselines
2. The crosscoder should show improved AUC when both $\lambda$ is small (slow decay)
   AND $\gamma$ is large (high amplitude) — these are the conditions where temporal
   context carries useful information
3. Upstream the HMM data generation to `src/data_generation/` once the crosscoder
   comparison validates the pipeline end-to-end

## Code

- **Experiment code**: `src/v5_hmm_sae_baseline/`
- **Results**: `results/v5_hmm_sae_baseline/`
- **Experiment plan**: [[plan]]
