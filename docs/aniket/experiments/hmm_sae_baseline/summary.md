---
author: aniket
date: 2026-03-22
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
| SAE TopK | 1 |
| SAE epochs | 300 |
| SAE learning rate | $10^{-4}$ |
| Seed | 42 |

## Results

### Gamma sweep at fixed $\mu = 0.05$

Swept $\gamma \in \{0, 0.053, 0.211, 0.474, 1.0\}$ by varying $(q, p_A, p_B)$, crossed
with $\lambda \in \{0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0\}$. All 35 configurations share
$\mu = 0.05$.

**Validation**: marginal sparsity matched theory within 0.1% for all configs with
$\lambda \geq 0.1$. At $\lambda = 0$, the high-$q$ configs ($q = 0.1, 0.2, 0.5$) show
larger deviations because the frozen-state initial draw dominates finite samples. The
pooled autocorrelation estimator (green) matches theory (red) at all $\lambda$ values
including $\lambda = 0$; the per-chain estimator (blue) correctly shows near-zero at
$\lambda = 0$ because each chain conditions on its (frozen) hidden state, making
within-chain emissions i.i.d.

![Empirical vs theoretical autocorrelation](../../../results/v5_hmm_sae_baseline/validation_autocorrelation.png)

**SAE performance**: with 300 epochs of training, the SAE fully converges. All 35
configurations achieve AUC = 0.944, mean\_max\_cos\_sim $\approx$ 0.998, and
frac\_recovered\_90 = 1.0. The SAE perfectly recovers all 10 ground-truth features
regardless of $\lambda$ or $\gamma$.

![Trade-off curves](../../../results/v5_hmm_sae_baseline/tradeoff_curves.png)

Convergence curves show all configs plateau by epoch ~150. Earlier runs with 50 epochs
showed under-training artifacts (AUC variation from 0.84 to 0.94 across configs);
300 epochs eliminates this.

![Convergence curves](../../../results/v5_hmm_sae_baseline/convergence.png)

## Sanity check summary

| Check | Status |
|-------|--------|
| Empirical $\mu$ matches theory ($\lambda \geq 0.1$) | Pass |
| Pooled autocorrelation matches theory (all $\lambda$) | Pass |
| Per-chain autocorrelation matches theory ($\lambda \geq 0.3$) | Pass |
| $\gamma = 0$ gives zero autocorrelation | Pass |
| $\gamma = 1$ recovers MC case | Pass |
| SAE trains without NaN/Inf | Pass |
| AUC = 0.944 for all 35 configs | Pass |

## Interpretation

A converged single-position SAE with TopK $k = 1$ achieves perfect feature recovery
(AUC = 0.944, r90 = 100%) uniformly across all $(\lambda, \gamma)$ conditions. The SAE
is blind to temporal structure: it sees one position at a time, so neither the decay rate
($\lambda$) nor the amplitude ($\gamma$) can help or hurt it. This flat baseline is
exactly what we need for the crosscoder comparison — any deviation from this line in the
crosscoder's results must come from exploiting (or failing to exploit) temporal structure.

## Pipeline upstreaming

The HMM emission step has been upstreamed into `src/data_generation/`:

- `EmissionConfig(p_A, p_B)` added to `src/data_generation/configs.py`
- `apply_emission` and updated `generate_support` in `src/data_generation/support.py`
- `hmm_marginal_sparsity`, `hmm_autocorrelation_amplitude`,
  `hmm_theoretical_autocorrelation` added to `src/data_generation/transition.py`
- Per-feature heterogeneity via `generate_support_per_feature` and
  `per_feature_from_pi_rho` in `src/shared/temporal_support.py`
- HMM emission support added to `temporal_crosscoders/data.py` via `p_A`/`p_B`
  parameters on `generate_sequences` and `CachedDataSource`

All changes are backwards-compatible: default emission config ($p_A = 0, p_B = 1$)
recovers the original MC behavior.

## Code

- **Experiment code**: `src/v5_hmm_sae_baseline/`
- **Results**: `results/v5_hmm_sae_baseline/`
- **Experiment plan**: [[plan]]
