---
author: aniket
date: 2026-03-19
tags:
  - proposal
  - todo
---

## Motivation

Our current data generation pipeline uses a simple Markov chain where the hidden state IS
the observation: state 0 deterministically emits $s = 0$ (OFF), state 1 deterministically
emits $s = 1$ (ON). The emission matrix is the identity and there is nothing "hidden."

We need to upgrade to a proper Hidden Markov Model (HMM) where each hidden state emits
$s = 1$ with some probability $p_A$ or $p_B$, not deterministically. Then we train a
standard TopK SAE on data generated from this HMM at several correlation levels to verify
the pipeline works end-to-end and establish baselines before the full crosscoder comparison.

This is a **self-contained experiment** in `src/v5_hmm_sae_baseline/`. It does NOT modify
the existing `src/data_generation/` pipeline. We will upstream the HMM changes later once
validated.

## Background

### Current model (MC with deterministic emissions)

Per feature $i$, hidden state $z_t \in \{0, 1\}$ follows a Markov chain with transition
matrix $P$:

$$P = \begin{bmatrix} 1 - \beta & \beta \\ 1 - \alpha & \alpha \end{bmatrix}$$

where $\alpha = P[1,1]$ (on-to-on persistence) and $\beta = P[0,1]$ (off-to-on rate).

Current emission: $s_t = z_t$ (deterministic). Observation IS the state.

With the reset process $T(\lambda) = (1-\lambda)I + \lambda R_S$:

- $\alpha = 1 - \lambda(1 - p)$, $\beta = \lambda p$
- Stationary distribution: $\pi_{\text{on}} = p$ (independent of $\lambda$)
- Autocorrelation: $\text{Corr}(s_t, s_{t+\tau}) = (1 - \lambda)^{|\tau|}$

### New model (HMM with stochastic emissions)

The hidden chain $z_t \in \{A, B\}$ still uses the same transition matrix $P$. But now:

$$s_t \mid z_t = A \sim \text{Bernoulli}(p_A), \quad s_t \mid z_t = B \sim \text{Bernoulli}(p_B)$$

The current MC is the special case $p_A = 0$, $p_B = 1$.

### Key derived quantities

Let $\pi = [\pi_A, \pi_B]$ be the stationary distribution. For the reset process with
stationary parameter $q$:

- $\pi = [1-q, \; q]$
- **Marginal firing probability**: $\mu = (1-q) \cdot p_A + q \cdot p_B$
- **Observed autocorrelation**: $\text{Corr}(s_t, s_{t+\tau}) = \rho^{|\tau|} \cdot \gamma$

where $\rho = 1 - \lambda$ is the second eigenvalue and:

$$\gamma = \frac{\pi_A \pi_B (p_B - p_A)^2}{\mu(1 - \mu)}$$

is the **amplitude prefactor** in $[0, 1]$.

The HMM decouples three quantities the MC conflated:

1. **Marginal sparsity** $\mu$ (controlled by $q$, $p_A$, $p_B$)
2. **Autocorrelation decay rate** (controlled by $\lambda$)
3. **Autocorrelation amplitude** (controlled by $|p_B - p_A|$)

## Experiment Design

### Sub-experiment A: Amplitude sweep at fixed sparsity

Fix $q = 0.5$, $\mu = 0.05$ (so $p_A + p_B = 0.1$). Sweep:

| $(p_A, p_B)$ | $\gamma$ | Label |
|---------------|----------|-------|
| (0.05, 0.05) | 0.000 | i.i.d. (no temporal info) |
| (0.04, 0.06) | 0.002 | low amplitude |
| (0.02, 0.08) | 0.019 | medium amplitude |
| (0.00, 0.10) | 0.053 | high amplitude |

Lambda sweep: $\lambda \in \{0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0\}$.

For each $(\lambda, p_A, p_B)$: generate data, validate statistics, train TopK SAE,
evaluate ground-truth recovery.

### Sub-experiment B: MC sanity check

$q = 0.05$, $p_A = 0$, $p_B = 1$, $\mu = 0.05$, $\gamma = 1$. Same lambda sweep. This
should exactly recover existing MC pipeline behavior.

### SAE hyperparameters (fixed)

| Parameter | Value |
|-----------|-------|
| $d_{\text{input}}$ | 64 |
| $n_{\text{latents}}$ | 64 |
| TopK $k$ | 5 |
| Epochs | 50 |
| Learning rate | $10^{-4}$ |
| Batch size | 256 |

### Data parameters

| Parameter | Value |
|-----------|-------|
| Features ($k$) | 10 |
| Ambient dimension ($d$) | 64 |
| Sequence length ($T$) | 128 |
| Sequences ($n_{\text{seq}}$) | 200 |
| Magnitude | $|N(0,1)|$ (half-normal) |
| Seed | 42 |

## Expected Results

1. **Validation**: empirical sparsity matches $\mu = 0.05$ and empirical autocorrelation
   matches $\rho^{\tau} \cdot \gamma$ for all configurations
2. **MC recovery**: sub-experiment B should match existing pipeline output exactly
3. **i.i.d. baseline**: $p_A = p_B$ produces zero autocorrelation regardless of $\lambda$
4. **SAE training**: feature AUC > 0.8 for the MC case with moderate $\lambda$; SAE
   performance should not degrade as amplitude decreases (the reconstruction task is the
   same, only temporal info changes)

## Code Structure

### Experiment modules

- `src/v5_hmm_sae_baseline/__init__.py`
- `src/v5_hmm_sae_baseline/hmm_data.py` -- HMM data generation
- `src/v5_hmm_sae_baseline/sae.py` -- TopK SAE model
- `src/v5_hmm_sae_baseline/metrics.py` -- evaluation metrics
- `src/v5_hmm_sae_baseline/train.py` -- training loop
- `src/v5_hmm_sae_baseline/run_experiment.py` -- main experiment script
- `src/v5_hmm_sae_baseline/plot_results.py` -- plotting

### Reused modules

- `src/data_generation/configs.py` -- TransitionConfig, FeatureConfig, etc.
- `src/data_generation/transition.py` -- build_transition_matrix, stationary_distribution
- `src/shared/temporal_support.py` -- generate_support_markov
- `src/data_generation/magnitudes.py` -- sample_magnitudes
- `src/shared/orthogonalize.py` -- orthogonalize
- `src/utils/logging.py` -- log()
- `src/utils/seed.py` -- set_seed()

### Output

Results saved to `results/v5_hmm_sae_baseline/`:

- `results.json` -- all metrics indexed by $(\lambda, p_A, p_B)$
- `validation_autocorrelation.png` -- empirical vs theoretical autocorrelation
- `tradeoff_curves.png` -- reconstruction loss and feature AUC vs $\lambda$
- `mc_sanity_check.png` -- sub-experiment B validation
