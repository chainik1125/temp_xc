---
author: Aniket
date: 2026-04-04
tags:
  - design
  - in-progress
---

## Richer HMM Extensions: Leaky Reset & Coupled Features

Motivated by Dmitry's concern (April 4 meeting) that the current generative process
is too simple -- local and global features are nearly redundant because each emission
depends on exactly one hidden state. Two extensions of increasing complexity.

## Extension 1: Leaky Reset

### Math

Replace the binary persist-or-resample transition with a leak parameter delta in [0,1].

**Current (delta=0):** With probability lambda, resample h_i(t) ~ Bernoulli(mu).
With probability 1-lambda, persist h_i(t) = h_i(t-1).

**Leaky (delta>0):** With probability lambda, resample
h_i(t) ~ Bernoulli((1-delta)*mu + delta*h_i(t-1)). With probability 1-lambda, persist.

The full transition matrix:

```text
P(h=1 | h=1) = (1-lambda) + lambda*((1-delta)*mu + delta) = 1 - lambda*(1-delta)*(1-mu)
P(h=1 | h=0) = lambda*(1-delta)*mu
```

Stationary distribution remains mu. Effective autocorrelation: rho_eff = 1 - lambda*(1-delta).

When delta=0 reduces to standard reset. When delta=1, chain is absorbing.

### Implementation

Just a new constructor: `TransitionConfig.from_leaky_reset(lam, p, delta)`.
No new generation code -- the existing Markov chain sampler handles any valid 2x2 matrix.

## Extension 2: Coupled Features

### Math

K hidden states h_i(t) in {0,1}, each an independent 2-state Markov chain (as before).
M emission features (M > K), connected via a coupling matrix C in {0,1}^{M x K}.

**Hidden dynamics:** Same pipeline. K independent chains.

**Coupled emission (OR gate):**

```text
s_j(t) = 1[ sum_i C_{ji} * h_i(t) >= 1 ]
```

Emission j fires if ANY parent hidden state is on.

**Coupled emission (sigmoid, for soft version):**

```text
s_j(t) ~ Bernoulli(sigmoid(alpha * (sum_i C_{ji} * h_i(t)) + beta_j))
```

**Observation:**

```text
x(t) = sum_{j=1}^{M} s_j(t) * m_j(t) * f_j
```

**Key property:** Observations live in span of M emission features {f_j}, but true
latent dimensionality is only K. Local feature recovery = recovering M emission dirs.
Global feature recovery = recovering K hidden-state-level directions.

### Coupling matrix design

Random bipartite graph:
- Each emission j has exactly `n_parents` parent hidden states (default 2)
- Each hidden state i drives approximately M*n_parents/K emissions
- Assignment is random but balanced

### Hidden feature directions

For each hidden state i, the "global" feature direction is the normalized mean of
the emission directions it controls:

```text
hidden_feature_i = normalize(sum_{j: C_{ji}=1} f_j)
```

### Output dictionary

```python
{
    "emission_features": (M, d),      # local ground truth
    "hidden_features": (K, d),        # global ground truth
    "hidden_states": (n_seq, K, T),   # K binary hidden chains
    "support": (n_seq, M, T),         # M binary emission support
    "coupling_matrix": (M, K),        # the C matrix
    "magnitudes": (n_seq, M, T),
    "activations": (n_seq, M, T),
    "x": (n_seq, T, d),
    "config": CoupledDataGenerationConfig,
}
```

## Evaluation Plan (document only -- not implementing yet)

When evaluating on coupled-feature data, compute TWO AUC scores:

1. **Local AUC**: decoder directions vs emission features (M directions)
2. **Global AUC**: decoder directions vs hidden features (K directions)

Phase transition hypothesis: at low k, models recover emission features (high local
AUC). At high k, TXCDRv2 transitions to recovering hidden-state features (global
AUC rises). This is the "figure 2" result for the paper.

## Suggested Sweep Parameters

### Leaky reset sweep

- delta in [0.0, 0.25, 0.5, 0.75] with fixed lam=0.5, p=0.05
- Compare TXCDRv2 vs baselines as delta increases

### Coupled features sweep

- K=10 hidden, M=20 emissions, n_parents=2 (default)
- Also try K=10, M=30, n_parents=3 (more entangled)
- rho in [0.0, 0.6, 0.9] as before
- k sweep as before to find phase transition
