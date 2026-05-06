---
author: aniket
date: 2026-03-07
tags:
  - proposal
  - in-progress
---

## Motivation

We need a modular, reusable data generation pipeline for temporal crosscoder experiments.
The pipeline generates activation vectors $\mathbf{x}_t = \sum_i a_{i,t} \mathbf{f}_i$ where
support variables $s_{i,t} \in \{0, 1\}$ follow a two-state Markov chain and magnitudes
$m_{i,t}$ are sampled i.i.d. The pipeline must accept **any** user-provided 2x2 transition
matrix, with Dmitry's reset process as a convenience constructor.

The reset process validation ([[reset_process_validation/summary|v4 summary]]) confirmed
our Markov chain implementation is correct. This pipeline builds on that validated code.

## Background

The data-generating process per feature $i$ at position $t$:

1. Run a two-state Markov chain: $s_{i,t} \in \{0, 1\}$
2. Sample magnitude: $m_{i,t} \sim |N(0, 1)|$ (half-normal)
3. Combine: $a_{i,t} = s_{i,t} \cdot m_{i,t}$
4. Sum over features: $\mathbf{x}_t = \sum_{i=1}^{k} a_{i,t} \mathbf{f}_i$

Features are independent of each other at each time step.

### Reset Process (default parameterization)

$$T(\lambda) = (1 - \lambda) I + \lambda R_S$$

- $\alpha = 1 - \lambda(1-p)$ (stay-on probability)
- $\beta = \lambda p$ (turn-on probability)
- Stationary probability: $p$
- Autocorrelation: $(1 - \lambda)^{|\tau|}$

## Pipeline Architecture

```text
src/data_generation/
  __init__.py
  configs.py          # Dataclass configs
  transition.py       # Transition matrix construction and validation
  support.py          # Support sequence generation (wraps shared code)
  magnitudes.py       # Magnitude sampling
  activations.py      # support x magnitudes -> activation coefficients
  dataset.py          # Full pipeline entry point
  test.py             # Smoke test script
  README.md           # Module documentation
```

## Module Specifications

### `configs.py`

- `TransitionConfig`: Holds either a raw 2x2 transition matrix or reset process
  parameters ($\lambda$, $p$). Validates rows sum to 1, entries in $[0, 1]$. Includes
  `from_reset_process(lam, p)` class method.
- `MagnitudeConfig`: Distribution type (`half_normal`), parameters ($\mu$, $\sigma$).
- `FeatureConfig`: $k$ (num features), $d$ (ambient dimension), `orthogonal` (bool).
- `SequenceConfig`: $T$ (sequence length), `n_sequences` (number of sequences).
- `DataGenerationConfig`: Combines all of the above. Every feature shares the same
  transition matrix.

### `transition.py`

- `build_transition_matrix(lam, p)` -> 2x2 tensor
- `validate_transition_matrix(P)`: checks rows sum to 1, entries in $[0, 1]$
- `stationary_distribution(P)` -> tensor
- `theoretical_autocorrelation(P, max_lag)` -> tensor
- `expected_holding_times(P)` -> dict

### `support.py`

- Wraps `src/shared/temporal_support.py` for reset process generation
- Adds `generate_support_markov(k, T, transition_matrix, stationary_prob, rng)` for
  arbitrary transition matrices
- Extends the shared module with the general Markov chain function

### `magnitudes.py`

- `sample_magnitudes(k, T, config, rng)` -> (k, T) tensor
- Supports half-normal $|N(\mu, \sigma^2)|$

### `activations.py`

- `generate_activations(support, magnitudes)` -> (k, T) tensor of $a_{i,t} = s_{i,t} \cdot m_{i,t}$

### `dataset.py`

- `generate_dataset(config, rng)` -> dict with keys: `features`, `support`, `magnitudes`,
  `activations`, `x`, `config`
- Single entry point for downstream SAE training code

## What to Reuse vs Write New

### Reused directly

- `src/shared/temporal_support.py` -- `generate_support_reset` for reset process
- `src/shared/orthogonalize.py` -- `orthogonalize` for feature direction generation
- `src/utils/seed.py` -- `set_seed` for reproducibility
- `src/utils/logging.py` -- `log` for progress messages

### Extended

- `src/shared/temporal_support.py` -- add `generate_support_markov` for arbitrary
  transition matrices

### New code

- `src/data_generation/configs.py` -- pipeline-specific config dataclasses
- `src/data_generation/transition.py` -- transition matrix utilities
- `src/data_generation/support.py` -- thin wrapper around shared module
- `src/data_generation/magnitudes.py` -- magnitude sampling
- `src/data_generation/activations.py` -- support x magnitude combination
- `src/data_generation/dataset.py` -- full pipeline orchestration
- `tests/test_data_generation.py` -- unit tests

## Testing

- `from_reset_process(lam=1.0, p=0.05)` gives same results as i.i.d. Bernoulli
- `from_reset_process(lam=0.0, p)` gives perfect memory
- Arbitrary valid transition matrices accepted
- Invalid matrices raise errors
- Output shapes correct for all pipeline outputs
- Stationary probability empirically correct (within tolerance)
- Full `generate_dataset` output has consistent shapes
- $x$ vectors are correct linear combinations of features

## Expected Output

The `generate_dataset` function returns a dict:

```python
{
    "features":     # (k, d) ground-truth feature directions
    "support":      # (n_sequences, k, T) binary support
    "magnitudes":   # (n_sequences, k, T) magnitude values
    "activations":  # (n_sequences, k, T) activation coefficients
    "x":            # (n_sequences, T, d) activation vectors
    "config":       # the DataGenerationConfig used
}
```
