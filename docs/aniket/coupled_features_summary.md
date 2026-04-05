---
author: Aniket
date: 2026-04-04
tags:
  - results
  - complete
---

## Coupled Features & Leaky Reset: Implementation Summary

Two HMM extensions implemented in the data generation pipeline to test whether
TXCDRv2's temporal structure recovery is robust beyond the simple independent-feature
reset process.

## What Was Implemented

### Extension 1: Leaky Reset

**Files changed:**
- `src/data_generation/configs.py` -- added `TransitionConfig.from_leaky_reset(lam, p, delta)`
- `src/data_generation/transition.py` -- added `build_leaky_transition_matrix()`

**API:**

```python
cfg = DataGenerationConfig(
    transition=TransitionConfig.from_leaky_reset(lam=0.5, p=0.05, delta=0.5),
    ...
)
```

No new generation code needed -- the existing Markov chain sampler handles any 2x2 matrix.
Verified: delta=0 reproduces standard reset, stationary distribution is p for all delta,
autocorrelation increases monotonically with delta.

### Extension 2: Coupled Features

**New files:**
- `src/data_generation/coupling.py` -- coupling matrix generation and application
- `src/data_generation/coupled_dataset.py` -- `generate_coupled_dataset()` pipeline

**New config classes:**
- `CouplingConfig` -- K_hidden, M_emission, n_parents, emission_mode
- `CoupledDataGenerationConfig` -- top-level config for coupled generation

**API:**

```python
from src.data_generation import CoupledDataGenerationConfig, CouplingConfig, generate_coupled_dataset

cfg = CoupledDataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),
    coupling=CouplingConfig(K_hidden=10, M_emission=20, n_parents=2),
    sequence=SequenceConfig(T=64, n_sequences=100),
    hidden_dim=64,
)
result = generate_coupled_dataset(cfg)
```

**Output includes:**
- `emission_features` (M, d) -- local ground truth
- `hidden_features` (K, d) -- global ground truth (normalized mean of controlled emissions)
- `coupling_matrix` (M, K) -- binary mapping
- Both `hidden_states` (n_seq, K, T) and `support` (n_seq, M, T)

**Coupling modes:**
- `"or"`: deterministic OR gate -- emission fires if any parent hidden state is on
- `"sigmoid"`: soft coupling via sigmoid(alpha * parent_sum + beta)

## Smoke Test Results

All 6 smoke tests pass (run with `uv run python src/data_generation/test.py`):

1. Standard MC dataset (existing)
2. Custom transition matrix (existing)
3. HMM stochastic emissions (existing)
4. **Leaky reset**: delta=0 reproduces standard; stationary dist preserved; autocorrelation increases with delta
5. **Coupled features (OR)**: shapes correct, x = activations.T @ features, coupling matrix has exactly n_parents per row, OR gate verified
6. **Coupled features (diagonal)**: K=M with n_parents=1 gives one parent per emission

## What's Next (For Han's Experiments)

The coupled features pipeline is ready for sweep experiments. Suggested setup:

- Default: K=10 hidden, M=20 emissions, n_parents=2
- More entangled: K=10, M=30, n_parents=3
- Sweep rho in [0.0, 0.6, 0.9] and k values to find phase transition
- Compute TWO AUC scores: local (vs emission_features) and global (vs hidden_features)

The phase transition should now be between M-feature local recovery at low k and
K-feature global recovery at high k -- a much cleaner demonstration than the current
setup where local and global features are nearly the same.

## Integration with src/bench/

The `src/bench/data.py` pipeline currently uses the simple independent-feature Markov
chain. To run coupled-feature sweeps, extend `DataConfig` with an optional
`CouplingConfig` and modify `build_data_pipeline()` to call `generate_coupled_dataset()`
when coupling is specified. The evaluation module already supports computing AUC against
arbitrary feature directions -- just pass `emission_features` or `hidden_features`
as the ground truth.
