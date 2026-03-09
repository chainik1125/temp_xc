## Data Generation Pipeline

Synthetic data generation for temporal crosscoder experiments. Generates activation
vectors $\mathbf{x}_t = \sum_i a_{i,t} \mathbf{f}_i$ with temporally correlated
binary support governed by a two-state Markov chain.

## Quick Start

```python
import torch
from src.data_generation import DataGenerationConfig, generate_dataset
from src.data_generation.configs import (
    TransitionConfig, MagnitudeConfig, FeatureConfig, SequenceConfig,
)

# Using the reset process (default)
cfg = DataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.05),
    features=FeatureConfig(k=10, d=64),
    sequence=SequenceConfig(T=128, n_sequences=100),
)
result = generate_dataset(cfg)

# Using a custom transition matrix
P = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
from src.data_generation.transition import stationary_distribution
pi = stationary_distribution(P)

cfg = DataGenerationConfig(
    transition=TransitionConfig(matrix=P, stationary_on_prob=pi[1].item()),
    features=FeatureConfig(k=10, d=64),
    sequence=SequenceConfig(T=128, n_sequences=100),
)
result = generate_dataset(cfg)
```

## Mathematical Formulation

Per feature $i$ at position $t$:

1. **Support**: $s_{i,t} \in \{0, 1\}$ from a two-state Markov chain with transition
   matrix $P$, where $P[\text{from}, \text{to}]$:
   - $P[0, 1] = \beta$ (off -> on)
   - $P[1, 1] = \alpha$ (on -> on)

2. **Magnitude**: $m_{i,t} \sim |N(\mu, \sigma^2)|$ (half-normal by default)

3. **Activation**: $a_{i,t} = s_{i,t} \cdot m_{i,t}$

4. **Observation**: $\mathbf{x}_t = \sum_{i=1}^{k} a_{i,t} \mathbf{f}_i$

### Reset Process (convenience constructor)

$$T(\lambda) = (1 - \lambda) I + \lambda R_S$$

where $R_S$ has every row equal to $[1-p, p]$.
- $\lambda = 0$: perfect memory (identity matrix)
- $\lambda = 1$: i.i.d. Bernoulli($p$)
- Autocorrelation: $(1 - \lambda)^{|\tau|}$

## Output Dictionary

`generate_dataset(config)` returns:

| Key | Shape | Description |
|-----|-------|-------------|
| `features` | `(k, d)` | Ground-truth feature directions (unit norm) |
| `support` | `(n_seq, k, T)` | Binary support variables |
| `magnitudes` | `(n_seq, k, T)` | Magnitude values |
| `activations` | `(n_seq, k, T)` | $a_{i,t} = s_{i,t} \cdot m_{i,t}$ |
| `x` | `(n_seq, T, d)` | Activation vectors (what the SAE sees) |
| `config` | `DataGenerationConfig` | Config used for generation |

## Configuration

### `TransitionConfig`

- `matrix`: 2x2 row-stochastic torch tensor
- `stationary_on_prob`: probability of being ON in the stationary distribution
- `from_reset_process(lam, p)`: convenience constructor

### `MagnitudeConfig`

- `distribution`: `"half_normal"` (default)
- `mu`: mean parameter (default 0.0)
- `sigma`: scale parameter (default 1.0)

### `FeatureConfig`

- `k`: number of features (default 10)
- `d`: ambient dimension (default 64)
- `orthogonal`: whether to orthogonalize features (default True)
- `target_cos_sim`: target pairwise cosine similarity (default 0.0)

### `SequenceConfig`

- `T`: sequence length (default 128)
- `n_sequences`: number of sequences (default 1)
