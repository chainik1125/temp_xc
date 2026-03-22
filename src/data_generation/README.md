## Data Generation Pipeline

Synthetic data generation for temporal crosscoder experiments. Generates activation
vectors $\mathbf{x}_t = \sum_i a_{i,t} \mathbf{f}_i$ with temporally correlated
binary support governed by a two-state Hidden Markov Model.

## Quick Start

```python
from src.data_generation import DataGenerationConfig, EmissionConfig, generate_dataset
from src.data_generation.configs import (
    TransitionConfig, MagnitudeConfig, FeatureConfig, SequenceConfig,
)

# MC case (deterministic emissions, original behavior)
cfg = DataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.05),
    features=FeatureConfig(k=10, d=64),
    sequence=SequenceConfig(T=128, n_sequences=100),
)
result = generate_dataset(cfg)

# HMM case (stochastic emissions)
cfg_hmm = DataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),
    emission=EmissionConfig(p_A=0.0, p_B=0.5),  # mu = 0.9*0 + 0.1*0.5 = 0.05
    features=FeatureConfig(k=10, d=64),
    sequence=SequenceConfig(T=128, n_sequences=100),
)
result_hmm = generate_dataset(cfg_hmm)
```

## Mathematical Formulation

Per feature $i$ at position $t$:

1. **Hidden state**: $z_{i,t} \in \{A, B\}$ from a two-state Markov chain with
   transition matrix $P$, where $P[\text{from}, \text{to}]$:
   - $P[0, 1] = \beta$ (A -> B, off -> on)
   - $P[1, 1] = \alpha$ (B -> B, on -> on)

2. **Emission** (HMM step): $s_{i,t} \sim \text{Bernoulli}(p_{z_{i,t}})$
   - In state A: $s \sim \text{Bernoulli}(p_A)$
   - In state B: $s \sim \text{Bernoulli}(p_B)$
   - With defaults $p_A = 0, p_B = 1$: observation = hidden state (MC case)

3. **Magnitude**: $m_{i,t} \sim |N(\mu, \sigma^2)|$ (half-normal by default)

4. **Activation**: $a_{i,t} = s_{i,t} \cdot m_{i,t}$

5. **Observation**: $\mathbf{x}_t = \sum_{i=1}^{k} a_{i,t} \mathbf{f}_i$

### Observed autocorrelation

$$\text{Corr}(s_t, s_{t+\tau}) = \underbrace{(1-\lambda)^{|\tau|}}_{\text{decay rate}} \cdot \underbrace{\gamma}_{\text{amplitude}}$$

where $\gamma = \pi_A \pi_B (p_B - p_A)^2 / [\mu(1-\mu)]$ is the amplitude prefactor.
With $p_A = 0, p_B = 1$: $\gamma = 1$, recovering the MC case.

### Reset Process (convenience constructor)

$$T(\lambda) = (1 - \lambda) I + \lambda R_S$$

where $R_S$ has every row equal to $[1-p, p]$.
- $\lambda = 0$: perfect memory (identity matrix)
- $\lambda = 1$: i.i.d. Bernoulli($p$)

## Output Dictionary

`generate_dataset(config)` returns:

| Key | Shape | Description |
|-----|-------|-------------|
| `features` | `(k, d)` | Ground-truth feature directions (unit norm) |
| `hidden_states` | `(n_seq, k, T)` | Hidden Markov chain states |
| `support` | `(n_seq, k, T)` | Observed (emitted) binary support |
| `magnitudes` | `(n_seq, k, T)` | Magnitude values |
| `activations` | `(n_seq, k, T)` | $a_{i,t} = s_{i,t} \cdot m_{i,t}$ |
| `x` | `(n_seq, T, d)` | Activation vectors (what the SAE sees) |
| `config` | `DataGenerationConfig` | Config used for generation |

## Configuration

### `TransitionConfig`

- `matrix`: 2x2 row-stochastic torch tensor
- `stationary_on_prob`: probability of being in state B (ON) at stationarity
- `from_reset_process(lam, p)`: convenience constructor

### `EmissionConfig`

- `p_A`: emission probability in state A (default 0.0)
- `p_B`: emission probability in state B (default 1.0)

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
