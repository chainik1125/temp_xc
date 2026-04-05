## Data Generation Pipeline

Synthetic data generation for temporal crosscoder experiments. Generates activation
vectors $\mathbf{x}_t = \sum_i a_{i,t} \mathbf{f}_i$ with temporally correlated
binary support governed by a two-state Hidden Markov Model.

## How to Use

### I just want data like we've always had (MC, deterministic emissions)

Nothing changed. The defaults recover the original behavior:

```python
from src.data_generation import DataGenerationConfig, generate_dataset
from src.data_generation.configs import TransitionConfig, FeatureConfig, SequenceConfig

cfg = DataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.05),
    features=FeatureConfig(k=10, d=64),
    sequence=SequenceConfig(T=128, n_sequences=100),
)
result = generate_dataset(cfg)
# result["support"] == result["hidden_states"] (deterministic emission)
```

### I want HMM data with stochastic emissions

The key change: the `p` in `TransitionConfig.from_reset_process(lam, p)` now controls
the hidden chain's stationary probability (renamed conceptually to $q$), and you add
an `EmissionConfig` to control stochastic emissions. The **marginal firing probability**
is $\mu = (1-q) p_A + q \cdot p_B$, NOT $q$ itself.

To maintain a target sparsity $\mu = 0.05$ while varying the amplitude $\gamma$:

```python
from src.data_generation import DataGenerationConfig, EmissionConfig, generate_dataset
from src.data_generation.configs import TransitionConfig, FeatureConfig, SequenceConfig

# Example: gamma ~ 0.47, mu = 0.05
# With p_A=0, p_B=0.5, q=0.1: mu = 0.9*0 + 0.1*0.5 = 0.05
cfg = DataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),  # p = q here
    emission=EmissionConfig(p_A=0.0, p_B=0.5),
    features=FeatureConfig(k=10, d=64),
    sequence=SequenceConfig(T=128, n_sequences=100),
)
result = generate_dataset(cfg)
# result["hidden_states"] != result["support"] (stochastic emission)
```

**Recipe for choosing $(q, p_A, p_B)$ to hit a target $(\mu, \gamma)$:**

1. Fix $p_A = 0$ (simplest case -- state A never fires).
2. Pick $p_B$ and solve $q = \mu / p_B$.
3. The resulting $\gamma = q(1-q) p_B^2 / [\mu(1-\mu)]$.
4. Larger $p_B$ (with smaller $q$) gives larger $\gamma$.

| Target | $p_A$ | $p_B$ | $q$ | $\gamma$ |
|--------|-------|-------|-----|----------|
| MC case | 0.0 | 1.0 | 0.05 | 1.000 |
| High amplitude | 0.0 | 0.5 | 0.1 | 0.474 |
| Medium amplitude | 0.0 | 0.25 | 0.2 | 0.211 |
| Low amplitude | 0.0 | 0.1 | 0.5 | 0.053 |
| No temporal info | 0.05 | 0.05 | 0.5 | 0.000 |

All rows above give $\mu = 0.05$.

### I want per-feature heterogeneity (different $\rho$ per feature)

Use the per-feature support generator directly. This follows Han's $(\pi, \rho)$
convention where each feature has its own stationary probability and autocorrelation:

```python
import torch
from src.shared.temporal_support import (
    generate_support_per_feature,
    per_feature_from_pi_rho,
)

# 10 features: 5 i.i.d., 5 persistent
pi = torch.tensor([0.05] * 10)                              # all same sparsity
rho = torch.tensor([0.0]*5 + [0.9]*5)                       # first 5 i.i.d., last 5 sticky
alpha, beta = per_feature_from_pi_rho(pi, rho)               # convert to transition probs

rng = torch.Generator().manual_seed(42)
support = generate_support_per_feature(10, 128, alpha, beta, pi, rng)  # (10, 128)
```

This is NOT integrated into `generate_dataset` yet -- you'd need to build the full
pipeline (magnitudes, features, x vectors) manually. The per-feature generator
produces the support tensor; the rest of the pipeline works the same.

### I want to use Andre's temporal_crosscoders/ sweep with HMM data

`generate_sequences` in `temporal_crosscoders/data.py` now accepts `p_A` and `p_B`:

```python
from data import generate_sequences

# MC case (default, same as before)
x_mc = generate_sequences(num_seqs=100, T=64, rho=0.9)

# HMM case -- pass p_stat=q (NOT mu)
x_hmm = generate_sequences(num_seqs=100, T=64, rho=0.9,
                            p_stat=0.1, p_A=0.0, p_B=0.5)
```

`CachedDataSource` also accepts `p_A`/`p_B`:

```python
cache = CachedDataSource(rho=0.9, toy_model=model, p_A=0.0, p_B=0.5)
```

Pre-configured emission sweeps are in `config.py` under `SWEEP_EMISSION`.

### I want to validate my data generation

Use the theoretical helpers to check empirical statistics against theory:

```python
from src.data_generation.transition import (
    hmm_marginal_sparsity,
    hmm_autocorrelation_amplitude,
    hmm_theoretical_autocorrelation,
)

P = cfg.transition.matrix
mu = hmm_marginal_sparsity(P, p_A=0.0, p_B=0.5)           # expected marginal sparsity
gamma = hmm_autocorrelation_amplitude(P, p_A=0.0, p_B=0.5) # amplitude prefactor
theory = hmm_theoretical_autocorrelation(P, 0.0, 0.5, max_lag=30)  # full curve

# Compare against empirical
empirical_mu = result["support"].mean().item()
assert abs(empirical_mu - mu) < 0.01
```

For autocorrelation validation, use the **pooled** estimator (correct at all $\lambda$
including $\lambda = 0$) rather than the per-chain estimator (which underestimates at
$\lambda = 0$ due to conditioning on the frozen hidden state):

```python
from src.v5_hmm_sae_baseline.metrics import (
    compute_empirical_autocorrelation,   # per-chain (use for lambda > 0)
    compute_pooled_autocorrelation,      # pooled (use for all lambda)
)
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

3. **Magnitude**: $m_{i,t} \sim |N(\mu_m, \sigma_m^2)|$ (half-normal by default)

4. **Activation**: $a_{i,t} = s_{i,t} \cdot m_{i,t}$

5. **Observation**: $\mathbf{x}_t = \sum_{i=1}^{k} a_{i,t} \mathbf{f}_i$

### Observed autocorrelation

$$\text{Corr}(s_t, s_{t+\tau}) = \underbrace{(1-\lambda)^{|\tau|}}_{\text{decay rate}} \cdot \underbrace{\gamma}_{\text{amplitude}}$$

where $\gamma = \pi_A \pi_B (p_B - p_A)^2 / [\mu(1-\mu)]$ is the amplitude prefactor.
With $p_A = 0, p_B = 1$: $\gamma = 1$, recovering the MC case.

### Parameterization equivalences

Our pipeline uses $(\lambda, q)$ via the reset process. Han uses $(\pi, \rho)$. Andre
uses $(\text{rho}, \text{p\_stat})$. They are all the same Markov chain:

| Our notation | Han's notation | Andre's notation | Meaning |
|--------------|---------------|-----------------|---------|
| $\lambda$ | $1 - \rho$ | $1 - \text{rho}$ | Mixing rate (0=memory, 1=i.i.d.) |
| $q$ | $\pi$ | `p_stat` | Stationary probability of state B |
| $\alpha$ | $\pi + \rho(1-\pi)$ | `alpha` | P(on -> on) |
| $\beta$ | $\pi(1-\rho)$ | `beta` | P(off -> on) |

### Reset process

$$T(\lambda) = (1 - \lambda) I + \lambda R_S$$

where $R_S$ has every row equal to $[1-q, q]$.
- $\lambda = 0$: perfect memory (identity matrix)
- $\lambda = 1$: i.i.d. Bernoulli($q$)

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

### I want leaky reset (softer transition boundaries)

Instead of the binary persist-or-resample, the leaky reset biases "reset" events
toward the current state. Parameter $\delta \in [0,1]$ controls the leak:

```python
cfg = DataGenerationConfig(
    transition=TransitionConfig.from_leaky_reset(lam=0.5, p=0.05, delta=0.5),
    features=FeatureConfig(k=10, d=64),
    sequence=SequenceConfig(T=128, n_sequences=100),
)
result = generate_dataset(cfg)
```

$\delta = 0$ recovers the standard reset. Higher $\delta$ = stickier states.
Stationary distribution stays at $p$ for all $\delta$.

### I want coupled features (many-to-many hidden-to-emission mapping)

K hidden states drive M > K emission features through a coupling matrix. This
creates a genuine separation between local (emission-level) and global
(hidden-state-level) structure.

```python
from src.data_generation import (
    CoupledDataGenerationConfig,
    CouplingConfig,
    generate_coupled_dataset,
)
from src.data_generation.configs import TransitionConfig, SequenceConfig

cfg = CoupledDataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),
    coupling=CouplingConfig(
        K_hidden=10,       # global latent dimensionality
        M_emission=20,     # local emission dimensionality
        n_parents=2,       # parents per emission
        emission_mode="or",  # OR gate: fires if any parent is on
    ),
    sequence=SequenceConfig(T=64, n_sequences=100),
    hidden_dim=64,
)
result = generate_coupled_dataset(cfg)

# Two sets of ground truth:
result["emission_features"]  # (20, 64) — local ground truth
result["hidden_features"]    # (10, 64) — global ground truth
result["coupling_matrix"]    # (20, 10) — which hidden states control which emissions
```

For evaluation, compute AUC against both `emission_features` (local) and
`hidden_features` (global) separately. The phase transition hypothesis:
at low k, models recover emission features; at high k, TXCDRv2 transitions
to recovering hidden-state features.

## Configuration

### `TransitionConfig`

- `matrix`: 2x2 row-stochastic torch tensor
- `stationary_on_prob`: probability of being in state B (ON) at stationarity
- `from_reset_process(lam, p)`: convenience constructor
- `from_leaky_reset(lam, p, delta)`: leaky reset constructor

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

### `CouplingConfig`

- `K_hidden`: number of hidden states (default 10)
- `M_emission`: number of emission features (default 20)
- `n_parents`: parent hidden states per emission (default 2)
- `emission_mode`: `"or"` (deterministic) or `"sigmoid"` (soft)
- `sigmoid_alpha`: sharpness for sigmoid mode (default 5.0)
- `sigmoid_beta`: bias for sigmoid mode (default -2.0)

### `CoupledDataGenerationConfig`

- `transition`: TransitionConfig for the K hidden chains
- `coupling`: CouplingConfig
- `magnitude`: MagnitudeConfig
- `sequence`: SequenceConfig
- `hidden_dim`: observation space dimension d (default 64)
- `target_cos_sim`: pairwise cosine sim for emission features (default 0.0)
- `seed`: random seed (default 42)
