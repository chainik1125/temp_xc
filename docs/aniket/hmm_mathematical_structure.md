---
author: Aniket
date: 2026-04-04
tags:
  - reference
  - complete
---

## HMM Data Generation: Mathematical Structure

This document specifies the complete mathematical structure of the synthetic data
generation pipeline, covering all three modes: standard (independent features),
leaky reset, and coupled features. It serves as a self-contained reference for the
team, connecting the theory to the code in `src/data_generation/`.

## Level 1: Standard Independent Features

The simplest case. $K$ features, each with an independent binary hidden state.

### Hidden state dynamics

For each feature $i \in \{1, \ldots, K\}$, the hidden state $h_i(t) \in \{0, 1\}$
follows a two-state Markov chain with transition matrix:

$$P = \begin{pmatrix} 1 - \beta & \beta \\ 1 - \alpha & \alpha \end{pmatrix}$$

where rows are "from" states (0 = off, 1 = on) and columns are "to" states.

### Reset process parameterization

The transition matrix is constructed from the **reset process**:

$$P(\lambda) = (1 - \lambda) I + \lambda R_S$$

where $R_S$ has every row equal to $[1-\mu, \mu]$.

This gives:

$$\alpha = 1 - \lambda(1 - \mu), \quad \beta = \lambda \mu$$

- $\lambda \in [0, 1]$: mixing rate. $\lambda = 0$ is perfect memory (identity),
  $\lambda = 1$ is i.i.d. Bernoulli($\mu$).
- $\mu$: stationary probability of the ON state.
- Second eigenvalue (lag-1 autocorrelation): $\rho = \alpha - \beta = 1 - \lambda$.

**Code:** `TransitionConfig.from_reset_process(lam, p)` in `configs.py`,
`build_transition_matrix(lam, p)` in `transition.py`.

### Stochastic emission (HMM layer)

The observed support $s_i(t)$ is sampled from the hidden state via:

$$s_i(t) \mid h_i(t) = 0 \sim \text{Bernoulli}(p_A)$$
$$s_i(t) \mid h_i(t) = 1 \sim \text{Bernoulli}(p_B)$$

With defaults $p_A = 0$, $p_B = 1$, the emission is deterministic: $s_i(t) = h_i(t)$.

**Marginal firing probability:**

$$\mu_{\text{obs}} = (1 - \mu) p_A + \mu \, p_B$$

**Observed autocorrelation:**

$$\text{Corr}(s_i(t), s_i(t+\tau)) = \gamma \cdot \rho^{|\tau|}$$

where $\gamma$ is the **amplitude prefactor**:

$$\gamma = \frac{\mu(1-\mu)(p_B - p_A)^2}{\mu_{\text{obs}}(1 - \mu_{\text{obs}})}$$

With $p_A = 0$, $p_B = 1$: $\gamma = 1$ (full temporal information).
With $p_A = p_B$: $\gamma = 0$ (no temporal information in observations).

**Code:** `EmissionConfig(p_A, p_B)` in `configs.py`, `apply_emission()` in `support.py`.

### Observation model

$$a_i(t) = s_i(t) \cdot m_i(t), \quad m_i(t) \sim |N(0, \sigma^2)|$$

$$\mathbf{x}(t) = \sum_{i=1}^{K} a_i(t) \, \mathbf{f}_i$$

where $\{\mathbf{f}_i\}_{i=1}^K \subset \mathbb{R}^d$ are orthogonalized feature
directions (unit norm, near-zero pairwise cosine similarity).

### Parameterization equivalences

Our pipeline uses $(\lambda, q)$. Han uses $(\pi, \rho)$. Andre uses
`(rho, p_stat)`. All describe the same 2-state Markov chain:

| Our notation | Han's notation | Andre's notation | Meaning |
|---|---|---|---|
| $\lambda$ | $1 - \rho$ | $1 - \text{rho}$ | Mixing rate |
| $q$ (= $\mu$) | $\pi$ | `p_stat` | Stationary ON probability |
| $\alpha$ | $\pi + \rho(1-\pi)$ | `alpha` | $P(1 \to 1)$ |
| $\beta$ | $\pi(1-\rho)$ | `beta` | $P(0 \to 1)$ |

## Level 2: Leaky Reset

Replaces the binary persist-or-resample with a **leak parameter** $\delta \in [0, 1)$
that biases reset events toward the current state.

### Transition matrix

With probability $\lambda$, resample $h_i(t) \sim \text{Bernoulli}((1-\delta)\mu + \delta \cdot h_i(t-1))$.
With probability $1 - \lambda$, persist $h_i(t) = h_i(t-1)$.

Writing out the full transition probabilities:

$$P(h=1 \mid h=1) = (1-\lambda) + \lambda[(1-\delta)\mu + \delta] = 1 - \lambda(1-\delta)(1-\mu)$$

$$P(h=1 \mid h=0) = \lambda(1-\delta)\mu$$

### Stationary distribution

**Claim:** The stationary distribution is $\pi_1 = \mu$ for all $\delta < 1$.

**Proof:** From $\pi_1 = \pi_0 \cdot P(1|0) + \pi_1 \cdot P(1|1)$ with $\pi_0 = 1 - \pi_1$:

$$\pi_1 = (1 - \pi_1) \lambda(1-\delta)\mu + \pi_1[1 - \lambda(1-\delta)(1-\mu)]$$

$$\pi_1 = \lambda(1-\delta)\mu - \pi_1\lambda(1-\delta)\mu + \pi_1 - \pi_1\lambda(1-\delta)(1-\mu)$$

$$0 = \lambda(1-\delta)\mu - \pi_1\lambda(1-\delta)[\mu + (1-\mu)]$$

$$0 = \lambda(1-\delta)\mu - \pi_1\lambda(1-\delta)$$

$$\pi_1 = \mu \quad \blacksquare$$

### Effective autocorrelation

The second eigenvalue:

$$\rho_{\text{eff}} = \alpha - \beta = [1 - \lambda(1-\delta)(1-\mu)] - [\lambda(1-\delta)\mu] = 1 - \lambda(1-\delta)$$

This is independent of $\mu$, just as in the standard case. The leak parameter
$\delta$ effectively rescales $\lambda \to \lambda(1-\delta)$, stretching temporal
correlations.

| $\delta$ | Effect | $\rho_{\text{eff}}$ at $\lambda = 0.5$ |
|---|---|---|
| 0 | Standard reset | 0.500 |
| 0.25 | Mild leak | 0.625 |
| 0.5 | Moderate leak | 0.750 |
| 0.75 | Strong leak | 0.875 |
| 1 | Absorbing (degenerate) | 1.000 |

### Limiting cases

- $\delta = 0$: recovers the standard reset process exactly.
- $\delta \to 1$: the chain becomes absorbing ($P(1|1) \to 1$, $P(1|0) \to 0$).
  The `stationary_distribution()` function returns $[0.5, 0.5]$ as a fallback
  since the chain has no ergodic stationary distribution.

**Code:** `TransitionConfig.from_leaky_reset(lam, p, delta)` in `configs.py`,
`build_leaky_transition_matrix(lam, p, delta)` in `transition.py`.

## Level 3: Coupled Features

This fundamentally changes the generative process by introducing a **many-to-many
mapping** between hidden states and observed emissions. This is the key extension
for testing whether temporal crosscoders discover latent structure rather than
just denoising per-token emissions.

### Setup

- $K$ hidden states $h_i(t) \in \{0, 1\}$, each an independent 2-state Markov
  chain (standard or leaky reset).
- $M$ emission features ($M > K$), connected via a **coupling matrix**
  $C \in \{0, 1\}^{M \times K}$.
- Each emission $j$ has exactly $n_{\text{parents}}$ parent hidden states
  ($C_{j,:}$ has exactly $n_{\text{parents}}$ ones).
- $M$ emission directions $\{\mathbf{f}_j\}_{j=1}^M \subset \mathbb{R}^d$
  (orthogonalized).

### Hidden dynamics

Same as Level 1 or 2. $K$ independent chains, each with its own transition matrix.

### Coupled emission (OR gate)

$$s_j(t) = \mathbb{1}\left[\sum_{i=1}^K C_{ji} \cdot h_i(t) \geq 1\right]$$

Emission $j$ fires if **any** parent hidden state is ON.

**Marginal emission probability** (at stationarity, with independent hidden chains
each having stationary probability $\mu$):

$$P(s_j = 1) = 1 - (1 - \mu)^{n_{\text{parents}}}$$

For $\mu = 0.1$, $n_{\text{parents}} = 2$: $P(s_j = 1) \approx 0.19$.

### Coupled emission (sigmoid, soft version)

$$s_j(t) \sim \text{Bernoulli}\left(\sigma\left(\alpha_s \sum_{i=1}^K C_{ji} h_i(t) + \beta_j\right)\right)$$

where $\sigma$ is the sigmoid function, $\alpha_s$ controls sharpness, and
$\beta_j$ controls base rate. As $\alpha_s \to \infty$ with appropriate $\beta_j$,
this recovers the OR gate.

### Observation model

$$\mathbf{x}(t) = \sum_{j=1}^{M} s_j(t) \cdot m_j(t) \cdot \mathbf{f}_j$$

where $m_j(t) \sim |N(0, \sigma^2)|$ as before.

### Emission autocorrelation (OR gate)

**Important subtlety:** The autocorrelation of emission $j$ is *not* simply
$\rho^{|\tau|}$. For the OR gate with independent parent chains:

$$P(s_j(t) = 0, s_j(t+\tau) = 0) = \prod_{i \in \text{parents}(j)} P(h_i(t) = 0, h_i(t+\tau) = 0)$$

Since each parent chain is independent, and for a 2-state chain:

$$P(h_i(t) = 0, h_i(t+\tau) = 0) = (1-\mu)^2 + \mu(1-\mu)\rho^{|\tau|}$$

This gives a more complex autocorrelation structure than the single-chain case.
The autocorrelation function of the emission decays geometrically but with a
different prefactor and possibly different effective decay rate than any individual
hidden chain.

### Ground truth feature directions

Two sets of ground truth for evaluation:

**Local (emission) features:** The $M$ emission directions $\{\mathbf{f}_j\}_{j=1}^M$.
These are what a standard SAE would recover — the directions that explain per-token
variance.

**Global (hidden) features:** For each hidden state $i$, the global direction is the
normalized mean of the emission directions it controls:

$$\mathbf{g}_i = \frac{\sum_{j: C_{ji}=1} \mathbf{f}_j}{\left\|\sum_{j: C_{ji}=1} \mathbf{f}_j\right\|}$$

When the emission features are orthogonal, the cosine similarity between hidden
features $\mathbf{g}_i$ and $\mathbf{g}_{i'}$ that share $c$ common emission
children (out of $n_i$ and $n_{i'}$ total) is:

$$\cos(\mathbf{g}_i, \mathbf{g}_{i'}) = \frac{c}{\sqrt{n_i \cdot n_{i'}}}$$

This means the hidden features are **not orthogonal** when hidden states share
emission children, making the global recovery problem genuinely harder.

### Coupling matrix design

The current implementation samples $n_{\text{parents}}$ distinct parents uniformly
at random for each emission independently. The number of children per hidden state
follows approximately $\text{Binomial}(M, n_{\text{parents}}/K)$.

For the default ($K=10$, $M=20$, $n_{\text{parents}}=2$): each hidden state controls
$\sim 4 \pm 1.9$ emissions on average. Some hidden states may get fewer children,
making their global feature direction dominated by a single emission.

**Note for paper experiments:** Consider using a balanced (round-robin) assignment
that guarantees each hidden state gets exactly $M \cdot n_{\text{parents}} / K$
children for cleaner results.

### The phase transition hypothesis

At low dictionary size $k$, models are capacity-constrained and recover the $M$
emission directions (high **local AUC**, low **global AUC**).

At high $k$, TXCDRv2's shared encoder aggregates information across positions and
discovers that certain emission features co-occur in temporally coherent patterns
(because they share hidden-state parents). It learns latents that represent the
$K$-dimensional hidden structure rather than the $M$-dimensional emission structure
(global AUC rises, local AUC may drop or plateau).

Standard SAEs and TFA should **not** exhibit this transition — they lack the
temporal aggregation mechanism needed to discover hidden causes from single-position
observations.

## Summary of code entry points

| Level | Config | Generator | Key function |
|---|---|---|---|
| Standard | `DataGenerationConfig` | `generate_dataset()` | `generate_support()` |
| Leaky reset | `TransitionConfig.from_leaky_reset()` | `generate_dataset()` | `build_leaky_transition_matrix()` |
| Coupled | `CoupledDataGenerationConfig` | `generate_coupled_dataset()` | `apply_coupling()` |

For the bench module (`src/bench/`), all three modes are accessible via `DataConfig`:

| Mode | Config | CLI |
|---|---|---|
| Standard | `DataConfig()` | `python -m src.bench.sweep` |
| Leaky reset | `DataConfig(markov=MarkovConfig(delta=0.5))` | `--delta 0.5` |
| Coupled | `DataConfig(coupling=CouplingConfig(...))` | `--coupled --K-hidden 10 --M-emission 20` |
