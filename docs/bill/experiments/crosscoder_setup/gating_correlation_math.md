---
author: bill
date: 2026-02-25
tags:
  - experimental design
---

# Comparing Correlation Channels in Sparse Bernoulli-Gated Variables

## Setup

Consider two random variables of the form

$$X_i = a_i \cdot B_i, \quad i \in \{1, 2\}$$

where $B_i \mid p_i \sim \text{Bernoulli}(p_i)$ is a binary activation gate and $a_i > 0$ is a magnitude. The pairs $(a_1, a_2)$ and $(p_1, p_2)$ are independent of each other, with $B_1 \perp B_2 \mid (p_1, p_2)$. Marginals are identically distributed: $a_1 \stackrel{d}{=} a_2$, $p_1 \stackrel{d}{=} p_2$.

We define two channels through which correlation between $X_1$ and $X_2$ can be introduced:

- **Amplitude channel**: correlating $(a_1, a_2)$ while keeping $(p_1, p_2)$ independent.
- **Activation channel**: correlating $(p_1, p_2)$ while keeping $(a_1, a_2)$ independent.

The question is: which channel is more effective at inducing correlation in the observed variables?

## General Covariance Decomposition

By the tower of expectation and the conditional independence of the Bernoullis:

$$\mathbb{E}[X_1 X_2] = \mathbb{E}[a_1 a_2]\,\mathbb{E}[B_1 B_2] = \bigl(\mu_a^2 + C_a\bigr)\bigl(\mu_p^2 + C_p\bigr)$$

where $C_a = \text{Cov}(a_1, a_2)$, $C_p = \text{Cov}(p_1, p_2)$, and we have used $\mathbb{E}[B_1 B_2] = \mathbb{E}[p_1 p_2]$. Since $\mathbb{E}[X_i] = \mu_a \mu_p$, we obtain:

$$\boxed{\text{Cov}(X_1, X_2) = \mu_p^2\, C_a \;+\; \mu_a^2\, C_p \;+\; C_a\, C_p}$$

This decomposes into three terms:

| Term | Interpretation |
|------|---------------|
| $\mu_p^2 \cdot C_a$ | Amplitude channel — weighted by co-occurrence probability |
| $\mu_a^2 \cdot C_p$ | Activation channel — weighted by mean signal strength |
| $C_a \cdot C_p$ | Cross-term (second-order, typically small) |

## Head-to-Head Comparison

Setting one channel at a time and dropping the cross-term:

| | Amplitude correlation only ($C_p = 0$) | Activation correlation only ($C_a = 0$) |
|---|---|---|
| $\text{Cov}(X_1, X_2)$ | $\mu_p^2 \cdot \rho_a \sigma_a^2$ | $\mu_a^2 \cdot \rho_p \sigma_p^2$ |

For equal induced correlation strength $\rho_a = \rho_p = \rho$, the ratio of effects is:

$$\frac{\text{Cov}_{\text{amplitude}}}{\text{Cov}_{\text{activation}}} = \frac{\mu_p^2\,\sigma_a^2}{\mu_a^2\,\sigma_p^2} = \frac{CV_a^2}{CV_p^2}$$

where $CV = \sigma / \mu$ is the coefficient of variation. **The channel whose underlying variable has a higher CV is the more powerful correlation lever.**

## Asymmetry from Boundedness

Since $p_i \in [0,1]$, its variance is bounded:

$$\sigma_p^2 \leq \mu_p(1 - \mu_p) \quad \Longrightarrow \quad CV_p^2 \leq \frac{1 - \mu_p}{\mu_p}$$

The amplitude $a_i$ (supported on $\mathbb{R}^+$) has no such ceiling, so $CV_a$ can be arbitrarily large. This creates distinct regimes:

### Amplitude channel dominates ($CV_a^2 \gg CV_p^2$)

- Heavy-tailed or high-variance amplitudes (e.g., log-normal $a_i$).
- **High $\mu_p$** (frequent activation). When $p \approx 1$, there is very little room for $p$ to vary, so $CV_p \to 0$ and the activation channel shuts down.

### Activation channel dominates ($CV_p^2 \gg CV_a^2$)

- **Low $\mu_p$** (sparse activation). As $\mu_p \to 0$, $CV_p \to \infty$ while the amplitude channel's prefactor $\mu_p^2 \to 0$, so gating correlation becomes the only effective mechanism.
- Near-constant amplitudes ($CV_a \approx 0$).

## Intuitive Summary

- **Amplitude correlation** controls *how much* co-varies, given both variables are active. Its effect is weighted by $\mu_p^2$ — the probability of co-occurrence. If activations are rare, you rarely get to "use" the amplitude correlation.

- **Activation correlation** controls *whether* the variables co-occur at all. Its effect is weighted by $\mu_a^2$ — the mean signal strength. It acts as a binary gate, so even perfect activation correlation produces zero signal if the amplitudes are tiny.

- The **sparsity regime** ($\mu_p \ll 1$) strongly favors the activation channel because $\mu_p^2$ crushes the amplitude channel, while the gating mechanism has maximal relative variability.

- The **dense regime** ($\mu_p \approx 1$) strongly favors the amplitude channel because $p$ has almost no room to vary, and nearly every draw is active anyway.

## Connection to Sparse Autoencoders

In the mechanistic interpretability setting, SAE features decompose naturally as $f_i(t) = a_i(t) \cdot \mathbf{1}[f_i(t) > 0]$, matching the $a \cdot B$ form. SAEs are designed for high sparsity (typical L0 of 50–200 out of thousands or millions of features), placing them squarely in the sparse regime where $\mu_p \ll 1$. The analysis predicts:

1. **Co-activation patterns across nearby tokens are dominated by the gating channel**, not the magnitude channel. Whether two adjacent tokens share the same active features matters far more than whether their magnitudes are correlated.

2. **Standard SAEs, which treat each position independently, miss the gating correlation entirely.** This may explain the bias towards shallow, token-specific features observed empirically.

3. **Temporal SAEs** (Bhalla et al. 2025) partially address this by encouraging consistent *activations* across adjacent tokens, but do not separately model the two channels or provide the theoretical justification for why gating dominance emerges in the sparse regime.