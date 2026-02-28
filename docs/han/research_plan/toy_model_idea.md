---
title: "Temporal crosscoders -- toy model setup"
author: Han Xuanyuan
date: 2026-02-16
type: document
---

We begin with a data-generating process that follows the Linear Representation Hypothesis, closely mirroring the setup in Chanin et al. (2025). This provides ground-truth knowledge of the underlying features, enabling direct evaluation of whether an SAE recovers them.

**Data generation.** Let $\mathbf{f}_1, \ldots, \mathbf{f}_k \in \mathbb{R}^d$ be $k$ unit-norm feature directions, taken to be mutually orthogonal ($\mathbf{f}_i^\top \mathbf{f}_j = 0$ for $i \neq j$, which requires $k \leq d$). Each activation vector $\mathbf{x} \in \mathbb{R}^d$ is a sparse linear combination of these features:
$$
\mathbf{x} = \sum_{i=1}^k a_i \, \mathbf{f}_i, \qquad a_i = s_i \, m_i,
$$
where $s_i \sim \mathrm{Bernoulli}(p)$ determines whether feature $i$ is active and $m_i \sim \mathcal{D}_m$ (e.g. $|\mathcal{N}(\mu, \sigma^2)|$) gives its activation magnitude conditional on being active. We set $p$ small to enforce sparsity, so that only a small number of features are active in any given sample.

**SAE objective.** We train an SAE directly on activations $\mathbf{x}$. The SAE has encoder weights $\mathbf{W}_\text{enc}^\text{SAE} \in \mathbb{R}^{h \times d}$, decoder weights $\mathbf{W}_\text{dec}^\text{SAE} \in \mathbb{R}^{d \times h}$, biases $\mathbf{b}_\text{enc} \in \mathbb{R}^h$, $\mathbf{b}_\text{dec} \in \mathbb{R}^d$, and activation function $\sigma$ (e.g. TopK or JumpReLU):
$$
\begin{align}
\mathbf{u} &= \sigma\!\left(\mathbf{W}_\text{enc}^\text{SAE}(\mathbf{x} - \mathbf{b}_\text{dec}) + \mathbf{b}_\text{enc}\right), \\
\hat{\mathbf{x}} &= \mathbf{W}_\text{dec}^\text{SAE} \, \mathbf{u} + \mathbf{b}_\text{dec}.
\end{align}
$$
The SAE is trained to minimize a reconstruction loss plus a sparsity penalty on the latent code $\mathbf{u}$:
$$
\mathcal{L} = \|\hat{\mathbf{x}} - \mathbf{x}\|_2^2 + \lambda \, \mathcal{R}(\mathbf{u}),
$$
where $\mathcal{R}$ is a sparsity-inducing regularizer (implicit in TopK, explicit in JumpReLU). If the SAE succeeds, the columns of $\mathbf{W}_\text{dec}^\text{SAE}$ should recover the true feature directions $\mathbf{f}_1, \ldots, \mathbf{f}_k$.

**Optional: introducing superposition via a bottleneck.** To test feature recovery under superposition, we can compress activations through a linear map $\mathbf{W}_\text{model} \in \mathbb{R}^{d' \times d}$ with $d' < k \leq d$, so that features are no longer orthogonal in the representation space $\mathbb{R}^{d'}$. The SAE then operates on $\mathbf{r} = \mathbf{W}_\text{model} \, \mathbf{x}$ rather than $\mathbf{x}$ directly. We treat this as a separate experimental condition.


## Extending the toy model with temporal correlations

We now introduce temporal structure. Let $\mathbf{x}_t \in \mathbb{R}^d$ denote the activation at sequence position $t \in \{1, \ldots, T\}$, generated as before:
$$
\mathbf{x}_t = \sum_{i=1}^k a_{i,t} \, \mathbf{f}_i, \qquad a_{i,t} = s_{i,t} \, m_{i,t}.
$$
In the baseline (non-temporal) setting, both $s_{i,t}$ and $m_{i,t}$ are i.i.d. across positions $t$, so activations at different positions are independent. Below, we describe three schemes for introducing temporal dependence, each targeting a different mechanism. These can be combined or used in isolation. In all cases, features remain independent of one another across feature index $i$; we discuss the important extension to cross-feature correlations at the end.

The goal is to induce nontrivial autocorrelation
$$
\mathrm{Corr}(a_{i,t},\; a_{i,t+\tau}) \neq 0 \quad \text{for } \tau > 0,
$$
which in turn induces temporal covariance in the activation vectors: $\mathrm{Cov}(\mathbf{x}_t, \mathbf{x}_{t+\tau}) \neq 0$.

---

### Scheme A: Temporally correlated magnitudes

This scheme introduces smooth variation in *how strongly* a feature activates over time, while the on/off support pattern remains i.i.d. across positions.

**Sampling procedure.** For each feature $i$, sample the full magnitude time-series jointly from a multivariate Gaussian with a user-chosen temporal covariance:
$$
\mathbf{m}_i := (m_{i,1}, \ldots, m_{i,T})^\top \sim \mathcal{N}(\mu_i \mathbf{1},\; \mathbf{C}_i^{(m)}), \qquad \mathbf{C}_i^{(m)} \in \mathbb{R}^{T \times T}.
$$
The diagonal entries $(\mathbf{C}_i^{(m)})_{t,t} = \sigma_i^2$ give the marginal variance of each magnitude, and the off-diagonal entries $(\mathbf{C}_i^{(m)})_{t,t'}$ encode how correlated magnitudes are across positions. For instance, the choice $(\mathbf{C}_i^{(m)})_{t,t'} = \sigma_i^2 \, \rho_i^{|t-t'|}$ with $\rho_i \in [0,1)$ gives exponentially decaying correlations controlled by a single parameter $\rho_i$.

Support variables are sampled independently of the magnitudes and independently across time:
$$
s_{i,t} \sim \mathrm{Bernoulli}(p_i) \qquad \text{independently over } t.
$$
Then set $a_{i,t} = s_{i,t}\,m_{i,t}$ and $\mathbf{x}_t = \sum_i a_{i,t}\,\mathbf{f}_i$ as before.

**How much temporal correlation survives the gating?** The magnitudes $m_{i,t}$ and $m_{i,t'}$ are correlated by construction, but the activations $a_{i,t} = s_{i,t}\,m_{i,t}$ are the product of these magnitudes with i.i.d. Bernoulli gates. A natural question is how much of the magnitude correlation survives this gating. We now derive this exactly.

For $t \neq t'$, since the support process is independent of the magnitude process, and the supports are i.i.d. across time, the second moment factors as
$$
\mathbb{E}[a_{i,t}\,a_{i,t'}] = \mathbb{E}[s_{i,t}]\,\mathbb{E}[s_{i,t'}]\,\mathbb{E}[m_{i,t}\,m_{i,t'}] = p_i^2\,\mathbb{E}[m_{i,t}\,m_{i,t'}].
$$
Combined with $\mathbb{E}[a_{i,t}] = p_i\,\mu_i$, this gives the covariance:
$$
\mathrm{Cov}(a_{i,t},\,a_{i,t'}) = p_i^2\,\mathbb{E}[m_{i,t}\,m_{i,t'}] - p_i^2\,\mu_i^2 = p_i^2\,\mathrm{Cov}(m_{i,t},\,m_{i,t'}).
$$
So the *covariance* is attenuated by exactly $p_i^2$. This holds for any choice of $\mathbf{C}_i^{(m)}$, with no assumptions on its structure.

However, the *correlation* (the normalized quantity) behaves differently, because the variance of $a_{i,t}$ is also affected by the gating. Using $s_{i,t}^2 = s_{i,t}$ (since $s_{i,t} \in \{0,1\}$):
$$
\mathrm{Var}(a_{i,t}) = \mathbb{E}[s_{i,t}\,m_{i,t}^2] - p_i^2\,\mu_i^2 = p_i(\mu_i^2 + \sigma_i^2) - p_i^2\,\mu_i^2 = p_i\,\sigma_i^2 + p_i(1-p_i)\,\mu_i^2.
$$
Writing $\mathrm{Cov}(m_{i,t}, m_{i,t'}) = \sigma_i^2\,\rho_{t,t'}^{(m)}$ where $\rho_{t,t'}^{(m)} = \mathrm{Corr}(m_{i,t}, m_{i,t'})$ is the magnitude correlation, the activation correlation becomes:
$$
\boxed{\mathrm{Corr}(a_{i,t},\, a_{i,t'}) \;=\; \gamma_i \;\cdot\; \rho_{t,t'}^{(m)}, \qquad \gamma_i := \frac{p_i\,\sigma_i^2}{\sigma_i^2 + (1-p_i)\,\mu_i^2}.}
$$
The factor $\gamma_i \in [0, p_i]$ is the attenuation: it tells us what fraction of the magnitude correlation is visible in the actual activations.

**Understanding the attenuation.** The factor $\gamma_i$ depends on the ratio $\mu_i / \sigma_i$ between the mean and standard deviation of the magnitude distribution. When the mean is large relative to the standard deviation ($\mu_i \gg \sigma_i$), the denominator is dominated by $(1-p_i)\mu_i^2$ and $\gamma_i \approx \frac{p_i\,\sigma_i^2}{(1-p_i)\mu_i^2} \approx 0$. The intuition is that in this regime, the variance of the activation $a_{i,t}$ is dominated by whether the feature is on or off (since the magnitude is nearly constant at $\mu_i$ when active), and the on/off process is i.i.d., so almost all variance is temporally uncorrelated.

In the best case for this scheme ($\mu_i = 0$, so the magnitude fluctuates symmetrically around zero), $\gamma_i$ reaches its maximum of $p_i$. Even then, for sparse features with $p_i = 0.05$, the activation correlation is only 5\% of the magnitude correlation. The fundamental reason is that two adjacent positions can only both "see" the correlated magnitudes if the feature happens to be active at both, which occurs with probability $p_i^2$.

**Implications for experiments.** This analysis shows that Scheme A produces weak temporal correlations in the activation signal $a_{i,t}$ whenever features are sparse. This makes it a useful baseline condition for testing whether a temporal SAE architecture can extract temporal structure even when that structure is faint. However, Schemes B and C, which correlate the support directly, will produce substantially stronger temporal signal and are likely the more informative regimes for evaluating temporal SAE architectures.



#### A concrete parameterization: AR(1) magnitudes

The covariance matrix $\mathbf{C}_i^{(m)}$ in Scheme A can in principle be any positive-definite $T \times T$ matrix. For our toy model, we want the simplest nontrivial choice that gives us a single knob controlling the temporal correlation strength. The first-order autoregressive process (AR(1)) is the natural candidate: it is the unique stationary Gaussian process whose correlations decay exponentially with lag and which is Markov in time (i.e. the future is conditionally independent of the past given the present).

**Definition as a recurrence.** For each feature $i$, define the magnitude process by the recurrence
$$
m_{i,t} = \mu_i + \rho_i\,(m_{i,t-1} - \mu_i) + \varepsilon_{i,t}, \qquad \varepsilon_{i,t} \sim \mathcal{N}(0,\; \sigma_i^2(1 - \rho_i^2)),
$$
where $\rho_i \in [0, 1)$ is the autocorrelation parameter and $m_{i,1} \sim \mathcal{N}(\mu_i, \sigma_i^2)$ is drawn from the stationary marginal. At each step, the magnitude is pulled toward the mean $\mu_i$ with strength $1 - \rho_i$, and receives a Gaussian innovation $\varepsilon_{i,t}$ whose variance $\sigma_i^2(1 - \rho_i^2)$ is chosen so that the marginal variance remains $\sigma_i^2$ at every position. The parameter $\rho_i$ directly controls memory: when $\rho_i = 0$ the magnitudes are i.i.d., and as $\rho_i \to 1$ the process becomes increasingly persistent.

**Resulting covariance structure.** The recurrence above implies the following joint statistics, which can be verified by induction:
$$
\mathbb{E}[m_{i,t}] = \mu_i, \qquad \mathrm{Cov}(m_{i,t},\, m_{i,t'}) = \sigma_i^2\,\rho_i^{|t - t'|}.
$$
This is exactly the covariance matrix $(\mathbf{C}_i^{(m)})_{t,t'} = \sigma_i^2\,\rho_i^{|t-t'|}$ from Scheme A, now derived from a concrete generative process rather than posited directly. The recurrence form also makes sampling straightforward: rather than drawing from a $T$-dimensional Gaussian (which requires factoring a $T \times T$ matrix), we can generate the process sequentially in $O(T)$ time.

**Activation-level correlations.** Applying the general attenuation result derived above (which holds for any choice of $\mathbf{C}_i^{(m)}$), the correlation of the gated activations $a_{i,t} = s_{i,t}\,m_{i,t}$ under this scheme is
$$
\mathrm{Corr}(a_{i,t},\, a_{i,t'}) = \gamma_i \cdot \rho_i^{|t-t'|}, \qquad \gamma_i = \frac{p_i\,\sigma_i^2}{\sigma_i^2 + (1-p_i)\,\mu_i^2}.
$$
The activation correlation inherits the same exponential decay structure as the magnitude correlation, but with amplitude reduced by the factor $\gamma_i \leq p_i$. This means the effective autocorrelation parameter of the activation process is still $\rho_i$ (the decay rate is unchanged), but the overall correlation is scaled down. For sparse features ($p_i = 0.05$) with $\mu_i = \sigma_i$, we get $\gamma_i \approx 0.026$: a magnitude correlation of $\rho_i = 0.9$ between adjacent positions translates to an activation correlation of only $\approx 0.023$.

**Choosing $\rho_i$ in practice.** The attenuation result tells us that if we want the activation-level temporal correlations to be detectable (say, $\mathrm{Corr}(a_{i,t}, a_{i,t+1}) \geq 0.01$), we need $\rho_i \geq 0.01 / \gamma_i$, which for the numbers above requires $\rho_i \geq 0.38$. This means we should sweep $\rho_i$ over the range $[0.5, 0.99]$ to explore the regime where temporal structure in magnitudes is potentially exploitable. Values of $\rho_i$ below $\sim 0.3$ will be effectively indistinguishable from the i.i.d. baseline for sparse features.

---




### Scheme B: Temporally correlated support via a Gaussian copula

This scheme introduces persistence in *which* features are active, which is arguably the more natural form of temporal dependence in language (a topic doesn't flicker on and off randomly; it stays active for a stretch of tokens).

For each feature $i$, define a latent Gaussian vector:
$$
\mathbf{z}_i := (z_{i,1}, \ldots, z_{i,T})^\top \sim \mathcal{N}(\mathbf{0},\; \mathbf{C}_i^{(s)}), \qquad \mathbf{C}_i^{(s)} \in \mathbb{R}^{T \times T},
$$
and obtain support indicators by thresholding:
$$
s_{i,t} = \mathbb{I}\!\left[z_{i,t} \leq \Phi^{-1}(p_i)\right],
$$
where $\Phi$ is the standard normal CDF. By construction, $\mathbb{P}(s_{i,t} = 1) = p_i$ for all $t$, while $\mathrm{Corr}(s_{i,t}, s_{i,t+\tau})$ is controlled by $\mathbf{C}_i^{(s)}$.


---

### Scheme C: Temporally correlated support via a two-state Markov chain

This is a simpler and more interpretable model of persistent feature activation, directly parameterizing the "stickiness" of the active state.

For each feature $i$, the support process $(s_{i,t})_{t=1}^T$ follows a two-state Markov chain:
$$
\begin{align}
s_{i,1} &\sim \mathrm{Bernoulli}(\pi_i), \\
\mathbb{P}(s_{i,t} = 1 \mid s_{i,t-1} = 1) &= \alpha_i, \\
\mathbb{P}(s_{i,t} = 1 \mid s_{i,t-1} = 0) &= \beta_i,
\end{align}
$$
where $\alpha_i$ controls the persistence of the active state and $\beta_i$ controls the turn-on rate. Setting $\pi_i$ to the stationary probability $\pi_i = \beta_i / (1 - \alpha_i + \beta_i)$ ensures stationarity from $t = 1$. The autocorrelation of the support has a simple closed form:
$$
\mathrm{Corr}(s_{i,t}, s_{i,t+\tau}) = (\alpha_i - \beta_i)^{|\tau|},
$$
so the persistence lengthscale is directly controlled by $\alpha_i - \beta_i \in (-1, 1)$. For $\alpha_i$ close to 1 and $\beta_i$ close to 0, the feature tends to stay in long contiguous "on" segments — the regime most relevant for modeling persistent semantic features.

Magnitudes are again sampled i.i.d. to isolate the support effect. This scheme is the simplest to implement and reason about analytically, and we recommend it as the default for initial experiments.




**Caveat on achievable correlations for Bernoulli variables.** Depending on the choice of marginal, the full range of Bernoulli correlation might not be achievable. Example: let $(S,S'\in\{0,1\})$ with
$$
\Pr(S=1)=\Pr(S'=1)=p.
$$
Let
$$
q := \Pr(S=1,S'=1).
$$
* $\Pr(1,1)=q$
* $\Pr(1,0)=p-q$
* $\Pr(0,1)=p-q$
* $\Pr(0,0)=1-2p+q$

All entries must be $\ge 0$, so $q \in [\max(0,2p-1),\ p]$. If $p\le 1/2$, this simplifies to $q\in[0,p]$. Now
$$
\mathrm{Corr}(S,S')=\frac{\mathrm{Cov}(S,S')}{\sqrt{\mathrm{Var}(S)\mathrm{Var}(S')}} = \frac{q-p^2}{p(1-p)}.
$$

* This is **increasing in $q$**.
* So the minimum correlation occurs at the smallest feasible $q$, i.e. $q=0$ (when $p\le 1/2)$:

$$
\rho_{\min}=\frac{0-p^2}{p(1-p)}=-\frac{p}{1-p}.
$$

* The maximum occurs at $q=p$:

$$
\rho_{\max}=\frac{p-p^2}{p(1-p)}=1
$$

So indeed, for (p\le 1/2),
$$
\rho \in \left[-\frac{p}{1-p},\ 1\right].
$$

For (p=0.05), the lower bound is
$$
-\frac{0.05}{0.95}\approx -0.05263,
$$
