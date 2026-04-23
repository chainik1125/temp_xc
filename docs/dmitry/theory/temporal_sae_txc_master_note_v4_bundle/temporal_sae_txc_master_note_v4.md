
# Temporal SAE / TXC note — reorganized benchmark ladder

This version rewrites the earlier master note around the hierarchy you suggested:

1. **Exact solution for static Chanin-style SAEs.**
2. **Generalization to HMMs that only add temporal correlation.**
3. **Generalization to richer HMMs that reveal new latent information.**
4. **Continuous-state matched teachers (Jordan blocks) as a useful non-HMM analogue.**

The stylistic rule in this rewrite is:

- every benchmark starts with a **generator table** listing the objects and their dimensions;
- every genuinely temporal generator that is an HMM gets a **state diagram**;
- every benchmark has the same internal rhythm:
  1. referenced setup,
  2. pedagogical minimal example,
  3. exact solution,
  4. tiny numerical theory-vs-experiment check,
  5. takeaway.

The main notation is fixed globally:

| symbol | shape | meaning |
|---|---:|---|
| \(d\) | integer | activation dimension, so \(x_t\in\mathbb R^d\) |
| \(g\) | integer | number of **ground-truth** features in the data generator |
| \(m\) | integer | number of **learned SAE latents** |
| \(r\) | integer | temporal-driver width or hidden-state dimension, depending on the section |
| \(T\) | integer | temporal **window length** used by the TXC |
| \(L\) | integer | full sequence length when needed |
| \(\chi\) | integer | number of hidden states in a shared-chain HMM |
| \(f_i\) | \(\mathbb R^d\) | true feature direction \(i\) in activation space |
| \(F=[f_1,\dots,f_g]\) | \(\mathbb R^{d\times g}\) | true feature matrix / true embedding matrix |
| \(x_t\) | \(\mathbb R^d\) | observed activation at token \(t\) |
| \(a_t\) | \(\mathbb R^m\) | learned SAE code at token \(t\) |
| \(g_t\) | \(\mathbb R^r\) | learned temporal-driver state |
| \(P\) | square matrix | transition matrix of an HMM / Markov chain |
| \(\pi\) | scalar | stationary on-probability of a binary support process |
| \(\rho\) | scalar | lag-1 correlation of a binary support process |

Two global conventions remove most earlier ambiguity.

1. **True features are columns of \(F\).**  
   If \(c_t\in\mathbb R^g\) is the ground-truth coefficient vector, then
   \[ x_t = F c_t. \]
   If you prefer the language “there is an Embed layer acting on the ground-truth features”, then \(F\in\mathbb R^{d\times g}\) is exactly that **fixed** embedding / dictionary matrix.

2. **“Orthogonal” means orthogonal in activation space.**  
   When I say the true features are orthogonal, I mean
   \[ f_i^\top f_j = 0\quad (i\neq j). \]
   When I say “orthonormal”, I additionally mean \(\|f_i\|_2=1\), so
   \[ f_i^\top f_j=\delta_{ij}. \]

---

## 1. Exact solution for static Chanin-style SAEs

### 1a. Referenced explanation of the setup

The right zeroth-order benchmark is the static toy from **Chanin & Garriga-Alonso, _Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders_** ([arXiv:2508.16560](https://arxiv.org/abs/2508.16560)). Their core point is that when the target \(L0\) is set incorrectly, MSE can favor **mixed** latent directions over the true underlying features.

The clean way to write their generator is:

| object | shape | what it is |
|---|---:|---|
| \(s\) | \(\{0,1\}^g\) | support pattern of true features for one sample |
| \(h\) | \(\mathbb R_{\ge 0}^g\) | nonnegative true amplitudes |
| \(c=s\odot h\) | \(\mathbb R^g\) | ground-truth coefficient vector |
| \(F=[f_1,\dots,f_g]\) | \(\mathbb R^{d\times g}\) | fixed true embedding / true feature dictionary |
| \(x=F(s\odot h)\) | \(\mathbb R^d\) | observed activation fed to the SAE |

So the full sample generator is

\[ x = F(s\odot h)=\sum_{i=1}^g s_i h_i f_i. \]

This answers the “is there an Embed layer?” question precisely:

- yes, there is a linear map from ground-truth feature coefficients to activations;
- in this note I denote that fixed map by \(F\in\mathbb R^{d\times g}\);
- the columns \(f_i\in\mathbb R^d\) are the true feature embeddings / true decoder directions.

There is **no time** in this benchmark. If we want to force it into HMM language, the hidden state is simply the full support pattern \(s\), redrawn iid each step from a distribution \(q(s)\). Equivalently, it is the \(\lambda=1\) reset limit on the joint support-pattern state space.

![Static toy as a reset-on-every-step process over joint support patterns](./temporal_sae_txc_v4_assets/hmm_static_joint_reset.png)

### 1b. Pedagogical minimal example: two orthonormal true features and a Top-1 SAE

We now solve the smallest nontrivial example exactly.

#### Generator

Take two true features

\[ f_1,f_2\in\mathbb R^d,\qquad f_1^\top f_2=0,\qquad \|f_1\|_2=\|f_2\|_2=1, \]

and four event types:

\[ x=0\quad(P_0),\qquad x=\mu_1 f_1\quad(P_1),\qquad x=\mu_2 f_2\quad(P_2),\qquad x=\mu_1 f_1+\mu_2 f_2\quad(P_{12}), \]

with

\[ P_0+P_1+P_2+P_{12}=1,\qquad \mu_1,\mu_2>0. \]

The true average support budget is

\[ L0_{\mathrm{true}} = P_1+P_2+2P_{12}. \]

So a **Top-1** SAE is under-budget whenever \(L0_{\mathrm{true}}>1\).

#### Learned dictionary ansatz

We take a tied Top-1 SAE with unit-norm decoder directions

\[ \ell_2=f_2,\qquad \ell_1(\alpha)=\frac{\alpha f_1+(1-\alpha)f_2}{\sqrt{\alpha^2+(1-\alpha)^2}}, \qquad \frac12\le \alpha\le 1. \]

Interpretation:

- \(\alpha=1\): the first learned latent is exactly the true feature \(f_1\);
- \(\alpha<1\): the first learned latent is mixed, containing some of \(f_2\).

Because the SAE is tied and Top-1, if the chosen decoder is \(\ell\), the reconstruction is

\[ \hat x = (\ell^\top x)\,\ell. \]

#### Case A: only feature 1 fires

If \(x=\mu_1 f_1\), Top-1 picks \(\ell_1\), so

\[ \hat x = \frac{\mu_1\alpha}{\sqrt{\alpha^2+(1-\alpha)^2}}\ell_1 = \frac{\mu_1\alpha^2}{\alpha^2+(1-\alpha)^2}f_1 + \frac{\mu_1\alpha(1-\alpha)}{\alpha^2+(1-\alpha)^2}f_2. \]

Hence

\[ L_1(\alpha) = \|\mu_1f_1-\hat x\|_2^2 = \mu_1^2\frac{(1-\alpha)^2}{\alpha^2+(1-\alpha)^2}. \]

So mixing hurts isolated \(f_1\) events.

#### Case B: only feature 2 fires

If \(x=\mu_2 f_2\), the best decoder is \(\ell_2=f_2\), so

\[ L_2(\alpha)=0. \]

#### Case C: both features fire

If \(x=\mu_1f_1+\mu_2f_2\), then in the regime \(\alpha\ge 1/2\), Top-1 picks \(\ell_1\). The reconstruction is

\[ \hat x = \frac{\mu_1\alpha+\mu_2(1-\alpha)}{\alpha^2+(1-\alpha)^2} \big(\alpha f_1+(1-\alpha)f_2\big). \]

A short orthogonality calculation gives

\[ L_{12}(\alpha) = \frac{(\mu_1(1-\alpha)-\mu_2\alpha)^2}{\alpha^2+(1-\alpha)^2}. \]

This is the term that makes the mixed solution win:

- at \(\alpha=1\), the latent only tracks \(f_1\), so it misses the \(f_2\) piece and
  \[ L_{12}(1)=\mu_2^2; \]
- if \(\mu_1=\mu_2\) and \(\alpha=1/2\), then the mixed latent aligns exactly with \(f_1+f_2\), so
  \[ L_{12}(1/2)=0. \]

### 1c. Full analytic solution

#### Equal-magnitude case \(\mu_1=\mu_2=\mu\)

In the symmetric equal-magnitude case, the expected loss is

\[ \mathcal E(\alpha) = \mu^2\, \frac{P_1(1-\alpha)^2 + P_{12}(1-2\alpha)^2}{\alpha^2+(1-\alpha)^2}. \]

Differentiating gives

\[ \mathcal E'(\alpha) = 2\mu^2 \frac{P_1\alpha^2-P_1\alpha+2P_{12}\alpha-P_{12}} {(\alpha^2+(1-\alpha)^2)^2}. \]

Two immediate consequences:

1. At the disentangled point \(\alpha=1\),
   \[ \mathcal E'(1)=2\mu^2P_{12}>0 \qquad\text{whenever }P_{12}>0, \]
   so the correct dictionary is **not even a local minimum** once co-firing occurs.

2. The exact stationary point is
   \[ \alpha^\star = \frac{P_1-2P_{12}+\sqrt{P_1^2+4P_{12}^2}}{2P_1}. \]

This is the exact mixed optimum in the simplest two-feature low-\(L0\) benchmark.

#### General unequal-magnitude case

If \(\mu_1\neq \mu_2\), the expected loss is

\[ \mathcal E(\alpha) = \frac{ P_1\mu_1^2(1-\alpha)^2 + P_{12}\big(\mu_1(1-\alpha)-\mu_2\alpha\big)^2 }{ \alpha^2+(1-\alpha)^2 }. \]

The stationary point is still analytic: it is the admissible root of the quadratic equation

\[ A\alpha^2+B\alpha+C=0, \]

with

\[ A=P_1\mu_1^2+P_{12}(\mu_1^2-\mu_2^2), \]

\[ B=-P_1\mu_1^2-P_{12}\mu_1^2+2P_{12}\mu_1\mu_2+P_{12}\mu_2^2, \]

\[ C=-P_{12}\mu_1\mu_2. \]

So even the unequal-amplitude two-feature benchmark is analytically tractable.

### 1d. Tiny numerical theory-vs-experiment check

Take

\[ P_1=P_2=0.25,\qquad P_{12}=0.30,\qquad P_0=0.20,\qquad \mu=1. \]

Then

\[ L0_{\mathrm{true}} = 0.25+0.25+2(0.30)=1.10, \]

so a Top-1 SAE is indeed under-budget.

The exact optimizer is

\[ \alpha^\star = \frac{0.25-0.60+\sqrt{0.25^2+4(0.30)^2}}{0.50} = 0.60. \]

The theory predicts

\[ \mathcal E(1)=0.30,\qquad \mathcal E(0.60)=0.10. \]

The small Monte Carlo experiment we ran gave approximately

- disentangled loss: \(0.299\),
- mixed-optimum loss: \(0.100\),

which matches the analytic values to sampling error.

![Static low-\(L0\) loss curve](./temporal_sae_txc_v4_assets/static_sparse_but_wrong_loss_vs_alpha.png)

### 1e. Takeaway

This is the correct **Level 0** benchmark.

- It isolates **static feature hedging**.
- It has **no temporal information**.
- Any temporal architecture must reduce to a static one here.
- So any gain above this level is a genuinely temporal gain, not just the same static low-\(L0\) phenomenon in disguise.

---

## 2. Generalization to HMMs that only add temporal correlation

In this section we keep the latent ontology simple. The hidden variable is still essentially “feature \(i\) is on or off”, but now its support process is persistent across time.

### 2a. Common generator: the two-state leaky-reset / reset HMM

We start from the simplest temporally correlated support process.

| object | shape | what it is |
|---|---:|---|
| \(s_t\) | \(\{0,1\}\) | on/off support bit at time \(t\) |
| \(P\) | \(2\times 2\) | transition matrix of the binary Markov chain |
| \(\pi\) | scalar | stationary on-probability, \(\pi=\Pr(s_t=1)\) |
| \(\rho\) | scalar | lag-1 correlation of \(s_t\) |
| \(\xi_t=s_t-\pi\) | scalar | centered support process |
| \(C_\tau\) | scalar | autocovariance, \(C_\tau=\mathbb E[\xi_t\xi_{t+\tau}]\) |

In reset-process form,

\[ P = (1-\lambda)I+\lambda R, \]

with stationary distribution \(r=(\pi,1-\pi)\). Then

\[ \rho = 1-\lambda, \qquad C_\tau = C_0\rho^{|\tau|}, \qquad C_0=\pi(1-\pi). \]

The state diagram is:

![Two-state leaky-reset HMM](./temporal_sae_txc_v4_assets/hmm_two_state_reset.png)

To connect this to an activation vector, the simplest orthogonal generator is

\[ x_t = \sum_{i=1}^g s_{t,i} f_i, \qquad f_i^\top f_j=\delta_{ij}, \]

where each \(s_{t,i}\) is an independent copy of the same two-state process (possibly with feature-specific \(\pi_i,\rho_i\)).

### 2b. Pedagogical minimal example: one persistent feature, one true direction

Before doing the architectures, reduce to the cleanest single-feature case:

| object | shape | meaning |
|---|---:|---|
| \(f\) | \(\mathbb R^d\) | one true orthonormal feature direction |
| \(s_t\) | \(\{0,1\}\) | persistent support bit |
| \(x_t=s_t f\) | \(\mathbb R^d\) | observed activation |
| \(u\) | scalar | learned local aligned gain |

Because \(f\) is orthonormal and there is only one feature, every aligned model reduces to a scalar problem. This is why the exact gradient-descent calculations close.

### 2c. Exact solution for a regular SAE

In the aligned monosemantic basin, choose the learned encoder and decoder to point along the true feature direction \(f\), and parameterize the effective local gain by \(u\in(0,1)\). Then the ReLU code is

\[ a_t = u\,s_t. \]

The population reconstruction+sparsity loss is

\[ \mathcal L_{\mathrm{SAE}}(u) = \pi\Big[\frac12(1-u)^2+\lambda_h u\Big], \]

where \(\lambda_h\) is the sparsity penalty.

The crucial fact is:

\[ \mathcal L_{\mathrm{SAE}}(u) \text{ depends on }\pi, \text{ but not on }\rho. \]

So a time-local SAE trained on single tokens is population-equivalent to training on iid samples from the stationary one-token marginal.

The gradient is

\[ \frac{\partial \mathcal L_{\mathrm{SAE}}}{\partial u} = \pi(u-1+\lambda_h), \]

hence full-batch gradient descent with learning rate \(\eta\) obeys

\[ u_{n+1} = (1-\eta\pi)u_n+\eta\pi(1-\lambda_h). \]

This has exact solution

\[ u_n = (1-\lambda_h)+\big(u_0-(1-\lambda_h)\big)(1-\eta\pi)^n, \]

and fixed point

\[ u^\star_{\mathrm{SAE}}=1-\lambda_h. \]

#### Tiny numerical check

Take

\[ \pi=0.2,\qquad \lambda_h=0.1,\qquad \eta=0.4,\qquad u_0=10^{-3}. \]

Then

\[ u_{n+1}=0.92u_n+0.072. \]

After 20 full-batch steps,

\[ u_{20} = 0.9+(10^{-3}-0.9)(0.92)^{20} \approx 0.730365. \]

Holding \(\pi\) fixed and varying \(\rho\) gives almost exactly the same trajectory, as predicted:

| \(\rho\) | empirical \(\pi\) | theory \(u_{20}\) | experiment \(u_{20}\) | abs. error |
|---:|---:|---:|---:|---:|
| 0.0 | 0.200051 | 0.730365 | 0.730436 | \(7.2\times 10^{-5}\) |
| 0.5 | 0.200441 | 0.730365 | 0.730998 | \(6.3\times 10^{-4}\) |
| 0.8 | 0.199643 | 0.730365 | 0.729815 | \(5.5\times 10^{-4}\) |

![Regular SAE: full-batch trajectory is blind to persistence](./temporal_sae_txc_v4_assets/hmm_sae_fullbatch_scalar.png)

The only place persistence re-enters is through minibatch noise. For contiguous minibatches of size \(B\),

\[ \mathrm{Var}(\bar g_B) = \frac{(u-1+\lambda_h)^2\pi(1-\pi)}{B^2} \left[ B + 2\sum_{\tau=1}^{B-1}(B-\tau)\rho^\tau \right], \]

so the effective batch size is approximately

\[ B_{\mathrm{eff}}\approx B\frac{1-\rho}{1+\rho}. \]

This matches experiment too:

![Regular SAE: contiguous-batch gradient variance grows with persistence](./temporal_sae_txc_v4_assets/hmm_sae_batch_variance.png)

### 2d. Exact solution for Proposal 3A: a single temporal self-filter

Now we make the architecture genuinely temporal. We keep the same single hidden scalar \(\xi_t=s_t-\pi\), but allow the latent to read from the previous token.

| object | shape | meaning |
|---|---:|---|
| \(u\) | scalar | local gain |
| \(\beta\) | scalar | one-step temporal self-coupling |
| \(h_t=u\xi_t\) | scalar | local latent |
| \(\tilde h_t=h_t+\beta h_{t-1}\) | scalar | temporally corrected latent |

The centered target is \(\xi_t\). The population loss is

\[ \mathcal L_{3A}(u,\beta) = \frac12\Big( C_0 - 2u(C_0+\beta C_1) +u^2(C_0+2\beta C_1+\beta^2 C_0) \Big) +\lambda_h\pi u +\frac{\lambda_\beta}{2}\beta^2. \]

At fixed \(u\), the optimal temporal weight is

\[ \beta^\star(u) = \frac{u(1-u)C_1}{u^2C_0+\lambda_\beta}. \]

So with no explicit kernel penalty (\(\lambda_\beta=0\)),

\[ \beta^\star = \frac{C_1}{C_0}\frac{1-u^\star}{u^\star} = \rho\frac{1-u^\star}{u^\star}. \]

Plugging this back into the objective gives

\[ u^\star_{3A} = 1-\frac{\lambda_h\pi}{C_0-C_1^2/C_0}. \]

For the reset family \(C_1=\rho C_0\), this becomes

\[ u^\star_{3A} = 1-\frac{\lambda_h\pi}{C_0(1-\rho^2)}. \]

So unlike the regular SAE, the population optimum now depends explicitly on persistence.

#### Tiny numerical check

Take empirical moments from a long run of the reset HMM:

\[ \pi=0.2,\qquad \rho\approx 0.706,\qquad C_0\approx 0.161285,\qquad C_1\approx 0.113829, \]

and choose

\[ \lambda_h=0.15,\qquad \lambda_\beta=0. \]

Then

\[ u^\star_{3A} = 1-\frac{0.15\cdot 0.2}{C_0-C_1^2/C_0} \approx 0.629390, \]

and

\[ \beta^\star = \frac{C_1}{C_0}\frac{1-u^\star}{u^\star} \approx 0.415584. \]

This matches the direct numerical solve from the same moments.

### 2e. Exact solution for the simplest two-layer temporal XC

Now make the read-in multiview rather than single-view.

Suppose the same hidden scalar \(\xi_t\) is visible in two source layers:

\[ y_t^{(1)}=\xi_t,\qquad y_t^{(2)}=\xi_t. \]

We use scalar read-in weights \(c_1,c_2\), so the local read-in is

\[ h_t=(c_1+c_2)\xi_t=u\xi_t. \]

Then the same one-step temporal correction gives

\[ \tilde h_t=h_t+\beta h_{t-1}. \]

If the read-ins have \(L_2\) penalties \(\lambda_{e,1}\) and \(\lambda_{e,2}\), the exact optimal split at fixed total gain \(u=c_1+c_2\) is

\[ c_1^\star=\frac{\lambda_{e,2}}{\lambda_{e,1}+\lambda_{e,2}}u, \qquad c_2^\star=\frac{\lambda_{e,1}}{\lambda_{e,1}+\lambda_{e,2}}u. \]

So equal penalties imply

\[ c_1^\star=c_2^\star=u/2. \]

The two-layer read-in collapses to an effective ridge penalty

\[ \lambda_{\mathrm{eff}} = \frac{\lambda_{e,1}\lambda_{e,2}}{\lambda_{e,1}+\lambda_{e,2}}. \]

With total decoder energy \(E_D\), the optimal gain is

\[ u^\star_{\mathrm{XC}} = \frac{E_D(C_0-C_1^2/C_0)-\lambda_h\pi} {E_D(C_0-C_1^2/C_0)+\lambda_{\mathrm{eff}}}, \]

and

\[ \beta^\star_{\mathrm{XC}} = \frac{C_1}{C_0}\frac{1-u^\star_{\mathrm{XC}}}{u^\star_{\mathrm{XC}}}. \]

So the simplest two-layer TXC changes the exact solution in only one way: it introduces a **layer-allocation problem**, and that problem solves exactly.

#### Tiny numerical check

Using the same empirical moments and taking

\[ E_D=2,\qquad \lambda_{e,1}=\lambda_{e,2}=0.2, \]

we get:

| model | \(u^\star\) theory | \(u^\star\) experiment | \(\beta^\star\) theory | \(\beta^\star\) experiment |
|---|---:|---:|---:|---:|
| Proposal 3A | 0.629390 | 0.629390 | 0.415584 | 0.415584 |
| 2-layer TXC | 0.503618 | 0.503618 | 0.695626 | 0.695626 |

And the split is

| parameter | theory | experiment |
|---|---:|---:|
| \(c_1\) | 0.251809 | 0.251809 |
| \(c_2\) | 0.251809 | 0.251809 |

### 2f. Exact solution for a window-\(T\) temporal XC

The one-lag filter is only the first nontrivial truncation. The full windowed TXC is a regularized Wiener-filter / Toeplitz problem.

| object | shape | meaning |
|---|---:|---|
| \(y_t^{(1)},y_t^{(2)}\) | scalars here | two noisy source-layer views of the same hidden scalar |
| \(c_{1,\tau},c_{2,\tau}\) | scalars | lag-\(\tau\) read-in weights |
| \(\alpha_\tau=c_{1,\tau}+c_{2,\tau}\) | scalar | effective lag-\(\tau\) weight |
| \(\alpha\) | \(\mathbb R^T\) | full lag profile |
| \(K_T\) | \(\mathbb R^{T\times T}\) | Toeplitz covariance matrix \([C_{|i-j|}]\) |
| \(\kappa_T\) | \(\mathbb R^T\) | covariance vector \((C_0,\dots,C_{T-1})^\top\) |

Take two noisy copies of the same centered hidden scalar:

\[ y_t^{(1)}=\xi_t+\varepsilon_t^{(1)},\qquad y_t^{(2)}=\xi_t+\varepsilon_t^{(2)}, \]

with independent Gaussian noises of variances \(\sigma_1^2,\sigma_2^2\). A window-\(T\) TXC forms

\[ h_t = \sum_{\tau=0}^{T-1} \big(c_{1,\tau}y_{t-\tau}^{(1)}+c_{2,\tau}y_{t-\tau}^{(2)}\big). \]

After eliminating the read-in split, the exact scalar lag-profile problem is

\[ \mathcal L_T(\alpha) = \frac{E_D}{2}\Big(C_0 - 2\kappa_T^\top\alpha + \alpha^\top K_T\alpha\Big) + \frac{r_{\mathrm{eff}}}{2}\|\alpha\|_2^2, \]

where

\[ r_{\mathrm{eff}} = \frac{r_1r_2}{r_1+r_2}, \qquad r_1=E_D\sigma_1^2+\lambda_{e,1}, \qquad r_2=E_D\sigma_2^2+\lambda_{e,2}, \qquad \gamma_{\mathrm{eff}}=\frac{r_{\mathrm{eff}}}{E_D}. \]

Therefore the exact optimum is

\[ \alpha_T^\star = (K_T+\gamma_{\mathrm{eff}}I_T)^{-1}\kappa_T. \]

Full-batch gradient descent is linear:

\[ \alpha_{n+1} = \alpha_n-\eta E_D\big((K_T+\gamma_{\mathrm{eff}}I_T)\alpha_n-\kappa_T\big). \]

And the minimal loss is

\[ \mathcal L_T^\star = \frac{E_D}{2}\Big(C_0-\kappa_T^\top(K_T+\gamma_{\mathrm{eff}}I_T)^{-1}\kappa_T\Big). \]

The gain from increasing the window is exact and nonnegative. Writing

\[ K_{T+1}+\gamma I_{T+1} = \begin{bmatrix} Q_T & b_T\\ b_T^\top & C_0+\gamma \end{bmatrix}, \]

we get the Schur-complement identity

\[ \mathcal L_T^\star-\mathcal L_{T+1}^\star = \frac{E_D}{2} \frac{\big(C_T-b_T^\top Q_T^{-1}\kappa_T\big)^2} {C_0+\gamma-b_T^\top Q_T^{-1}b_T} \ge 0. \]

So larger windows can only help.

#### Tiny numerical check

Using a reset process with

\[ \pi=0.2,\qquad \rho=0.7,\qquad \sigma_1=0.4,\qquad \sigma_2=0.8,\qquad E_D=2,\qquad \lambda_{e,1}=\lambda_{e,2}=0.1, \]

the theory and empirical ridge regression optimum agree:

| \(T\) | loss theory | loss experiment | \(\|\alpha_{\rm exp}-\alpha_{\rm th}\|_2\) |
|---:|---:|---:|---:|
| 1 | 0.080506 | 0.080442 | 0.000155 |
| 2 | 0.069216 | 0.069155 | 0.000851 |
| 4 | 0.067067 | 0.067009 | 0.001387 |

Selected lag weights:

| \(T\) | lag | \(\alpha_{\rm theory}\) | \(\alpha_{\rm experiment}\) | abs. error |
|---:|---:|---:|---:|---:|
| 2 | 0 | 0.429912 | 0.430243 | 0.000331 |
| 2 | 1 | 0.199948 | 0.199690 | 0.000258 |
| 4 | 0 | 0.416563 | 0.416725 | 0.000163 |
| 4 | 1 | 0.171256 | 0.171176 | 0.000080 |
| 4 | 2 | 0.071521 | 0.071443 | 0.000078 |
| 4 | 3 | 0.033723 | 0.033691 | 0.000032 |

![Window-\(T\) TXC: optimal lag profiles](./temporal_sae_txc_v4_assets/temporal_xc_windowT_kernel_profiles.png)

![Window-\(T\) TXC: full-batch GD trajectories](./temporal_sae_txc_v4_assets/temporal_xc_windowT_gd_trajectories.png)

### 2g. Takeaway at the “simple HMM” level

At this level, the HMM only adds **persistence**. It does **not** create new hidden directions.

So the story is:

- regular SAE: population-blind to persistence;
- Proposal 3A: learns a true temporal filter;
- simplest two-layer TXC: same temporal filter plus an exactly solvable layer-allocation problem;
- window-\(T\) TXC: exact Toeplitz Wiener filter, where bigger \(T\) gives monotone denoising gains.

This level is about **filtering** and **denoising**, not about uncovering genuinely new latent information.

---

## 3. Generalization to richer HMMs

Now we move beyond “same hidden feature, just more persistent.” The right new question is:

> when does time reveal hidden information that a single token cannot reveal?

### 3a. The right structural notion: observability rank

For a finite-state HMM or state-space model, the useful object is the \(T\)-step observability matrix

\[ \mathcal O_T = \begin{bmatrix} B\\ BP\\ \vdots\\ BP^{T-1} \end{bmatrix}, \]

where

- \(P\in\mathbb R^{K\times K}\) is the transition matrix on \(K\) hidden states;
- \(B\in\mathbb R^{d_{\rm obs}\times K}\) maps hidden states to one-token observations.

Interpretation:

- if \(\operatorname{rank}(\mathcal O_T)=\operatorname{rank}(B)\), then a longer window only denoises;
- if \(\operatorname{rank}(\mathcal O_T)>\operatorname{rank}(B)\), then time reveals new hidden directions.

That is the clean dividing line between “simple HMMs” and “richer HMMs.”

### 3b. Pedagogical minimal example: a locally aliased HMM

Take a 3-state HMM in which two hidden states look identical locally.

| object | shape | meaning |
|---|---:|---|
| hidden state \(z_t\) | one-hot vector in \(\mathbb R^3\) | true hidden state |
| \(P\) | \(3\times 3\) | transition matrix |
| \(B\) | \(2\times 3\) | one-token observation matrix |
| \(x_t=Bz_t\) | \(\mathbb R^2\) | observed token-level representation |

A concrete example is

\[ B= \begin{bmatrix} 1 & 1 & 0\\ 0 & 0 & 1 \end{bmatrix}, \qquad P= \begin{bmatrix} 0.1 & 0.1 & 0.8\\ 0.8 & 0.1 & 0.1\\ 0.1 & 0.1 & 0.8 \end{bmatrix}. \]

States 1 and 2 are **locally aliased** because they have the same one-token observation.

The state diagram is:

![Locally aliased HMM](./temporal_sae_txc_v4_assets/hmm_aliasing_observability.png)

But the temporal futures of states 1 and 2 differ. Indeed,

\[ \operatorname{rank}(B)=2,\qquad \operatorname{rank}(\mathcal O_2)=3. \]

So a two-token window reveals a hidden distinction that one token cannot.

This is the first benchmark where a windowed TXC is not “just a denoiser.” It can uncover a direction that is **not present in the single-token observation space**.

#### Tiny numerical check

The observability-rank toy behaves exactly as expected:

![Observability-rank toy example](./temporal_sae_txc_v4_assets/hmm_observability_rank_vs_T.png)

### 3c. Exact solution for Proposal 5: a shared-chain HMM with factorial SAE emissions

Proposal 5 is the first architecture matched to a genuinely shared temporal mode.

| object | shape | meaning |
|---|---:|---|
| \(h_t\) | \(\{1,\dots,\chi\}\) | shared hidden state |
| \(P\) | \(\mathbb R^{\chi\times\chi}\) | hidden-state transition matrix |
| \(B\) | \([0,1]^{m\times\chi}\) | emission probabilities, \(B_{k,h}=\Pr(s_{t,k}=1\mid h_t=h)\) |
| \(\gamma_t^{(T)}\) | simplex point in \(\Delta^{\chi-1}\) | posterior over hidden states from a window of length \(T\) |
| \(q_{t,k}^{(T)}\) | scalar | posterior support probability of feature \(k\) |
| \(r_k\) | scalar | aligned learned reconstruction gain for feature \(k\) |

State diagram:

![Shared hidden chain with factorial feature emissions](./temporal_sae_txc_v4_assets/hmm_shared_chain_factorial.png)

Conditioned on \(h_t\), the feature supports are independent Bernoullis. Marginal correlations appear because different features share the same hidden chain.

The posterior support probability of feature \(k\) is

\[ q_{t,k}^{(T)} = \Pr(s_{t,k}=1\mid \mathcal F_t^{(T)}) = \sum_{h=1}^{\chi} B_{k,h}\,\gamma_t^{(T)}(h). \]

In the aligned orthogonal basin, reconstruct with

\[ \hat x_t = \sum_{k=1}^m r_k q_{t,k}^{(T)} f_k. \]

Define the posterior-explained support energy

\[ Q_{k,T}:=\mathbb E[(q_{t,k}^{(T)})^2]. \]

Then the population loss decouples by feature:

\[ \mathcal L_{P5}(r) = \frac12\sum_k \pi_k - \sum_k Q_{k,T}r_k + \frac12\sum_k(Q_{k,T}+\lambda)r_k^2 + \lambda_s\sum_k \pi_k. \]

So full-batch gradient descent is exact:

\[ r_{k,n+1} = (1-\eta(Q_{k,T}+\lambda))r_{k,n}+\eta Q_{k,T}, \]

with fixed point

\[ r_{k,\star}^{(T)}=\frac{Q_{k,T}}{Q_{k,T}+\lambda}. \]

A very useful identity is

\[ Q_{k,T} = \pi_k-\mathbb E\!\left[\operatorname{Var}(s_{t,k}\mid \mathcal F_t^{(T)})\right]. \]

So \(Q_{k,T}\) is literally the amount of feature-\(k\) support variance explained by the temporal inference module.

As \(T\) grows, \(Q_{k,T}\) grows monotonically:

\[ Q_{k,T+1}-Q_{k,T} = \mathbb E\Big[\big(q_{t,k}^{(T+1)}-q_{t,k}^{(T)}\big)^2\Big]\ge 0. \]

#### Tiny numerical check

With a binary hidden chain, one noisy observation channel, and ridge \(\lambda=0.1\):

| \(T\) | \(Q_T\) | \(r_\star\) theory | \(r_{20}\) theory | \(r_{20}\) experiment | abs. error |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.339451 | 0.772443 | 0.767038 | 0.766916 | 0.000122 |
| 3 | 0.375350 | 0.789628 | 0.786159 | 0.786508 | 0.000349 |
| 5 | 0.380171 | 0.791741 | 0.788476 | 0.788765 | 0.000289 |

![Proposal 5: posterior energy rises with window size](./temporal_sae_txc_v4_assets/proposal5_posterior_energy_vs_T.png)

![Proposal 5: exact GD trajectories](./temporal_sae_txc_v4_assets/proposal5_gd_vs_T.png)

### 3d. Exact solution for the direct sum of two leaky-reset blocks

This is the first HMM family where a single temporal feature can correspond directly to a **belief-state coefficient**.

| object | shape | meaning |
|---|---:|---|
| block label \(c\) | \(\{1,2\}\) | which block generated the sequence |
| \(P_1,P_2\) | square matrices | within-block transition operators |
| \(P=P_1\oplus P_2\) | block-diagonal matrix | full transition operator |
| \(\eta_{1,t},\eta_{2,t}\) | simplex points | within-block belief states |
| \(\omega_{1,t},\omega_{2,t}\) | scalars | posterior block weights, \(\omega_{1,t}+\omega_{2,t}=1\) |
| \(\eta_t\) | direct-sum state | full belief state |
| \(U_t^{(T)}\) | integer | switch count in a length-\(T\) window |

State diagram:

![Two-block direct sum HMM](./temporal_sae_txc_v4_assets/hmm_direct_sum_two_block.png)

Because the two blocks do not mix,

\[ P=P_1\oplus P_2, \]

the belief state decomposes exactly as

\[ \eta_t=\omega_{1,t}\eta_{1,t}\oplus \omega_{2,t}\eta_{2,t}. \]

So the genuinely nonlocal hidden coordinate is the block posterior coefficient \(\omega_{1,t}\).

In the symmetric deterministic-emission benchmark, the exact posterior from a window depends only on the switch count

\[ U_t^{(T)} = \#\{\tau\in\{t-T+2,\dots,t\}:x_\tau\neq x_{\tau-1}\}. \]

The exact block posterior is logistic in this scalar sufficient statistic:

\[ \omega_{1,t}^{(T)} = \sigma\big(\beta_0(T)+\beta_1 U_t^{(T)}\big). \]

So one scalar temporal feature is enough to recover the belief-state coordinate up to an affine map or a sigmoid.

If the reconstruction target is linear in the block label, then the best **linear** TXC feature is the centered switch-count mode. Writing

- \(\sigma_{\mathrm w}^2\) for the within-block variance of that mode,
- \(\Delta\) for the between-block mean difference,
- \(\kappa\) for its covariance with the target,

full-batch GD is exact:

\[ \alpha_{n+1} = \big(1-\eta(\sigma_{\mathrm w}^2+n\Delta^2)\big)\alpha_n+\eta\kappa, \qquad n=T-1. \]

The useful window scale is

\[ T_{\mathrm{sig}}-1 \sim \frac{\sigma_{\mathrm w}^2}{\Delta^2}. \]

So the window required for reliable recovery grows like an inverse square signal-to-noise ratio.

#### Tiny numerical check

The exact posterior collapses cleanly onto switch count:

![Direct-sum benchmark: posterior versus switch count](./temporal_sae_txc_v4_assets/direct_sum_reset_posterior_vs_switchcount.png)

A linear rank-1 TXC then improves with \(T\):

![Direct-sum benchmark: best rank-1 linear recoverability versus \(T\)](./temporal_sae_txc_v4_assets/direct_sum_reset_linear_R2_vs_T.png)

And the full-batch GD trajectories follow the exact recurrence:

![Direct-sum benchmark: GD trajectories versus \(T\)](./temporal_sae_txc_v4_assets/direct_sum_reset_gd_vs_T.png)

### 3e. Exact solution for \(K\)-block direct sums

The two-block family generalizes in two qualitatively different ways.

| object | shape | meaning |
|---|---:|---|
| \(c\) | \(\{1,\dots,K\}\) | block label |
| \(\omega=(\omega_1,\dots,\omega_K)\) | simplex point in \(\Delta^{K-1}\) | posterior over blocks |
| \(U\) | scalar or vector | temporal sufficient statistic(s) |
| \(M\) | integer | number of independent temporal channels |

Representative state diagram:

![Representative \(K\)-block direct-sum family](./temporal_sae_txc_v4_assets/hmm_direct_sum_k_block.png)

#### One-parameter \(K\)-block family: still rank 1

If blocks differ only by a single persistence parameter \(p_i\), then the sufficient statistic is still the scalar switch count \(U\). The exact posterior is

\[ \omega_i(U)=\operatorname{softmax}_i(\alpha_i+\beta_i U). \]

Even though there are \(K\) blocks, the posterior simplex only traces out a **1D curve**. So one scalar TXC feature is still enough.

#### Genuine simplex family: need \(K-1\) independent temporal signatures

To force a true \((K-1)\)-dimensional posterior family, we need \(K-1\) independent temporal statistics. A matched construction is:

- \(M\) independent temporal channels,
- block-specific switch-probability vectors \(p_i\in(0,1)^M\),
- count vector \(U\in\mathbb R^M\) summarizing the window.

Then the posterior log-odds are affine in \(U\), and the informative linear rank is the affine rank of the set \(\{p_i\}\).

The smallest genuinely nontrivial case is

\[ K=3,\qquad M=2. \]

That is the first direct-sum benchmark where one temporal feature is provably insufficient and a 2D posterior simplex appears.

#### Tiny numerical checks

One-parameter family: posterior stays on a 1D curve.

![One-parameter \(K\)-block family: posterior curves](./temporal_sae_txc_v4_assets/kblock_oneparameter_posteriors.png)

Rank-1 versus genuine simplex family:

![Informative singular values](./temporal_sae_txc_v4_assets/kblock_informative_singular_values.png)

Useful window scaling:

![Useful window scaling with \(K\)](./temporal_sae_txc_v4_assets/kblock_window_scaling_vs_K.png)

### 3f. Takeaway at the “richer HMM” level

This level is qualitatively different from Section 2.

- **Aliased-state HMMs:** time reveals new hidden directions via observability rank.
- **Proposal 5:** the relevant quantity is posterior-explained support energy \(Q_{k,T}\), not raw firing rate.
- **Direct sums:** a single temporal feature can literally become a posterior coefficient \(\omega_i\).
- **\(K\)-block simplex families:** temporal width becomes structurally necessary only when the posterior family has affine rank \(K-1\).

So the right notion of “harder HMM” is **not just larger persistence \(\rho\)**. It is larger **temporal observability** or larger **posterior-simplex dimension**.

---

## 4. Continuous-state matched teachers: the Jordan-block family

This section is not an HMM, but it is still valuable because it gives the cleanest exact “time reveals hidden coordinates” benchmark.

### 4a. Linear-Gaussian predictive-state teacher

The matched continuous-state teacher is

| object | shape | meaning |
|---|---:|---|
| \(z_t\) | \(\mathbb R^r\) | hidden continuous state |
| \(A\) | \(\mathbb R^{r\times r}\) | state transition matrix |
| \(x_t^{(1)},x_t^{(2)}\) | \(\mathbb R^{d_1},\mathbb R^{d_2}\) | two source-layer observations |
| \(C_1,C_2\) | \(\mathbb R^{d_1\times r},\mathbb R^{d_2\times r}\) | source-layer readout matrices |
| \(y_t\) | \(\mathbb R^{d_y}\) | target to reconstruct |
| \(F\) | \(\mathbb R^{d_y\times r}\) | target readout matrix |

with dynamics

\[ z_{t+1}=Az_t+\xi_t,\qquad x_t^{(1)}=C_1z_t+\varepsilon_t^{(1)},\qquad x_t^{(2)}=C_2z_t+\varepsilon_t^{(2)},\qquad y_t=Fz_t+\zeta_t. \]

Because the model is jointly Gaussian, the Bayes-optimal MSE predictor from a window is exactly linear:

\[ \hat y_t^\star = \Sigma_{yY}\Sigma_{YY}^{-1}Y_t. \]

So a window-\(T\) temporal XC with sufficient width can realize the population optimum exactly.

### 4b. The \(r\)-dimensional Jordan block

The cleanest exact family is

\[ z_{t+1}=J_r(\rho)z_t, \qquad J_r(\rho)=\rho I+N, \]

where \(N\) is the nilpotent Jordan off-diagonal shift.

Take the simplest single-view observation

\[ s_t=e_1^\top z_t. \]

Then the hidden coordinates are weighted discrete derivatives of the observed stream:

\[ (E-\rho)^j s_t = z_t^{(j+1)}, \qquad j=0,1,\dots,r-1, \]

where \(E\) is the time-shift operator \(Es_t=s_{t+1}\).

So in the noiseless single-view setting, the minimum useful window is exactly

\[ T_{\min}=r. \]

This gives the clean single-view scaling law:

- hidden-state dimension \(r\) grows;
- required temporal window grows linearly with \(r\).

For a \(q\)-view observation model, the correct object is again the observability index. If \(q\) independent instantaneous views are available, then

\[ \lceil r/q\rceil \le T_{\min}\le r. \]

So multiple layers can reduce the required window only when they add genuinely independent instantaneous views.

### 4c. Can Proposal 3 help here?

This gives a clean answer to the earlier question about Proposal 3.

- **Proposal 3A** mostly cannot create the missing Jordan coordinates from a single local channel. It helps denoise, but it does not beat the observability bound.
- **Proposal 3B / a temporal crosscoder with several temporal drivers** can realize the optimal window-to-state map, because the relevant object is a low-rank predictive-state representation.
- To match the Jordan family exactly, Proposal 3B should really learn a cross-driver recurrence
  \[ g_{t+1}=Mg_t+Ua_t \]
  with \(M\in\mathbb R^{r\times r}\) allowed to approximate a Jordan or companion form.

So the tensor-network / temporal-driver idea helps here by giving the right **low predictive rank**, not by lowering the information-theoretic minimum window.

#### Tiny numerical checks

Window scaling with \(r\):

![Jordan window scaling versus hidden dimension](./temporal_sae_txc_v4_assets/jordan_window_scaling_vs_r.png)

Conditioning / observability degradation at larger \(r\):

![Jordan observability conditioning](./temporal_sae_txc_v4_assets/jordan_observability_conditioning.png)

### 4d. Takeaway

The Jordan family is the clean continuous analogue of the aliased-HMM story:

- single-token evidence hides some latent coordinates;
- time reveals them through discrete derivatives / observability;
- the minimum useful window scales like the observability index.

---

## 5. Synthesis: what each benchmark is testing

Here is the whole ladder in one table.

| level | benchmark | what is hidden | what the exact solution shows | what a temporal model can gain |
|---|---|---|---|---|
| 0 | static Chanin toy | nothing temporal; only static co-firing structure | wrong \(L0\) makes mixed features MSE-optimal | nothing temporal; this is the static baseline |
| 1 | leaky-reset HMM + regular SAE | only persistence | population loss depends on \(\pi\), not \(\rho\) | nothing at population level; only SGD-noise effects |
| 2 | Proposal 3A on the same HMM | persistence | optimal temporal filter \(\beta^\star\propto \rho(1-u^\star)/u^\star\) | true temporal filtering |
| 3 | simplest two-layer TXC | persistence + multiview read-in | exact harmonic-mean layer split | same filter, plus exactly solvable layer allocation |
| 4 | window-\(T\) TXC on simple HMM | persistence over many lags | exact Toeplitz Wiener filter | monotone denoising gains with \(T\) |
| 5 | aliased-state HMM | hidden state distinctions invisible in one token | observability rank can jump with \(T\) | genuinely new recoverable directions |
| 6 | Proposal 5 shared-chain HMM | shared temporal mode controlling many features | exact GD in terms of \(Q_{k,T}\) | posterior-informed feature ranking |
| 7 | two-block direct sum | block identity / belief coefficient | one temporal feature can equal \(\omega_1\) | recover a belief-state coefficient |
| 8 | \(K\)-block simplex family | \((K-1)\)-dimensional posterior simplex | width \(K-1\) becomes structurally necessary | temporal width matters for structural reasons |
| 9 | Jordan block | hidden continuous coordinates | \(T_{\min}\) set by observability index | exact matched teacher for TXCs |

### What I think is genuinely solved

In the aligned / orthogonal basin, these pieces are analytically clean:

- static Chanin two-feature low-\(L0\) benchmark,
- regular SAE on reset/HMM supports,
- Proposal 3A one-feature temporal filter,
- simplest two-layer TXC,
- window-\(T\) TXC on a single persistent hidden scalar,
- aliased-state / observability benchmark,
- Proposal 5 gain dynamics via \(Q_{k,T}\),
- two-block and \(K\)-block direct sums,
- Jordan-block teacher.

### What is still genuinely open

The first really hard setting is where **component assignment itself** changes because of time, not just denoising or gain splitting:

- multiple ambiguous true features,
- finite latent budget,
- non-orthogonal / superposed decoder geometry,
- ReLU or TopK basin changes,
- several competing temporal explanations.

That is the point where a temporal architecture could discover **different** features, not just cleaner estimates of the same feature.

---

## 6. Practical reading guide

If you want the shortest path through the note, I think the clean order is:

1. **Section 1:** understand the static Chanin low-\(L0\) effect.  
2. **Section 2c–2f:** see exactly why regular SAEs are blind to persistence, while Proposal 3A and TXCs are not.  
3. **Section 3b:** see the first benchmark where time reveals a new hidden direction.  
4. **Section 3d–3e:** see the first benchmarks where a TXC feature is naturally a posterior coordinate \(\omega_i\).  
5. **Section 4:** use Jordan blocks when you want the sharpest scaling law for \(T\) versus hidden-state complexity.

---

## 7. Source map

This note consolidates the earlier working notes and the user-provided source notes:

- the temporal architecture note,
- the modular-addition dynamics note,
- the reset-process / HMM note,
- and the derived notes on static Chanin, regular SAE, Proposal 3A, simplest TXC, window-\(T\) TXC, Proposal 5, direct sums, \(K\)-block families, and Jordan blocks.

The main change here is stylistic rather than mathematical: the benchmark ladder is now explicit, and every object is introduced with its dimension and role before it is used.
