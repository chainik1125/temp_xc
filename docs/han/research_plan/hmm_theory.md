---
title: "The Reset Process: A Framework for HMMs with Controlled Temporal Persistence"
author: Han Xuanyuan
date: 2026-02-16
type: document
---


## 1. Background: The Standard Two-State HMM

A hidden Markov model (HMM) consists of a **hidden state** $z_t$ that evolves over time according to a Markov chain, and an **emission** $x_t$ that is produced at each time step as a function of the current hidden state. The two defining components are:

1. A **transition model** $P(z_{t+1}\mid z_t)$, specifying how the hidden state evolves.
2. An **emission model** $P(x_t\mid z_t)$, specifying how observations are generated from the hidden state.

We will work throughout with two hidden states ${A,B}$ and two possible emissions ${0,1}$, with the convention (used later) that “active” corresponds to state $A$ and emission $1$, while “inactive” corresponds to state $B$ and emission $0$.

### 1.1 Transition model

Write

* $p_A = P(z_{t+1}=A \mid z_t=A)$ for the probability of staying in state $A$,
* $p_B = P(z_{t+1}=B \mid z_t=B)$ for the probability of staying in state $B$.

The transition matrix is
$$
\mathbf{T}=
\begin{pmatrix}
p_A & 1-p_A\\
1-p_B & p_B
\end{pmatrix},
$$
where rows correspond to the current state and columns to the next state, ordered $(A,B)$.

If the chain is ergodic (in particular, not trapped in absorbing classes), the stationary distribution $\pi$ satisfies $\pi \mathbf{T}=\pi$ and is
$$
\pi_A=\frac{1-p_B}{2-p_A-p_B},\qquad
\pi_B=\frac{1-p_A}{2-p_A-p_B}.
$$
(When the chain is not ergodic, there may be multiple stationary distributions, and the long-run behavior depends on initialization.)

### 1.2 Emission model

The emission model specifies $P(x_t\mid z_t)$. We will consider two cases: deterministic and non-deterministic emissions.

## 2. Example 1: Deterministic Emissions

Suppose state $A$ always emits $1$ and state $B$ always emits $0$:
$$
P(x_t=1\mid z_t=A)=1,\qquad P(x_t=1\mid z_t=B)=0.
$$

Then the emission is a deterministic function of the state:
$$
x_t=\mathbf{1}[z_t=A].
$$
So the emission sequence $x_1,x_2,\ldots$ is a relabeling of the state sequence $z_1,z_2,\ldots$.

### 2.1 Lag-1 autocorrelation

Let $\mu=\mathbb{E}[x_t]$ under stationarity. Since $x_t=\mathbf{1}[z_t=A]$, we have $\mu=\pi_A$. Also
$$
\mathbb{E}[x_t x_{t+1}]
= P(z_t=A,z_{t+1}=A)
= \pi_A,p_A.
$$

Using $\rho = \frac{\mathbb{E}[x_t x_{t+1}]-\mu^2}{\mathrm{Var}(x_t)}$ and $\mathrm{Var}(x_t)=\mu(1-\mu)$, one obtains
$$
\rho = p_A+p_B-1.
$$

Interpretation:

* If $p_A$ and $p_B$ are both close to $1$, the chain rarely transitions and the emissions are highly autocorrelated ($\rho\approx 1$).
* If $p_A=1-p_B$, then $P(z_{t+1}=A\mid z_t)$ is the same for both current states, so $(z_t)$ (and hence $(x_t)$) is i.i.d., and $\rho=0$.


## 3. Example 2: Non-Deterministic Emissions

Now suppose emissions are stochastic given the state:
$$
P(x_t=1\mid z_t=A)=q_A,\qquad
P(x_t=1\mid z_t=B)=q_B,
$$
for some $q_A,q_B\in[0,1]$.

Let $\mu=P(x_t=1)$ under stationarity:
$$
\mu = \pi_A q_A + \pi_B q_B.
$$

### 3.1 Emission autocorrelation

The lag-1 autocorrelation of the emission process factors into a “state persistence” term and an “emission distinguishability” term:
$$
\rho_{\text{emission}}
=
\rho_{\text{state}}\cdot
\frac{\pi_A\pi_B (q_A-q_B)^2}{\mu(1-\mu)},
\qquad
\rho_{\text{state}}=p_A+p_B-1.
$$

Equivalently,
$$
\frac{\pi_A\pi_B (q_A-q_B)^2}{\mu(1-\mu)}
=
\frac{\mathrm{Var}(\mathbb{E}[x_t\mid z_t])}{\mathrm{Var}(x_t)},
$$
so it lies in $[0,1]$ and quantifies how much of the emission variance is explained by the hidden state.

Consequences:

* If emissions are deterministic ($q_A=1,q_B=0$), the factor becomes $1$ and $\rho_{\text{emission}}=\rho_{\text{state}}$.
* If emissions are identical ($q_A=q_B$), the factor is $0$ and the emissions are i.i.d. even if the hidden chain is highly persistent.

### 3.2 Subtlety: the “frozen chain” limit

If $p_A=p_B=1$, then $\rho_{\text{state}}=1$ and the hidden state never changes. Conditioned on the (random) initial state, $(x_t)$ is i.i.d., but **marginally** (averaging over which state you are frozen in) consecutive emissions are dependent because they share the same latent variable. The expression above still matches that marginal dependence if you interpret $\pi$ as the mixing distribution over the absorbing states rather than as a unique stationary distribution.


## 4. A Convenient Reparametrization: Sparsity and Autocorrelation

In the deterministic-emission case ($q_A=1,q_B=0$), we have

* $\pi_1=P(x_t=1)=\pi_A$ (marginal activation probability),
* $\rho=p_A+p_B-1$ (lag-1 autocorrelation).

Inverting these relationships yields:
$$
p_A = 1-(1-\pi_1)(1-\rho),\qquad
p_B = 1-\pi_1(1-\rho).
$$

It is often useful to work with the “off $\to$ on” and “on $\to$ off” transition probabilities. With the convention that “off” corresponds to state $B$ (emission $0$) and “on” corresponds to state $A$ (emission $1$), define
$$
p_{01} = P(z_{t+1}=A \mid z_t=B)=1-p_B,\qquad
p_{10} = P(z_{t+1}=B \mid z_t=A)=1-p_A.
$$
Then
$$
p_{01}=\pi_1(1-\rho),\qquad
p_{10}=(1-\pi_1)(1-\rho).
$$

This gives “orthogonal” control in the deterministic case:

* $\pi_1$ controls sparsity (how often $1$ appears),
* $\rho$ controls persistence (how sticky the chain is),
  subject to the constraints $p_{01},p_{10}\in[0,1]$.


## 5. The Reset Process Formulation

The reset process is an alternative way to parametrize Markov chains that makes persistence control especially transparent.

### 5.1 Core idea

At each time step:

* With probability $1-\lambda$: **stay** in the current state.
* With probability $\lambda$: **reset**—jump to a state drawn from a fixed target distribution, independent of the current state.

Here $\lambda\in[0,1]$ controls how frequently resets occur.

### 5.2 Formal definition

Let the state space have $K$ states and let $r\in\mathbb{R}^K$ be a target distribution on the simplex (so $r_i\ge 0$ and $\sum_i r_i=1$). Define the reset matrix $R$ as the $K\times K$ matrix with every row equal to $r$:
$$
R=
\begin{pmatrix}
\text{--- }  r \text{ ---}\\
\text{--- }  r  \text{ ---}\\
\vdots
\end{pmatrix}.
$$

The reset-process transition matrix is
$$
T = (1-\lambda)I + \lambda R.
$$

Key property: for any probability row vector $\eta$ (so $\sum_i \eta_i=1$),
$$
\eta R = r,
$$
because $R$ replaces the current distribution by the target distribution.

### 5.3 Properties

**Stationary distribution (state-independent reset).** If $\lambda>0$, the chain is ergodic (assuming $r$ has full support on the relevant communicating class) and
$$
\pi = r.
$$
(If $\lambda=0$, the chain never moves and every distribution concentrated on a single state is stationary.)

**Autocorrelation (two-state, deterministic emissions).** Specialize to $K=2$ with states ordered $(A,B)$ and set the reset target to match the desired marginal activation:
$$
r = (\pi_1,1-\pi_1).
$$
In the deterministic-emission case ($A\mapsto 1$, $B\mapsto 0$), the transition probabilities are
$$
P(B\to A)=\lambda \pi_1,\qquad P(A\to B)=\lambda(1-\pi_1).
$$
Hence
$$
\rho = 1 - P(B\to A) - P(A\to B) = 1-\lambda.
$$
So $\lambda$ directly controls lag-1 autocorrelation: $\lambda=0$ gives $\rho=1$ (perfect persistence), $\lambda=1$ gives $\rho=0$ (i.i.d.), and intermediate values interpolate linearly.

**Equivalence to Section 4.** Setting $\rho=1-\lambda$ gives
$$
p_{01}=\lambda \pi_1=\pi_1(1-\rho),\qquad
p_{10}=\lambda(1-\pi_1)=(1-\pi_1)(1-\rho),
$$
which exactly matches the reparametrization in Section 4.

### 5.4 Example: two-state chain with $\pi_1=0.1$, $\lambda=0.2$

Take $r=(0.1,0.9)$ in the $(A,B)$ ordering. Then
$$
T
=

0.8
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}
+
0.2
\begin{pmatrix}
0.1&0.9\\
0.1&0.9
\end{pmatrix}
=

\begin{pmatrix}
0.82&0.18\\
0.02&0.98
\end{pmatrix}.
$$
The stationary probability of state $A$ (and hence of emission $1$ in the deterministic case) is $\pi_1=0.1$, and the lag-1 autocorrelation is $\rho=1-\lambda=0.8$.

---

## 6. Emission-Dependent Transitions

The standard HMM assumes $P(z_{t+1}\mid z_t)$. A natural generalization is to allow
$$
P(z_{t+1}\mid z_t,x_t),
$$
so the transition depends on the current emission.

The reset-process idea extends by allowing a different reset target for each emission value $x\in\mathcal{X}$. For each $x$, define a target distribution $r_{\mathcal{S},x}$ and corresponding reset matrix $R_{\mathcal{S},x}$ (each row equal to $r_{\mathcal{S},x}$). Define
$$
T(x) = (1-\lambda)I + \lambda R_{\mathcal{S},x}.
$$

A generative step is then:

1. Given $z_t$, sample $x_t\sim P(x_t\mid z_t)$.
2. Transition $z_{t+1}\sim T(x_t)[z_t,\cdot]$.

When $r_{\mathcal{S},x}$ is the same for all $x$, this reduces to the emission-independent reset process from Section 5.

(With emission-dependent targets, the stationary distribution of $z_t$ is generally not equal to any single $r_{\mathcal{S},x}$; it depends on both emissions and transitions.)

---

## 7. Belief Updates and the Forward Algorithm

So far, we have described generative dynamics of $(z_t,x_t)$. An observer who sees only the emissions and wants to infer the hidden state maintains a belief state. It is useful to distinguish:

* **Predictive prior** (before seeing $x_t$):
  $$
  \eta_t(i)=P(z_t=i\mid x_{1:t-1}).
  $$

* **Filtered posterior** (after seeing $x_t$):
  $$
  \gamma_t(i)=P(z_t=i\mid x_{1:t}).
  $$

### 7.1 Exact Bayesian update (forward algorithm)

Given an emission model $e_x(i)=P(x_t=x\mid z_t=i)$, the Bayesian correction step is
$$
\gamma_t(i)=\frac{\eta_t(i),e_{x_t}(i)}{\sum_j \eta_t(j),e_{x_t}(j)}.
$$

Then, using the emission-dependent transition matrix $T(x_t)$,
$$
\eta_{t+1} = \gamma_t,T(x_t).
$$

This is the standard forward recursion adapted to transitions that depend on the observed emission.

### 7.2 Specialization to reset-form transitions

If
$$
T(x)=(1-\lambda)I+\lambda R_{\mathcal{S},x},
$$
then for any row probability vector $v$ we have $vT(x)=(1-\lambda)v+\lambda r_{\mathcal{S},x}$. Applying this to $v=\gamma_t$ gives an especially simple predictive update:
$$
\eta_{t+1}
=

(1-\lambda)\gamma_t+\lambda r_{\mathcal{S},x_t}.
$$
This is exact, but note the dependence on the **posterior** $\gamma_t$, not the prior $\eta_t$.

### 7.3 When does the “no-Bayes-step” affine update appear?

You sometimes see an update of the form
$$
\eta_{t+1}=(1-\lambda)\eta_t+\lambda r_{\mathcal{S},x_t}.
$$
This is not, in general, the exact Bayesian filtering recursion when $\eta_t$ is defined as $P(z_t\mid x_{1:t-1})$. It can arise exactly in either of the following situations:

1. **Deterministic emissions (state revealed).** If $x_t$ determines $z_t$ (as in Section 2), then $\gamma_t$ becomes a vertex distribution, and the update reduces to a deterministic function of $x_t$ anyway.

2. **“Emissions folded into transitions” (joint operator form).** If one works with joint matrices
   $$
   \tilde T(x)_{ij} = P(z_{t+1}=j,x_t=x\mid z_t=i),
   $$
   then an unnormalized forward message satisfies
   $$
   \tilde \eta_{t+1} = \tilde \eta_t\tilde T(x_t),
   $$
   followed by normalization if you want a probability vector.

Outside such cases, omitting the Bayes correction typically yields an approximation rather than the exact filter.

---

## 8. The Continuous State Interpretation

The mathematics of Section 7 admits a second interpretation that goes beyond the standard discrete-state HMM. Instead of treating $\eta_t$ as an observer’s belief over a discrete hidden state, we can treat $\eta_t$ as the **actual state** of the system: a point on the simplex.

Under this interpretation:

* The state $\eta_t$ lives on the simplex $\Delta^{K-1}$ (not at a vertex).
* The emission $x_t$ is sampled stochastically from $\eta_t$, e.g. for binary emissions one could take $P(x_t=1)=\eta_{t,A}$.
* The state then updates deterministically via a reset-form rule driven by the emitted symbol:
  $$
  \eta_{t+1}=(1-\lambda)\eta_t+\lambda r_{\mathcal{S},x_t}.
  $$

This is no longer a standard HMM; it is a deterministic dynamical system on the simplex with stochastic emissions.

### 8.1 Key differences from the discrete model

In the **discrete model** (standard HMM), the hidden state $z_t$ is always one of the $K$ basis states, and transitions are stochastic. Emissions may be deterministic or stochastic given $z_t$. In the deterministic-emission case, state persistence directly creates emission autocorrelation.

In the **continuous model**, the state update is deterministic given the emission, and stochasticity enters through emission sampling. The state typically lives in the simplex interior.

### 8.2 Autocorrelation behavior in the continuous model

The relationship between $\lambda$ and emission autocorrelation can **reverse** relative to the discrete deterministic-emission case:

* Continuous model with $\lambda=0$: $\eta_t$ never changes, so emissions are i.i.d. draws from a fixed distribution, giving $\rho=0$.

* Continuous model with $\lambda=1$: $\eta_{t+1}=r_{\mathcal{S},x_t}$, so emissions form a first-order Markov chain. In the binary case,
  $$
  P(x_{t+1}=1\mid x_t=x)=\big(r_{\mathcal{S},x}\big)_A,
  $$
  which typically yields nonzero autocorrelation when $r_{\mathcal{S},0}\neq r_{\mathcal{S},1}$.

This reversal occurs because the two models place the source of randomness in different places: in the discrete model, randomness is in state transitions; in the continuous model, randomness is in emissions, and the deterministic state update feeds back into future emission probabilities.

### 8.3 When the continuous model is useful

For generating binary feature supports (feature on/off each time step), the discrete Markov chain model is the natural choice. The continuous model is useful when the “hidden state” is genuinely graded—e.g., a continuously evolving context vector that drifts in response to each token. The reset-process framework supports both interpretations; the difference is whether the system’s state is sampled to a vertex each step (discrete) or allowed to remain in the simplex interior (continuous).
