---
author: Dmitry
date: 2026-05-06
tags:
  - reference
  - in-progress
---

## Setup B HMM — noisy independent emissions, in computational-mechanics presentation

We follow the convention of Crutchfield and collaborators
\[crutchfield1989inferring; crutchfield2012between; shai2024transformers\]
and present Setup B as an **edge-labelled HMM**

$$
\mathcal H \;=\; \big(\mathcal X,\; \mathcal S,\; \boldsymbol{\eta}^{\emptyset},\; \{T^{(x)}\}_{x \in \mathcal X}\big),
$$

where $\mathcal X$ is the observed alphabet, $\mathcal S$ is the latent
state space, $\boldsymbol{\eta}^{\emptyset} \in \Delta(\mathcal S)$ is
the initial state distribution, and the symbol-labelled transition
matrices $T^{(x)}$ have entries

$$
T^{(x)}_{s, s'} \;=\; \Pr\!\big(s_{t+1} = s',\; x_t = x \;\big|\; s_t = s\big).
$$

Setup B is a **factorised** HMM: $N = 20$ mutually independent
two-state chains. We therefore describe the per-feature HMM
$\mathcal H_i$ first; the joint bench is the Kronecker product
$\mathcal H = \bigotimes_{i=1}^{N} \mathcal H_i$.

## Parameters (per-feature)

From `configs/datasources.yaml::toy_markov_n20_d40_noisy`:

| symbol | meaning | value |
|---|---|---|
| $\rho$ | self-transition correlation (uniform across features) | 0.7 |
| $\pi$ | stationary on-rate | 0.5 |
| $p_A$ | $\Pr(x = 1 \mid s = 0)$ — false-alarm rate | 0 |
| $p_B$ | $\Pr(x = 1 \mid s = 1)$ — hit rate | 0.625 |
| $\gamma$ | noise level $(1-p_B)/p_B$ | 0.25 |
| $T_{\text{seq}}$ | sequence length | 64 |
| $d$ | residual-stream dim | 40 |
| $\{\mathbf f_i\}_{i=1}^N$ | feature directions in $\mathbb R^d$, mutually orthogonal | random QR |
| $m_{i,t}$ | per-token magnitude | $\lvert\mathcal N(1, 0.15^2)\rvert$ |

## Per-feature HMM $\mathcal H_i$

- **Latent state space**: $\mathcal S_i = \{0, 1\}$ (binary on/off).
- **Observed alphabet**: $\mathcal X_i = \{0, 1\}$ (firing indicator).
- **Initial state**: $\boldsymbol{\eta}^{\emptyset}_i = (1-\pi,\, \pi) = (0.5,\, 0.5)$.
  This is the stationary distribution, so the chain is stationary
  from $t = 0$.

The latent-state transition matrix (rows index $s_t$, columns index
$s_{t+1}$) is

$$
P_h \;=\;
\begin{pmatrix}
1 - \pi(1-\rho) & \pi(1-\rho) \\[2pt]
(1 - \pi)(1-\rho) & 1 - (1-\pi)(1-\rho)
\end{pmatrix}
\;=\;
\begin{pmatrix}
0.85 & 0.15 \\
0.15 & 0.85
\end{pmatrix},
$$

and the emission matrix (rows index $s_t$, columns index $x_t$) is

$$
B \;=\;
\begin{pmatrix}
1 - p_A & p_A \\
1 - p_B & p_B
\end{pmatrix}
\;=\;
\begin{pmatrix}
1 & 0 \\
0.375 & 0.625
\end{pmatrix}.
$$

The symbol-labelled transition matrices are the elementwise product
$T^{(x)}_{s, s'} = B[s, x] \cdot P_h[s, s']$:

$$
T^{(x = 0)} \;=\;
\begin{pmatrix}
(1 - p_A)\,(1 - \pi(1-\rho)) & (1 - p_A)\,\pi(1-\rho) \\[2pt]
(1 - p_B)\,(1 - \pi)(1-\rho) & (1 - p_B)\,(1 - (1-\pi)(1-\rho))
\end{pmatrix}
\;=\;
\begin{pmatrix}
0.85 & 0.15 \\[2pt]
0.05625 & 0.31875
\end{pmatrix},
$$

$$
T^{(x = 1)} \;=\;
\begin{pmatrix}
p_A\,(1 - \pi(1-\rho)) & p_A\,\pi(1-\rho) \\[2pt]
p_B\,(1 - \pi)(1-\rho) & p_B\,(1 - (1-\pi)(1-\rho))
\end{pmatrix}
\;=\;
\begin{pmatrix}
0 & 0 \\[2pt]
0.09375 & 0.53125
\end{pmatrix}.
$$

By construction $T^{(x=0)} + T^{(x=1)} = P_h$, and each row of $P_h$
sums to one. The total transition operator
$T = \sum_{x} T^{(x)} = P_h$ governs the marginal evolution of the
hidden state, while the split into $T^{(0)}$ and $T^{(1)}$ encodes the
emission stochasticity.

### Edge-labelled state diagram

![Per-feature edge-labelled HMM for Setup B](../../figs/c2/setup_b_hmm.png)

Each directed edge carries the label $x \,|\, T^{(x)}_{s, s'}$. The
$h{=}0$ self-loop has only the $x{=}0$ channel because $p_A = 0$
makes false alarms impossible; the $h{=}1$ self-loop and the $1 \to 0$
transition both split into two parallel channels (one per emission
symbol) because emissions from the on-state are stochastic.

The TikZ source for the figure lives at `images/setup_b_hmm.tikz`
and is included by `\input{images/setup_b_hmm}` in the paper.

### Sequence likelihood

For an observed firing sequence $\mathbf x_{1:T_{\text{seq}}} \in \{0,1\}^{T_{\text{seq}}}$ on a
single chain, the comp-mech likelihood follows the standard
edge-labelled HMM presentation:

$$
\Pr\!\big(\mathbf x_{1:T_{\text{seq}}}\big)
\;=\;
\boldsymbol{\eta}^{\emptyset \top}\,
T^{(x_1)}\,
T^{(x_2)}\,
\cdots\,
T^{(x_{T_{\text{seq}}})}\,
\mathbf 1,
$$

where $\mathbf 1$ is the all-ones column vector over $\mathcal S_i$.

## Joint HMM

The 20 chains are mutually independent, so the joint Setup-B HMM is

$$
\mathcal H \;=\; \bigotimes_{i=1}^{N} \mathcal H_i,
\qquad
\mathcal S = \{0, 1\}^{20},\;\mathcal X = \{0, 1\}^{20},
$$

with joint symbol-labelled transition matrices

$$
T^{(\mathbf x)} \;=\; \bigotimes_{i=1}^{N} T_i^{(x_i)},
\qquad \mathbf x = (x_1, \ldots, x_N) \in \{0,1\}^{20}.
$$

The hidden state space has cardinality $|\mathcal S| = 2^{20} \approx 10^6$
and the emission alphabet has the same size. The factorisation makes
this tractable: any quantity that decomposes additively across chains
(activations $\mathbf x(t) = \sum_i s_i(t)\,\lvert m_{i,t}\rvert\,\mathbf f_i$,
log-likelihoods, sufficient statistics) can be computed at per-chain
cost rather than at $|\mathcal S|^2$ cost.

## Activation construction

Activations are built from the **noisy** observations $\mathbf x_t = (x_{1,t}, \ldots, x_{N,t})$,
not the hidden states $\mathbf s_t$:

$$
\mathbf a(t) \;=\; \sum_{i=1}^{N} x_{i, t}\,\lvert m_{i, t}\rvert\,\mathbf f_i,
\qquad \mathbf a(t) \in \mathbb R^{d}.
$$

(We use $\mathbf a$ for the residual-stream activation here to avoid
clashing with the comp-mech symbol $x$ for the emitted symbol.)
$\mathbf a(t)$ contains *no* information about $s_{i, t}$ when
$x_{i, t} = 0$; the only way an autoencoder can recover the hidden
state is by aggregating across positions of the same chain.

## Why this is the cleanest denoising bench

Setup B has a **single** set of feature directions
$\{\mathbf f_i\}_{i=1}^N$: both the noisy observation $x_i \in \{0,1\}$
and the clean hidden state $s_i \in \{0, 1\}$ project onto the same
direction $\mathbf f_i$. So the geometric AUC against "emission
features" and against "hidden features" is the same number by
construction — the eAUC vs gAUC distinction collapses.

The discriminating signal lives at the **latent code level**: does an
encoder activation $z_j(t)$ correlate more strongly with $x_i(t)$
(noisy observation) or with $s_i(t)$ (clean hidden state)?

- $\bar r_{\rm local}(j, i) = \mathrm{Corr}\!\big(z_j(t), x_i(t)\big)$,
  averaged over the best matching latent $j$ for each $i$.
- $\bar r_{\rm global}(j, i) = \mathrm{Corr}\!\big(z_j(t), s_i(t)\big)$,
  averaged the same way.

Setup B plots $(\bar r_{\rm local}, \bar r_{\rm global})$ as a scatter
across (architecture, $T$, $k_{\rm pos}$) cells. Points on the
diagonal $y = x$ track the noisy observation as well as the hidden
state ("no denoising"); points above $y = x$ track the hidden state
better than the observation ("denoising"). The per-token denoising
floor is $\sqrt{p_B} \approx 0.79$, achieved when $z_j$ tracks the
noisy observation and the noisy observation is the best per-token
estimator of the hidden state.

## Empirical headline (from c2.md)

Decoder AUC vs $k_{\rm pos}$, mean ± std over 3 seeds:

- **TopK SAE**: 0.39 at $k = 1$, peaks 0.99 only at $k = 10$ — needs
  high per-token capacity to recover all 20 directions because each
  observation is sparse.
- **TXC-base $T = 5$**: 0.93 at $k = 1$, saturates 0.99 at $k \geq 3$.
- **TXC-base $T = 12$**: 0.98 at $k = 1$ — windowing is doing the
  heavy lifting, not per-token capacity.

The denoising effect strengthens monotonically with $T$ in the
single-latent and probe scatters: TXC-base reaches the diagonal at
$T = 2$ (partial denoising) and crosses above at $T \geq 4$ (full
denoising), confirming temporal aggregation as the mechanism rather
than parameter count.

## TikZ source (also at `images/setup_b_hmm.tikz`)

```latex
\begin{tikzpicture}[
    >=Latex,
    node distance=4.2cm,
    state/.style={
        circle, draw, thick, minimum size=1.15cm,
        font=\normalsize, inner sep=0pt
    },
    edge_lab/.style={font=\scriptsize, align=center, fill=white, inner sep=1pt},
    every loop/.style={looseness=8, draw, thick, ->},
    transition/.style={draw, thick, ->},
]
    \node[state, fill=gray!10] (off) {$h{=}0$};
    \node[state, fill=blue!12, right=of off] (on) {$h{=}1$};

    \path[transition] (off) edge[loop left]
        node[edge_lab] {$0\,|\,0.85$} (off);
    \path[transition] (on) edge[loop right]
        node[edge_lab] {$0\,|\,0.319$ \\ $1\,|\,0.531$} (on);

    \path[transition] (off) edge[bend left=20]
        node[edge_lab, above] {$0\,|\,0.15$} (on);
    \path[transition] (on) edge[bend left=20]
        node[edge_lab, below] {$0\,|\,0.056$ \\ $1\,|\,0.094$} (off);
\end{tikzpicture}
```

## References

- Generator: `purified/src/temp_bench/data/toy/markov.py`.
- Datasource config: `purified/configs/datasources.yaml::toy_markov_n20_d40_noisy`.
- Provenance: `docs/han/research_logs/2026-03-30-experiment1c-noisy-emissions.md`
  at commit `118fde8` (wasteland Phase 2 Experiment 1c noisy-emissions).
- Component writeup: `purified/docs/components/c2.md` § "Setup B".
- Comp-mech HMM presentation conventions:
  Crutchfield 1989; Crutchfield 2012; Shai et al. 2024.
- Plots (mirrored locally to `temp_xc_tex/figs/c2/`):
  `c2_noisy_auc_vs_kpos.png`, `c2_noisy_singlelatent_scatter.png`,
  `c2_noisy_probe_scatter.png`, `c2_noisy_denoising_panels.png`.
