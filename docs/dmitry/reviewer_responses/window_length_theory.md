---
author: Dmitry
date: 2026-07-23
tags:
  - design
  - in-progress
---

## When does performance *provably* improve with window length?

A theory note for the reviewer response. The reviewers' central complaint is
that the paper never shows temporal aggregation (cross-position weight sharing)
is *responsible* for the gains, and one reviewer notes that "changing the
temporal window length has almost no effect on performance" in sparse probing.
This note sets out the theoretical frame we can stand on: there are exactly two
mechanisms by which window length improves recovery, they produce two
characteristic curve shapes, and we already have three tasks — one in the paper,
two run in sprints — that instantiate them. The polynomial clock in particular
turns window length into a *tunable, theorem-backed* causal axis and, as a
by-product, delivers the Stacked-SAE isolation the reviewers demanded.

### The one-line frame

Window length can only help through one of two routes:

1. **Identifiability threshold (a floor).** Below a critical window $W^\star$ the
   target is information-theoretically invisible — $I(Y; \mathbf{x}_{1:W}) = 0$ —
   so *every* architecture is pinned at chance. This is the "zero before the
   threshold."
2. **Evidence integration (a slope).** Above threshold the target is present but
   noisy; more observations average the noise down, so error keeps falling with
   $W$. This is the "slowly improving after."

A task that combines both — a hard floor, then a slow climb — is the shape the
clock experiments show. Everything below makes these two mechanisms precise and
maps our tasks onto them.

### Scope: synthetic observations versus transformer residuals

The impossibility statements apply to the observations supplied to the
dictionary learner. In the synthetic clock, a single observation
$\mathbf{x}_t$ is generated directly from $Q_t$, so the Markov relation
$Y \rightarrow Q_t \rightarrow \mathbf{x}_t$ makes the single-position
information claim exact. A transformer residual is different: at layer
$\ell$, attention has already made
$\mathbf{r}^{(\ell)}_t = F_\ell(\mathbf{x}_{\leq t})$, so a single residual
position can contain information integrated from many earlier tokens. Once its
effective receptive field crosses the clock threshold, the per-position
impossibility proof no longer applies.

This suggests a depth-dependent trade-off between **information availability**
and **information accessibility**. Early residuals may expose relatively
legible ingredients that remain distributed across positions; later residuals
contain more of the model's contextual computation, but the resulting
representation may be compressed into an entangled or otherwise non-SAE-friendly
basis. These are distinct properties:

$$
\text{information present}
\;\not\Rightarrow\;
\text{linearly decodable}
\;\not\Rightarrow\;
\text{sparsely and interpretably represented}.
$$

The expected real-model signature is therefore a layer-by-window trade-off:
the window required for prediction should shrink with depth as attention
integrates context, while sparse interpretability need not improve and may peak
at an intermediate layer. TXCs are most useful in the intermediate regime,
where predictive evidence is available across positions but has not yet become
a locally accessible sparse feature. The synthetic experiments deliberately
remove the upstream model so that the temporal information boundary itself can
be isolated.

### Mechanism 1 — the identifiability threshold

**General statement.** Let label $Y$ (uniform on an alphabet of size $q$) and a
length-$W$ window $\mathbf{x}_{1:W}$ be jointly distributed. Suppose there is a
critical $W^\star$ with $I(Y; \mathbf{x}_{1:W}) = 0$ for $W < W^\star$. Then for
any decoder — SAE, TXC, MLP, anything — the Bayes accuracy for $Y$ is exactly
chance $1/q$ below $W^\star$. By the data-processing inequality, any learned
representation $\mathbf{z} = \phi(\mathbf{x}_{1:W})$ also has $I(Y;\mathbf{z}) = 0$,
so the floor is **architecture-independent**. This is the strongest possible
statement of "window length is causally necessary": it is not that more context
helps, it is that less context makes the task *impossible*.

**Instance A — the polynomial clock (exact algebraic threshold).**
Over a prime field $\mathbb{F}_q$, sample coefficients $B_0,\dots,B_{h-1}, Y$
i.i.d. uniform and emit $Q_t = \sum_{j=0}^{h-1} B_j t^j + Y\,t^h \bmod q$, observed
as $\mathbf{x}_t = \mathbf{u}_{Q_t} + \sigma\boldsymbol\varepsilon_t$. The target is
the leading coefficient $Y$.

- *Floor.* For any $W \le h$, the $W$ observations impose $W$ independent linear
  constraints (distinct Vandermonde nodes) on the $h$ nuisance coefficients, so
  for every fixed $Y=y$ the window $Q_{0:W-1}$ is *uniform* on $\mathbb{F}_q^W$ —
  identical across $y$. Hence $I(Y; \mathbf{x}_{1:W}) = 0$ and every encoder is at
  chance $1/q$. The nuisance coefficients are a finite-field one-time pad.
- *Recovery.* At $W = h+1$, $h+1$ distinct points determine a degree-$h$
  polynomial uniquely (Lagrange interpolation), so $Y$ is a deterministic
  function of the window. Noiseless Bayes accuracy jumps $1/q \to 1$.

So $W^\star = h+1$ **exactly**, and it is *tunable by the polynomial degree $h$*.
This is the sharpest lever we have: a family of tasks whose window threshold we
set by hand.

**Instance B — frequency resolution (soft, SNR-modulated threshold).**
On the circle embedding (the geometry real LLMs use for cyclic concepts, Engels
et al. 2024), hidden velocity $y$ becomes a temporal tone $f_y = y/M$. Two tones
$f, f'$ can be separated from a length-$W$ window only above the Rayleigh limit
$|f - f'| \gtrsim 1/W$ (Dirichlet-kernel width). For a pair separated by
$\Delta f$ the threshold is $W^\star \approx 1/\Delta f$. Unlike the clock's exact
algebraic cliff, this is a *resolution* threshold softened by SNR — a nice
contrast to present alongside the clock.

### Mechanism 2 — evidence integration

**General statement.** Above threshold, $Y$ is present but observed through noise.
If the window yields $n(W)$ conditionally-independent noisy measurements of $Y$
(equivalently, accumulated Fisher information growing with $W$), the Bayes error
decreases monotonically in $W$. This is the smooth climb.

**Instance A — the noisy polynomial clock.** For $W > h+1$ the system is
*over-determined*: $W$ noisy evaluations of a degree-$h$ polynomial with $h+1$
unknowns. Least-squares interpolation averages the $W-h$ excess constraints, so
the leading-coefficient error scales like $\sigma/\sqrt{W-h}$ (up to Vandermonde
conditioning). For degree one this is transparent: $Y = Q_{t+1}-Q_t$ holds for
*every* consecutive pair, giving $W-1$ i.i.d. noisy estimates of $Y$ to average.

**Instance B — sticky-HMM smoothing (the Denoising bench).** A hidden two-state
chain $s_t$ with self-transition $\rho$, seen through a noisy Bernoulli emission,
is optimally estimated by the forward–backward (Baum) smoother from
$a_{t-W:t+W}$. Posterior error falls with $W$ but **saturates at the correlation
time $\tau_c \sim 1/(1-\rho)$**: past $\tau_c$, distant emissions are nearly
independent of $s_t$ and add no information. The useful window scales as
$1/(1-\rho)$ — a clean prediction for a $\rho$-sweep.

**Instance C — periodogram coherent integration.** For a single tone in white
noise the periodogram peak SNR grows $\propto W$, so estimation error falls with
$W$ — the Mechanism-2 companion to the Rayleigh (Mechanism-1) resolution limit.

### The composite curve, and what we actually measured

The noisy polynomial clock is Mechanism 1 **then** Mechanism 2: chance for
$W < h+1$, then error $\propto \sigma/\sqrt{W-h}$ for $W \ge h+1$. The degree-one
run (`results/v6_colored_sources/polynomial_clock_h1_q31.json`, $q=31$,
$\sigma=0.1$, chance $=0.032$, $W^\star=2$) shows exactly this — validation
accuracy, best-$k$ per architecture:

| architecture | W=1 | W=2 | W=3 | W=4 |
|---|---|---|---|---|
| local SAE (per-token) | 0.037 | 0.037 | 0.034 | 0.039 |
| window-linear probe ("Stacked SAE") | 0.041 | 0.097 | 0.171 | 0.198 |
| **TXC** | **0.037** | **0.157** | **0.479** | **0.923** |

The TXC sits at the chance floor at $W=1$ exactly as the theorem demands, then
climbs monotonically to 0.92. The higher-degree files are not yet a clean test
of the predicted shifted floor. Their probe split is over sampled anchors, not
over polynomial episodes: anchors are sampled with replacement from 4096
sequences and then randomly divided into train and validation sets. The same
episode can therefore appear on both sides of the split. Correspondingly, the
TXC is implausibly above the theorem's chance floor below threshold
($0.149$ versus $1/11=0.091$ at $h=2,W=2$; about $0.22$ versus
$1/7=0.143$ at $h=3,W=2,3$). These results are descriptive only until the
probes and model evaluation use episode-disjoint data.

Subject to that caveat, $h=2$ ($q=11$, $W^\star=3$) climbs to 0.60 by $W=5$;
$h=3$ ($q=7$, $W^\star=4$) is **not learned by the current recipe** (0.29 at
$W=6$). A clean rerun should combine episode-disjoint evaluation with more
steps, seed averaging, and a controlled comparison of the proposed
$k_{\text{win}}=1$ global-budget bottleneck against the larger-$k$ sweep.

The FreqBench Rayleigh scan gives the *graded* companion on the circle embedding:
TXC linear accuracy $0.59 / 0.80 / 0.95 / 1.00$ at $W = 4 / 8 / 16 / 32$, with the
per-frequency failure region sitting left of $1/W$ in every panel.

### Why this answers the three reviewers

- **"Window length has almost no effect."** True *only* when the task has no
  genuine cross-token structure ($W^\star = 1$) or the model has already done the
  temporal binding internally before the hookpoint (cf. the sprint's GPT-2
  day-stride result: after one attention block, single positions already decode
  stride). On a task with $W^\star > 1$, window length is causally necessary and
  the effect is exactly the threshold plus the post-threshold slope. The clock
  makes $W^\star$ a dial, so we can *show* the effect appearing and moving.
- **"Isolate the temporal contribution."** The local SAE is *provably* at chance
  at every $W$ (Mechanism 1) — the cleanest possible isolation.
- **"Where is the Stacked SAE?"** The window-linear probe *is* the Stacked-SAE
  control: temporal aggregation living in the linear readout, with no
  cross-position weight sharing. It is the middle row above — well above chance
  but far below the TXC — which isolates cross-position weight sharing (not mere
  aggregation) as the operative ingredient. This is precisely the decomposition
  Reviewer 1 asked for, on a task where the theory says the crosscoder must win.

### Testable predictions (what each task should show if the frame is right)

1. **Clock, floor.** Local raw probe and per-token SAE at chance $1/q$ for all
   $W \le h$; symbolic interpolation oracle at 1.0 for $W = h+1$ (noiseless).
2. **Clock, slope.** Above $W^\star$, TXC accuracy rises with $W$; error tracks
   $\sigma/\sqrt{W-h}$. Threshold moves right by one as $h$ increments.
3. **Stacked-SAE gap.** Window-linear probe $\ll$ TXC at every $W$ above
   threshold; the gap is the cross-position-weight-sharing contribution.
4. **Denoising $\rho$-sweep.** Saturation window $\propto 1/(1-\rho)$: at
   $\rho=0.7$ saturate by $T\!\approx\!4$; at $\rho=0.9$ by $T\!\approx\!10$; at
   $\rho=0.97$ by $T\!\approx\!30$.
5. **Rayleigh scan.** Per-frequency deficit boundary at $f \approx 1/W$; overall
   accuracy monotone in $W$ toward the periodogram oracle.

### Open questions to settle at the pause

- **Headline choice.** Clock (sharpest theorem, most new work) vs. Rayleigh scan
  (graded, real-geometry, already clean) vs. Denoising $\rho$-sweep (cheapest,
  already in the paper) as the centerpiece figure. They are not mutually
  exclusive — the strongest rebuttal is probably the clock as headline with the
  other two as support.
- **Recipe fix for $h\ge 2$.** Does the global $k_{\text{win}}=1$ bottleneck plus
  more steps recover the degree-2/3 threshold, or is degree-3 (2401 templates)
  genuinely out of reach at this dictionary budget?
- **Framing.** Do we present the two mechanisms as a standalone theory
  subsection, or fold them into the existing synthetic section as the
  justification for the window sweeps?

Related: [[reviewer_responses]] (the reports), the FreqBench sprint
(`docs/dmitry/sprints/2026-06-10_freqbench_sprint/`), and Aniket's clock
proposal (`docs/aniket/experiments/synthetic/notes/polynomial_clock_experiment.tex`).
