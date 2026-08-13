---
author: Claude (with Dmitry)
date: 2026-08-10
tags:
  - proposal
---

## BIRD posterior codes: deriving the optimal temporal latent code from information-restricted diffusion

> **Results update (2026-08-10):** Phase A (A1–A4) ran the same day on Modal
> — see [[2026-08-10_bird_clock_results]]. A1/A2 confirmed exactly (entropy
> and $L_0$ laws to numerical precision; frontier tracks prediction); A3
> transition located exactly at $W = h{+}2$ with new defect-kinetics on top;
> A4's naive cash-out of §3.5 falsified — reconstruction fails to *require*
> binding but TopK window encoders volunteer it; refined design queued.
> B1 (same day): the posterior head + DSM closes **94% of the Bayes gap**
> vs 7% for the TopK TXC at matched everything; the 2×2 shows neither the
> architecture nor the objective suffices alone.
>
> **Novelty status** (see [[2026-08-10_bird_novelty_check]]): the softmax
> estimator and σ-ladder claims are prior art (Smart et al. 2502.05164;
> BIRD itself) — demoted to method. What's ours: the DSM-vs-reconstruction
> ablation + Bayes-gap yardstick on sequential data, and the $W{=}h{+}2$
> generative transition. Must-cite-and-distinguish: Yun et al. 2607.15693
> (DSM dictionary on images, posted one week ago, no objective ablation).

### 0. TL;DR

The Ganguli group's BIRD framework (Hunt, Kamb & Ganguli 2026, arXiv:2607.08041,
building on the ELS machine of Kamb & Ganguli, ICML 2025, arXiv:2412.20292)
studies denoisers in which each unit sees only a *restricted observation* of the
noisy data and inverts diffusion by computing an exact Bayesian posterior over
which training example produced its observation. Mapped onto our synthetic
temporal settings, this framework is not an analogy but an **exact
correspondence**:

- our global latents (polynomial coefficients $\beta$, tone velocity $Y$,
  hidden class $C$) play the role of BIRD's training set $\mathcal{D}$;
- a TXC's temporal window is BIRD's restricted observation channel
  $\mathcal{C}_{t,W}$;
- the *optimal latent code* the user's framing asks for — "a local activation
  is a linear combination of noised global latents" — is exactly the BIRD
  restricted posterior: the posterior-mean denoiser writes every local
  activation as a softmax-weighted linear combination of global atoms
  evaluated locally, and the softmax weights **are** the code.

On the polynomial clock the correspondence is exact and closed-form: BIRD's
entropy law $S = \ln|\mathcal{D}| - I$ holds with equality (not just
asymptotically), the local-impossibility theorem is the memorization–
generalization phase boundary, the optimal code's $L_0$ obeys
$q^{\,h+1-W}$, and the framework makes one genuinely new, sharp prediction: the
*coding* transition sits at $W = h{+}1$ but the *generative-consistency*
(anti-chimera) transition sits at $W = h{+}2$. It also answers a foundational
question we had left implicit: **why a denoising objective (diffusion) is the
right temporal dictionary objective at all** — at $\sigma = 0$ reconstruction
provably cannot distinguish a temporally-bound code from a stacked per-token
code (both reconstruct perfectly), while at $\sigma > 0$ optimal denoising
*requires* the cross-position binding. Noise is what makes temporal structure
loss-bearing.

### 1. Imported theory: ELS and BIRD in one page

**Ideal (unrestricted) score.** For a finite training set $\mathcal{D}$ and
forward process $\phi_t = \sqrt{\bar\alpha_t}\,\varphi + \sqrt{1-\bar\alpha_t}\,\epsilon$,
the optimal denoiser is the posterior mean over training examples,

$$M_t[\phi] = \sum_{\varphi\in\mathcal{D}} \varphi\; W_t(\varphi|\phi),\qquad
W_t(\varphi|\phi) = \operatorname{softmax}_\varphi\!\left(-\tfrac{\|\phi-\sqrt{\bar\alpha_t}\varphi\|^2}{2(1-\bar\alpha_t)}\right),$$

which at $t\to 0$ memorizes: it can only reproduce training points.

**Local score (LS/ELS).** Kamb & Ganguli's resolution of the
creativity-vs-memorization paradox: convolutional denoisers are *local* and
*equivariant*, so each pixel $x$ computes the same posterior but conditioned
only on its receptive field $\Omega_x$:

$$M_t[\phi](x) = \sum_{\varphi} \varphi(x)\; W_t(\varphi_{\Omega_x}|\phi_{\Omega_x}),$$

with equivariance additionally pooling training patches over spatial
positions. Outputs are **locally consistent patch mosaics**: every patch
matches some training patch, but the global image is novel. This analytic
machine matches trained UNets with median $r^2 \approx 0.90$–$0.96$.

**BIRD generalization** (Hunt, Kamb & Ganguli 2026). Abstract the receptive
field to an arbitrary restricted channel $\mathcal{C}_{x,t}:\mathbb{R}^d\to\mathbb{R}^{n_{x,t}}$
per unit; the unit holds the Bayesian posterior
$P(\varphi\,|\,\mathcal{C}_{x,t})$ over training samples. The central result
is an information-theoretic phase transition. In the large-dataset limit the
posterior entropy obeys

$$S\big[P(\varphi|\mathcal{C})\big] \;=\; \max\big(0,\; \ln|\mathcal{D}| - I(\varphi;\mathcal{C})\big),$$

with **memorization** (posterior concentrated on one training point) iff
$I \ge \ln|\mathcal{D}|$ and **generalization/creativity** otherwise. For
spatially-local channels this defines a critical patch size $L_c(\sigma_t)$
(scaling like $\sigma_t\sqrt{\ln|\mathcal{D}|}$ for their image ensembles),
and generation "proceeds near the edge of memorization": the effective patch
scale tracks $L_c(\sigma_t)$ as noise anneals.

### 2. The dictionary between BIRD and our synthetic settings

| BIRD (images) | temporal crosscoders (this repo) |
|---|---|
| training image $\varphi \in \mathcal{D}$ | global latent / clean trajectory template $\Phi_\beta$ (atom bank) |
| pixel $x$ | token position $t$ |
| receptive field / patch $\Omega_{x,L}$ | temporal window $\Omega_{t,W}$ |
| diffusion noise $\sigma_t$ | observation noise $\sigma$ + inference-time annealing $\sigma_\tau$ |
| restricted posterior $P(\varphi\,\vert\,\mathcal{C}_{x,t})$ | **the latent code** $z_\beta(t)$ |
| posterior-mean denoiser | "local activation = LC of global latents" |
| memorization phase | window identifies the episode's latent |
| patch-mosaic creativity | chimeric trajectories (locally valid, globally invalid) |
| critical patch scale $L_c(\sigma)$ | critical window $W_c(\sigma)$ |
| translation equivariance | time-shift sharing of temporal atoms |

One conceptual difference, which turns out to be a feature: in BIRD,
$\mathcal{D}$ is an empirical sample and generalization means composing
beyond it. In our synthetic settings the atom bank is the **exact support of
the clean data distribution** ($q^{h+1}$ polynomial trajectories;
$M\cdot|\Omega|$ tone templates). So for us "memorization" is not a failure
mode — it is *successful inference* of the episode latent, and BIRD's phase
boundary becomes the boundary between "the window determines the latent" and
"it provably does not". The same formalism covers both readings.

### 3. Exact correspondence on the polynomial clock

Setting (see `src/v6_colored_sources/polynomial_clock.py`): hidden
$\beta = (B_0,\dots,B_{h-1},Y) \sim \mathrm{Unif}(\mathbb{F}_q^{h+1})$,
symbols $Q_t = P_\beta(t) = \sum_k \beta_k t^k \bmod q$, observations
$x_t = u_{Q_t} + \sigma\varepsilon_t$ with orthonormal alphabet
$\{u_a\}_{a\in\mathbb{F}_q}$. Clean trajectory template
$\Phi_\beta = (u_{P_\beta(0)},\dots,u_{P_\beta(T-1)})$; atom bank size
$M_{\text{atoms}} = q^{h+1}$.

#### 3.1 The optimal local code is the restricted posterior — and it is attention

Condition on a window $x_{\text{win}} = x_{[t, t+W)}$. Since every restricted
template has equal norm ($\|\Phi_\beta^{\text{win}}\|_F^2 = W$), the Gaussian
likelihood's quadratic term is constant across $\beta$ and the posterior is a
pure inner-product softmax:

$$z_\beta(t)\;=\;P\big(\beta\,\big|\,x_{\text{win}}\big)\;=\;\operatorname{softmax}_\beta\!\left(\frac{\langle \Phi_\beta^{\text{win}},\, x_{\text{win}}\rangle_F}{\sigma^2}\right).$$

Three readings of this one formula:

- **It is the user's "LC of noised global latents" claim, made exact.** The
  posterior-mean denoiser at each position in the window is

  $$\hat{x}_{t'} \;=\; \mathbb{E}[u_{Q_{t'}} \mid x_{\text{win}}] \;=\; \sum_\beta \Phi_\beta(t')\, z_\beta,$$

  i.e. every local activation is optimally explained as a **linear
  combination of global atoms evaluated locally, with position-shared
  coefficients** $z$. The observed activation is this LC plus noise. The
  "best latent code" is the coefficient vector, and Bayes says it is a
  softmax, not an $\ell_1$-sparse vector.
- **It is a single cross-attention layer.** Query $=$ the window, keys/values
  $=$ the atom bank, temperature $= \sigma^2$. The BIRD-optimal temporal
  encoder is literally attention over a global template memory with
  noise-tied temperature. (TopK-TXC is its hard-assignment limit; see §3.2
  for what $k$ should be.)
- **It is a sufficient statistic.** $z(t)$ carries all window information
  about anything downstream ($Y$, future symbols, other positions), so every
  rung of the screen's probe ladder is bounded by probes on $z$.

#### 3.2 Exact phase structure: entropy law, $L_0$ law, coding threshold

In the noiseless limit the posterior is **uniform on the consistency coset**
$S(x_{\text{win}}) = \{\beta : P_\beta(t') = Q_{t'}\ \forall t' \in \text{win}\}$,
an affine subspace of $\mathbb{F}_q^{h+1}$ of dimension $\max(0,\,h{+}1{-}W)$
(Lagrange interpolation). Consequences, each an exact finite-$q$ instance of a
BIRD asymptotic:

- **Entropy law with equality.** $I(\beta; x_{\text{win}}) = \min(W, h{+}1)\ln q$
  and $S[z] = (h{+}1{-}W)^+\ln q = \ln M_{\text{atoms}} - I$. BIRD's
  large-dataset formula holds exactly here because the posterior is exactly
  uniform on a coset — no sub-exponential-tail assumption needed.
- **The impossibility theorem is the phase boundary.** For $W \le h$ the coset
  projects *uniformly* onto the $Y$ coordinate — that is precisely the repo's
  local-impossibility statement ($I(Y;\text{window}) = 0$) — and at
  $W = h{+}1$ we hit $I = \ln M_{\text{atoms}}$: BIRD memorization =
  identifiability of the episode latent. The clock sits exactly *on* BIRD's
  critical manifold $\ln|\mathcal{D}| = I$.
- **$L_0$ law for the ideal code.** The optimal code has exactly
  $q^{\,h+1-W}$ active atoms (the coset), stepping geometrically to one-hot
  at $W = h{+}1$. This prescribes TopK's $k$ as a function of window length,
  and predicts the *measured* participation ratio of a well-trained code —
  a falsifiable architecture-independent target.
- **Optimal dictionary size.** The set of achievable posteriors is indexed by
  the observed window symbols, so a *coset dictionary* needs only
  $q^{\min(W,\,h+1)}$ atoms — the number of distinct realizable clean windows.
  This is the clock version of FreqBench P5's template count
  ($|\Omega|\cdot M = 1010$), and BIRD explains why that number is the
  memorization threshold for window dictionaries.

#### 3.3 The noise frontier $W_c(\sigma)$

With noise, identifiability needs a margin. Two distinct degree-$\le h$
polynomials agree on at most $h$ points, so restricted templates satisfy
$\|\Phi_\beta^{\text{win}} - \Phi_{\beta'}^{\text{win}}\|_F^2 \ge 2(W-h)$:
**atom separation grows linearly in $W$ while per-symbol separation is fixed
at $\sqrt2$**. A union bound over the bank gives reliable identification when
$(W-h)/\sigma^2 \gtrsim 2\ln M_{\text{atoms}}$, i.e.

$$W_c(\sigma) \;\approx\; h \;+\; 2\sigma^2 (h{+}1)\ln q .$$

This is the clock's analogue of BIRD's $L_c(\sigma) \sim \sigma\sqrt{\ln|\mathcal{D}|}$;
the exponent differs (here $\sigma^2$, there $\sigma$) because our alphabet is
orthonormal while their image ensembles have power-law spectra — the
$W_c(\sigma)$ scaling exponent is a property of the data spectrum, and our
synthetic families let us dial it. Between the per-symbol failure noise and
the atom-identification noise there is a regime where **local decoding is
destroyed but the temporal code survives** — the cleanest possible
demonstration that windows buy robustness, not just information.

#### 3.4 A new sharp prediction: identifiability at $h{+}1$, generative consistency at $h{+}2$

ELS Theorem 4.1 says converged local-score samples are locally-consistent
mosaics. For the clock this splits into two distinct thresholds:

- Any $W \le h{+}1$ symbol values interpolate to *some* degree-$\le h$
  polynomial, so for $W \le h{+}1$ **every** symbol sequence is locally
  consistent — local windows constrain generation not at all, and
  reverse-diffusing with a $W$-local score produces chimeric trajectories
  (each window explained by some $\beta$, the sequence by none).
- For $W \ge h{+}2$, adjacent windows overlap in $\ge h{+}1$ positions, which
  pins the *same* interpolating polynomial in both; induction forces one
  global $\beta$. Chimeras become impossible.

So the **coding transition** ($W^\* = h{+}1$: the window determines $\beta$)
and the **generative transition** ($W^{\*\*} = h{+}2$: locally-valid implies
globally-valid) are separated by exactly one step. This is invisible to probe
ladders (which only test coding) and is a genuinely diffusion-native
prediction: it can only be measured by *generating* with a local score and
counting globally-valid outputs. It is the clock's version of "creativity":
for $W \le h{+}1$ the local score machine is maximally creative (it can
produce all $q^T$ symbol sequences from a training support of $q^{h+1}$).

#### 3.5 Why denoising forces temporal binding and reconstruction cannot

A point the correspondence makes embarrassingly clear in hindsight. Consider
the two codes:

- **product code**: per-position symbol posteriors (what a stacked per-token
  SAE represents), reconstruction $\hat x_{t'} = \sum_a u_a P(Q_{t'}{=}a \mid x_{t'})$;
- **coset code**: the BIRD posterior $z_\beta$ over global atoms.

At $\sigma_{\text{train}} = 0$ (plain autoencoding) both reconstruct the clean
window *identically* — reconstruction loss is blind to whether the code binds
positions together. This is exactly why TXC and Stacked SAE arrive at similar
FVU and must be separated by probes (the whole reason the screen's dictionary
ladder exists). Under a **denoising objective at noise where single symbols
are ambiguous**, the optimal prediction of position $t'$ is the
coset-marginal $P(Q_{t'}{=}a\mid x_{\text{win}}) = \sum_{\beta: P_\beta(t')=a} z_\beta$,
which *requires* propagating the polynomial constraint across positions:
cross-position binding stops being representationally optional and becomes
loss-bearing. **The noise axis is what converts temporal structure from
"reconstruction-neutral" to "objective-critical" — this is the principled
reason a diffusion approach is the right temporal dictionary objective.**
Quantitatively, per-symbol decoding fails at fixed $\sigma \sim 1$ while the
window atom survives to $\sigma^2 \sim (W-h)/\ln M_{\text{atoms}}$: the
denoising gain of the temporal code over the local code is unbounded in $W$.

#### 3.6 The $\sigma$-ladder is the window ladder

BIRD's "generation at the edge of memorization" — receptive scale tracking
$L_c(\sigma_t)$ during annealing — imports directly: a *single* DSM-trained
model across a $\sigma$-ladder implements our entire window-length sweep
(`experiments/temporal_screen/ladder.py::window_length_sweep`, FreqBench's
$W$-scan) as a noise sweep, with $W_c(\sigma)$ as the exchange rate. This
suggests a second, diffusion-native definition of a feature's timescale,
complementary to FreqBench's frequency response: **the noise scale at which
the feature's code first becomes necessary for denoising**. Slow/global
features should switch on at large $\sigma$ (large effective windows), local
ones at small $\sigma$ — a "timescale spectrometer" needing one trained model
instead of a $W$-grid of dictionaries.

### 4. Continuous atom families: the circle tasks

For FreqBench's circle embedding and the screen's `order_clock`, the atom bank
is a continuous family $\Phi_{S,\theta_0}(t) = R\,r(\theta_0 + S\omega t)$
indexed by sign $S$ and nuisance phase $\theta_0$. The BIRD posterior
marginalizes the nuisance analytically: with $c_t = R^\top x_t$ the complex
plane projection and $\hat A(f) = \sum_t c_t e^{-2\pi i f t}$ the window DFT,

$$P(S \mid x_{\text{win}}) \;\propto\; \int \! d\theta_0\, e^{\kappa \operatorname{Re}[e^{-i\theta_0}\hat A(S f_\omega)]} \;=\; I_0\!\big(\kappa\,|\hat A(S f_\omega)|\big),$$

a Bessel function of the **periodogram amplitude** at the two candidate
frequencies. Three payoffs:

- FreqBench's oracle (Rife–Boorstyn periodogram argmax) is *derived* rather
  than chosen: it is the BIRD posterior for this atom family.
- The required nonlinearity for the AC-sign task is characterized exactly —
  phase marginalization produces a quadratic (energy-detector) invariant.
  The screen's hand-derived verdicts for its four calibration cases
  (`experiments/temporal_screen/synthetic.py`) all fall out of one
  computation: `local` → posterior factorizes; `smoothing` → posterior
  depends on the window only through its mean (order-free, linear);
  `linear_temporal` → posterior linear in a position difference;
  `order_clock` → phase-marginalized Bessel posterior (order-critical,
  nonlinear). One framework, four verdicts.
- **Embedding geometry decides whether a generalization phase exists.** Under
  the circle embedding the atom family is low-dimensional, so a small
  dictionary can tile it ($H = 64 \ll 1010$ templates suffices — FreqBench
  §4.5); under the random embedding the ratio-invariance theorem (P3) says
  the bank has no usable metric structure, so *only the memorization phase
  exists* — matching the observed all-or-nothing jump at
  $H > 1010$. BIRD turns FreqBench's empirical "capacity routes" dichotomy
  into a phase diagram.

### 5. Retrodictions: existing results the framework already explains

| existing result (where) | BIRD reading |
|---|---|
| local impossibility for $W \le h$, identifiability at $h{+}1$ (`polynomial_clock.py`, proposal §4–5) | exact phase boundary $\ln M_{\text{atoms}} = I$; entropy law with equality |
| atom margin $1/(h{+}1)$ and constructive sparse solution | atom separation $2(W{-}h)$; posterior concentration at the memorization edge |
| TXC ≈ Stacked SAE on FVU, separated only by probes (screen; paper c-series) | reconstruction degeneracy of §3.5 — $\sigma{=}0$ objectives cannot price binding |
| window dictionaries memorize at $H > 1010$ templates; conv ($L{=}3$) never does (FreqBench §4.5) | memorization phase requires channel information $I \ge \ln(\text{templates})$; a 3-tap channel is below threshold at any $H$ |
| multiband beats vanilla under superposition, 0.96 vs 0.91 (FreqBench §4.6) | factored posterior over a product latent space vs a flat softmax over $10^9$ templates — flat memorization is information-theoretically unavailable, structured channels win |
| per-token SAE + MLP probe = 1.0 but linear = chance (FreqBench §4.1) | the information bound is channel-level, not code-level; conversion-to-linearity = moving the posterior computation (Bessel/coset) into the encoder |
| periodogram oracle ≈ 0.99 and its $W{=}4$ Rayleigh collapse | BIRD posterior for the circle atom family; Rayleigh cell = posterior overlap of adjacent atoms |
| backtracking signal is low-frequency; DC branch detector wins (FreqBench §4.8) | the restricted channel matched to where $I$ lives — channel design, not just window design |

That a single framework retrodicts this many independent findings is the main
evidence it is the right analytic language for the temporal problem.

### 6. Proposed experiments

Phase A is fully analytic (no training, CPU-scale: $q{=}31$, $h \in \{1,2\}$
gives $961$–$29{,}791$ atoms — trivially enumerable).

- **A1 — posterior codes vs the probe ladder.** Implement the LS machine for
  the clock: $z = \operatorname{softmax}(\langle \Phi_\beta^{\text{win}}, x\rangle/\sigma^2)$.
  Feed $z$ through the existing screen ladder as a new rung ("R6: analytic
  BIRD code"). Predictions: $Y$-probe accuracy steps at $W = h{+}1$ to the
  oracle rung; participation ratio of $z$ tracks $q^{h+1-W}$; trained TXC
  (R4) is upper-bounded by R6 everywhere.
- **A2 — the $(W,\sigma)$ phase diagram.** Identification accuracy and
  posterior entropy on a $(W, \sigma)$ grid; overlay the predicted frontier
  $W_c(\sigma) = h + 2\sigma^2(h{+}1)\ln q$ and the entropy law
  $S = \ln M - I$. This is the clock's version of BIRD's Fig. on
  edge-of-memorization, and doubles as the "windows buy robustness" figure
  ($\exists\,\sigma$: local decoding at chance, temporal code near-perfect).
- **A3 — the creativity experiment.** Reverse-diffuse from noise with a
  $W$-local score (the ELS machine, atoms shared across anchors) and measure
  (i) global-polynomial validity rate and (ii) the $\beta$-drift rate along
  generated sequences, vs $W$. Prediction: chimeras for all $W \le h{+}1$,
  hard transition to global validity at $W = h{+}2$ — one step *after*
  identifiability. This is the experiment only the diffusion framing can even
  state.
- **A4 — denoising vs reconstruction at matched everything.** Same
  architecture, same capacity (below the template count), trained (a) as an
  autoencoder at $\sigma_{\text{train}}{=}0$ and (b) with DSM across a
  $\sigma$-ladder. Prediction from §3.5: (b)'s codes carry $Y$ linearly,
  (a)'s need not — the first controlled demonstration that the *objective*,
  not the architecture, forces temporal binding.

Phase B (small trained models):

- **B1 — the posterior head as an architecture.** One cross-attention layer
  over a learned atom memory with $\sigma^2$-tied temperature, weight-tied
  decoder $\hat x_{t'} = \sum_\beta \Phi_\beta(t') z_\beta$, trained by DSM
  across the ladder. Compare against TopK-TXC / Stacked SAE on the clock, the
  circle suite, and the screen's calibration quartet at matched $H$. The
  theory says this is the Bayes-form of a TXC; measure how much of the gap to
  R6 it closes, and whether the learned temperature tracks $\sigma^2$.
- **B2 — timescale spectrometer via noise.** For one DSM model trained across
  $\sigma$, measure the effective receptive field (Jacobian support) vs
  $\sigma$ against $W_c(\sigma)$; then read per-feature switch-on noise
  scales and compare with FreqBench's frequency-response curves on the same
  data — two independent definitions of "feature timescale" that the theory
  says must agree.

Phase C (defer until A/B verdicts): real activations — run B1 on the
backtracking screen windows where the low-frequency/DC result already tells
us where $I$ lives.

### 7. Deferred: subtracting the local part

The user's suggestion — model the local activation as (global LC) plus a
local component to be subtracted — has a natural BIRD formulation we
deliberately postpone: the $W{=}1$ posterior (symbol identity) is the "local
part", and the temporal content of the code is the *refinement* from the
product of local posteriors to the joint coset posterior. On the clock this
decomposition is maximally clean because the $Y$-information is **purely
synergistic** (any $\le h$ positions carry zero; all $h{+}1$ carry
everything — a $q$-ary generalization of parity), so local subtraction is
exactly the removal of the product-code component. This also connects to the
planned Baum–Welch SAE: the BIRD window posterior is a restricted E-step, and
the $W \to T$ limit with overlapping windows is forward–backward message
passing. Both are follow-ons once A1–A4 land.

### 8. Prior synthetic work in this repo (context)

- `docs/dmitry/synthetic/2026-05-06_overnight/results.md` — 11-benchmark
  synthetic suite over regular_sae / txc_base / txcdr_t5, including
  `bench_e_denoising_recon` and `e1_pure_smoother`: denoising appears there
  as an *evaluation*, never as a training objective — this proposal is the
  repo's first diffusion-native treatment.
- `docs/aniket/hmm_mathematical_structure.md` — the HMM generative modes
  (standard / leaky-reset / coupled) behind that suite; the BIRD window
  posterior is the restricted E-step for these models (§7's Baum–Welch
  connection).
- `docs/aniket/experiments/synthetic/notes/` — TeX proposals incl.
  `polynomial_clock_experiment.tex` (the clock's origin) and
  `txc_smoother_filter_experiment_plan.tex` — the smoother/filter framing is
  the closest prior idea to the posterior-code view taken here.
- `experiments/temporal_screen_1/` — screening *language* tasks for temporal
  structure (rollout-decay curves, Cagnetta-style exponents); the LLM-side
  complement to the synthetic settings used here.

### 9. References

- Hunt, Kamb & Ganguli 2026, *An exact information theory of generalization
  phase transitions in Bayesian diffusion models*
  ([arXiv:2607.08041](https://arxiv.org/abs/2607.08041)) — the BIRD framework.
- Kamb & Ganguli 2025, *An analytic theory of creativity in convolutional
  diffusion models* ([arXiv:2412.20292](https://arxiv.org/abs/2412.20292)),
  ICML 2025 — the LS/ELS machines.
- This repo: `src/v6_colored_sources/polynomial_clock.py` (clock + atoms +
  impossibility), `experiments/temporal_screen/` (ladder, calibration suite),
  FreqBench sprint `docs/dmitry/sprints/2026-06-10_freqbench_sprint/summary.md`
  (P1–P5, capacity routes, spectral crosscoder, periodogram oracle).
