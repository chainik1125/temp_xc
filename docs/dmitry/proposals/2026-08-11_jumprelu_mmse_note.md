---
author: Claude (with Dmitry)
date: 2026-08-11
tags:
  - reference
---

## JumpReLU as the zero-noise limit of Bayesian denoising: a pedagogical derivation

This note derives, step by step, the claim made in the diffusion-SAE
discussion: **under a denoising objective, the optimal activation function
is not a design choice but a posterior-inference step — and JumpReLU
emerges as its zero-noise limit, provided features have a minimum firing
strength.** Along the way we get, for free: why the gate's temperature is
$\sigma^2$, where Wiener shrinkage (and hence the DSM reconstruction tax)
comes from, why the softmax posterior head is the same theorem for a
different prior, and a classical-statistics pedigree (MMSE shrinkage
operators; the Donoho–Johnstone universal threshold) for the whole story.

Notation: $\varphi(t) = \tfrac{1}{\sqrt{2\pi}}e^{-t^2/2}$ is the standard
normal density and $\mathcal N(y;\mu,v)$ the normal density with mean
$\mu$, variance $v$ evaluated at $y$.

### 1. The setup: one feature, one noisy observation

Everything happens in a scalar channel first; §7 explains why an SAE
latent reduces to this. A single feature has coefficient $a$ with a
**spike-and-slab prior**:

$$a = \begin{cases} 0 & \text{with probability } 1-\pi \quad (\text{feature off — the spike}) \\ a \sim f(a) & \text{with probability } \pi \quad (\text{feature on — the slab}), \end{cases}$$

where $f$ is the distribution of firing magnitudes. We observe the
coefficient through Gaussian noise at level $\sigma$:

$$y = a + \sigma\varepsilon, \qquad \varepsilon \sim \mathcal N(0,1).$$

The denoising objective asks for the function $\hat a(y)$ minimising
$\mathbb E\,(\hat a(y) - a)^2$. A standard fact (worth re-deriving once:
expand the square, condition on $y$, minimise pointwise) is that the
minimiser is the **posterior mean**:

$$\hat a(y) = \mathbb E[a \mid y].$$

So "what activation should a denoising-trained SAE use" becomes "what
does $\mathbb E[a\mid y]$ look like".

### 2. Splitting the posterior mean into gate × magnitude

Condition on whether the feature is on. Since $a = 0$ contributes nothing
to the mean:

$$\hat a(y) = \underbrace{P(\text{on} \mid y)}_{\text{the gate } g(y)} \; \times \; \underbrace{\mathbb E[a \mid y, \text{on}]}_{\text{the magnitude } m(y)}.$$

This factorisation is the whole conceptual content: **every MMSE
activation is a gate times a magnitude estimate.** The rest of the note
just computes the two factors and takes limits.

**The gate**, by Bayes' rule with the two hypotheses "off" (likelihood
$\mathcal N(y;0,\sigma^2)$, prior $1-\pi$) and "on" (likelihood
$M_f(y) := \int f(a)\,\mathcal N(y;a,\sigma^2)\,da$, prior $\pi$):

$$g(y) = \frac{\pi M_f(y)}{(1-\pi)\,\mathcal N(y;0,\sigma^2) + \pi M_f(y)} = \frac{1}{1 + e^{-\ell(y)}}, \qquad \ell(y) = \ln\frac{\pi M_f(y)}{(1-\pi)\,\mathcal N(y;0,\sigma^2)}.$$

The gate is **always a logistic function of the log-odds** $\ell(y)$ —
the only question is what $\ell$ looks like for a given slab.

### 3. Warm-up: the magnitude factor for a Gaussian slab (Wiener shrinkage)

Take $f = \mathcal N(\mu, s^2)$. Then $a \mid y,\text{on}$ is the
posterior of a Gaussian mean under a Gaussian likelihood — the one
conjugate computation everyone should do by hand once. Multiply the two
exponentials and complete the square in $a$:

$$f(a)\,\mathcal N(y;a,\sigma^2) \propto \exp\!\left(-\frac{(a-\mu)^2}{2s^2} - \frac{(y-a)^2}{2\sigma^2}\right) \propto \exp\!\left(-\frac{(a - \bar a)^2}{2\bar v}\right)$$

with

$$\bar v = \left(\frac{1}{s^2} + \frac{1}{\sigma^2}\right)^{-1}, \qquad \bar a = \mu + \frac{s^2}{s^2 + \sigma^2}\,(y - \mu).$$

So the magnitude factor is

$$m(y) = \mu + \underbrace{\frac{s^2}{s^2+\sigma^2}}_{\text{Wiener factor} \,<\, 1}(y-\mu),$$

and the slab-convolved likelihood needed for the gate is
$M_f(y) = \mathcal N(y;\mu, s^2 + \sigma^2)$ (variances add under
convolution).

Two observations to bank:

- As $\sigma \to 0$ the Wiener factor $\to 1$: the magnitude estimate
  becomes **pass-through**, $m(y) \to y$. This is where JumpReLU's linear
  part will come from.
- At finite $\sigma$ the estimate is *shrunk toward the prior*. A network
  trained σ-blind across a noise ladder bakes in an averaged shrinkage —
  this is precisely the Wiener-shrinkage account of the DSM
  reconstruction tax measured in the scale-up runs (NMSE 0.303 vs 0.279,
  recovered to 0.291 by annealing σ downward).

### 4. Worked example A: the narrow slab, in full

Let the slab be sharply peaked at a known firing strength: $f = \delta(a-\mu)$
with $\mu > 0$ (equivalently take $s \to 0$ above). Then
$M_f(y) = \mathcal N(y; \mu, \sigma^2)$ and the log-odds can be expanded
completely:

$$\ell(y) = \ln\frac{\pi}{1-\pi} + \ln\frac{\mathcal N(y;\mu,\sigma^2)}{\mathcal N(y;0,\sigma^2)} = \ln\frac{\pi}{1-\pi} + \frac{y^2 - (y-\mu)^2}{2\sigma^2} = \ln\frac{\pi}{1-\pi} + \frac{\mu}{\sigma^2}\left(y - \frac{\mu}{2}\right).$$

(The middle step: $y^2 - (y-\mu)^2 = 2\mu y - \mu^2$.) The gate is a
logistic in $y$:

$$g(y) = \operatorname{logistic}\!\left(\frac{y - \theta_\sigma}{T}\right), \qquad \boxed{T = \frac{\sigma^2}{\mu}}, \qquad \theta_\sigma = \frac{\mu}{2} - \frac{\sigma^2}{\mu}\ln\frac{\pi}{1-\pi}.$$

Read the three pieces:

- **Temperature $T = \sigma^2/\mu$**: the gate's softness scales with the
  *noise variance*. This is the same $\sigma^2$-temperature that appeared
  in the clock's posterior head — there for a categorical family, here
  for a binary one. It is not a coincidence; both are Gaussian likelihood
  ratios.
- **Threshold $\to \mu/2$** as $\sigma \to 0$: the decision boundary sits
  at the *midpoint* between "off" (0) and "on" ($\mu$), nudged by the
  prior odds at finite noise. Rare features ($\pi$ small) push the
  threshold up — you demand more evidence for an a-priori-unlikely
  feature.
- As $\sigma\to 0$, $T \to 0$: the logistic hardens into a **step
  function** at $\mu/2$.

For the delta slab the magnitude factor is trivially $m(y) = \mu$, so the
limit is a step *times a constant* — a gate, but not yet JumpReLU's
pass-through of $y$. The pass-through needs a slab with width, which is
the next section.

### 5. The theorem: broad slab bounded away from zero ⟹ JumpReLU

Now the realistic case: firing magnitudes vary, but a feature that fires
does so with at least strength $m_0 > 0$ — the slab $f$ is supported on
$[m_0, \infty)$ with $f(m_0) > 0$, and is broad compared to the noise
($s \gg \sigma$, where $s$ is the slab's scale). Claim:

$$\hat a(y) \;\xrightarrow{\sigma\to 0}\; y \cdot \mathbb 1\!\left[y > \tfrac{m_0}{2}\right] \qquad \text{— JumpReLU with threshold } \tfrac{m_0}{2}.$$

*Proof sketch, each step elementary.*

1. **Magnitude factor.** For $y$ in the slab's bulk and $\sigma \ll s$,
   the likelihood $\mathcal N(y;a,\sigma^2)$ is far narrower than $f$, so
   the posterior over $a$ concentrates at $a \approx y$ and
   $m(y) \to y$ (this is the Wiener factor $\to 1$ argument of §3, made
   local: within a window of width $\sigma$ around $y$, $f$ is
   approximately constant, so the posterior is approximately
   $\mathcal N(y, \sigma^2)$ restricted to the support). Pass-through. ✓
2. **Gate, above the support** ($y > m_0$): $M_f(y) \approx f(y)$ (same
   localisation), which is $O(1)$, while
   $\mathcal N(y;0,\sigma^2) = \varphi(y/\sigma)/\sigma$ is doubly
   exponentially small. So $\ell(y) \to +\infty$: gate open. ✓
3. **Gate, in the gap** ($0 < y < m_0$): both hypotheses must reach $y$
   through their tails. The "on" likelihood is dominated by the nearest
   slab point $a = m_0$ (Laplace/dominant-point approximation — the same
   move as step 2 of the atom-margin arguments in the clock work):

   $$M_f(y) \approx f(m_0)\,\sigma\varphi\!\left(\frac{m_0 - y}{\sigma}\right)\cdot\frac{1}{m_0 - y}\cdot c,$$

   with $c$ an $O(1)$ constant; the only part that matters is the
   Gaussian factor. The log-odds become, up to $O(\ln \sigma)$ terms
   swamped by the leading one,

   $$\ell(y) \approx \frac{y^2 - (m_0 - y)^2}{2\sigma^2} + O(\ln\sigma) = \frac{m_0}{\sigma^2}\left(y - \frac{m_0}{2}\right) + O(\ln\sigma)$$

   — *exactly the delta-slab formula of §4 with $\mu = m_0$*: a logistic
   gate with temperature $\sigma^2/m_0$ and threshold
   $\tfrac{m_0}{2} + O\!\big(\tfrac{\sigma^2}{m_0}\ln\tfrac1\sigma\big)$.
   As $\sigma \to 0$ the temperature vanishes, the log-correction to the
   threshold vanishes, and the gate hardens to a step at $m_0/2$. ✓
4. Combine: step gate at $m_0/2$, pass-through magnitude. $\blacksquare$

The interpretability content of the hypothesis deserves emphasis: **the
exact-JumpReLU limit requires features to have a minimum firing
strength.** If you believe JumpReLU is the right SAE activation, you are
implicitly claiming activations-when-active are bounded away from zero —
a falsifiable statement about superposition geometry, and arguably the
cleanest justification anyone has offered for that architecture.

### 6. Worked example B: numbers you can follow by hand

Take $\pi = 0.05$, slab uniform on $[1, 2]$ (so $m_0 = 1$, $f(m_0)=1$),
and watch the gate harden as σ falls. For $y < 1$, using the §5 step-3
formula $\ell(y) \approx \frac{1}{\sigma^2}(y - \tfrac12) + \kappa_\sigma$
with the constant $\kappa_\sigma = \ln\frac{\pi f(m_0)\sigma\sqrt{2\pi}}{1-\pi}$
(collecting the prior odds and the Laplace prefactor):

At $\sigma = 0.2$: $\kappa \approx \ln(0.0264) \approx -3.6$, gate
temperature $\sigma^2/m_0 = 0.04$, so the gate's midpoint sits at
$y^\* = 0.5 + 0.04 \times 3.6 \approx 0.64$ and (to two decimals, from
the logistic formula):

| $y$ | 0.50 | 0.60 | 0.70 | 0.80 | 1.50 |
| --- | --- | --- | --- | --- | --- |
| $g(y)$ | 0.03 | 0.24 | 0.80 | 0.98 | ≈1 |
| $\hat a(y) \approx g\cdot y$ | 0.01 | 0.14 | 0.56 | 0.78 | 1.50 |

At $\sigma = 0.05$: temperature drops to $0.0025$, $\kappa \approx -5.0$,
midpoint $y^\* \approx 0.51$ — the transition region is now $\sim 0.01$
wide: visually a step at $\approx 0.5$ with pass-through above. The
hardening and the threshold's drift toward exactly $m_0/2$ are the whole
theorem in one table.

### 7. Contrast example: a slab touching zero ⟹ no JumpReLU, but a famous threshold

Suppose instead firing magnitudes can be arbitrarily small — slab density
$f(0) > 0$ (say exponential). Rerun §5 step 3: the dominant slab point
for small $y$ is now $a \approx y$ itself, so the "on" likelihood is
$O(1)\cdot f(y)$ and the log-odds are

$$\ell(y) \approx \frac{y^2}{2\sigma^2} + \ln\frac{\pi f(y)\,\sigma\sqrt{2\pi}}{1-\pi}.$$

Setting $\ell = 0$ gives the gate's threshold

$$\theta_\sigma \approx \sigma\sqrt{2\ln\frac{1-\pi}{\pi f(0)\sigma\sqrt{2\pi}}} \;\sim\; \sigma\sqrt{2\ln(1/\sigma)},$$

which $\to 0$ absolutely (the pointwise limit of $\hat a$ is a plain
ReLU) but $\to\infty$ *in units of the noise* — every finite-σ gate still
rejects fluctuations several σ tall. Readers from the wavelet-denoising
world will recognise $\sigma\sqrt{2\ln(\cdot)}$ as the **Donoho–Johnstone
universal threshold**; the classical sparse-estimation literature derived
this whole family of shrinkage nonlinearities in the 1990s, with priors
playing exactly the role they play here (Laplace slab → soft
thresholding, i.e. a shifted ReLU; spike-and-slab → the sigmoid-gated
forms above). Our contribution is only the transplant: *SAE activations
are shrinkage operators, and the denoising objective is what makes that
identification exact rather than analogical.*

### 8. From the scalar channel to an SAE

An SAE latent sees $p_i = \langle w_i, \tilde x\rangle + b_i$ where
$\tilde x = x + \sigma_{\text{train}}\varepsilon$ and
$x = \sum_j a_j d_j + \text{noise}$. If decoder directions are
approximately orthonormal and the encoder is roughly matched
($w_i \approx d_i$), then

$$p_i \approx a_i + \underbrace{\textstyle\sum_{j\neq i} a_j \langle w_i, d_j\rangle}_{\text{interference}} + \sigma_{\text{train}}\,\langle w_i, \varepsilon\rangle.$$

The last term is exactly Gaussian; the interference term is a sum of many
small, weakly dependent contributions, hence approximately Gaussian by
the CLT — this is the "superposition interference is the noise"
identification. So each latent is the scalar channel of §1 with an
effective σ, and the per-latent MMSE activation is the gate-times-
magnitude of §2 with that σ. Three consequences, mapped to experiments:

- **JumpReLU** = the $\sigma\to0$ slice under the minimum-strength prior
  (§5) — hence the post-hoc gate-swap ablation (evaluate trained TopK
  dictionaries under threshold gates) is a test of the *slice*, currently
  running; the σ-conditioned `bayes_gate` architecture is the full
  family.
- **TopK**, in this language, is MAP inference under an "exactly $k$
  active" prior — a prior mismatched both to per-token variability and to
  the soft mid-σ posteriors, with the added optimisation pathology that
  hard winner-take-all gates give dead latents zero gradient (measured:
  31–57% dead at 100M tokens; AuxK exists to patch precisely this, and a
  sigmoid gate would not need it).
- **The categorical row of the table**: if the prior says exactly one of
  $M$ templates is active (the polynomial clock), the same derivation
  yields softmax over templates with temperature $\sigma^2$ — the
  posterior head, whose measured behaviour (temperature floor ~0.2–0.4,
  94% Bayes-gap closure on template data) is this note's theorem in its
  natural habitat.

### 9. Tweedie cross-check (optional but satisfying)

Tweedie's formula says the MMSE denoiser can be written *without ever
mentioning the prior*, via the marginal density $p_\sigma$ of $y$:

$$\hat a(y) = y + \sigma^2 \frac{d}{dy}\ln p_\sigma(y).$$

Exercise for the delta-slab case: the marginal is the two-component
mixture $p_\sigma(y) = (1-\pi)\mathcal N(y;0,\sigma^2) + \pi\mathcal N(y;\mu,\sigma^2)$.
Differentiate its log, simplify with the same $2\mu y - \mu^2$ algebra as
§4, and the right-hand side reduces to $g(y)\,\mu + (1-g(y))\cdot 0$ plus
a vanishing correction — the §4 estimator recovered from the score. This
is the bridge to diffusion proper: a trained denoiser *is* an estimate of
$\nabla \ln p_\sigma$, and everything above is a statement about what
that score implies for code shapes.

### 10. Summary table

| feature prior | MMSE activation at noise σ | $\sigma\to0$ limit |
| --- | --- | --- |
| spike + slab bounded off zero | logistic gate (T = $\sigma^2/m_0$) × Wiener-shrunk linear | **JumpReLU**, threshold $m_0/2$ |
| spike + slab touching zero | same, threshold $\sigma\sqrt{2\ln(1/\sigma)}$ | ReLU (pointwise) |
| spike + delta slab | logistic gate × constant $\mu$ | step × $\mu$ |
| Laplace (no spike) | soft threshold | shifted ReLU |
| categorical (one of $M$) | softmax, temperature $\sigma^2$ | argmax / one-hot |
| "exactly $k$ active" | — (MAP: TopK) | TopK |

The practical moral, one sentence: *train across the noise ladder with
the activation the prior actually implies — a σ-conditioned soft gate —
and JumpReLU is what you deploy on clean inputs, not what you train
with.*
