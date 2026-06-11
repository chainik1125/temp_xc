---
author: Claude (10h unsupervised sprint)
date: 2026-06-10
tags:
  - results
  - complete
---

## FrequencyBench sprint — theory notes

Working notes; the polished versions of these go into `summary.md`.

### Setup and notation

Cyclic process: $M$ prime (default 101), velocity set $\Omega \subset \Z_M$,
$Y \sim \mathrm{Unif}(\Omega)$, phase $B \sim \mathrm{Unif}(\Z_M)$, symbols
$Q_t = B + Yt \bmod M$ for $t = 0,\dots,W-1$. Embedding $u : \Z_M \to \R^d$;
activations $x_t = u_{Q_t} + \sigma \epsilon_t$ with $\epsilon_t \sim N(0, I_d)$
i.i.d., independent of $(Y, B)$.

Two embedding families:

- **random**: $(u_a)_a$ = random orthonormal frame (rows of a QR-orthogonalized
  Gaussian matrix), $d = M$. Key property: *exchangeable* — relabeling symbols
  does not change the embedding distribution.
- **circle**: $u_a = R\,(\cos\theta_a, \sin\theta_a)^\top$, $\theta_a = 2\pi a/M$,
  $R$ a random $d\times 2$ isometry ($d=8$). Symbol space now has the geometry of
  a circle; nearby symbols have nearby embeddings.

### P1 — single-token ceiling (restated from the proposal)

For any fixed $y$, $Q_t = B + yt$ with $B$ uniform, so $Q_t \mid Y=y \sim
\mathrm{Unif}(\Z_M)$ for every $y$. Hence $I(Y; Q_t) = 0$, and since
$x_t = f(Q_t, \epsilon_t)$ with $\epsilon_t \perp Y$, also $I(Y; x_t) = 0$.
Single-token Bayes accuracy $= 1/|\Omega|$, for **any** (nonlinear) encoder.

### P2 — temporally additive readouts cannot perfectly decode velocity

**Proposition.** Let $\phi_t : \R^d \to \R^m$ be arbitrary measurable per-token
maps (e.g. any per-token SAE encoder), and let a classifier score class $y$ by an
additive-over-time functional
$$S_y(X) = \sum_{t<W} f_{y,t}(x_t), \qquad f_{y,t}(x) = \langle w_{y,t},
\phi_t(x)\rangle + b_{y,t},$$
(this covers any linear probe on concatenated or mean-pooled per-token codes).
Then for any two velocities $y \neq y'$, the classifier cannot satisfy both
$\Prb(S_y > S_{y'} \mid Y{=}y) = 1$ and $\Prb(S_{y'} > S_y \mid Y{=}y') = 1$.

*Proof.* Since $Q_t$ is marginally uniform on $\Z_M$ conditional on $Y$ being
*any* fixed velocity (P1 argument, per token),
$$\E[S_y \mid Y] = \sum_t \frac1M \sum_{q \in \Z_M} \E_\epsilon
f_{y,t}(u_q + \sigma\epsilon)$$
is the same for every conditioning velocity $Y$. So $D := S_y - S_{y'}$ has
$\E[D \mid Y{=}y] = \E[D \mid Y{=}y'] =: \Delta$. If $D > 0$ a.s. under $Y{=}y$
then $\Delta > 0$ unless $D = 0$ a.s.; if $D < 0$ a.s. under $Y{=}y'$ then
$\Delta < 0$ likewise. Contradiction. $\square$

Comments:

- The bottleneck is *additivity across time*, not linearity in $x$: each token
  may be processed by an arbitrarily powerful encoder; what is forbidden is any
  interaction between tokens before the readout.
- The theorem only rules out *perfect* separation (equal means do not pin down
  tail behavior). Empirically the gap is much larger: linear probes on stacked
  per-token codes sit at chance (see results). State both honestly.
- A single layer of *temporal filters + nonlinearity* (one TXC atom = windowed
  linear filter followed by TopK/ReLU) breaks additivity, and suffices: e.g.
  template/transition detectors make velocity exactly linearly decodable from
  codes. So "dictionary that mixes time before its nonlinearity" is precisely
  what converts temporal structure into linearly decodable codes.

### P3 — exact ratio invariance for random embeddings

**Proposition.** Let the embedding be exchangeable (random frame). For
$c \in \Z_M^\times$, let $cT$ denote the task with velocity set
$c\Omega = \{cy\}$. Then the joint distribution of (windows, labels) of task
$T(\Omega)$ and of $T(c\Omega)$ (labels transported by $y \mapsto cy$) are
identical. Consequently any training + evaluation pipeline (any architecture,
any probe, any sample size) has identical performance distribution on the two.

*Proof.* Apply the symbol relabeling $\psi_c(a) = ca \bmod M$: it maps
$Q_t = B + yt$ to $cB + (cy)t$ with $cB$ uniform, i.e. class-$y$ data of
$T(\Omega)$ becomes class-$cy$ data of $T(c\Omega)$ exactly. Exchangeability
of the embedding absorbs the relabeling. $\square$

*Implementation caveat.* A Haar-distributed orthonormal frame is exchangeable
(uniform measure on the Stiefel manifold is invariant under the right
$O(M)$-action, which includes permutations). Our generator takes Q from
numpy's QR of a Gaussian matrix, which is Haar only up to per-vector sign
conventions whose distribution is weakly column-order-dependent; strict
exchangeability is therefore an idealization of the implementation. The
prediction it licenses (all pair tasks of equal ratio class equally hard, and
empirically *all* random-embedding pairs uniformly hard) holds to within seed
noise (§ pair-task results), so the deviation is negligible in practice.

**Corollary (two-class tasks).** The difficulty of the two-velocity task
$\{y, y'\}$ depends only on the ratio orbit $\{r, r^{-1}\}$, $r = y'/y \bmod M$
(taking $c = y^{-1}$ or $y'^{-1}$; the reflection $a \mapsto -a$ gives sign
freedom). **There is no notion of "frequency" in the random-embedding task**:
$\{1, 2\}$ is exactly as hard as $\{8, 16\}$ and $\{50, 100\}$; difficulty is
*multiplicative*, not metric.

For the 10-class task the full joint is not exactly invariant (multiplication
does not fix $\Omega$ as a set), but pairwise confusability inherits the ratio
structure. Empirical prediction: the 10-class confusion matrix on random
embeddings is organized by ratio (e.g. the five ratio-2 pairs in
$\Omega = \{0,1,2,4,8,16,24,32,40,50\}$: (1,2),(2,4),(4,8),(8,16),(16,32)),
NOT by $|y - y'|$.

### P3 (second part) — coincidence-line geometry (why ratio governs confusability)

Windows $X(B, y)$ and $X(B', y')$ share a symbol at slot pair $(t, s)$ iff
$yt - y's \equiv B' - B$. Solutions form the line $t \equiv \delta + r s$
($r = y'/y$, $\delta = (B'-B)/y$) on the $(t,s)$ torus. For uniform phases the
*expected* number of shared slots is $W^2/M$ for every pair, but the
*distribution* depends on $r$ alone. Small-integer ratios give aligned lines
with up to $W/\max(|p|,|q|)$ in-window solutions for $r = p/q$: e.g. $r = 2$
windows can share $W/2 = 8$ of 16 symbols, while generic ratios scatter
(max 2–3 shared). Bag-of-symbols (smoothing) detectors confuse exactly the
high-overlap pairs. Numerically: compute $\max_\delta N_r(\delta)$ per pair in
$\Omega$ and correlate with measured confusion.

### P4 — circle embedding restores frequency semantics

Project onto the circle plane: $c_t := R^\top x_t \in \R^2 \cong \mathbb C$
gives $c_t = e^{i(\theta_B + 2\pi (y/M) t)} + \sigma\tilde\epsilon_t$ — a unit
tone at temporal frequency $f_y = y/M$ cycles/token with uniform random phase,
in white Gaussian noise. Hence:

1. Velocity *is* temporal frequency; $\Omega = \{0,1,2,4,8,16,24,32,40,50\}$
   becomes the frequency ladder $f \in \{0, .0099, .0198, .0396, .079, .158,
   .238, .317, .396, .495\}$ — DC to Nyquist.
2. The ML classifier (uniform phase) is the **periodogram argmax**:
   $\hat y = \arg\max_{y} |\sum_t c_t e^{-2\pi i y t/M}|$ — classical tone
   detection (Rife–Boorstyn). Our oracle implements exactly this.
3. **Window resolution limit**: tones $y, y'$ have window correlation
   $|D_W((y{-}y')/M)|$ (Dirichlet kernel) ⇒ Rayleigh resolution $\Delta f
   \approx 1/W$. At $W = 16$, classes $\{0, 1, 2, 4\}$ all sit within one
   Rayleigh cell ($\Delta f = 0.0625$); they are distinguishable only via SNR.
   Prediction: capacity-limited architectures fail on the *low*-frequency
   cluster first — and per-class oracle normalization
   $S(\omega) = (A(\omega) - 0.1)/(A_{oracle}(\omega) - 0.1)$ is essential
   to separate "architecture is low-pass" from "low frequencies are
   intrinsically harder at this window length".
4. In DCT-index units ($W=16$, index $w \leftrightarrow$ freq $w/32$): the
   ladder maps to $w \approx \{0, 0.32, 0.63, 1.27, 2.5, 5.1, 7.6, 10.1, 12.7,
   15.8\}$ — neatly spanning the four bands (DC | 1–5 | 6–10 | 11–15) of the
   multiband model. Branch-specialization is therefore sharply testable.

### P5 — constructive spectral atoms (capacity separation, sketch)

Clean velocity-$y$ windows form $M$ templates per class ($|\Omega| \cdot M$
total). An unstructured window dictionary can solve the task by *template
memorization* with $\Theta(|\Omega| M)$ atoms. A spectral dictionary needs only
a quadrature pair (or a small bank of phase-shifted windowed tones) per
frequency: $O(|\Omega| \cdot W/\text{(phase tiling)})$ atoms, independent of
$M$. At dictionary sizes $H \ll |\Omega| M = 1010$ the memorization route is
unavailable and spectral structure should win; at $H \gg 1010$ both routes are
available. This is why we run $H \in \{256, 2048\}$.

### Sign task remark (phase, not energy)

$S = \pm 1$ sequences are time reversals of each other, so they have identical
window power spectra; the sign lives in the *relative phase* (quadrature) of
the two circle channels — equivalently the sign of the rotation. Energy
detectors (power-spectrum features, band-energy summaries) are provably blind
to it; phase-sensitive filters (signed transition detectors, quadrature pairs)
are not. The AC-sign benchmark is a phase-detection task.
