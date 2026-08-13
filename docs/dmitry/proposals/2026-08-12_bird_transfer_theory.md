---
author: Claude (theory agent)
date: 2026-08-12
tags:
  - proposal
---

## BIRD transfer theory: deriving the measured transfer phenomena from the posterior-code framework

Companion to [[2026-08-10_bird_temporal_codes]] (the framework),
[[2026-08-10_bird_clock_results]] (the confirmed laws),
[[2026-08-11_backtracking_detection_dsm]] and
[[2026-08-11_backtracking_steering_dsm]] (the LLM measurements), and
[[2026-08-12_arc_review]] (the claim taxonomy). The MMSE gate derivation
lives at `experiments/diffusion_txc/2026-08-11_jumprelu_mmse_note.md`.

This note asks: do the five transfer phenomena the program has *measured*
follow from the framework it started with — the restricted Bayesian
posterior over a template bank, the softmax encoder with noise-tied
temperature $\sigma^2$, the entropy law $S = (h{+}1{-}W)^+\ln q$, the
$L_0$ law $q^{\,h+1-W}$, and the DSM optimum as posterior mean / score of
the *training* density? Where they do, the derivation is given in full,
pedagogical-style, with the polynomial clock as the exactly-solvable
sandbox and small CPU numerics as checks. Where they do not, the missing
assumption is named rather than papered over.

Scoreboard, one line per fact:

- **F1 (direction-deep OOD collapse)** — derived, *given one extension*:
  the spike ("no template") hypothesis from the MMSE note. The DSM
  encoder is a likelihood-ratio test against a $\sigma$-ladder-widened
  null; off-distribution the ratio is negative almost surely, and the
  gate's positivity floor makes that unrecoverable by any threshold.
  Pure BIRD (closed softmax bank) cannot produce silence at all — see §9.
- **F2 (objective×domain interaction)** — derived cleanly; the strongest
  derivation in the note. Reconstruction's optimum is a support/second-
  moment functional; DSM's is a full-density functional with the prior
  entering the preactivation as $\sigma^2\ln\pi_\beta$.
- **F3 (live pool does not widen on-domain)** — derived: the per-domain
  pool is a covering-number law, not a health metric, and the framework
  *retrodicts the failed pool-revival prediction*. New quantitative
  cash-out: two independent measured live-ratios agree on an effective
  alphabet ratio $q_{\text{web}}/q_{\text{trace}} \approx 1.9$ to 1%,
  with an implied temporal saturation depth $h_{\text{eff}} \approx 4$.
- **F4 (model identity dominates text domain)** — framed, and it yields
  the cheapest testable prediction of the note (linear pullback between
  base and distill `resid_L10`); the affine-drift assumption itself is
  extra-theoretic.
- **F5 (handles are domain-born for both objectives)** — allocation half
  derived (best off-domain atom alignment $\sqrt{2\ln H/d} \approx
  0.07$); the causal half needs a bridge assumption BIRD does not supply.

Section map: §1 imports, §2 F2, §3 F1, §4 F3, §5 F4, §6 F5, §7 the
$\sigma$-conditioned gate aside, §8 pre-registration, §9 honest
failures, §10 numerical appendix.

### 1. Imports: the three optima in one place

Everything below manipulates three objects.

**The BIRD posterior code.** For a template bank $\{\Phi_\beta\}$ of
equal-norm restricted templates and a window $x$,

$$z_\beta(x) = \operatorname{softmax}_\beta\!\left(\frac{\langle \Phi_\beta, x\rangle}{\sigma^2} + \ln \pi_\beta\right),$$

where $\pi_\beta$ is the prior over templates — uniform in the original
clock work, *not* uniform once training corpora are mixtures (§2). On
the clock ($q$, $h$ known, $W$-windows) the noiseless posterior is
uniform on the consistency coset: per-window support $q^{\,h+1-W}$ for
$W \le h{+}1$, one-hot above; realizable distinct windows number
$q^{\min(W,\,h+1)}$; entropy $S = (h{+}1{-}W)^+\ln q$. Three different
counts — active-per-window, realizable-window bank, and width $H$ —
which §4 is largely about not conflating.

**The DSM optimum.** Training $f$ to predict clean $x$ from
$y = x + \sigma\varepsilon$ under density $p$ gives

$$f^*_p(y) = \mathbb{E}_p[x \mid y] = \frac{\int x\, p(x)\, \mathcal N(y;x,\sigma^2 I)\,dx}{\int p(x)\, \mathcal N(y;x,\sigma^2 I)\,dx} = y + \sigma^2 \nabla_y \ln p_\sigma(y),$$

the posterior mean, equivalently (Tweedie) the score of the
noise-smoothed *training* density. Per latent, the MMSE note factorises
this into gate × magnitude: $\hat a_i(y) = g_i(y)\, m_i(y)$ with the
gate a logistic in the log-odds
$\ell_i(y) = \ln\frac{P(y \mid i\ \text{on})\,\pi_i}{P(y \mid i\ \text{off})\,(1-\pi_i)}$,
temperature $\sigma^2/\mu$, threshold $\to m_0/2 > 0$ as
$\sigma \to 0$. Crucially the deployed gate has a **positivity floor**:
`posthoc_gate_evals.py` computes `relu(pre) * (pre > theta)`, and the
MMSE reading agrees ($\theta \to m_0/2 > 0$; firing magnitudes are
nonnegative). A latent whose preactivation never crosses zero cannot
fire at *any* threshold. This floor is load-bearing for F1.

**The reconstruction optimum.** $\min \mathbb{E}_p\|x - D\,z_E(x)\|^2$
with a TopK gate. No noise hypothesis is ever priced; the gate is a
*rank* gate (top-$k$ of the preactivations, whatever their absolute
level), and the preactivation is a matched-filter score with no null
subtraction. Both differences do work below.

### 2. F2 — the two objectives are functionals of different things

Measured (mixed-corpus pair, [[2026-08-11_backtracking_detection_dsm]]):
trace-text training flips DSM detection from worst to best
(sentence S8 $0.181 \to 0.208$, S32 $0.209 \to 0.242$, reaching stage-B
TXC level) while moving recon barely (S8 $0.196 \to 0.190$, S32
$0.215 \to 0.228$). An interaction, not a main effect. Derivation in two
propositions.

**Proposition R (reconstruction reads the support).** Let
$L[\theta; p] = \mathbb{E}_{x\sim p}\, e_\theta(x)$ with
$e_\theta(x) = \|x - D z_E(x)\|^2 \ge 0$, and suppose the dictionary
class has $\varepsilon$-sufficient capacity on $\operatorname{supp}(p)$:
some $\theta$ achieves $e_\theta(x) \le \varepsilon^2$ pointwise there.
Then for **any** reweighted $\tilde p$ with
$\operatorname{supp}(\tilde p) \subseteq \operatorname{supp}(p)$, every
$p$-minimizer $\theta_p$ satisfies
$L[\theta_p; \tilde p] \le \sup_{\operatorname{supp} p} e_{\theta_p} \le \varepsilon^2$,
i.e. it is $\varepsilon^2$-optimal for $\tilde p$ too. *Proof*: the loss
is a nonnegative integrand weighted by the density; a pointwise-small
error is small under every reweighting. $\blacksquare$

So with capacity to spare, the reconstruction optimum is determined by
$p$ only through its **support**; density-within-support enters only
when capacity binds, and then only through low moments of the residual
(for a linear autoencoder, exactly $\Sigma_p$ — PCA). Moving the corpus
from FineWeb to 72/28 trace/web changes the support little (both are
base-Llama activations; trace text is inside the model's input
distribution) and the second moments modestly — the recon dictionary,
and hence its detection number, barely moves. Measured: $0.196 \to
0.190$.

**Proposition D (DSM reads the density).** $f^*_p(y)$ above is a *ratio
of density-weighted integrals*: reweighting $p$ changes it pointwise at
every $y$ within noise-reach of the reweighted region — even at
infinite capacity. There is no "capacity to spare" escape, because even
the ideal solution depends on the density, not the support. In gate
form the dependence is exactly the prior term of §1's softmax: with a
non-uniform template prior,

$$z_\beta \propto \pi_\beta\, e^{\langle \Phi_\beta, x\rangle/\sigma^2} \quad\Longleftrightarrow\quad \text{preactivation shift } +\,\sigma^2 \ln \pi_\beta.$$

A template with vanishing training mass has its gate threshold raised
by $\sigma^2 \ln(1/\pi_\beta)$ — for $\pi_\beta \to 0$ the gate is
effectively welded shut *even when the direction is perfectly in the
span*. Conversely, raising a domain's mixture weight from $\approx 0$
to $0.72$ lowers every one of its templates' thresholds by
$\sigma^2 \ln(0.72/\epsilon)$, opening their gates at realistic
evidence levels.

**Application to the measured interaction.** The FineWeb-trained DSM
dictionary carried trace-domain templates at $\pi \approx 0$: its
live-on-trace pool was web templates firing by accident of projection,
carrying no label information — detection at the raw floor (0.181).
Mixing traces into the corpus did not need to (and per §4 *could not*)
widen the pool; it changed which templates own the gates. The live pool
became the trace domain's own template set, which is
backtracking-discriminative — detection jumps to 0.208/0.242.
Reconstruction, a support functional, had nothing comparable to gain:
$0.196 \to 0.190$. The interaction sign pattern — DSM moves a lot,
recon barely — is exactly the two propositions side by side.

*Worked clock scale.* At $\sigma = 0.3$ ($\sigma^2 = 0.09$), moving a
template's prior from $10^{-6}$ to $0.72/155 \approx 4.6\times10^{-3}$
shifts its preactivation by $0.09 \times \ln(4.6\times10^{3}) \approx
0.76$ — the same order as the clock's atom score margin ($W - h = 2$ at
$W = 3$), i.e. prior mass is worth as much as a full symbol of
evidence. Density is not a tiebreaker; it is a first-order term of the
decision statistic.

### 3. F1 — the OOD collapse is direction-deep because the DSM gate is a likelihood-ratio test

Measured (recalibration probe, [[2026-08-11_backtracking_steering_dsm]]):
the FineWeb-trained `w6_dsm` on distill-model windows has ~214/16,384
latents with positive preactivation, ~1.4 per window; reconstruction
$\approx b_{\text{dec}}$ plus 1–2 latents at NMSE 0.26; per-latent rate
recalibration revives 214 → 215 (nothing) at realized $L_0$ 1.36, while
the recon twin revives 8,586 → 16,046 at $L_0 \approx 111$. Why is the
DSM failure in the *directions* and the recon failure in the
*thresholds*?

**Step 1 — what the DSM preactivation is.** By the MMSE factorisation,
the optimal DSM encoder output for latent $i$ is a monotone function of
the log-likelihood-ratio $\ell_i(y)$ between "feature $i$ present,
embedded in $p$-typical interference" and the null "feature $i$ absent"
— where, because training draws $\sigma \sim \mathrm{LogUniform}(0.05,
1.0)\times\mathrm{RMS}$, **the null includes Gaussian displacements up
to the full RMS of the data**. The trained null hypothesis is not
"$y = $ background"; it is a *diffuse* hypothesis covering a thick
isotropic shell around $\operatorname{supp}(p)$.

**Step 2 — acceptance regions are small and $p$-anchored.** A
likelihood-ratio test between a *specific* hypothesis (feature $i$ on:
slab scale $s$, support concentrated near $\mu_i$ + $p$-typical
context) and a *diffuse* one (isotropic $\sigma_{\max}$-noise off
$p$) accepts only on a set of measure that shrinks like
$(s/\sigma_{\max})^{d_{\text{eff}}}$ — a small region glued to the
training density's own signal manifold. Everywhere else — including,
generically, the entire support of a shifted/rotated eval density $r$ —
the diffuse hypothesis wins and $\ell_i(y) < 0$. The σ-ladder is what
*teaches* the encoder to answer "noise, gate closed" for any
displacement not along a trained signal direction. This is motivation
4's density/anomaly monitoring, experienced from the wrong side: the
dictionary is functioning as the OOD detector it was trained to be, and
it is flagging the deployment distribution.

**Step 3 — the positivity floor makes it unrecoverable.** With
$\ell_i(y) < 0$ for $r$-almost-every $y$, the preactivation never
crosses zero. The gate semantics (`relu(pre) * (pre > theta)`; MMSE
threshold $m_0/2 > 0$) mean recalibration can only move $\theta$ *down
toward zero*, never below it in effect. Hence 214 → 215: no threshold
choice exists. The predicted per-window positive count is
$H \times \Pr_r[\ell_i > 0] = O(1)$ — measured 1.4.

**Step 4 — why the pool is tiny even though TopK force-fires 96.** The
top-$k$ union over windows is small iff the preactivation *ordering* is
frozen across windows. Across eval windows,
$\operatorname{Var}_r(\langle w_i, y\rangle) = w_i^\top \Sigma_r w_i$:
the trained discriminants $w_i$ span a subspace adapted to
$\Sigma_p$-signal, and if fine-tuning moved the informative directions,
$w_i^\top \Sigma_r w_i \ll w_i^\top \Sigma_p w_i$. The preactivation
field over eval windows is then a *quenched* pattern (set by the fixed
misalignment, not by window content), the same ~600 latents win the
top-96 everywhere, and the union pool is 605/16,384 — as measured at
the steering site.

**Step 5 — the recon contrast.** The recon preactivation is a raw
matched-filter score with no null subtraction: under $r$ it straddles
zero (its sign tracks the correlation of the window's fluctuation with
$d_i$, which survives partial subspace overlap), the rank gate keeps
the pool wide (65.1% live, measured), and a per-latent quantile
threshold with the positivity floor revives every latent with positive
tail mass — 8,586 → 16,046, at a mild NMSE cost. Threshold-shallow, as
measured.

**Step 6 — the deceptively low NMSE.** With gates shut, the posterior
mean reverts to the trained prior: $\hat x \approx b_{\text{dec}}$ plus
the 1–2 templates that weakly match. If the shared component of the
activation second moment (mean + dominant common directions) carries a
fraction $\rho$ of $\|x\|^2$, then NMSE $\approx 1 - \rho$ with zero
window-specific transfer; the measured 0.26 says $\rho \approx 0.74$,
which is unremarkable for a residual-stream site. Tweedie sharpens the
reading: the learned score field points *back toward*
$\operatorname{supp}(p)$, so used as a projector on OOD activations the
dictionary actively rewrites them into training-typical content — which
is why the denoise-after-steer arm destroyed generation at $\alpha = 0$.
Corollary, now derived rather than observed: **NMSE without a live-pool
count is uninformative** about transfer.

**Clock check** (full listing §10): $q{=}31$, $h{=}1$, $W{=}3$,
$\sigma{=}0.05$; Bayes preactivation $\ell_\beta = (\langle \Phi_\beta,
x\rangle - W/2)/\sigma^2$, recon proxy = raw score with rank gate. A
model change is a rotation of the alphabet. On-domain: 1.00 fires per
window, pool = bank, identification 1.000. Rotated: fires per window
0.0000, LLR pool 0, max score 1.28 against threshold 1.5 — while the
rank gate's pool stays at 755/961 and quantile recalibration revives
**0/961 for the Bayes arm vs 961/961 for the recon arm**. The measured
215-vs-16,046 asymmetry, from mechanism.

### 4. F3 — the live pool is a covering number, not a health metric

Measured: `w6mix_dsm`, trained on a 72/28 trace/web mix, logs
dead-fraction $\approx 0$ on its training stream yet is **94.8% dead on
trace-only windows (~849 live)** — and that pool carries stage-B-level
detection. The FineWeb-trained `dsm` is ~0% dead on FineWeb. The
pre-registered prediction "domain match revives the pool" failed while
the outcome succeeded. The framework, applied correctly, says the
prediction *had* to fail.

**Step 1 — three counts, not one.** The framework has three distinct
numbers: the per-window active count ($L_0$ law, $q^{\,h+1-W}$), the
**realizable-window bank** ($q^{\min(W,h+1)}$ distinct clean windows —
the dictionary-size law), and the width $H$. The live pool of a trained
dictionary on a domain $A$ is the second one: the set of atoms whose
acceptance regions intersect $\operatorname{supp}(p_A)$, i.e. the
**covering number $N_A$ of the domain's window set at the dictionary's
resolution**. It is a property of the *domain*, not of the dictionary's
health.

**Step 2 — mixtures.** Training on $\lambda\, p_A + (1{-}\lambda)\,
p_B$ keeps every atom alive through its own domain:
$\text{pool}(\text{mix}) = \text{pool}(A) \cup \text{pool}(B)$. If
$N_B \gtrsim H$ (web) the union saturates $H$ and training
dead-fraction reads $\approx 0$ — while $\text{pool}(A) = N_A$ stays
exactly as small as domain $A$'s diversity dictates. Extreme per-domain
concentration is the **correct restricted-posterior behaviour**; the
entropy law *is* the concentration. Clock check (§10): 72/28 mixture of
a 155-template subdomain with the full 961-bank — training
dead-fraction 0.003, subdomain-eval pool **exactly 155** ($=|A|$),
dead-fraction 83.9% on-domain. The 94.8% phenomenon, exactly, in a
system where it is provably optimal. A monitoring lesson falls out for
free: log per-domain dead-fraction, since the aggregate is blind to
this by construction.

**Step 3 — the quantitative cash-out: a ratio law.** Model the two text
domains as clocks with effective per-token alphabets $q_{\text{web}} >
q_{\text{tr}}$ and shared correlation depth $h_{\text{eff}}$. The
live-ratio between domains at window length $W$ is then

$$r(W) \;=\; \frac{\text{live}_{\text{web}}(W)}{\text{live}_{\text{tr}}(W)} \;=\; \left(\frac{q_{\text{web}}}{q_{\text{tr}}}\right)^{\min(W,\ h_{\text{eff}}+1)}.$$

Two independent measurements exist (both at the detection site,
base-Llama activations; different hookpoints and dictionaries):

- per-token DSM SAEs (`ln1_L10`): live 95.4% on FineWeb vs 49.5% on
  traces → $r(1) = 1.93$;
- `w6_dsm` ($W{=}6$, `resid_L10`): live 98.8% vs 3.8% → $r(6) = 26.0$.

Unsaturated compounding would give $r(1)^6 = 51.6 \ne 26.0$; instead
$\ln r(6)/\ln r(1) = 4.96$, i.e. the saturating law with
$\min(6, h_{\text{eff}}{+}1) \approx 5$, an effective correlation depth
$h_{\text{eff}} \approx 4$ tokens. Cross-check: the law then demands
$q_{\text{web}}/q_{\text{tr}} = r(6)^{1/5} = 1.92$ against the directly
measured $r(1) = 1.93$ — **agreement to 1% between two dictionaries at
two hookpoints**. The implied diversity gap is
$\log_2 1.92 \approx 0.94$ bits/token: R1 traces are about one bit per
token more templated than FineWeb as seen by these dictionaries.
Caveats stated plainly: different hookpoints and budgets; identifying
activation-pool ratios with token-alphabet ratios assumes comparable
quantization resolution across domains. §8 pre-registers the two cheap
tests this law makes falsifiable ($W{=}3$ interpolation; $H$-invariance).

**Step 4 — what ~849 implies.** If the code were one-hot, the trace
domain would have $N_{\text{tr}} \approx 850$ realizable window types
at dictionary resolution. With $k = 96$ compositional slots the pool is
the *atom alphabet* and the expressible window set is
combinatorially vast ($\binom{849}{96}$-scale), so 849 atoms is nowhere
near a capacity bottleneck for the domain either way. The honest
statement is the bracket, plus the invariances (§8): the pool tracks
the domain, not $H$. External anchors worth checking (proposed, not
done): the count of distinct coarse-grained sentence-opening windows
covering ~90% of the 25,528 judged trace sentences should be
$O(10^2$–$10^3)$; and note the *trace-trained* stage-B per-token SAE is
itself 42.5% dead on traces — even a recon dictionary trained on the
domain leaves much of $H$ unused, the same low-diversity signature.

**Step 5 — why the tiny pool carries full signal (the controls).** The
detection controls (`detection_controls.json`) measured: `w6mix_dsm`
restricted to its 849 live latents scores sentence S8 0.203 / far S32
0.346 vs full `w6mix_recon` (15,274 live) at 0.201 / 0.321 — **~18× the
per-latent signal, equal totals** — while `w6mix_recon` subsampled to
849 *random* live latents collapses to 0.123–0.146, at or below the
label-shuffle floor (0.144) and the untrained-random-dictionary arm
(0.128). Derivation of all three numbers at once:

- *Equal totals are forced, given sufficiency.* Restricting `w6mix_dsm`
  to its live pool discards nothing — on this domain the other 15,535
  latents never fire, so dsm@849 *is* the dsm code. If both codes are
  approximately sufficient statistics of the window for the domain
  (both reconstruct trace windows at their operating fidelity), the
  probe on either is capped by the same $I(\text{window};
  \text{label})$, and totals must match. The framework's nontrivial
  claim is that the covering set is sufficient *despite* being 5.2% of
  $H$ — which is Step 1.
- *The 18× per-latent ratio is then arithmetic*: equal totals divided
  by pool sizes, $15{,}274/849 \approx 18$. Not a coincidence; an
  identity given sufficiency.
- *The random-subsample collapse is the substantive asymmetry.* Recon
  spreads window information democratically across a redundant,
  variance-aligned frame (feature splitting); a random 1/18th of a
  democratic frame is not sufficient, and top-$S$ selection by
  per-latent t-statistic needs *individual* latents to carry label
  signal. DSM's basis is identity-aligned — templates are window
  types, and the label (a sentence type) is a union of templates — so
  single latents are label features. This is B3's atom-purity result
  ([[2026-08-10_bird_clock_results]]) expressed behaviourally.

**Step 6 — the failed prediction, retrodicted.** Domain training
changes *which* templates own gates (§2's prior term), not *how many*
the domain supports (this section's covering law). The framework
predicts exactly the split verdict the mix run delivered: detection
flips, pool stays ~850 (605 → 849, a factor 1.4, not 15×). The
program's own intuition ("the pool will revive") was the part the
framework never licensed.

### 5. F4 — model identity is a coordinate change of the template bank

Measured: the same trace text through base Llama vs the R1-distill
fine-tune yields no transfer — `w6mix_dsm` on distill activations sits
at NMSE 0.807 with 5.2% live, despite trace-*text* training fixing the
detection site the same morning.

**Step 1 — text overlap bounds nothing.** The activation density is a
pushforward: $p_{M,\text{text}} = (g_M)_*\,\mu_{\text{text}}$, where
$g_M$ maps context to the layer-10 window. The code is built from inner
products $\langle \Phi_\beta, x\rangle$ — the template bank lives in
the *model's* coordinate system. Two models define two pushforwards of
the same text measure, and nothing in the framework (or in measure
theory) makes their overlap track the text overlap: the overlap is a
property of $g_{M}$ vs $g_{M'}$, i.e. of the fine-tune. On the clock,
"fine-tune" is an alphabet rotation $u_a \mapsto R\,u_a$: template
scores collapse from $W$ to $\mathcal N(0, W/d)$-scale fluctuations,
and the §3/§10 numeric — fires 0.0000, LLR pool 0 — doubles as the F4
sandbox. NMSE 0.807 / 5.2% live is that signature at LLM scale.

**Step 2 — the same number, two mechanisms.** The mix arm is ~5.2%
live at *both* sites (849 on base-Llama trace windows, ~850 on distill
windows). Numerical coincidence, different mechanisms, and the
framework says which is which: on-domain the pool is the covering set
and the active set varies with window content (§4); cross-model the
preactivation field is quenched (§3 Step 4) and the active set is
frozen. Discriminating statistics, cheap to compute from the cached
windows: NMSE already dissociates them (ordering-gate 0.448 on-domain
vs 0.807 cross-model); the mean pairwise Jaccard of active sets across
windows should be content-varying on-domain and near-constant
cross-model. §8 pre-registers the direction.

**Step 3 — the measurable coupling.** If the fine-tune's effect on the
layer-10 representation is approximately affine on the relevant support,

$$x_{\text{distill}} \approx A\,x_{\text{base}} + c + \xi,$$

then the bank is not destroyed, only re-expressed: encoding
$\hat A^{-1}(x' - \hat c)$ restores the inner products up to the map
residual $\xi$ — and the residual is exactly the kind of isotropic-ish
perturbation the $\sigma$-ladder trained the dictionary to absorb
(corruption up to $1.0\times$RMS). This is the one transfer setting
where DSM's noise training should *pay* rather than punish. The test is
far cheaper than retraining: one forward pass of the same trace text
through both models, positions aligned, ridge-fit $A, c$, re-run the
pre-flight gates through the pullback. Predictions and the kill
condition are pre-registered in §8 (P5/K5). The framework does **not**
derive that fine-tuning is affine — that is an added assumption (§9);
what it derives is the sharp dichotomy: high-$R^2$ map + revival ⇒
same bank, rotated coordinates; high-$R^2$ map + no revival ⇒ the bank
itself changed (feature birth), and the BIRD reading of F4 dies.

### 6. F5 — handles are allocation, and allocation follows training mass for both objectives

Measured: the FineWeb-trained `w6_recon`, **65% alive** at the steering
site, has zero directional handle (excess-anti $-0.02$,
CI $[-0.13, +0.11]$-scale), while the trace-trained stage-B crosscoder
has $+0.42$ $[+0.31, +0.52]$; the trace-trained per-token SAE sits at
$+0.069$.

**Step 1 — both objectives allocate atoms by functionals of $p$.**
Recon places atoms to cover the support / span the variance (Prop. R's
binding-capacity case); DSM places gates by density (Prop. D). They
disagree about *which* functional, but agree about the domain of
integration: **neither allocates an atom to a direction carrying no
training mass.** The backtracking contrast direction $v$ — what
separates D+ from D− windows in the model's geometry — is a
trace-domain object; under FineWeb training it has $\approx$ zero
variance and $\approx$ zero density mass, so neither objective points
an atom at it.

**Step 2 — the off-domain best atom is a random-alignment extreme
value.** For a bank of $H$ atoms adapted to an unrelated density, the
alignment with a generic $v \in \mathbb R^d$ is extreme-value
statistics:

$$\mathbb{E}\ \max_{i \le H} |\cos(d_i, v)| \approx \sqrt{\frac{2\ln H}{d}} = \sqrt{\frac{2\ln 16384}{4096}} \approx 0.069$$

(one numerical draw: 0.065). Steering with the best-mined atom then
injects $\le 7\%$ of its norm on-target and $\ge 99.5\%$ of its
*energy* off-target — phenomenologically a norm perturbation, which is
precisely what the random-direction control produces, and the measured
wave-2 outcome: every window-source CI spans zero, and the mined sign
fails to modulate the steering direction (the fingerprint of a
nonspecific push). Aliveness could not have rescued this: encoder-side
aliveness only requires score regions to intersect the eval support
(§3 Step 5 — recon's matched filters do this for free); a *handle*
requires a decoder atom concentrated on $v$. Different projections of
the dictionary; the first is cheap, the second is bought only with
training mass.

**Step 3 — on-domain, the same argument allocates the handle.**
Backtracking is *frequent* in the trace corpus (base rate 12.4% of
sentences), so $v$ carries $O(1)$ variance and density there — both
objectives allocate atom(s) with $O(1)$ cosine to it. The stage-B
crosscoder (recon-trained, trace-trained) duly has the +0.42 handle:
the framework is objective-agnostic here, exactly as F5 states —
handles are *domain-born*. The measured ladder (conventional/DoM
$\approx 0$ → trace per-token $+0.069$ → trace windowed $+0.42$) then
reads as allocation (needs domain mass) × binding (the window
concentrates a temporally-extended contrast into one atom); the
framework derives the allocation gate and, via §3.5 of the proposal,
motivates but does not quantify the binding multiplier (§9).

### 7. Aside — why the $\sigma$-conditioned gate's dictionary is less noise-robust than TopK-DSM's

Measured (`topk_vs_topkdiff`): trained `bayes_gate` arms win absorption
(0.137, best of any arm) and k=5 probing (+7–11 points) but their
support-Jaccard under input perturbation is the worst of any arm (0.52
vs TopK-dsm's 0.76), and neither $\sigma$-matched conditioning
(0.54–0.55) nor a rank readout (0.55–0.56) recovers it — the fragility
is dictionary-borne. Formal account:

- **Conditioning factorises the optimum.** With $\sigma$ given, the
  Bayes solution is a *family* $f^*(y, \sigma)$: all
  $\sigma$-dependence lives in closed form in the gate parameters
  (temperature $\sigma^2/\mu$, Wiener factor $s^2/(s^2+\sigma^2)$,
  threshold shift), so the *weights* only ever need to implement the
  $\sigma$-independent clean discriminant. Conditioning weakly reduces
  Bayes risk (side information never hurts), and the weights specialise
  to the sharpest, highest-gain discriminants — the same property that
  buys the absorption and probing wins.
- **The blind arm is forced to internalise the ladder.** TopK-DSM has
  one weight set for all $\sigma$; its objective literally contains the
  fragility eval — produce a stable code for $y = x + \sigma\varepsilon$
  across the ladder — so the optimal blind filter is the Bayes response
  to a *composite* hypothesis ($\sigma$ unknown): gates become
  $\sigma$-mixtures, filters are Wiener-averaged over the ladder,
  margins widen. Robustness lives in $w$ because nothing else could
  carry it.
- **Why $\sigma$-matched evaluation cannot restore it.** The
  conditioning channel enters through the gate parameters, not the
  directions; at any conditioning value the deployed $w$ are still the
  specialised clean discriminants. Support stability under an
  *unmodelled* perturbation is a margin property of $w$, and the
  conditioned objective never prices it: training only ever visits the
  diagonal (conditioning = true $\sigma$), so the off-diagonal response
  is unconstrained. The information-flow summary: the blind weights
  must satisfy $I(\sigma\text{-robustness}; w) > 0$; the conditioned
  weights are sufficient *given* $\sigma$ and factor it out.
- **Consequence and a concrete fix.** Specialisation and robustness are
  the same currency spent two ways; the framework predicts they cannot
  co-occur in one conditioned arm unless the objective adds an explicit
  off-diagonal term. Pre-registerable fix: conditioning-jitter (train
  with $u \ne \varepsilon^2$ on a fraction of batches, or conditioning
  dropout) should trade a small part of the probing win for a
  substantial Jaccard recovery; if Jaccard does not move under
  conditioning-jitter, this account of the regression is wrong.

### 8. PRE-REGISTRATION — the distill-captured run, predicted before its data exists

Registered 2026-08-12, before the distill-captured training experiment
(arc review experiment 2) has produced any data. Setup assumed: the w6
DSM recipe ($H = 16384$, $k = 96$, $W = 6$) trained on
DeepSeek-R1-Distill-Llama-8B `resid_L10` activations over trace text,
then the three deployment gates re-run.

- **P1 — live pool on distill traces (the headline prediction).** The
  covering law (§4) says the pool follows the *domain*, not the fix:
  predicted live pool **600–2,500 latents (3.7–15% of $H$; central
  5–10%)**, i.e. the same order as the 849 base-model trace pool —
  emphatically **not** a jump toward 100%. The distill model processing
  its own generations should be at least as templated as base Llama
  processing them, so the central estimate sits at $\approx 850 \times
  [1, 2]$. On a trace-only stream the *training* dead-fraction itself
  will be high ($\ge 80\%$); on a mixed distill stream it will read
  $\approx 0$ while the trace-eval pool stays in the band above.
- **P2 — NMSE regime and the projector gate.** On its own training
  distribution: NMSE **0.05–0.25** at a $\ge$15k-step budget (the
  FineWeb-on-FineWeb regime, 0.046–0.061, plus trace-corpus margin);
  $\le 0.45$ if the budget is the mix run's 8k steps. The $\alpha = 0$
  projector-damage test **passes** (coherent generation, mean Sonnet
  grade $\ge 2$ for the majority of prompts) — *despite* the ~90–95%
  dead pool. This is the framework's sharpest counter-intuitive
  prediction: it decouples pool size from on-domain fidelity, directly
  against the earlier reading that "a ~600-atom sub-dictionary cannot
  substitute for the residual stream". The failure was OOD-ness, not
  pool size.
- **P3 — the frozen-vs-varying discriminator.** On-domain, per-window
  positive-preactivation counts of order $10^2$–$10^3$ (vs 1.4
  cross-model), and the active set varies with content: mean pairwise
  Jaccard of top-96 sets across windows well below its cross-model
  (near-frozen) value.
- **P4 — detection at the matched site.** The live pool carries
  stage-B-band signal: sentence S8 **0.19–0.25** (instrument-limited;
  the weakest of these predictions and not a framework-killer either
  way).
- **P5 — the linear pullback (cheaper than any retrain).** Ridge-fit
  $x_{\text{distill}} \approx A\,x_{\text{base}} + c$ on paired
  same-text forwards: predicted $R^2 \ge 0.7$; encoding pulled-back
  distill windows with the *existing* `w6mix_dsm` revives the live pool
  to $\ge 300$ latents ($\ge$ a third of the base-trace pool) and NMSE
  $\le 0.55$ (from 0.807), the map residual being absorbed by the
  trained $\sigma$-range.
- **Auxiliary registrations from the ratio law (§4).** (i) A $W{=}3$
  FineWeb-trained DSM dictionary is **80–90% dead on traces** (central
  86%, from $r(3) = 1.93^3 = 7.2$). (ii) $H$-invariance: retraining
  `w6mix_dsm` at $H = 8192$ leaves the trace pool at $\approx 850$
  (dead $\approx 89.6\%$), i.e. the pool does not scale with $H$.
  (iii) Base-Llama per-token log-loss on trace text is lower than on
  FineWeb by **0.6–1.3 bits/token** (central 0.9, from the measured
  alphabet ratio 1.92). (iv) A *selected* (top-$|t|$) 849-subset of
  `w6mix_recon` roughly matches full recon, unlike the random subset —
  concentration-by-selection, completing §4 Step 5.

**Falsification conditions** — result patterns that kill the
framework's account, not merely disappoint it:

- **K1 (kills the F1 density account).** The distill-captured DSM
  dictionary, evaluated on its *own training distribution*, still shows
  the collapse signature: pool $< 300$ with near-frozen preactivations
  **and** NMSE $\ge 0.5$ **and** the $\alpha=0$ projector still destroys
  generation. Then the collapse was never distributional and something
  intrinsic to DSM on LLM activation geometry is at fault (the
  isotropic-Gaussian corruption model becomes the lead suspect, per
  B3-P4's discrete-structure warning).
- **K2 (kills the F3 covering account).** The on-domain pool jumps to
  $\ge 50\%$ of $H$, recon-like. Concentration would then be an OOD
  artifact or optimizer pathology, not domain entropy — note this
  "vindication" pattern for the program would *falsify* the framework's
  entropy story, which is why P1's number is registered now.
- **K3 (kills the ratio law).** The $W{=}3$ arm lands far outside
  80–90% dead on traces, or the $H = 8192$ pool scales with $H$.
- **K4 (kills the F2 asymmetry).** A distill-captured *recon* twin
  moves its detection number as much as the DSM twin moved under
  text-domain training — the support/density asymmetry between the
  objectives would then be wrong, not just noisy.
- **K5 (kills the F4 coordinate-change reading).** The fitted map has
  $R^2 \ge 0.7$ but the pullback revives $< 100$ latents / leaves NMSE
  $\ge 0.7$. Same bank rotated ⇒ falsified; the fine-tune changed the
  bank (feature birth) — a more interesting world, but a different
  theory.

### 9. What the framework cannot derive — honest failures

- **F1 needs the spike extension, and pure BIRD refuses to be silent.**
  The restricted posterior over a *closed* template bank is a softmax:
  it sums to 1 on any input, so a genuinely OOD window gets
  *relabelled*, never rejected. Deriving direction-deep silence
  requires the spike/"no template" null with the $\sigma$-ladder giving
  it wide support — the MMSE note's spike-and-slab, an extension beyond
  BIRD proper (principled, already in the repo, but not optional: §3
  collapses without it).
- **SGD $\approx$ Bayes is assumed throughout.** Every derivation reads
  the trained encoder as implementing the MMSE gate/LLR of its training
  density. B1's 94% Bayes-gap closure and B3's purity results support
  this on synthetic data; on LLM activations it is an article of faith
  that the whole note is conditional on.
- **Absolute pool sizes are retrodicted, not predicted.** The framework
  predicts ratios, saturation, and invariances (§4, §8), but ~849
  ab initio would require the trace domain's covering number at the
  dictionary's resolution — not independently measured. The
  sentence-template count is the proposed independent anchor; until it
  exists, 849 calibrates the theory rather than testing it.
- **F5's causal half has no bridge.** BIRD is a theory of *codes*
  (inference), silent on what activation additions do to downstream
  computation. "Decoder atom aligned with the behaviour-contrast
  direction ⇒ causal handle" is an added assumption; the +0.42
  magnitude, the sign inversion against mining, and the U-shaped
  nonspecific response are all outside the framework.
- **F4's affine-drift assumption is extra-theoretic.** The framework
  motivates the pullback test but says nothing about how SGD fine-tuning
  actually moves representations; K5 is the honest split of outcomes.
- **Recon pool widths are not quantitative.** "Support functional"
  explains recon's insensitivity (§2) but does not predict its
  feature-splitting factor (why trace-trained stage-B recon is 42.5%
  dead while mixed recon is 6.8% dead on traces); the framework has no
  model of splitting economics.
- **The corruption model is a modelling choice.** All gate geometry in
  §3 inherits isotropic Gaussian at $\sigma\cdot$RMS; the one synthetic
  setting with quasi-discrete event structure falsified B3-P4. K1 is
  where this bill arrives if it arrives.

### 10. Numerical appendix

CPU-only, seconds, exact clock sandbox ($q{=}31$, $h{=}1$, $W{=}3 =
h{+}2$, $d{=}64$, $\sigma{=}0.05$; bank $M = 961$; Bayes preactivation
$\ell_\beta = (\langle\Phi_\beta, x\rangle - W/2)/\sigma^2$, recon
proxy = raw matched-filter score with rank gate; model change = random
orthogonal rotation of the alphabet; mixture = 72/28 over a
155-template subdomain vs the full bank):

```python
import numpy as np

rng = np.random.default_rng(0)
q, h, d, W, sigma = 31, 1, 64, 3, 0.05
M = q ** (h + 1)
Q_, _ = np.linalg.qr(rng.standard_normal((d, q)))
A_ = Q_.T                                             # orthonormal alphabet
B0, Y = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
betas = np.stack([B0.ravel(), Y.ravel()], 1)
sym = (betas[:, :1] + betas[:, 1:] * np.arange(W)[None, :]) % q
Phi = A_[sym].reshape(M, -1)                          # ||Phi||^2 = W

def windows(idx, alphabet):
    x = alphabet[sym[idx]] + sigma * rng.standard_normal((len(idx), W, d))
    return x.reshape(len(idx), -1)

thr, n = W / 2.0, 3000
S = windows(rng.integers(0, M, n), A_) @ Phi.T        # on-domain
R, _ = np.linalg.qr(rng.standard_normal((d, d)))
S2 = windows(rng.integers(0, M, n), A_ @ R) @ Phi.T   # rotated model
q_hi = np.quantile(S2, 1 - 96.0 / M, axis=0)          # rate recalibration
maskA = np.isin(betas[:, 1], np.arange(5))            # 155-template domain
Aidx = np.where(maskA)[0]
Str = windows(np.concatenate([rng.choice(Aidx, 15000),
                              rng.integers(0, M, 6000)]), A_) @ Phi.T
SA = windows(rng.choice(Aidx, 3000), A_) @ Phi.T
```

Outputs (seed 0):

```text
F1 on-domain : LLR fires/win 1.0000  pool 922/961  id-acc 1.000
F1 rotated   : LLR fires/win 0.0000  pool 0    rank-gate pool 755
               max score 1.28 < thr 1.5 ; recalibration revives
               Bayes 0/961 vs recon 961/961
F3 mix-train : dead frac 0.003
F3 A-eval    : pool 155 = |A| exactly ; dead frac 0.839 ; L0/win 1.00
ratio law    : r1 1.93, r6 26.0, r1^6 51.6 (unsaturated, wrong)
               implied depth ln r6 / ln r1 = 4.96 -> h_eff ~ 4
               q_web/q_trace = 26.0^(1/5) = 1.92 vs r1 = 1.93 (1%)
               entropy gap log2(1.92) = 0.94 bits/token
               W=3 prediction: r1^3 = 7.2 -> 86.2% dead on traces
F5           : E max |cos| over 16384 atoms in R^4096 = 0.069
               (one draw: 0.065)
```

The on-domain pool of 922/961 is coupon-collector coverage of the bank
at 3000 draws ($961(1 - e^{-3000/961}) \approx 919$), not leakage.
Script: `bird_transfer_checks.py` in the session scratchpad; it is
fully specified by the listing above.

## P5 measured (same day)

The pullback probe ran within hours of pre-registration
(`modal_pullback.py`; volume `ooc_recal/pullback*.json`). Verdict: **K5
pattern — the coordinate-change reading of F4 is falsified.** Held-out
map R² = 0.86 (affine drift confirmed, threshold 0.7 passed), but the
pulled-back activations revive nothing (424 → 415 live for w6mix_dsm;
ΔNMSE ≈ 0.02). The base→distill shift is 86% linear, yet the dictionary's
acceptance regions are tight enough (the F1 LLR geometry) that correct
coordinates do not produce accepted evidence — the fine-tune's
behaviourally-relevant component lives in the unmapped residual (feature
birth). Instrument note: a probe-vs-gate NMSE discrepancy (0.2 vs 0.8)
traced to normalization convention (energy-about-mean vs
variance-about-mean); the live-pool count is the convention-free metric
and agrees across instruments — use it for all cross-instrument claims.
Distill-captured training (P1/P2) is now the only remaining route to the
steering site, exactly as this note's main line predicts.

## P1/P3 measured (2026-08-13 morning)

The distill-captured pair trained overnight (8000 steps, identical 72/28
stream to w6mix, only the capture model changed; `txc_w6_distill/`).
Scorecard probe (`modal_p1p3.py`, volume `ooc_recal/p1p3_scorecard.json`,
60 traces teacher-forced through the distill, both NMSE conventions):

| pre-registration | predicted | measured (w6dist_dsm) | verdict |
| --- | --- | --- | --- |
| P1 live pool | 600–2,500 (5–10%) | **1,063 (6.5%)** | hit, mid-range |
| P3 pos. preacts/window | 10²–10³, content-varying | 162 (p10–p90 145–185) | hit |
| P2 (NMSE half) | ≤0.45 at 8k steps | 0.16 (var conv.) | hit |
| K2 kill (pool ≥50%) | would falsify covering law | 6.5% | not triggered |

The recon twin: 14,894 live (90.9%), NMSE 0.102 — democratic allocation
as its own theory requires. The naive coverage-revives-the-pool reading
is now falsified twice (w6mix on-domain, w6dist on-model); concentration
survives full distribution match, as the covering-number law demands,
while the previously welded-shut gates open (1.4 → 162 positive
preactivations per window). Remaining: P2's α=0 projector cell (running)
and K4 (matched-site detection twins).

## P2 projector cell measured (2026-08-13): FAILED — partial K1

With the distill-captured DSM projector (pool alive, gates open,
on-domain), the α=0 denoise-after-steer cell scores mean Sonnet grade
0.25, 0/20 prompts above floor (unprojected reference: 2.85, 20/20). The
density-blind recon twin control is less bad on every metric (0.50,
4/20, gc 0.40 vs 0.15). Both projectors fail; the ordering is opposite
to motivation 1b's claim. Effective code L0 is 96.0/96 for both —
density separates nothing. Failure mode shifted from word salad
(base-captured) to fluent-but-content-corrupted (garbled numerals,
misread problem statements): distribution matching repaired surface
statistics, not computation-carrying precision.

Verdict for the framework: the encoder story (density-gated posterior,
covering-law concentration — P1/P3) stands; the decoder-side
intervention story (Tweedie projection preserves computation at k=96
fidelity) is falsified. Substituting any 96-sparse reconstruction at
every position destroys the residual stream's computational payload;
manifold-projected steering in the substitute-reconstruction form is
dead on-domain for both objectives, not merely untested.
