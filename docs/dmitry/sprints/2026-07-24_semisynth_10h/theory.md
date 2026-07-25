---
author: Claude theory agent
date: 2026-07-24
tags:
  - design
  - in-progress
---

## Coverage, resolution, and timescale: what a window-$W$ steering handle buys

This note formalises the sprint's centrepiece. The empirical arc so far
([[semisynthetic_language_tasks]]) is: naturalistic ordered-generation tasks are
*mode-dominated* and a per-token broadcast beats a windowed template; trajectory-profile
tasks with multiset-matched foils invert that, with template growing linearly in $k$ and
broadcast pinned at zero. The open objections are that per-slot additivity makes the
$k$-sweep look mechanical, and that a per-token SAE with an externally supplied schedule
reproduces the template. Both are answered by separating **two different things a window
buys**, which the $k$-sweep confounds:

- **Coverage** — how many segments *one knob* can reach (Section 2). More $W$ is better.
- **Resolution** — how fast the write can change *inside* that reach (Section 3). More
  $W$ is worse, and the crossover is set by the task's intrinsic timescale $\ell$.

A per-token SAE deployed the standard way (one coefficient held fixed over the
generation) is the $W\!=\!k$ corner of the resolution axis, which is exactly the
broadcast arm. A "scheduled SAE" is the $W\!=\!1$, $k$-knob corner of the coverage axis.
The two axes cross at the full template, and the whole claim is about the price in knobs.

### The setting

A document is $k$ contiguous segments $t = 1\ldots k$. Each carries an attribute
$a_t \in \{-1,+1\}$ (EN/FR, calm/tense). A **profile** $\pi \in \{-1,+1\}^k$ is the
target trajectory. A **handle** writes a field $V \in \mathbb{R}^{k \times d}$, adding
$V_t$ to the residual stream at layer $L$ across segment $t$'s token span. The
teacher-forced metric is the diff-in-diff margin
$\Delta = [\mathrm{lp}(T) - \mathrm{lp}(F)]_{\text{steered}} - [\mathrm{lp}(T) -
\mathrm{lp}(F)]_{\text{base}}$, with foil $F$ realising a permutation $\sigma(\pi)$
(`full_modal.py`).

### 1. When a per-segment template must beat a broadcast

Three conditions, each of which the harness already enforces or can be made to.

- **(V) Within-window variation.** $\pi$ is non-constant. Otherwise template $=$
  broadcast identically and there is nothing to measure.
- **(M) Multiset matching.** $F$ is a permutation of $T$'s attributes, so every bag
  statistic agrees between them.
- **(R) Per-episode randomness.** $\pi$ is resampled per episode with high entropy, so
  no fixed schema is cacheable and $\pi_{t+1}$ is not inferable from $\pi_{1:t}$.

**Claim (first-order broadcast nullity).** Write the broadcast field $V_t = m\,u$ for all
$t$. To first order in $m$, $\Delta_{\text{bc}} \approx m\,\langle \sum_t g_t(T) - \sum_t
g_t(F),\, u\rangle$ where $g_t$ is the gradient of the segment log-prob with respect to
the residual at segment $t$. If sensitivity is *slot-local* — $g_t$ depends on $a_t$ but
not on $t$ or on the rest of the profile — then $\sum_t g_t = \sum_a n_a g_a$ with $n_a$
the multiset counts, and under (M) the two sums cancel: $\Delta_{\text{bc}} = 0$
*exactly*. Any nonzero broadcast effect is therefore second-order, arising only from
positional heterogeneity of $g_t$, profile-dependent context interaction, or curvature.

This explains the measured $\approx 0$ broadcast and, importantly, the *slightly
negative* alt\_phase values: with a fixed target profile A and fixed foil B, the residual
positional asymmetry has a consistent but arbitrary sign. It is not evidence that
broadcasting hurts.

- **Sharp cheap control:** swap the target and foil labels in alt\_phase. Prediction —
  $\Delta_{\text{bc}}$ **flips sign** with roughly equal magnitude, while
  $\Delta_{\text{template}}$ is unchanged. If broadcast stays negative under the swap,
  the DC write genuinely damages the behaviour and the "second-order asymmetry"
  explanation is wrong.

**The two empirical failure mechanisms**, restated as violations:

- **(F1) Mode dominance** violates the spirit of (M). LMs cache window-constant
  properties ("listing weekdays", "tense scene") as a single broadcastable direction. If
  the profile has any DC component $\bar\pi \neq 0$, a broadcast write loads on it and
  the comparison is contaminated. Diagnostic: $\bar\pi$, and
  $\cos(\text{per-slot direction}, u_{\text{DC}})$. Requirement: **balanced profiles**.
- **(F2) Auto-fill** violates (R). The model completes what the prefix determines, so a
  cheap low-bandwidth handle that sets the prefix inherits credit for the rest. This is a
  confound for the $W$-sweep and *the signal* for Section 4.

### 2. Knob-budget theory of the $W$-sweep (the coverage axis)

**Definitions.** A **window-$W$ knob** at offset $o$ writes the correct per-segment
schedule inside its span and nothing outside: $V_t = s\,\pi_t\,u$ for
$t \in [o, o{+}W{-}1]$, zero elsewhere, with a *single* free scalar $s$. A budget of $M$
disjoint knobs covers $c = \min(MW, k)$ segments. $W\!=\!1$ is a per-token latent;
$W\!=\!k, M\!=\!1$ is one latent writing the entire trajectory.

**Axiom (A), per-slot additivity.** $\Delta = \sum_{t \in \text{covered}} \delta_t$ with
$\delta_t = \delta$ independent of $t$ and of which other slots are covered. Then

$$\Delta(W; M, k) \;=\; \Delta_{\text{full}} \cdot \frac{\min(MW,\,k)}{k}.$$

Three predictions with no free parameters once $\Delta_{\text{full}}$ is measured:
linear in $W$ **through the origin** at $M\!=\!1$; a knee at $W = k/M$ that **moves left
as $M$ grows**; and — the strongest form — every $(W, M)$ cell **collapses onto the
identity line** when plotted against coverage $c/k$. The knee-location prediction is the
one hardest to reproduce by any confound, so the $M$-sweep is worth its cost.

**Lemma (offset averaging).** If block offsets are drawn uniformly over the $k$ cyclic
positions, each slot is covered by exactly $W$ of the $k$ offsets, so
$\mathbb{E}_o[\Delta] = \frac{1}{k}\sum_o \sum_{t \in \text{block}(o)} \delta_t =
\frac{W}{k}\sum_t \delta_t = \Delta_{\text{full}}\,W/k$. **Positional heterogeneity does
not bias the offset-averaged curve at all** — it only adds variance across offsets. This
is why rotating placement (already in the design) is not cosmetic: it makes the linear
law robust to the most likely violation of (A). Report the per-offset fan as a secondary
result; it *measures* $\delta_t$'s profile.

**When (A) fails, and the signature of each failure.**

- **Interference between adjacent steered slots.** A contiguous $W$-block puts all its
  writes next to each other; $W$ scattered single-slot knobs do not. Destructive
  interference (the model normalising away a sustained push) makes $\Delta(W)$
  **concave** and makes scattered singles beat a contiguous block at matched coverage.
  Constructive interference (a run of consistent writes read as a stronger commitment)
  makes $\Delta(W)$ **convex** and contiguous beat scattered. *Convexity is the most
  interesting outcome available tonight*: it would say the window is not mere
  bookkeeping but a genuinely super-additive handle. Control: contiguous-vs-scattered at
  matched coverage.
- **Saturation of the attribute direction.** $\delta_t$ is not linear in magnitude, and
  the full run already plateaued by frac $0.35$–$0.5$. Signature: the normalised
  $\Delta(W)/\Delta_{\text{full}}$ curve **depends on frac**, compressing toward 1 at
  high frac. Diagnostic: run $\ge 2$ fracs; the curves should superimpose after
  normalisation. Prediction: linear at frac $0.2$, compressed at $0.5$.
- **The metric is additive by construction.** The teacher-forced margin is a *sum* of
  per-token log-probs, so near-additivity is close to a property of the measurement
  rather than of the model. This is the honest reading of the "looks mechanical"
  objection, and it should be conceded in writing. It also says where the informative
  experiment is: **generation mode** (Section 4) and **the resolution axis** (Section 3),
  where slots interact through sampled text and additivity is not baked in.

**Why $W\!=\!1$ with $k$ knobs equals the full template — and what survives.** Under (A),
$\Delta(1; k) = \Delta_{\text{full}} = \Delta(k; 1)$. So window length buys *nothing in
principle* if you are given $k$ knobs and the correct schedule. The $W$-sweep is
therefore not an expressivity claim; it is a **control-bandwidth-per-knob** claim, and
the honest headline is: *bandwidth per knob $\propto W$; the number of knobs needed to
express a length-$k$ trajectory is $\lceil k/W \rceil$.*

Three responses to the "scheduled SAE" objection, in increasing force:

1. **Concede the equivalence.** With $k$ free scalars and an oracle schedule, a per-token
   SAE reproduces the template. Say so. The measured quantity is the price: $k/W$ control
   parameters per trajectory.
2. **Ask where the schedule comes from.** The scheduled SAE needs an external controller
   that already knows $\pi$ — in our harness, the experimenter handing over `intents[j]`.
   When the schedule is what you want to *discover* ("what time-course of this attribute
   produces backtracking?"), it is not available, and its search space is $k$ scalars per
   latent. A crosscoder decoder row block $W_{\text{dec}}[:, t, :]$ *is* a schedule
   already learned unsupervised from the model's own activations. Falsifiable version:
   mine the schedule from an unsupervised dictionary and check it steers as well as the
   oracle schedule.
3. **Note that the current tasks do not test direction diversity.** A scheduled SAE
   writes $s_t u$ — one direction, varying gain. A TXC can vary direction with $t$. Our
   own $\cos(t_{\text{dir}}, u_{\text{DC}}) \approx \pm 1$ diagnostics say the present
   tasks live in the scheduled-knob regime, so this is *not* currently a live advantage
   and should not be claimed. A multi-attribute profile (slot 1 language, slot 2
   intensity, slot 3 topic) would separate them, and is the right follow-up.

The dictionary-level statement that does survive is a sparsity one, and it connects
steering back to reconstruction: **window-$W$ atoms cut the $L_0$ needed to express a
length-$k$ trajectory by a factor $W$** — which is exactly what a sparse dictionary is
trained to do. Prediction on learned dictionaries: on $\ell$-timescale data, TXC atoms
should acquire structure at scale $\ell$, and reconstruction $L_0$ should fall with $W$
up to $W \approx \ell$ and then flatten.

### 3. The $\ell$-timescale family and the $(W, \ell)$ phase diagram

**Design.** Profiles are square waves of run length $\ell$ (period $2\ell$), balanced
overall, $k = 12$. Two handle classes:

- **Full per-segment template** (resolution 1): $V_t = m\,\pi_t\,u$. Reference, $=1.0$.
- **Block-constant window-$W$ handle**: partition into $k/W$ blocks; inside block $b$
  write a *constant* $c_b\,u$, with $c_b$ free. This is resolution-limited, not
  coverage-limited — the complement of Section 2, and it is also the **energy-matched
  control** for the whole programme: same span, same write energy, no schedule.

**Attenuation law.** With per-slot contribution $\delta \pi_t c_t$ and block mean
$\mu_b = \frac{1}{W}\sum_{t \in b} \pi_t$, we get $\Delta = \delta W \sum_b c_b \mu_b$.
Under a **per-slot magnitude cap** $|c_b| \le 1$ the optimum is $c_b = \mathrm{sign}(\mu_b)$:

$$R(W, \ell) \;=\; \frac{\Delta_{\text{block}}}{\Delta_{\text{full}}} \;=\; \operatorname*{mean}_b |\mu_b|.$$

Under a **matched total write energy** instead, Cauchy–Schwarz gives $c_b \propto \mu_b$
and $R = \sqrt{\operatorname{mean}_b \mu_b^2}$ (RMS). The two normalisations differ
wherever the $|\mu_b|$ are heterogeneous, so reporting both discriminates which budget
the model actually obeys.

**Closed form (blocks aligned to run boundaries).** $R = 1$ if $W \mid \ell$; if
$\ell \mid W$ with $q = W/\ell$, then $R = 1/q$ for $q$ odd and $R = 0$ for $q$ even;
otherwise mixed (table below). Note this corrects a loose statement in the brief: the
condition for full recovery is $W \mid \ell$, **not** $W \le \ell$.

Exact predicted matrix, $k = 12$, blocks aligned, relative to full template $= 1.0$
(magnitude-cap normalisation; RMS normalisation in parentheses where it differs):

| $\ell \backslash W$ | 1 | 2 | 3 | 4 | 6 | 12 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.000 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 |
| 2 | 1.000 | 1.000 | 0.333 | 0.000 | 0.333 | 0.000 |
| 3 | 1.000 | 0.667 (0.816) | 1.000 | 0.333 (0.408) | 0.000 | 0.000 |
| 6 | 1.000 | 1.000 | 1.000 | 0.667 (0.816) | 1.000 | 0.000 |

If profile phase is randomised rather than fixed, the prediction smooths to the
phase-averaged matrix (e.g. $\ell\!=\!6$: $1.000, 0.833, 0.778, 0.667, 0.500, 0.000$),
which is *less* falsifiable. **Recommendation: fix phase $=0$ for the teacher-forced
matrix** (a fixed profile is fine there — alt\_phase already does this) and randomise
only in generation mode.

Three things to read off this table:

- **The $W\!=\!12$ column is identically zero for every $\ell$.** Block-constant at
  $W = k$ *is* the broadcast arm, and balance makes it null at every timescale. The
  family therefore interpolates continuously from template ($W\!=\!1$) to broadcast
  ($W\!=\!k$), with the crossover set by $\ell$. That is the headline of this section.
- **$R$ is non-monotone in $W$**, and this is the signature to hunt for: at $\ell\!=\!2$,
  $W\!=\!6$ (0.333) *beats* $W\!=\!4$ (0.000); at $\ell\!=\!1$, $W\!=\!3$ (0.333) beats
  $W\!=\!2$ (0.000); at $\ell\!=\!6$, $W\!=\!6$ (1.000) beats $W\!=\!4$ (0.667). A wider
  window doing better than a narrower one is purely combinatorial (whether the block
  spans an even or odd number of runs). Reproducing the zig-zag would confirm the
  additive model far more strongly than any monotone curve could.
- **$R$ is model-independent.** After normalising by $\Delta_{\text{full}}(\ell)$ the
  matrix has zero free parameters, so cell-by-cell agreement is a direct measurement of
  additivity — 24 cells, one prediction each.

Two secondary predictions for the same run: $\Delta_{\text{full}}(\ell)$ should be
roughly constant in $\ell$ under (A), but plausibly *increasing* (longer runs fight the
model's continuity prior less); and the **unsteered base margin should be $\approx 0$ for
every $\ell$** — a nonzero value means the model has an innate run-length preference,
which must be reported and subtracted.

Optional variant worth one cell: use a square wave of a *different* $\ell'$ as the foil
instead of a random permutation. The multiset still matches, and the margin then measures
control over the **timescale itself** rather than the specific pattern.

### 4. Entrainment: steer a prefix, let the model finish

**Design.** Generation mode. Steer only the first $W$ segments with the correct schedule,
then release. Score per-slot attribute accuracy against the intended profile, split into
**steered slots** ($t \le W$) and **unsteered slots** ($t > W$). Chance $= 0.5$.

**The prediction that matters.** Unsteered-slot accuracy can exceed chance only if
$\pi_{t>W}$ is predictable from $\pi_{1:W}$:

- **Random balanced profile** (condition (R) holds): unsteered accuracy is $0.5$ for
  *every* $W$, by construction. This is a hard null and therefore a **bug detector** —
  any above-chance value means profile leakage, a scoring error, or a judge artefact,
  not entrainment.
- **Periodic profile with run length $\ell$, phase fixed:** observing $W \le \ell$
  segments shows a single unbroken run, which is consistent with every $\ell' \ge W$, so
  the first flip position is unknown and the continuation is undetermined. At
  $W = \ell + 1$ the first flip is visible and the period is revealed. So the
  **entrainment threshold is $W^\star = \ell + 1$ exactly**, and it *moves right by one
  per unit of $\ell$* — the same structure as the polynomial clock's $W^\star = h+1$ in
  [[window_length_theory]], transplanted from decodability into steering. $\ell = 1$
  (alt\_phase) gives $W^\star = 2$.

**Predicted shapes.**

- *Steered slots, both tasks:* roughly flat in $W$ at the full-template value
  ($\approx 0.75$–$0.85$ at frac $0.35$, anchored on the measured $0.812$).
- *Unsteered slots, random profile:* flat at $0.5$.
- *Unsteered slots, periodic profile:* $0.5$ for $W \le \ell$, a **step** at
  $W = \ell + 1$, then flat. Within the unsteered tail, per-slot accuracy should decay
  with distance $d = t - W$ as $0.5 + A e^{-d/\lambda}$; expect a coherence length
  $\lambda$ of order 1–3 sentences at 1.5B.
- *All-slot accuracy vs $W$:* random profile traces the coverage law
  $0.5 + (a_s - 0.5)\,W/k$ — a straight line; the periodic profile follows the same line
  up to $W = \ell$ and then jumps above it. **The gap between the measured curve and the
  linear coverage line is the entrainment**, on one pair of axes. That is the figure.

**Two controls this needs.** Alternation is a strong generic prior, so a model may
alternate without entraining: score **phase-matched** accuracy, which is at chance under
a prior-only account, and measure the unsteered alternation base rate at $m = 0$.
Second, a 1.5B model is unlikely to infer a period for $\ell \ge 3$ from a bare prefix;
the honest expectation is strong entrainment at $\ell\!=\!1$, weak at $\ell\!=\!2$,
negligible above. Supplying the periodic structure via a few-shot prefix converts the
missing prior into in-context evidence and should restore $W^\star = \ell + 1$ — that is
the variant to run if the bare version is flat.

### 5. Predictions table

| Experiment | Metric | Predicted | Falsified by | More interesting than confirmation |
| --- | --- | --- | --- | --- |
| $W$-sweep, $M=1$, $k=12$ | $\Delta/\Delta_{\text{full}}$ vs $W$ | Linear through origin: $W/12$ | Nonzero intercept, or flat curve | **Convexity** — super-additive contiguous writes |
| $W$-sweep, $M \in \{1,2,3\}$ | knee location | Knee at $W = k/M$; all cells collapse onto $c/k$ | Knee independent of $M$ | Collapse fails only for large $W$ — a genuine span effect |
| Contiguous vs scattered, matched coverage | $\Delta$ | Tie | Either wins | Contiguous wins: window $\neq$ bookkeeping |
| $W$-sweep at 2 fracs | normalised curves | Superimpose | Compression at high frac (saturation) | Low-frac curve superlinear |
| Broadcast label swap (alt\_phase) | sign of $\Delta_{\text{bc}}$ | Flips sign, same magnitude | Stays negative | Stays negative — DC write truly harmful |
| $(W,\ell)$ matrix, $k=12$ | $R(W,\ell)$, 24 cells | The table in Section 3, zero free params | Monotone-in-$W$ rows | Zig-zag present but amplitude off — measures interference |
| $(W,\ell)$, cap vs energy | $R$ at $(\ell,W) = (3,2), (3,4), (6,4)$ | mean-absolute form: 0.667 / 0.333 / 0.667 | the RMS values 0.816 / 0.408 / 0.816 | RMS fits: the model obeys an energy budget, not a per-slot cap |
| $\Delta_{\text{full}}(\ell)$ | peak $\Delta$ vs $\ell$ | Flat | Rising | Rising — quantifies the continuity prior |
| Base margin vs $\ell$ | unsteered $\mathrm{lp}(T)-\mathrm{lp}(F)$ | $\approx 0$ | Nonzero | Nonzero — an innate timescale preference to subtract |
| Entrainment, random profile | unsteered accuracy | $0.5$ at every $W$ | Above chance | Above chance $\Rightarrow$ a leak; fix before anything else |
| Entrainment, $\ell$-periodic | unsteered accuracy vs $W$ | Step at $W = \ell + 1$ | Smooth ramp, or flat | Step present for $\ell \ge 3$: in-context period inference |
| Entrainment, all slots | accuracy vs $W$ | Random: linear coverage line. Periodic: line then jump | Both linear | Random profile also jumps $\Rightarrow$ (R) is violated |

### Design corrections the experiment must apply

- **$\ell = 4$ is unbalanced at $k = 12$.** The square wave gives three runs
  (`++++----++++`) with DC $= +1/3$, which reintroduces exactly the broadcastable mode
  (F1) the design exists to eliminate. Exact balance needs $k/\ell$ **even**, so at
  $k = 12$ the admissible family is $\ell \in \{1, 2, 3, 6\}$. Run $\ell = 4$ at
  $k = 24$ if it is wanted (its aligned row is $1.000, 1.000, 0.667, 1.000, 0.333,
  0.000, 0.333, 0.000$ for $W = 1,2,3,4,6,8,12,24$), or at $k = 8$.
- **Report $\bar\pi$ for every profile actually sampled.** Balance in expectation is not
  balance per episode; a per-episode DC component is a per-episode broadcast advantage.
- **Fix phase for the teacher-forced matrix, randomise it for generation.** The aligned
  matrix carries the falsifiable zig-zag; the phase-averaged one washes it out.
- **Normalise every $R$ cell by $\Delta_{\text{full}}$ measured at the same $\ell$ and
  frac**, not by a global constant — otherwise a real $\Delta_{\text{full}}(\ell)$ trend
  leaks into the phase diagram and looks like an additivity violation.
