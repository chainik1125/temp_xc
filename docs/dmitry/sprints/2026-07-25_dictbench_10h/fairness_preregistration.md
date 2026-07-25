---
author: Claude review agent
date: 2026-07-25
tags:
  - design
  - in-progress
---

## Pre-registration — every way the dictionary benchmark can be unfair, written before it runs

Scope: the sprint defined in [[start]]. Read against
[[2026-07-24_semisynth_10h/review_audit|last sprint's audit]],
[[2026-07-24_semisynth_10h/summary|its summary]], and the code that will actually be
run: `src/bench/architectures/crosscoder.py`, `src/bench/architectures/topk_sae.py`,
`src/bench/saebench/configs.py`, `src/bench/data.py`, and the existing prototype
`experiments/temporal_screen/trajectory_steering/dict_modal.py` (written last sprint,
never executed — no `results/temporal_screen/dictcmp.json` exists).

Everything below is stated *before* any dictionary is trained. Where a prediction is
given it is falsifiable and the number is committed to here, not chosen afterwards.

### Why this document is shaped the way it is

Last sprint produced four successive confounded positives — coverage, sign composition,
position, and a degenerate adjacency statistic — and each returned a number that looked
like the hoped-for answer. The common structure was not carelessness. In each case the
*quantity that varied between the compared arms* was not the quantity named in the claim,
and the metric was sensitive enough to whatever else varied that a large significant
number appeared anyway.

This benchmark has the same structure and more knobs. Two architectures differ in
parameter count, activation scale, decoder normalisation, training token consumption,
selection-space size, and write support — all at once. Any of those can carry the whole
result. So the ranking below is by **probability of producing a believable number that is
not about temporal dictionaries**, not by how badly each item offends.

Terminology used throughout, fixed here so the arms cannot drift:

- **slot** — one segment (sentence) of the trajectory; `k = 12` of them.
- **position** — one token index inside a dictionary window; `T` of them.
- **latent** — one dictionary feature; `m` of them are allowed.
- **coefficient** — one scalar the operator sets at steering time.
- **write** — the `k × d` matrix actually added to the residual stream.
- **fidelity** — `Δmargin(write) / Δmargin(full ground-truth template)`, on the same
  eval pairs, per pair, as in `linfit_modal.py`.

---

### C1 — The m-budget buys coverage and position weight, not dictionary structure

**Mechanism.** Last sprint established that `Δmargin ≈ Σ_t a_t c_t π_t`: additive over
slots, with unequal per-slot weights. The weights are measured and stored
(`results/temporal_screen/weights_ell3.json`): `a_t` runs from **0.28 to 10.32** with
mean 4.28, and among the nine slots resolved at more than 2 SEM the strongest is **4.0×**
the weakest. So *which slots a write touches* determines most of its effect before any
dictionary property enters. Now look at what the m-budget does in
`dict_modal.py:224-237`: with `m < k` the per-token arm sets `per_seg = m // k = 0` and
writes only `min(m, k)` slots, leaving the rest exactly zero, while the window arm writes
all `k` slots at every `m`. At `m = 4, k = 12` the token arm covers a third of the
trajectory and the window arm covers all of it.

Coverage alone, with no dictionary content whatsoever, produces this fidelity from the
measured weights:

| m | fidelity from best-m slots | from worst-m slots | uniform |
| --- | --- | --- | --- |
| 1 | 0.201 | 0.005 | 0.083 |
| 2 | 0.352 | 0.038 | 0.167 |
| 4 | 0.571 | 0.143 | 0.333 |
| 6 | 0.724 | 0.276 | 0.500 |

A 4× spread at `m = 4` is available from position choice alone. The registered prediction
"TXC > SAE at small m, converging as m → k" is *exactly the shape coverage produces*, and
it will be produced whether or not a temporal dictionary is worth anything.

**Direction of bias.** Toward TXC, strongly, at every `m < k`.

**Cheapest control.** Zero GPU: for every arm, compute the additive prediction
`Σ_t a_t c_t π_t` from the write itself using the stored `a_t`, and report observed
fidelity against it. Then one GPU arm: **matched support** — force both arms to write all
`k` slots (the SAE arm gets `m` latents whose per-slot coefficients are free, or one
latent broadcast, but never a zero slot) and re-run the m-sweep.

**Pre-registered prediction.** At matched support and matched injected norm, the
TXC−SAE fidelity gap at every `m` will be **≤ 0.10**, and the additive prediction from
`a_t` will track observed fidelity at χ²/dof ≤ 3 (last sprint's weighted fit reached
2.48 on 31 conditions). A gap **> 0.15 at matched support that survives the additive
subtraction** is the only version of this result I would believe.

---

### C2 — The schedule is supplied by the least-squares fit, so no arm reads it off a decoder

**Mechanism.** This is not a bias, it is a validity failure, and it is the reason a clean
C1-corrected win would still not support the sprint's claim. [[start]] is explicit that
the gap to close is that "last time the *schedule came from ground truth*… The claim only
becomes about dictionaries when the schedule is **read off a decoder** rather than
supplied." But in `dict_modal.py:220` and `:271-285` the coefficients are
`np.linalg.lstsq(A, ideal(prof))` — fitted per episode against the *known target
profile*. The profile enters both arms through the coefficients. What the dictionary
supplies is a basis, not a schedule. Under that design the honest claim is "window-shaped
atoms span the trajectory template in fewer coefficients than per-token atoms", which is a
statement about two linear subspaces and can be settled on CPU with no language model
involved.

**Direction of bias.** Neutral between arms; fatal to the framing. It also silently
inflates the TXC's apparent information advantage, because the description length of the
write includes a per-episode target that is not counted against either budget.

**Cheapest control.** The **frozen-write arm**, which is the experiment the sprint
actually wants: pick one latent per architecture on a selection split, fix one global
coefficient and one global sign, and apply the *same* write to every eval episode with no
per-episode refitting. The TXC's `T` decoder rows then supply the schedule and nothing
else does; the SAE has one direction and must broadcast it. Cost is one extra arm.

**Pre-registered prediction.** The TXC frozen-write fidelity will be **> 0.20** and the
SAE broadcast frozen-write fidelity will be **≈ 0** (a constant write is inert against a
multiset-matched foil by construction — this is last sprint's most robust fact, +266 vs
−8 at k = 12). If the TXC frozen arm is also ≈ 0, the dictionary claim fails outright and
the lstsq arms cannot rescue it, because they are all target-informed.

---

### C3 — Decoder normalisation makes `alpha × decoder_row` mean two different things

**Mechanism.** `TopKSAE._normalize_decoder` (`topk_sae.py:43-46`) divides each decoder
**column** by its own norm: every SAE direction has norm exactly 1.
`TemporalCrosscoder._normalize_decoder` (`crosscoder.py:47-50`) divides each latent's
whole `(T, d)` block by its **Frobenius** norm: the *sum over positions* of squared row
norms is 1. So a TXC latent that spreads its energy evenly over `T = 5` positions has
per-position row norm **0.447**, and one that concentrates on a single position has row
norm 1.0 there and ~0 elsewhere.

Three consequences, all silent:

- At matched `alpha`, the SAE injects `√T` more total magnitude over the window (2.24× at
  T = 5, 3.46× at T = 12).
- The normaliser *penalises temporal latents specifically*. The more genuinely
  window-spanning a TXC latent is, the smaller each of its writes. A TXC latent that has
  degenerated into a per-token feature gets full magnitude.
- Rescaling the TXC's rows to unit norm per position "fixes" the magnitude but **destroys
  the decoder's learned relative magnitude schedule across positions** — which is the
  temporal structure under test. There is no neutral choice here; each choice decides part
  of the answer.

`dict_modal.py:286` chose energy matching (`sc = ‖T‖ / ‖R‖`) and that is the least-bad
default, but it must be stated as a decision rather than inherited silently.

**Direction of bias.** Matched `alpha` biases toward SAE. Per-position renormalisation
biases toward TXC and voids the claim. Matched total energy is neutral only if coverage is
also matched (see C1).

**Cheapest control.** Report, for every arm, the three numbers that make writes
comparable: total injected Frobenius norm, per-slot injected norm, and injected norm as a
fraction of the mean residual-stream norm (`base_norm ≈ 54.85` at Qwen-1.5B L14). Run the
headline comparison at **matched total injected norm** and additionally report each arm's
full dose-response curve so that a matched-dose and a matched-effect reading are both
available.

**Pre-registered prediction.** The mean per-position row norm of trained TXC latents will
be between 0.25 and 0.55 at T = 5 (i.e. energy is spread, not concentrated), and reporting
at matched `alpha` instead of matched norm will move the TXC−SAE gap by more than 0.15
fidelity — enough on its own to flip the headline.

---

### C4 — Dose convexity inflates whichever arm has the larger projection

**Mechanism.** Last sprint measured a convex response: fidelity `R` rises with dose in
**12 of 12 cells across two models**, with implied exponent `p` falling 1.38 → 1.14
(1.5B) and 1.34 → 1.13 (7B) as dose rises. If `Δ ∝ ⟨c, π⟩^p` with `p > 1`, then two arms
compared at the same dose but with different achieved projections have their *ratio*
exaggerated by `p`. A genuine 1.3× advantage in projection reads as 1.4–1.5× in Δmargin,
and none of the excess is about dictionaries. Compounding it, `dict_modal.py:293` takes
`best = max(per, key=mean)` over the frac grid **per arm**, so each arm also collects a
max-selection bias, larger for the noisier arm; and every peak in every experiment last
sprint sat at the grid edge, so "peak" has meant "where we stopped looking".

**Direction of bias.** Toward whichever arm writes the larger projection at the tested
dose — expected to be TXC under C1.

**Cheapest control.** Report at a **matched frac** as the headline, with the per-arm
optimum alongside, never one convention in one table and the other in the next (last
sprint's O5). Extend the frac grid until at least one arm has an interior maximum. Best of
all: plot **Δmargin against the write's own predicted projection** `Σ_t a_t c_t π_t`,
pooling every arm, dose and `m` on one axis.

**Pre-registered prediction.** All arms will fall on a **single** curve in that plot, with
per-arm residual scatter smaller than the spread between arms along the curve. If they do,
the benchmark has measured projection and convexity, and the dictionary identity adds
nothing. A visible arm-dependent offset at matched projection is the positive result —
and it is a far stronger one than a fidelity ranking.

---

### C5 — Feature selection: three separate ways the rule is not matched

The sprint's rule is "largest activation difference between tense and calm segments,
applied identically to both architectures". Applying the same *formula* to two
architectures is not the same as applying the same *rule*.

**Mechanism (a) — the statistic is not scale-free.** TXC pre-activations are
`einsum("btd,tds->bs")`, a sum over `T` positions with `W_enc` initialised at
`1/√d_in`; SAE pre-activations are a single `x @ W_enc.T` with Kaiming init. The two
activation scales differ by construction and by training dynamics. "Largest activation
difference" is therefore an architecture-dependent quantity, and comparing raw differences
across architectures is meaningless.

**Mechanism (b) — pooling changes the estimator, not just the noise.** A TXC latent's
activation is one scalar per `T`-token window. To label a window tense or calm you must
either restrict to windows lying entirely inside one segment (which selects the *DC
intensity* latent — precisely the non-temporal one) or accept mixed windows (in which case
the difference is diluted, or you are selecting a *transition* detector, a different
object). Either way the rule does not select "the temporal feature"; it selects whatever
its labelling convention makes selectable. Separately, pooling `T` tokens reduces the
variance of the difference estimate, so any selection statistic with a variance in the
denominator (a t-statistic, an AUC at small n) hands the TXC an advantage that is purely
about effective sample size.

**Mechanism (c) — winner's curse, unequal across arms.** Selecting the max over
`d_sae ≈ 12288` candidates on `n ≈ 24` episodes is a max-statistic; the expected fidelity
of the winner is well above zero even under a pure null. The TXC's candidate family is
richer (each latent carries `T` rows, so the selected object is a latent *and* a shape),
so its max-statistic is larger under the null too.

**Direction of bias.** (a) unknown and unstable; (b) toward TXC for any variance-normalised
statistic, toward SAE for any raw-difference statistic that ignores dilution; (c) toward
TXC.

**Cheapest control.** Three things, all cheap. Select on a **held-out split** never used
for eval. Use a **scale-free** statistic — per-latent standardised difference (Cohen's d)
or single-latent AUC — computed on identically constructed labelled examples for both
arms. And run a **permutation null**: shuffle the tense/calm labels, re-run the identical
selection rule, and report the fidelity the winner achieves by selection noise alone,
**per architecture**.

**Pre-registered prediction.** The permutation-null selected fidelity will be **> 0.10**
for at least one architecture, and larger for TXC than for SAE. Any reported advantage
smaller than the difference of the two nulls is not evidence. I also predict that a rule
based on all-tense-vs-all-calm windows selects a TXC latent whose decoder rows are
**near-constant across positions** (mean pairwise cosine between its own rows > 0.7),
i.e. the selection rule finds the DC feature — which would make the TXC's temporal
advantage untestable with that rule.

---

### C6 — Capacity: matched k is not matched anything, and the two protocols are the same model

**Mechanism.** At `d_sae` matched and `T = 5`, the TXC has **5× the decoder parameters**
(`d_sae × T × d_in` versus `d_sae × d_in`; at `d_sae = 8 × 1536 = 12288` that is 94.4M
versus 18.9M) and 5× the encoder parameters. It also consumes `T` tokens per training
sample: at matched steps and batch size it sees **5× the tokens**. This repo has already
been bitten by exactly this — the purified-sampling bug where the SAE received ~25× more
tokens per step than the TXC went unnoticed until 2026-05-05.

Worse, the protocol machinery does not do what [[start]] says it does. `configs.py:67-80`
returns a **per-position** `k`, and `TemporalCrosscoder.__init__` (`crosscoder.py:36`)
multiplies it by `T`. So:

- Protocol A: `tempxc_k_at(5) = 100` → `window_k = 500`.
- Protocol B: `tempxc_k_at(5) = max(1, 500 // 5) = 100` → `window_k = 500`.

**The two protocols instantiate the identical model at T = 5.** `configs.py:76-77` says so
outright ("At T=5, B coincides with A by design"). Meanwhile [[start]] describes Protocol B
as "TempXC k = 100·T", which if passed to the constructor gives `window_k = 2500` — a
third thing that matches neither protocol. Running "both protocols" as described in
[[start]] will either produce two identical runs differing only by seed, or one run at a
budget nobody registered.

And the accounting is ambiguous even when correct. At `window_k = 500`, the TXC spends 500
scalar coefficients per 5-token window and the SAE spends `100 × 5 = 500` — matched in
*coefficients*. But the TXC's 500 latents each write all `T` positions, so it spends
**2500 (latent, position) contributions** against the SAE's 500 — a 5× mismatch in
position-writes at the same nominal protocol.

**Direction of bias.** Toward TXC on every axis: parameters, tokens seen, and
position-writes.

**Cheapest control.** Assert the realised `window_k` and parameter count of both trained
models in the run log rather than trusting the protocol name. Match **training tokens**,
not steps. Add a **matched-parameter SAE** at `d_sae_SAE = T × d_sae_TXC` as a third
dictionary — this is the single most informative extra training run available, and at
these sizes it is cheap.

**Pre-registered prediction.** Protocols A and B at T = 5 will differ by less than seed
noise (I will treat a fidelity difference < 0.05 as confirmation that they are the same
model). The matched-parameter SAE will close **at least half** of any TXC advantage
measured at matched `d_sae`. **Which matching is the honest headline:** none of the three
named ones. Matched `d_sae` flatters TXC on capacity; matched parameters flatters SAE on
sparsity; matched `k` is ambiguous between coefficients and position-writes. The honest
headline is **matched reconstruction** — train each dictionary to equal per-token FVU on
the same cache, then ask what it can write — reported next to the matched-parameter row,
with matched-`d_sae` as a supplementary row rather than the number in the abstract.

---

### C7 — The steering budget and the sparsity budget are different currencies

**Mechanism.** "A TXC latent writes `T` positions for one coefficient" is only an
advantage if a *knob* is the right unit. Count knobs three ways and the answer changes:

- **Latents** — TXC wins by construction: `m` latents cover `m × T` positions versus `m`.
- **Scalar coefficients the operator sets** — a tie by construction, if the SAE is allowed
  free per-position coefficients.
- **Description length of the write** — TXC wins only if its per-position shape is *fixed
  by the dictionary* and not re-chosen per episode. Under `dict_modal.py`'s lstsq the shape
  is re-fit against the known target every episode, so the description includes the target
  and the advantage is bookkeeping (this is C2 in budget language).

The sparsity budget is a *training-time* constraint on the encoder. The steering budget is
an *operator-time* constraint on the write. Matching one does not match the other, and the
sprint's `m` currently conflates them.

**Direction of bias.** Whichever currency is chosen decides the winner before data is
collected. Latent-counting favours TXC; coefficient-counting favours SAE.

**Cheapest control.** Report a **two-axis Pareto frontier** — fidelity against number of
latents *and* fidelity against number of scalar coefficients — rather than a single
m-sweep. Both curves come free from the same runs. Then the frozen-write arm from C2 is
what distinguishes real structure from accounting.

**Pre-registered prediction.** TXC will dominate on the latent axis and the two arms will
be within 0.10 on the coefficient axis. If that is what happens, the correct sentence is
"a temporal dictionary buys the operator fewer knobs for the same write, not a better
write", which is a real and publishable claim and is *not* the claim [[start]] registers.

---

### C8 — Training distribution: the mix ratio, the run-length family, and the window/slot granularity

**Mechanism (a) — mix ratio.** Too much task data and the dictionary contains a latent
that *is* the task, proving nothing; too little and neither dictionary learns anything
task-relevant and the sprint returns an underpowered null. There is a mix ratio that
maximises the TXC−SAE gap, and choosing it after seeing results is a forking path.

**Mechanism (b) — the run-length family is reused at eval.** `dict_modal.py:154-157` draws
`ell` from `{1, 2, 3, 6}` for the training cache *and* for the eval targets. The window
dictionary's leading atoms are then literally the principal components of the eval target
family. At `m = 1` the top window atom is close to the most common eval profile shape by
construction.

**Mechanism (c) — granularity, the most load-bearing and least visible.** The dictionary's
window is `T` **tokens**. The trajectory's slot is one **sentence**, roughly 8–12 tokens,
and the steering hook writes one vector across a whole segment span
(`dict_modal.py:96-98`). At `T = 5` a window usually lies *inside* one sentence, so the TXC
never sees a tense→calm transition and has no temporal structure available to learn. To
give it reach you must either set `T ≳ 100` (all 12 slots) or train on segment-pooled
activations (a different cache from the one described) or invent a mapping from `T`
decoder rows onto `k` segments (stretch, tile, truncate — whoever picks it picks the
winner).

**Direction of bias.** (a) toward TXC as the task fraction rises; (b) toward the window
dictionary; (c) either way, depending on an unstated choice.

**Cheapest control.** Pre-register **two** mix ratios (50/50 and 10/90) and report both.
Train on `ell ∈ {1, 2, 3}` and evaluate on **held-out** `ell = 6` plus unstructured random
profiles. State the token-to-slot mapping explicitly in the summary, and report the
distribution of **within-window class transitions** in the training cache — one histogram
that says whether the TXC could have learned anything temporal at all.

**Pre-registered prediction.** The TXC−SAE gap will be monotone increasing in task
fraction, and at 10% task data the TXC frozen-write fidelity will be **< 0.15**. On
held-out `ell = 6` the window dictionary's advantage will fall by **at least half**. If
fewer than 20% of training windows contain a class transition, I predict no temporal
advantage at any `m`, and the run should be re-scoped to a `T` that spans slots before it
is treated as a test of the hypothesis.

---

### C9 — The DoM incumbent is the regression target, not a competitor

**Mechanism.** In `dict_modal.py:198-211` the target both dictionaries are fitted to is
`ideal(prof) = ±u_dc`, built from the intensity direction. DoM *is* that direction. So a
"three-way comparison" in which one arm defines the target is not a comparison: DoM scores
fidelity 1.0 by construction and the dictionaries can only approach it from below. Add the
supervision asymmetry — DoM is fit on labels and handed the ground-truth schedule, the
dictionaries are unsupervised — and reporting the three arms on one axis is actively
misleading in both directions at once.

**Direction of bias.** Structurally toward DoM as a ceiling; simultaneously unfair to DoM
if a dictionary ever beats it, since beating your own regression target can only be noise.

**Cheapest control.** Reframe rather than re-run. Report DoM **with the ground-truth
schedule as the ceiling** (the denominator of fidelity, which is what it already is), and
add a genuinely competitive DoM arm: **DoM direction with a schedule the operator must
also supply from `m` knobs** — e.g. block-constant at width `W`, whose achievable fidelity
is exactly last sprint's phase diagram. That arm is a fair incumbent because it pays the
same operator cost, and its performance is already known to within 0.05, which makes it a
calibration check on the whole harness.

**Pre-registered prediction.** The block-constant DoM arm at `W = k/m` will reach the
fidelity last sprint's `(W, ℓ)` law predicts, `mean_b|μ_b|`, to within **0.08**. If the
harness does not reproduce that, the harness is broken and no dictionary result from it
should be believed. This is the cheapest available positive control and it should run
first, in the H1 smoke.

---

### C10 — Metric properties already known to distort comparisons

All four were established last sprint and all four apply here unchanged.

- **Extensivity.** `Δmargin` sums log-probs over all `k` slots and the foil differs in
  ~`k/2` of them, so raw Δ is extensive in the number of differing slots and varies pair to
  pair. **Control:** report the *paired ratio* to `Δ_full` on the same eval pairs, as
  `linfit_modal.py:205-213` does, never raw Δ. `dict_modal.py:289-292` currently reports
  raw Δ.
- **Unequal position weights.** Two arms on different supports differ by
  `Σ_{t∈C} a_t c_t − Σ_{t∈S} a_t c_t` even under perfect additivity. This is C1's
  mechanism and it is the reason a *matched-multiset* write is not a matched write.
- **Dose dependence.** Covered in C4.
- **Target-versus-foil composition.** `Δmargin = Δlp(T) − Δlp(F)`; an arm can score by
  destroying the foil rather than inducing the target. **Control:** store both components
  separately — free, the quantities already exist inside `margin()`.

**Pre-registered prediction.** The target-side component will be **at least 40%** of
Δmargin for every arm. If one arm's advantage is carried by foil collapse and the other's
is not, the arms are doing different things and a single fidelity number cannot rank them.

---

### C11 — Reconstruction quality is not matched, and dictionary health is unreported

**Mechanism.** Two dictionaries at matched `k` will not have matched FVU. The TXC
reconstructs a window jointly and can exploit cross-token redundancy, so its *per-token*
FVU at matched per-token `k` may be lower for reasons that have nothing to do with temporal
features. Better reconstruction means decoder rows better aligned with the data manifold,
which means better steering. A dictionary-quality difference would read as a temporal-
dictionary difference. Separately, TopK dictionaries at `d_sae = 12288` trained for a few
thousand steps typically have a substantial dead fraction, and feature-splitting differs by
architecture; neither is visible in a fidelity number.

**Direction of bias.** Toward whichever dictionary happens to train better on this cache —
likely TXC.

**Cheapest control.** Report per-token FVU, realised L0, and dead-latent fraction for both
models. If FVU differs by more than 10% relative, add an SAE trained at whatever `k` equals
the TXC's FVU (the matched-reconstruction arm from C6).

**Pre-registered prediction.** Per-token FVU will differ by more than 10% relative between
the two architectures at nominal Protocol A, so the matched-reconstruction arm will be
needed rather than optional.

---

### C12 — Statistical hygiene and the size of the grid

**Mechanism.** The design has arms × `m` × frac × protocol × task ≈ 3 × 5 × 2 × 2 × 2 ≈ 120
cells at `n ≈ 24` episodes each. Something will be significant. Last sprint's specific
failures also recur here unless deliberately closed: per-pair deltas were repeatedly not
stored, so the one contrast that mattered could only be tested unpaired
(`wsweep_modal.py`, `convex_modal.py`); arms drew different episodes because the RNG was
shared rather than per-episode; and no `m = 0` arm existed, so the null was asserted rather
than measured.

**Cheapest control.** All free: store **per-pair deltas** for every arm and cell; use a
**per-episode RNG** (`Random(base + episode)`) so all arms see identical profiles,
carriers and sentence draws; include an **`m = 0` arm**; include a **random-latent arm per
architecture**; declare the headline cell before looking; bootstrap the paired ratio for
CIs.

**Pre-registered prediction.** The two architectures' random-latent floors will *not* be
equal: the SAE's random arm will sit at fidelity ≈ 0 with small variance (a constant write
is inert against a multiset-matched foil), while the TXC's random arm writes a random
schedule and will show |fidelity| up to ≈ `1/√k` ≈ 0.29 by chance. **Comparing either arm
against a shared zero is wrong; each must be read against its own null.**

---

### The single most likely way this sprint returns a positive that is not real

**The m-sweep will show TXC beating SAE at small m because the TXC arm writes every slot
and the SAE arm writes only `m` of them, and last sprint measured that the effect is
essentially additive over slots with per-slot weights spanning 4×.** Coverage produced the
first false positive of the previous sprint, in a design nobody thought was about
coverage; here it is built into the budget definition itself, it points the same way as the
registered prediction, and it produces exactly the predicted shape (advantage at small `m`,
convergence as `m → k`). It will look like the hypothesis confirming.

**The one control that catches it.** Run the **matched-support arm**: both dictionaries
write all `k` slots at matched total injected norm, so `m` buys only the *shape* of the
schedule and never the number of slots touched — and report, alongside it, the additive
prediction `Σ_t a_t c_t π_t` computed for each arm's own write from the stored `a_t` in
`results/temporal_screen/weights_ell3.json`. If the TXC advantage disappears at matched
support, it was coverage. If it survives matched support but is fully predicted by the
additive model, it is position weight. Only an advantage that survives both is about
temporal dictionaries. Both cost one arm and zero new machinery.

---

### What I will accept as a win, stated now

A dictionary result is real if **all** of the following hold. This bar supersedes any bar
in the earlier audit.

- The **frozen-write** TXC arm (one latent, one global coefficient, no per-episode
  refitting) beats its own **permutation-null selection floor** by ≥ 0.15 fidelity.
- The advantage holds at **matched support** and **matched total injected norm**.
- It survives subtraction of the additive prediction built from measured `a_t` — i.e. the
  arms do not lie on one curve in the Δ-versus-projection plot (C4).
- It appears at **both** mix ratios and on **held-out run lengths**.
- It is **not** closed by the matched-parameter SAE.
- It is reported at a matched dose with an interior optimum, with per-pair deltas and
  paired bootstrap CIs, against an `m = 0` arm and per-architecture random-latent nulls.

If instead the honest outcome is "TXC needs fewer knobs, writes no better", that is worth
saying plainly and is not a failed sprint. The failure mode to avoid is a fidelity ranking
whose mechanism is coverage, magnitude, or selection.

### Order I would run these in

1. **H1 smoke, extended by one check**: the block-constant DoM positive control (C9). If
   the harness does not reproduce the known `(W, ℓ)` law to 0.08, stop and fix the harness.
2. **Free, before any dictionary trains**: assert realised `window_k`, parameter counts and
   token counts (C6); wire per-pair deltas, per-episode RNG, `m = 0` and random-latent arms
   (C12); switch the metric to the paired ratio and store the target/foil decomposition
   (C10).
3. **Selection hygiene** (C5): held-out selection split, scale-free statistic, permutation
   null — this must exist before any latent is chosen, not after.
4. **Frozen-write arm** (C2) and **matched-support arm** (C1) as the headline pair.
5. **Matched-parameter SAE** and FVU reporting (C6, C11).
6. Mix ratio and held-out `ell` robustness (C8), budget last.

### Code-level landmines found by reading, before anything runs

- `src/bench/saebench/configs.py:67-80` with `crosscoder.py:36` — Protocols A and B
  produce the **identical** `window_k = 500` at `T = 5`. Running "both" is running one
  twice. [[start]]'s description of Protocol B (`k = 100·T`) matches neither.
- `crosscoder.py:36` — the constructor multiplies `k` by `T`. Any script that passes a
  window budget where a per-position budget is expected is silently off by `T`.
- `crosscoder.py:47-50` versus `topk_sae.py:43-46` — Frobenius-over-`(T, d)` versus
  per-column normalisation. `alpha × decoder_row` is not a comparable quantity across the
  two architectures (C3).
- `dict_modal.py:224-237, 274-285` — `per_seg = m // k` leaves slots exactly zero for the
  per-token arm at `m < k`. This is the coverage confound in one line.
- `dict_modal.py:289-292` — raw Δmargin stored, no per-pair deltas, no ratio to `Δ_full`.
- `dict_modal.py:293` — `max` over frac per arm, with a two-point frac grid; both peaks
  will be at the grid edge.
- `dict_modal.py:102` — one shared `rng` across cache construction, targets, foils and
  carriers; arms are not guaranteed identical episodes.
- `dict_modal.py:200-208` — the eval targets are drawn from the same `struct_profile`
  family as the training cache (C8b).
- `src/bench/data.py:410-424` — the window generator draws `n = batch_size // (seq_len −
  T + 1) + 1` sequences and takes `batch_size` windows from all `(seq_len − T + 1)`
  offsets, while `gen_flat` draws `batch_size` independent tokens. At equal `batch_size`
  and equal steps the window path touches a different number of underlying tokens than the
  flat path. Verify token counts explicitly; this is the same class of bug as the 2026-05-05
  purified-sampling issue.
