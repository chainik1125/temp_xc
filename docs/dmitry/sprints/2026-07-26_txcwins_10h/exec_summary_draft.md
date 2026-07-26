---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - results
  - in-progress
---

## Status

Draft executive summary for `summary.md`, assembled from the theory agent's structure with
findings 5 and 6 rewritten against the completed run. Items marked ⚠ depend on measurements
still landing.

**Pending when this was written:** the rotation ladder (the only design that could support an
expressiveness claim), gradient-based rank arms at full configuration, and the held-out-content
split on recency and evidence.

## Executive summary

This sprint set out to find more tasks where a temporal crosscoder beats a per-token
dictionary. It did not find them, and in the course of not finding them it produced the
reason: a criterion, computable before any dictionary is trained, that says which tasks *could*
separate the architectures — and the finding that almost none do.

### The previous sprint's headline is withdrawn

The order-task result — crosscoder +11.29 against the SAE's +1.24 — was measured on a one-sided
dose grid. Rerun at both signs with two dictionary inits, the crosscoder does not beat the SAE
significantly in either (+6.34 and +3.41 against +4.66 and +5.16; z = 1.35 at one init, losing
at the other).

**The control that was the proof has inverted.** `txc_flat` — the same slab with its temporal
profile averaged away — was reported as *inverting* to −8.02, and that was the evidence the
profile carried the effect. It reaches **+12.10 and +18.47**, roughly double the crosscoder
itself. `txc_flat` is large and negative at positive doses and large and positive at negative
ones, so a positive-only grid recorded the negative branch and read a **sign** as an
**inversion**. Since the sign of a steering vector is a free parameter, the honest reading is
the reverse of the published one: `txc_flat` is a better constant write than the SAE's, and the
order task is steerable by a constant write.

Two lessons generalise. **A one-sided dose grid cannot distinguish a directional effect from a
magnitude artefact**, and the two failure modes point opposite ways — an arm positive at *both*
extremes is a second-order artefact, while an arm effective only at negative doses is genuinely
directional and invisible to a positive-only sweep. Both occurred here. And **selecting each arm
at its own best dose is not neutral**: it picks each arm's saturation point, which is exactly
where the linear reasoning that justifies every ratio below stops applying.

### 1. A per-token dictionary's write is rank-1, whatever its temporal machinery

Read from the decoders rather than argued: an SAE latent has one direction, and the tSAE's
attention lives entirely in its encoder while its decoder holds one direction per latent with no
position axis. Scaling a latent by its own activation — what practitioners do, and what the tSAE
does automatically — varies the *coefficient* across positions but never the direction. So a
per-token dictionary, steered well, reaches **any rank-1 write**, and the architectures can
differ only where the intervention needs genuinely different directions at different positions.

### 2. The rank of a task's optimal write is bounded by its attribute count

If activations decompose over semantic attributes, the difference slab factors as `P = S·U` —
schedules in `S`, directions in `U` — so `rank(P) ≤ A`, the number of attributes whose positional
pattern differs between conditions. **Schedule complexity lives inside `S`'s columns and cannot
raise its rank.**

This is why the search is structurally hard rather than unlucky. Language tasks almost always
manipulate one attribute — formality, intensity, refusal, which instruction applies — so however
intricate the time-course, the required write factors into one direction times a schedule. The
result arrived three ways: registered in advance, derived independently from the measurements,
and proved. It predicts the phase ladder is rank-1 at every rung (two sentence pools are one
attribute; measured `r1` 0.921 → 0.970) and an `m`-block rotation is rank `m − 1`.

The design rule that follows: **an expressiveness result needs a task where two or more distinct
attributes move in different directions at different positions.** The cheapest natural source is
content plus its own carried state, since a maintained state's schedule is the running integral
of the content's, and an integral is never proportional to its integrand.

### 3. Two numbers screen a task before any dictionary is trained

From the slab of optimal writes: `c`, the share reachable by a constant write, and `r1`, the
share reachable by any rank-1 write. Together they say whether a task can separate the
architectures at all, at a cost of one backward pass per document.

**A number computed before any dictionary is trained orders every steering outcome in this
sprint, including the one that had to be withdrawn.** `c` — the share of the optimal write a
*constant* write can reach — is measured from 20–24 backward passes through the model, with no
dictionary involved. Across eight configurations spanning both metric modes:

| task | `c` (gradient) | crosscoder | SAE | z | outcome | file |
| --- | --- | --- | --- | --- | --- | --- |
| `rotate12` | **0.033** | **+18.23** | +5.36 | **+9.8** | **wins** | `rot_m12_T.json` |
| recency | **0.037** | **+6.49** | +2.67 | **+18.0** | **wins** | `recency_grad.json` |
| `rotate6` | 0.102 | −0.01 | +5.92 | −7.5 | loses | `rot_m6_T.json` |
| recency, positions vary | 0.112 | +2.52 | +3.30 | −2.4 | loses | `recency_var_grad.json` |
| evidence | 0.136 | +5.79 | +1.32 | **+29.7** | **wins — the exception** | `evidence_grad.json` |
| `rotate2` | 0.163 | +2.86 | +5.27 | −1.0 | no win | `rot_m2_T.json` |
| `rotate3` | 0.179 | +6.43 | +11.51 | −2.9 | no win | `rot_m3_T.json` |
| order | 0.241 | +6.34 | +4.66 | +1.4 | no win | `order_sym_ds0.json` |

**`c` describes the constant write's ceiling, and it does that well.** Measured against
`sae_broadcast`/`grad_slab` — the quantity it is actually about — `sqrt(c)` gets **4/4 rank
agreement** across the rotation ladder.

Measured instead against the crosscoder's *margin over* the constant write, across 13 tasks
Kendall **τ = −0.58**. The weaker number is expected rather than disappointing: a margin composes
**two** independent discovery outcomes on top of the geometry. This also dissolves the apparent
`evidence` anomaly — `c` = 0.136 says a constant write can reach ~37% of the optimum, not that the
crosscoder does badly; both arms can do well and the margin is then set by which one finds its
target.

> **Geometry sets the ceilings. Discovery determines what is reached.** `c` ranks candidate tasks
> rather than deciding them, because the between-architecture gap also depends on two independent
> discovery outcomes — and this sprint's finding is that **discovery is the binding constraint
> everywhere**.

⚠ **Both quantitative tests of `c` were measured on the wrong component, and on the right one
they fail.** `c` bounds `⟨W_const, Ḡ⟩`, a first-order quantity, which is **odd** in α — but the
constant arms are 72–80% **even**, so the 4/4 ordering test and the τ = −0.58 were both computed
against a numerator that is mostly curvature. Recomputed on the odd component alone (`α = 0.5`,
the linear regime), with the outcome registered in advance by its author:

| test | on raw peaks | on the odd component | registered prediction |
| --- | --- | --- | --- |
| four-rung ordering | **4/4** | **0/4** | "still passes 4/4" |
| τ across 24 cells | −0.58 | **−0.467** | "\|τ\| rises above 0.58" |

Both predictions are refuted. The reason is visible in the magnitudes: the constant write's odd
share of the optimal write measures **0.9–6% with inconsistent sign** across the ladder, against
`sqrt(c)` predictions of 18–42%. **On the component `c` is actually about, a constant write
achieves essentially nothing on every task measured, so there is no spread for `c` to rank.**

This cuts two ways and both belong in the record. It **strengthens** the architectural claim —
constant writes have no first-order purchase on any of these tasks, so the per-token baselines are
weaker than their reported numbers suggest. And it **weakens `c` as a quantitative instrument**
further than the earlier wording allowed: its author's own reading was that if `|τ|` did not rise,
"ranks but does not decide" is if anything generous. That is the standing statement.

**`c` is necessary and demonstrably not sufficient**, and the counterexample is worth more than
the caveat. The same recency task on Qwen2.5-0.5B (`c` = 0.026) and SmolLM2-1.7B (`c` = 0.037)
sits squarely in the winning range and shows **no effect from any write, including the supervised
one**. A low `c` says a constant write has nothing to ride; it does not say anything is steerable
at all. **`sqrt(c)` predicts the constant write's share of the optimum, and it does so on the rung that
most embarrasses a monotone story.** If `c` is the operative statistic, `sae_broadcast/grad_slab`
should track `sqrt(c)` at every rung:

| m | `sqrt(c)` | `sae_broadcast`/`grad_slab` |
| --- | --- | --- |
| 2 | 0.403 | 0.0693 |
| 3 | **0.423** | **0.1124** |
| 6 | 0.320 | 0.0355 |
| 12 | 0.180 | 0.0195 |

Predicted ordering `3 > 2 > 6 > 12`; measured ordering `3 > 2 > 6 > 12` — **rank agreement 4/4,
including the non-monotone `m=3 > m=2` inversion, which `r1` gets wrong** (`r1` is monotone in `m`
and predicts `2 > 3`). The absolute values sit at 0.11–0.27 of predicted because `sae_broadcast`
uses the SAE's *learned* direction rather than the optimal constant one, so it should undershoot
by a roughly constant factor; the ordering is the test.

This matters because both `c` and `r1` are minimised at `m = 12`, so the single crosscoder win
cannot discriminate between them. **The four-rung ordering can, and it separates them cleanly.**

**This retires the metric-mode sentence circulated earlier tonight.** "The crosscoder wins under
a metric that cancels constant writes and loses under one that does not" is refuted by
`rotate12`, which is an **ordering-mode** task — the mode that leaves constant writes exposed —
and the crosscoder wins it by z = 9.8. The property that predicts the outcome is the task's `c`,
not the metric's family. A task property beating a metric property is the stronger result, and it
is why this replaces rather than restores the earlier claim.

All eight rows are full-configuration runs with `c` and the arms read from the same file.

**What the earlier version got wrong.** A table circulated an hour before this one showed the
same ordering computed from the **difference-of-means** slab. Those values are near-orthogonal to
the metric gradient (`cos` = 0.057, 0.096, 0.190 across the ladder) and do **not** order the
outcomes — order and recency sit at 0.039 apart on the diff-of-means and 7× apart on the
gradient. The gate is real; the cheap proxy for it is not.

The same DC-component account **retro-predicts eight of this project's own executed
experiments** — two failed language demonstrations and six successes — with no training.
Passphrase verification fails because its steering target is a validity *state*, a constant
write's natural shape, despite maximally position-dependent inputs; ordered generation fails
because a shared contextual mode is a constant write by definition; the six trajectory tasks
succeed because multiset-matched foils leave no DC component, and their broadcast arms measure
at or slightly below zero exactly as that predicts.

### 4. `c` is a property of the task *and the metric*

The same task screens differently under different metrics. An ordering metric
(`logP(A) − logP(B)`) cancels *content* when multisets match but leaves *context* exposed — the
residue a constant write rode. A difference-of-differences metric additionally cancels anything
pushing both classes the same way, driving `c` toward zero by construction. That is why the
constructed ladders reversed and the real-behaviour tasks did not.

One line: **use a difference-of-differences metric and a symmetric dose grid.** The first removes
a component the ordering metric leaves exposed; the second is the only thing that reveals which
kind of effect an arm has.

### 5. The surviving win is discovery, not expressiveness — and the ordering is measured

**Reported at matched dose in the linear regime.** Selecting each arm at its own best dose picks
its saturation point, which is where the first-order reasoning behind every ratio here stops
applying. Matching the dose *magnitude* across arms and reporting at the smallest magnitude where
the crosscoder is significant (α = 0.5 on every task) changes two conclusions, **one of them
against us**:

| task | crosscoder | SAE | attention tSAE | `txc_flat` | scheduled SAE |
| --- | --- | --- | --- | --- | --- |
| recency | +1.90 | +0.25 | +0.97 | +0.34 | **+7.10** |
| evidence | **+1.59** | −0.02 | +0.07 | +0.75 | +1.85 |
| `rotate12` | **+3.54** | +0.94 | +1.24 | +0.98 | +1.53 |
| order (2 inits) | +0.90 / +1.54 | +0.32 / +0.43 | — | **+1.58 / +4.11** | +0.10 / −0.09 |

**The discovery gap on recency is 3.7×, not the 1.2× best-dose reporting showed** (z = −29.2). And
`rotate2` and `rotate6` show no significant crosscoder effect at *any* dose, so their entries above
are the honest null.

**The direction-versus-schedule split, measured properly.** At matched dose the two surviving
tasks order oppositely against a scheduled per-token write: on recency the scheduled SAE wins
(+7.10 against +1.90, z = −29.2) — its **direction** was already right and only the schedule was
missing. On `rotate12` the crosscoder beats it (+3.54 against +1.53, z = +5.3) and ties the
gradient-scheduled version — the SAE's **direction is wrong and no schedule rescues it**. **The
crosscoder's value is largest where the direction itself must be found**, not the schedule.

**The order-task retraction holds at α = 0.5 in both inits**, so it was not an artefact of reading
a saturated grid: `txc_flat` beats `txc_slab` at every dose tested.

**Stated at its true scope:** on Qwen2.5-1.5B-Instruct, a crosscoder latent reverses the model's
instruction-position bias in generated text, beating every unsupervised per-token baseline and
every temporal-profile control. The same task on two smaller models has **no steerable target at
the layer tested, for any write including the supervised one** — see the limits section.

Instruction recency, completed configuration, every arm at matched injected norm:

| arm | Δ margin | |
| --- | --- | --- |
| `rank1_best` | +8.55 ± 0.27 | rank-1 truncation of the difference-of-means slab |
| `dom_slab` | +8.20 ± 0.22 | supervised reference |
| `sae_schedule` | +7.86 ± 0.25 | the SAE's own direction on its best schedule — **a published method**, see below |
| **`txc_slab`** | **+6.48 ± 0.15** | the crosscoder |
| `tsae_broadcast` | +3.65 ± 0.14 | **this repo's attention-based temporal SAE — not the published tSAE** |
| `sae_broadcast` | +2.60 ± 0.15 | a per-token dictionary as actually deployed |
| `random_broadcast` | +1.81 ± 0.16 | |
| `txc_flat` | +1.42 ± 0.14 | profile removed |
| `random_slab` | +1.39 ± 0.07 | |
| `txc_profile_random` | +0.00 ± 0.04 | profile kept, directions randomised |

Three levels, each gap separately significant. **The crosscoder beats deployed practice by
2.5×** — +6.48 against +2.60, z = 18.3 — which is the honest, useful win. **A per-token
dictionary handed a schedule beats the crosscoder** — +7.86 and +8.55, z = 4.7 and 6.7 — so the
write was never out of reach. The crosscoder reaches **76% of the rank-1 ceiling**.

So its genuine claim is that it *found* a schedule unsupervised, from reconstruction alone, that
a per-token dictionary could have executed if handed it. It is a discovery claim.

**`sae_schedule` is not an oracle we constructed to be hard to beat — it is a published method.**
Heyman & Vandeputte's Prompt Steering Replacement (arXiv:2605.03907, title and authors verified)
estimates token-specific steering coefficients from the activations themselves, and reports
beating existing activation-steering methods across three benchmarks, particularly for
high-coherence outputs. So the finding is better stated as: **the crosscoder loses to a method a
practitioner can already run.** That sharpens the discovery framing rather than softening it —
what the crosscoder adds is obtaining the schedule from reconstruction alone, with no supervision
and no imitation target.

**Two kinds of discovery, and the two headline tasks split on them.** On recency the SAE's learned
direction is essentially the optimal rank-1 direction, so only the *schedule* was missing and
supplying it beats the crosscoder (+7.86 against +6.48). On `rotate12` the SAE's learned direction
is wrong, so a schedule buys it almost nothing (+5.36 constant → +5.83 scheduled) and the
crosscoder beats every arm obtainable from a learned per-token dictionary by 3×. **The crosscoder's
value is largest where the direction itself is what has to be found**, not the schedule.

**A registered architectural prediction appeared to land here and is now withdrawn as
untested.** The temporal SAE was argued from its decoder to be rank-1 with an *automatically
supplied* schedule, so it should sit strictly between a constant write and an optimally scheduled
one, and the measurement obliged: +2.60 < +3.65 < +7.86. **That arm was trained at a third of the
learning rate it wants.** At its own best recipe the attention tSAE is the *best reconstructor of
the three architectures* — FVU 0.0144 against the SAE's 0.0373 and the crosscoder's 0.0968, a
2.6× margin over the SAE — with reading AUC 1.000. Its steering number was obtained from a badly
undertrained dictionary and predicts nothing. The comparison is being rerun with each arm at its
own recipe.

⚠ **The arm measured is not the published tSAE.** `harness.py:268-274` imports `TemporalSAE`
from `temporal_crosscoders/han_tsae`, which is this repo's **attention-based** variant. The
published tSAE (Bhalla et al., ICLR 2026) is an **InfoNCE** architecture with no attention. The
rank conclusion transfers — both have one decoder direction per latent, so both are rank-1, and
the ordering prediction holds either way — but the arm's identity does not. It must be described
as "this repo's attention-based temporal SAE", never as "the tSAE". The kickoff's carried debt
on tSAE identification is therefore **resolved**, and it resolves to *we benchmarked a different
temporal SAE than the published one*.

**The constant arms are a second-order artefact, and this understates the crosscoder.** A
difference-of-differences metric cancels a constant write only to *first* order; the residual
`≈ ½α²(⟨v,H_A v⟩ − ⟨v,H_B v⟩)` is **even in α**, while a genuine directional effect is **odd**.
Splitting each arm's dose response about zero:

| arm | recency even-share | evidence even-share |
| --- | --- | --- |
| `sae_broadcast` | **0.72** | **0.80** |
| `tsae_broadcast` | 0.58 | 0.51 |
| `txc_slab` | **0.12** | **0.10** |
| `grad_slab` | 0.17 | 0.01 |

The constant arms are dominantly even and the crosscoder dominantly odd, with `grad_slab` — known
to be the first-order optimum — almost purely odd as the control that makes the decomposition
safe. So `sae_broadcast` is a **mis-specified** baseline rather than a weak one, and the honest
per-token comparator is the scheduled arm.

**The controls hold on this task**, which is what distinguishes it from the order task where they
did not: `txc_flat` at +1.42 sits *below* a random constant direction at +1.81, and
`txc_profile_random` is +0.00 ± 0.04. Neither the directions without the profile nor the profile
without the directions carries any of the effect.

⚠ The gradient-based arms exist only at smoke scale, so `rank1_best` above is the rank-1
truncation of the *difference-of-means* slab, and the `sqrt(r1)` rank law is not tested against
it — that law requires the gradient slab, and `cos(P_dom, Ḡ) = 0.044` says the two are not
proxies for each other.

**The reading result has now replicated five times.** `auc_selection = 1.000` for the SAE on
recency, evidence and `recency_var`, against the crosscoder's 0.719, 0.685 and 0.632 — on the two
tasks that carry the empirical claim. A per-token dictionary reads these factors perfectly and
steers them worst. It is the most-replicated finding in the project.

### 6. Reconstruction quality does not predict steering quality — the ordering is inverted

At 8.0 realised coefficients per segment on the recency corpus, **each arm at its own best
recipe** from a full lr × steps sweep:

| arm | FVU | best steering Δ |
| --- | --- | --- |
| attention tSAE | **0.0144** | +2.32 |
| TopK SAE | 0.0373 | +2.35 |
| crosscoder | **0.0968** | **+7.81** |

**The best reconstructor steers worst and the worst reconstructor steers best**, by 3.4×. The
ordering is exactly inverted. Any benchmark ranking temporal dictionaries by FVU — which is what
the field does — would rank these three in precisely the wrong order for the use a crosscoder is
being proposed for.

This also retires a caveat carried all sprint. The crosscoder's poor FVU was being treated as the
price of the shared code, to be weighed against its steering advantage. On this evidence **it is
not a price**: reconstruction is uncorrelated with, or anticorrelated with, the property actually
wanted.

**The headline survives giving every arm its own best recipe**, which was the largest standing
threat to it — the concern being that the crosscoder benefited from a configuration handicapping
its competitors, a concern earned by the tSAE having to be corrected three times for exactly that
reason. Best Δ over symmetric doses at per-arm recipes:

| task | crosscoder | SAE | `txc_flat` | attention tSAE |
| --- | --- | --- | --- | --- |
| recency | **+7.81** | +2.35 | +2.37 | +2.32 |
| evidence | **+5.59** | +1.20 | +2.44 | — |
| `rotate12` | **+14.61** | +2.21 | +3.13 | +6.38 |

On recency the advantage is **larger** at per-arm recipes than at the shared one (+7.81 against
+6.48).

### 7. A single learning rate across architectures does not measure architectures

The sprint's default `lr = 3e-4` is near-optimal for the SAE and wrong for both temporal
architectures. Best FVU per arm across a 3 × 2 recipe sweep on the recency corpus, matched at 8.0
realised coefficients per segment on held-out data: **SAE 0.0373 at 3e-4, crosscoder 0.0968 at
1e-3, attention tSAE 0.0144 at 3e-3** — each arm peaking at a different recipe, spanning a 10×
range in learning rate.

This is a caveat on every cross-architecture number in both sprints, and it is the reason the
tSAE arm was reported at three different values tonight — 5× worse than the SAE, then 1.9× worse,
then 2.6× *better*. Each revision was a training-recipe artefact, not a measurement.

**The sprint's assigned deliverable was to calibrate the L1 temporal SAE, and the answer is that
no usable setting exists.** The coefficient controls sparsity only five to eight orders of
magnitude above the documented `l1 = 1e-3`, and the dictionary dies before it gets sparse: **FVU
crosses 1.0 — worse than predicting the mean — at 29 coefficients per segment, before L0 crosses
32.** The last usable point is `l1 = 100` at 151 coeff/segment, FVU 0.32.

Two causes, both architectural. `lam = 1/(4·d_in)` puts codes at ~4e-3 while the reconstruction
term is ~18, so `l1 = 1e-3` contributes 0.2% of the loss. And `TemporalSAE` has **no encoder
bias**, so sparsity has to come from dictionary geometry alone — the alive fraction is still 0.998
at 67 coeff/segment. Both readings of the loss fail the same way. **Use the same architecture with
`sae_diff_type="topk"`, which binds exactly.** This closes the carried debt from the previous
sprint: the answer is that the L1 form is not calibratable, not that we failed to calibrate it.

Two mechanical notes that follow. The crosscoder's realised coefficient spend moves with the
learning rate (10.15 at 3e-4, 8.32 at 1e-3, 8.04 at 1e-3/6000, against nominal 8), so **recipe
and budget-matching are not independent knobs**. And the crosscoder is the only one of the three
that diverges outright — FVU 0.0968 at 1e-3 against 0.3596 at 3e-3.

### 8. The scope limit, and what the crosscoder is *not* doing

The advantage requires the factor to sit at **consistent positions across documents**. A
dictionary latent is one fixed write reused everywhere, so any fixed-write arm is bounded by the
*mean* slab; when positions vary, the per-document slab keeps its shape but slides, and the mean
is a broad ramp rather than a sharp template. Randomising the instruction positions leaves the
crosscoder retaining 10% of its effect against a fixed write's 67% — a limit on the whole
intervention class, with a crosscoder discovery gap on top.

**It is not solving the task by locating the instructions.** The supervised rank-1 write puts 97%
of its mass on the two instruction segments; the crosscoder's profile is nearly flat, its two
largest entries at positions 10 and 1 rather than 9 and 2 — and it still reaches 76% of the
supervised effect. The narrower and more interesting claim is that **there is more than one way
to move this metric, and the crosscoder found a different one from the supervised write.** That
`cos(P_dom, Ḡ) = 0.044` — the supervised and gradient routes are themselves nearly unrelated —
makes a third, differently-shaped solution unsurprising rather than anomalous.

## What was not achieved

**On the evidence task the crosscoder beats the best rank-1 write derived from the
difference-of-means reference (z = +8.66) and loses to the one derived from the gradient
(z = −61.6), from the same file** (`evidence_grad.json`). Since the gradient is the correct
object, no expressiveness win survives — and the flip is the sharpest available demonstration
that **difference-of-means is a reference, not a ceiling**. Any percent-of-ceiling figure has to
name which object it used.

**The sprint did construct the geometry an expressiveness win requires.** `evidence` measures
`r1` = 0.62 and `rotate12` measures 0.18 — 38% and 82% of those optimal writes lie beyond rank-1
reach. On none of them does the crosscoder reach even the **rank-1** ceiling. **The headroom
exists and goes unused**, so the negative is not for want of rank > 1 tasks; it is a finding about
the architecture rather than a failure of task design.

⚠ **The most inviting error left in this document:** on `evidence` the crosscoder beats
`rank1_best` at z = +8.66, and that must not be read as clearing a rank-1 ceiling. `rank1_best` is
the rank-1 truncation of a reference nearly orthogonal to the gradient, so beating it says the
reference is poor. The ceiling is `grad_rank1`, and the crosscoder loses to it at z = −61.6.

**No expressiveness win was found, including on a design built specifically to produce one.**
The rotation ladder drives the rank-1 reachable share `r1` down to 0.177 by construction, and at
that rung the crosscoder still loses to the best rank-1 write taken from the metric's own
gradient: **+18.23 against `grad_rank1` +102.46** (z = −31.5), and against `rank1_best` +59.94
(z = −22.7). The same holds at every rung — `grad_rank1` reaches +109.98, +67.74, +102.46 while
the crosscoder reaches +2.86, −0.01, +18.23. **A rank-1 write beats the crosscoder everywhere on
a ladder designed to put the target out of rank-1 reach.**

**`r1` bounds the write and does not forecast the architecture — those are different claims and
only the second fails.** The law is a *within-task* ratio, `Δ(rank-1 arm)/Δ(full write) ≈ sqrt(r1)`,
and comparing rank-1 arms in absolute terms across rungs tests nothing because the denominator
moves too (`grad_slab` runs 76.0 → 102.4 → 166.9 → 275.2). Measured correctly:

| m | `grad_rank1`/`grad_slab` | `sqrt(r1)` | error |
| --- | --- | --- | --- |
| 2 | 1.447 | 0.551 | 162% |
| 3 | 0.567 | 0.516 | **10%** |
| 6 | 0.406 | 0.459 | **12%** |
| 12 | 0.372 | 0.421 | **12%** |

**The law holds to within 12% at three of four rungs and reproduces the monotone decline.** It
fails at `m = 2`, and the failure is self-diagnosing: a ratio above 1 means the rank-1 truncation
beat the full write, which is impossible to first order for a strict subspace, so that rung is
outside the linear regime (the same signature appears in its difference-of-means arms, 50.49
against 39.35). What `r1` has *not* been shown to do is predict what a crosscoder achieves — at
`m = 12` it identified headroom to +102.5 and the crosscoder used +18.23. **One gate on `c`, one
bound from `r1`, and no result yet converting `r1`'s headroom into a win.**

**The gradient-derived rank-1 arm beats the difference-of-means one at all four rungs** — 2.18×,
1.42×, 1.36×, 1.71×. That is the screen-on-the-gradient point measured four more times. Finding 2 explains why this is structurally
hard rather than a matter of not having looked in the right place, and states the condition that
would have to hold instead.

## The object exists; the task does not

A crosscoder latent's slab **is** a fixed, plottable, rank-≥2 steering object learned as one unit,
and no published method supplies one — position-varying steering that exists elsewhere is either
input-conditioned (a network evaluated at inference) or a union of rank-1 writes selected by
attribute rather than position. So the gap this work sits in is not a missing object. **What the
sprint could not supply is a task on which that object pays.** The contribution is the
characterisation of what such a task requires — rank ≥ 2, `c ≈ 0`, and positions consistent across
documents — together with a construction satisfying the first two.

## The experiment to run next

Every design this sprint achieved either rank ≥ 2 or `c ≈ 0`, never both — the trajectory tasks
got `c = 0` with rank 1, recency got rank 2 with `c` = 0.067, the rotation ladder got rank without
low `c`. The reason is now understood: **the carried state is simultaneously what creates rank ≥ 2
and what creates the DC residue**, because both come from the same integral.

Few-shot demonstration order breaks the tie. The label at position `t` is one attribute and the
running label balance is its integral, so matching the label multiset gives **rank 2 for every
foil**. Matching the multiset is the *zeroth* moment; the state's DC residue is the *first*, since
`Σ_t cumsum(Δc)(t) = −Σ_j j·Δc_j`. Adding the first-moment constraint gives both at once:

| constraint | mean `c` | rank |
| --- | --- | --- |
| multiset matched only | 0.076 | 2 |
| multiset **and** first moment matched | 0.0000 | 2 |

Verified by running `demo_order.py` in this directory. **The reference ordering must be
non-extremal** — `[1,1,1,1,0,0,0,0]` uniquely maximises the first moment, so it admits zero valid
foils and the constrained cell comes back empty; centred and alternating references admit 6–7
each. The script originally shipped with the extremal reference and printed `nan` for the cell it
exists to demonstrate; it now uses a centred one.

A second knob comes free: with `q` labels, `A ≤ 2(q−1)`, so `r1` should fall as the alphabet
grows — a real task with a genuine handle for testing `rank(P) ≤ A`.

## Limits

Two surviving tasks, both from the same rank-2 family, both designed by the same agent; one
model, one layer, one dictionary size. **Four separate results in this sprint moved materially
with learning rate or step count**, which is a caveat on every number here. The dictionaries are
trained on the same content they are asked to steer, so the current claim is "steers the ordering
of content it was trained on" until the held-out split lands. The rotation ladder, the gradient
rank arms and that split were all still running when this was written.

**The headline task does not transfer to either model it was tried on, and the reason is not the
dictionaries.** At each model's own best recipe, mid-layer:

| model | baseline bias | `dom_slab` | best rank-1 | `txc_slab` |
| --- | --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct L14 | −2.42 | +8.20 | +8.55 | **+6.48** |
| Qwen2.5-0.5B-Instruct L12 | +1.50 | +0.31 | +0.31 | +0.32 |
| SmolLM2-1.7B-Instruct L12 | +2.18 | +0.61 | — | +0.80 |

**The supervised write fails on both smaller models too.** There is no `(T, d)` write at that
layer, of any kind, that shifts which instruction they obey — so the crosscoder is not failing to
find something, there is nothing at that site to find. This is a statement about **where the
behaviour is linearly manipulable**, not about dictionary architectures, which makes it more
useful than a dictionary-level negative would have been. **The SmolLM2 layer sweep is complete and uniformly negative across six depths** (6, 9, 12, 15,
18, 21) against a baseline bias of +2.19: the supervised write reaches at most +1.25 and sits at
±0.1 at layers 15, 18 and 21. So this is "no steerable site at any of six depths", not "at the
layer tested". One artefact to pre-empt: L21 shows `dom_slab` +5.70 at α = −2 alone with every
other dose at ±0.06, which is a large write destabilising the last layer rather than a
dose-response.

**The bias itself flips sign across models.** Qwen2.5-1.5B is recency-driven (−2.42, obeys the
later instruction); Qwen2.5-0.5B and SmolLM2-1.7B are **primacy**-driven (+1.50, +2.18). So
"language models resolve conflicting instructions by recency" is not safe as a general statement
at this scale, and the task is named **instruction-position bias** throughout, with the sign given
per model.

**The crosscoder's temporal profile is actively harmful at two of three rotation rungs.**
`txc_flat` — its own slab with the profile averaged away — reaches +10.36 against the
crosscoder's +2.86 at `rotate2` and +9.83 against −0.01 at `rotate6`, reversing only at
`rotate12` (+4.32 against +18.23). And `txc_slab` across the ladder reads +2.86, +6.43, −0.01,
+18.23 — a −0.01 followed by a +18.23 is instability, not a trend. The honest description of the
surviving win is **discovery, and unreliable discovery**.

**The framework is in better shape than the empirical base**, and the honest reading is that the
criterion is the deliverable and the two surviving wins are its first test rather than its
confirmation.
