---
author: Claude (Fable 5), for Dmitry Manning-Coe
date: 2026-07-24
tags:
  - results
  - complete
---

## Steering a shape, not a level: what the width of a steering handle buys

*10-hour sprint, 2026-07-24 21:58 → 2026-07-25 07:58 PDT. Qwen-2.5-1.5B-Instruct and
Qwen-2.5-7B-Instruct, layer 14, difference-of-means directions, Modal A10G, ~$14.*

## Executive summary

A steering vector added at a token position sets a **level** — more refusing, calmer,
more French, everywhere at once. Much of what we want to control has a **shape**:
answer the safe part of a request and decline the step that crosses the line; bring an
escalating exchange back down. This sprint measures what a steering handle spanning W
positions buys over one acting at a single position, on tasks where the target and its
foil are **the same sentences in a different order**, so a constant write is inert by
construction and only shape is under test.

**1. Positions add up, and a steering write moves the model by roughly its projection
onto the target — including when it opposes the target.** A handle limited to one
constant per block of W segments can only deliver the part of the target that survives
averaging inside those blocks. Across 36 random combinations of profile, block width and
coefficients, predictions span −0.42 to +0.42 with nothing fitted and the measured effects
track them at **slope 0.94, intercept −0.01, mean error 0.073**. Thirteen conditions
predict a *negative* effect and the model duly moves the wrong way in proportion — the
part that could most easily have failed. Good on average, not exact: χ²/dof is 3.6,
because positions carry unequal weight — measured directly, the strongest segment is
2.2–4.2× the weakest, and the first segment is always the weakest.

- ![linearity](../../../../plots/2026-07-24_trajectory_steering/linearity.png)

**2. So the useful handle width is the target's own timescale.** Once positions add up,
the achievable fraction for a square-wave target of run length ℓ is fixed by
arithmetic. The resulting (W, ℓ) grid is reproduced to a mean error of **0.053** (1.5B)
and **0.045** (7B) over its six at-risk cells at the top of the dose grid — 0.099 and
0.094 at the lowest dose, a systematic drift taken up below. The grid's other 18 cells
are algebraic identities and serve as plumbing checks. Read as control cost: fidelity
per control parameter is maximised at W ≈ ℓ, where full fidelity costs `k/ℓ` parameters
rather than k. Fidelity is **not monotone in W**, since a wider handle wins when it
aligns with whole runs and collapses when it straddles them, so "wide enough to match
the timescale" is the right summary of window size rather than "wider is better".

- ![phase diagram](../../../../plots/2026-07-24_trajectory_steering/phase_diagram.png)

**3. It reaches a behaviour with safety shape: the order of declining and helping within
one response.** The scheduled handle shifts the model's preference between a declining and
a helping continuation the intended way on **96.9%** of slots (mean shift 1.00 nats/token)
against **51.2%** for the same direction at constant strength, and flips which continuation
it actually prefers on **53.6%** against a 50% floor — large, reliably-signed pressure on
the choice rather than a decisive change of it. That 51.2% is the constant write's expected
value, not a malfunction: pushing every slot toward declining is right on exactly the half
meant to decline. A random direction at matched magnitude does nothing. *(Dose 0.5 of the
mean residual norm, 160 slots.)*

- ![stance](../../../../plots/2026-07-24_trajectory_steering/stance.png)

**4. What matters is which sign lands where, not how much is written.** Permuting the
schedule inside a block — coverage, contiguity and injected norm held exactly fixed,
only the placement changed — collapses the effect from +55.3 to −1.2 at W=8 (*n = 28*).
That rules out any "you simply added more push" reading. An additive response predicts
the collapse exactly, so it is a check the design passes rather than a mechanism it
reveals, and it says nothing about *adjacency* — a separate question taken up below.

- ![controls](../../../../plots/2026-07-24_trajectory_steering/controls.png)

**Three of the claims we started the night with did not survive their own controls** —
growth with trajectory length, our first window sweep, and superadditivity. Each, and
the weaker claim that replaced it, is in *What we corrected* below.

## What problem this is, and why it is interesting

Activation steering is usually described as pushing a model along a direction: find a
refusal direction, add α times it, get more refusal. That fits a **level**. The
interesting safety cases are not levels. "Refuse everything" is not the useful refusal
control; "answer the safe part and decline the specific step that crosses the line" is.
"Be calmer" is not the useful de-escalation control; "come down over the next four
turns" is. Those targets are shapes over time, and their good and bad versions are
frequently built from the same material in a different arrangement.

That observation makes a clean design available. If target and foil are **permutations
of one another**, every bag-of-segments statistic is identical between them, so a
constant write — what a per-token steering vector applies when used across a span —
cannot separate them except by accident. Whatever does separate them is acting on
arrangement. This is what makes the constant-write arm a genuine null rather than a
weak baseline, and it is used throughout.

The prior context is a temporal-crosscoder project whose steering claim had not
appeared in natural-language behaviours: on ordered generation (days of the week,
counting), a single direction *broadcast* at every position matched or beat a
per-position schedule, because those behaviours ride a shared contextual mode that a
broadcast reinforces everywhere. The tasks here are built to remove that mode.

## The experiments

**The resolution family (findings 1 and 2).** Profiles are square waves of run length
ℓ ∈ {1,2,3,6} at k=12; ℓ=4 is excluded because it leaves a DC component at k=12, which
would reintroduce the broadcastable mode the design exists to remove. Two handle
classes: the full per-segment template, and a block-constant handle writing one scalar
per block of W. Coverage is pinned at all 12 segments in every cell, so nothing here
can be a coverage effect; every cell in a row injects the same total norm; and
normalising by the full-template effect on the *same* eval pairs divides out both the
extensivity of the metric and per-pair variation in target-foil distance. Those three
properties are why this design replaced our first one.

What the grid actually contains matters for reading the fit. Of 24 cells, 18 are
algebraic identities: 9 where W divides ℓ, making the block-constant write
byte-identical to the full template (R ≡ 1), and 9 where every block straddles equal
halves, so the coefficient is exactly zero and no vector is written (Δ ≡ 0). They are
worth running — the zero-write cells are a real plumbing test, and all 18 pass — but
the law was never at risk on them. The six informative cells carry the result, and one of
them — ℓ=3, W=4, undershooting at 1.9σ — is where the grid first hinted at the
position-weight heterogeneity that the linearity test later established.

Because the predicted fraction follows from linearity by algebra, this grid measures
**how linear the steering response is in the schedule** — and six cells at two distinct
predicted values is a thin basis for that. Square waves are the reason: every block mean
is a simple rational, so no choice of W, ℓ, phase or k spreads the predictions (checked
on CPU — k=24 buys 12 at-risk cells still spanning only {1/6, 1/3, 2/3}).

**The linearity test (finding 1)** removes that limitation by dropping square waves. For
any balanced profile and *any* block-constant coefficient vector, linearity predicts an
effect equal to the projection of the write onto the target, which takes a continuum of
values and goes negative whenever the write opposes the target. Sampling 36 random
(profile, width, coefficient) conditions at k=12, each normalised against the
full-schedule effect on the same eval pairs, gives 12 distinct predictions spanning
−0.42 to +0.42 and the regression above. It is the same claim as the grid, tested with
about six times the leverage.

The regression also shows precisely *which* part of "linear" holds, and the distinction
matters more than the fit. Two claims live inside the word. **Additivity** says what is
written at one position does not change what another position contributes; **homogeneity**
says every position carries the same weight, so the effect is the *unweighted* projection.
Homogeneity is refuted here: χ²/dof is 3.56, ten of the 36 predictions fall outside their
own 95% intervals (probability 7.5 × 10⁻⁶ if the model were right), the residual is
systematic in block width, and three conditions whose projection onto the target is
exactly zero measure significantly non-zero (z = −3.2, −3.1, +2.6), which a pure
projection model cannot produce. The phase sweep says the same thing from another angle:
two cells sharing a predicted 0.333 landed at 0.145 and 0.676 with non-overlapping
intervals. Additivity, the load-bearing claim, is untouched by any of it — every deviation
found is consistent with additivity plus unequal per-position weights.

That reframing also settles what a "span effect" would even be. Adjacency, coherent
transitions, a state that carries forward — these are all names for one position's write
changing another position's contribution, which is exactly a failure of additivity. So the
span question and the additivity question are the same question, and a two-arm
contiguous-versus-scattered contrast is the wrong instrument for it: under additivity with
unequal weights, such a contrast is non-zero whenever the two supports sit on different
positions, which they always do. Our first attempt at it was confounded by sign
composition and a sign-matched rebuild would have been confounded by position — three
false positives in a row from the same family of designs. The right test measures the
per-position weights from single-position writes, predicts every multi-position condition
additively, and asks whether any residual tracks adjacency; that run
(`weights_modal.py`) was in flight when the compute window closed.

**Declining and helping in order (finding 3).** About 40 template-generated requests
about the user's own property, a 12-sentence declination bank and a 12-sentence
content-free procedural bank, each split into disjoint halves so the direction is fit
on one half and evaluated on the other. Prompts go through the chat template. Target
and foil place the identical drawn sentences in different orders. Four arms: scheduled,
constant, single-segment, and a random direction at matched magnitude. The behavioural
metric avoids a free-text classifier: at each sentence boundary the model scores two
held-out candidates, one declining and one helping, and we measure whether steering
shifts that choice the intended way with the model's intrinsic preference differenced
out.

Two limits belong with this result. The direction separates *sentences that decline*
from *sentences that help*; its cosine with a prompt-level refusal direction is 0.108,
above the 0.026 expected for unrelated directions in 1536 dimensions but explaining
about 1% of variance. On the criterion registered before the run, that makes it a
**declination-register** direction rather than the refusal *decision* direction, so we
name it that way. Second, the model's dynamics here are strongly asymmetric: a seeded
refusal is abandoned almost at once (P(help | just declined) = 0.87) while compliance
is nearly absorbing (P(decline | just helped) = 0.026). Steering *into* declining
mid-response is the direction that would matter for recovery, and it is the hard one.

**Entrainment: a null we first mistook for a result.** We steered the first W of six
sentences, released, and scored the *unsteered* tail, asking whether a wide enough
prefix lets the model carry a predictable pattern on by itself. Scored against the
analytic persistence null, a period-2 profile appeared to jump at exactly the predicted
threshold W\* = ℓ+1 (+0.208 ± 0.045 at W=3, t = 4.6) while both unpredictable families
stayed at their nulls. It looked like the cleanest positive of the night.

It is not one. The run already contained an unsteered (m = 0) control, and scored
against *that* — the honest reference for "would the tail have looked like this anyway"
— the same cell gives **+0.002**. The persistence null falls from 0.600 to 0.000 across
the widths, and the apparent effect is that fall rather than any improvement in the
model: raw tail accuracy runs 0.578, 0.502, 0.542, 0.221 while the unsteered baseline
sits flat at 0.540. At W=4 the tail is 0.319 *below* baseline. The alternating family
disagrees with period-2 as well. There is no entrainment here at this scale.

- ![entrainment](../../../../plots/2026-07-24_trajectory_steering/entrainment.png)

The methodological yield is real and worth carrying. With **balanced** profiles the
correct persistence reference is **0.400**, not 0.500: a French-heavy steered prefix
forces an English-heavy tail by construction, so a model that merely persists scores
*below* chance, and an earlier version of this experiment read that as a failure when
the measurements were sitting exactly on their null. Switching to i.i.d. profiles
restores a true 0.500 null and the measurements sit on that too (0.470–0.528 against
0.496–0.501). Two lessons: score windowed-steering experiments against an unsteered
control rather than an analytic null wherever one is available, and derive the null
before believing either a positive or a negative.

**Controls (finding 4).** Fixed-Hamming foils; per-position marginals with a
superadditivity index; within-block schedule scrambling; contiguous versus scattered
writes at matched coverage; and an SVD of the per-position template matrix.

## What we corrected, and how

Three claims were retracted or rescoped during the sprint, each by a control run
against our own result.

- **"Grows with trajectory length" → "constant in length."** The margin sums log-probs
  over all k segments and a permuted foil differs in ~k/2 slots, so a constant per-slot
  effect produces a linear-in-k curve mechanically. With fixed-Hamming foils the curve
  is flat from k=4 on (+80.8, +79.6, +78.0, +77.5; k=2 sits lower at +71.8). The same
  normalisation turns the staged refusal sweep (+20.7 → +28.6) into a per-differing-slot
  *decline* of 10.4 → 7.0. Constant-in-length is still a real property, and it is what
  we claim.
- **"Improves with window size" (first version) → coverage.** Writing m consecutive
  blocks of width W occupies one contiguous span of mW segments, so that grid varied
  coverage. Across the ten distinct conditions with coverage ≥ 2, Δ per covered slot
  spans 17.6–23.7 for language and 7.1–8.4 for intensity with no trend in how the
  segments are grouped into knobs, while Δ itself tracks coverage closely. The
  matched-coverage contrast between contiguous and scattered writes shows no difference
  we can resolve (+18.94 ± 2.40 vs +17.42 ± 2.81 at 1.5B, +16.76 vs +13.79 at 7B;
  *n = 28*) — underpowered rather than proven equal.
- **"A wide knob is superadditive" → dose curvature.** A block appeared worth more than
  its parts (S = +3.6 at W=4), but the standard error we quoted was that of Δ(block)
  alone and ignored the uncertainty in the subtracted marginals; treating them as
  independent puts t at 1.6–1.7 rather than 2.6–2.8, so the effect is not distinguishable
  from zero. A single power law in the number of written segments, `Δ ∝ N^1.27`,
  reproduces the whole curve with no window term, and the normalisation we used made the
  widest point 1.00 by construction. The within-block scramble cannot rescue it: an
  additive response already predicts a scrambled schedule nets zero.

One further claim was scoped rather than retracted. The per-position template is
**rank-1** (σ₁ = 89% of the energy) — one direction with an externally supplied sign
schedule. Everything here is therefore about the *form of the control signal*: **a
schedule beats a level**. Whether a temporal dictionary beats a per-token one needs
trained dictionaries and is the main thing this sprint does not answer.

### Positions do not combine perfectly — and none of the deviations is a span effect

Three measurements show the imperfection. The response undershoots its predicted
projection at low dose and climbs toward it as dose rises, in **12 of 12 cells across
both models**. Ten conditions whose predicted effect is exactly zero measure between
**−0.127 and +0.124**. And two conditions sharing a predicted 0.333 land at 0.145 and
0.676.

The first is quantitatively accounted for by the response being a *convex function of
the write's projection onto the target*: fitting `R = R_predicted^p` gives an implied
exponent falling 1.38 → 1.18 → 1.14 across the dose grid at 1.5B and 1.34 → 1.17 → 1.13
at 7B — two models agreeing within 0.04 at every dose — and the exponent is separately
measurable from each arm's own dose-response curve, where it matches (1.76 implied
against 1.82 measured for ℓ=1, W=3). The second and third are accounted for by
per-position weights differing from one another.

Those weights are measured rather than inferred. Steering each of eight positions alone
gives per-position effects spanning **2.2× to 4.2×** between the weakest and strongest
position depending on dose, which rejects equal weighting at χ²/dof of 5.2 to 7.0 —
far beyond the individual standard errors. The profile is interpretable: the **first**
segment is consistently the weakest (0.51× the mean at dose 0.35), which is the position
with the least preceding text for a write to act on. Feeding those measured weights back
in predicts the multi-segment conditions well: a block's effect matches the sum of its
own constituent single-position effects to within 4–12% for every width from 2 to 8, at
all three doses. Additivity with unequal, measured weights is a good description of this
system; equal weighting is not.

**None of that is a span effect, and the distinction is not a technicality.** A span
effect means the response depends on how the written positions are *arranged* — that
writing at adjacent positions differs from writing at scattered ones with everything
else held fixed. A convex response to the net projection is arrangement-blind by
construction, since it depends only on a scalar that any rearrangement preserving the
projection leaves unchanged; unequal per-position weights are arrangement-blind too,
since they depend on *where* a write landed and never on what sits beside it. So a
cross-position effect is not evidence of a span effect. The only arrangement-shaped
signal anywhere in the sprint is a correlation of **+0.245** between our residuals and
block width (t = 1.48 on 34 dof, p ≈ 0.14) — not significant, and confounded with
position, because width determines which positions receive which coefficients.

The honest position is therefore narrower than "no span effect" and stronger than "we
could not tell". Nothing we measured requires arrangement sensitivity and everything we
measured is explained without it — and we never ran an experiment with the power to
detect it, because three successive attempts were confounded: first by coverage, then by
the sign composition of the compared writes, then by position. The test that would settle it is not a contrast between two arrangements at all: it is to
predict every multi-position condition from measured per-position weights and ask whether
any residual tracks adjacency. We have the first half of that — the weights, and the fact
that they predict block effects to within 4–12% — from the existing marginals. What is
missing is the same measurement on writes whose *arrangement* varies at matched positions,
which is what `weights_modal.py` adds; it was specified and queued but did not land inside
the compute window.

One structural note belongs with this. Teacher-forcing removes the mechanism a span
effect would most plausibly use: the text is pinned at every position, so a run of
consistent writes cannot establish a state that the model's own continuation carries
forward. The entrainment experiment asks the same question where that mechanism is
available; there the answer is also null. These are the pinned and unpinned versions of
one question rather than a tension.

**Graded amplitude control** did not survive its pre-registered gate: five urgency
levels fail to project onto the direction in order (L1 −5.06, L2 −7.35, L4 +3.49,
L5 +2.94), so we report sign scheduling only, and a sign-only handle already accounts
for the effect (binary +15.0 vs graded +11.8). The handle is cleanly bidirectional:
flipping the schedule drives the margin from +11.8 to −16.2.

## Map of the work

Code, all runnable as `modal run <path>`, in
`experiments/temporal_screen/trajectory_steering/`:

| file | what it does |
| --- | --- |
| `linfit_modal.py` | the linearity regression across a continuum of predictions — finding 1 |
| `lsweep_modal.py` | the (W, ℓ) resolution family — finding 2 |
| `stance_modal.py` | declining/helping order, teacher-forced, four arms |
| `stance_gen_modal.py` | menu-constrained behavioural metric + three-way pre-check |
| `controls_modal.py` | fixed-Hamming foils, SVD rank, calibrated stance shift |
| `convex_modal.py` | per-position marginals, superadditivity, schedule scrambling |
| `round2_modal.py` | phase sweep, stance fixed-Hamming, span-vs-dose, direction identity |
| `span2_modal.py` | span vs adjacency with the sign-composition confound removed |
| `entrain2_modal.py` | entrainment with analytic nulls and an unsteered control |
| `graded_modal.py` | graded amplitude with a monotonicity gate |
| `dict_modal.py` | window-spanning vs per-token dictionaries — written, not run (Modal capacity) |
| `wsweep_modal.py` | the retracted coverage sweep, kept for the record |

Results in `results/temporal_screen/*.json`, figures in
`plots/2026-07-24_trajectory_steering/`, plotting scripts in `scripts/plot_*.py`.
Process notes including every dead end in the order it happened are in [[log]]; the
task-design theory is in [[theory]]; the behaviour census and the literature bridge in
[[real_behaviors]]; and the adversarial audit that forced the retractions in
[[review_audit]].

## Limitations

- **Two models, one layer, one language pair, one intensity axis.** The linearity
  result replicates across 1.5B and 7B at layer 14; layer and attribute generality are
  untested.
- **Difference-of-means directions, not trained dictionaries.** This is the largest
  gap, and it bounds every claim to control-signal *form*.
- **Teacher-forced margins carry most of the evidence.** The behavioural results are
  smaller-n. One generation harness was rebuilt mid-sprint after its classifier proved
  artifact-prone; at the lower dose every arm in the rebuilt version sits at exactly
  0.500 (a model that always prefers one class scores half of a balanced profile), and
  only at the top dose does it separate — which is where the 53.6% flip rate quoted in
  finding 3 comes from. Free-generation stance, as opposed to forced choice between
  supplied continuations, remains unmeasured.
- **Doses are large.** Peaks sit at 0.35–0.5 of the mean residual norm, and steered free
  generation at those doses is code-mixed rather than fluent, so behavioural claims are
  about attribute identity per slot rather than text quality.
- **Every arm peaks at the top of its dose grid**, so absolute magnitudes are lower
  bounds rather than optima.
- **Linearity holds on average, with real scatter around it.** Only 26 of 36 bootstrap
  intervals cover their prediction, and the phase sweep found two conditions sharing a
  predicted 0.333 that landed at 0.145 and 0.676 with non-overlapping intervals. Where a
  write lands matters beyond how it projects onto the target, and we have not
  characterised that.

## What we would do next

Run `weights_modal.py`, which is specified and was queued when compute ran out: twelve
per-position weights measured from single-position writes, used to predict forty
multi-position conditions additively, with the residual regressed on adjacency. It is the
decisive test of whether any arrangement sensitivity exists, and it subsumes the ρ
measurement in `rho_modal.py`, which probes a specific mechanism for the dose-drift that
already has an independent cross-check.

Then train a temporal crosscoder and an L0/width-matched per-token SAE on one activation
cache and rerun the resolution family with decoder rows in place of the
difference-of-means direction. That single experiment converts "a schedule beats a
level" into a statement about dictionaries, which is the claim the wider project needs.
Then sweep layers and a third attribute to test whether `W* ≈ ℓ` holds generally, and
build the multi-turn version of the stance task, where the timescale is set by the
conversation rather than by us.
