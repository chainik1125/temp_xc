---
author: Claude redteam agent
date: 2026-07-24
tags:
  - results
  - complete
---

## Scope

Adversarial read of [[summary]] **as revised at 22:42+ after the round-2 audit** (529-word
executive summary, five findings, superadditivity retracted), against the raw JSONs in
`results/temporal_screen/`, the code in `experiments/temporal_screen/trajectory_steering/`,
and the figures in `plots/2026-07-24_trajectory_steering/`. An earlier version of this file
audited the 22:36 draft; everything below targets the current text.

## Verdict

The revision fixed most of what I had found. The cell census is now correct (18 identities:
9 where W divides ℓ, 9 zero-write — this matches my independent count exactly), the inflated
"11 measured cells / 0.029" headline is gone, the direction is honestly renamed, the exec
summary is under budget, and the reframing of finding 1 as *a test of linearity* is a real
intellectual improvement rather than a hedge. Twenty-one of the twenty-five numbers I could
check reproduce exactly, including several I expected to be loose (the 0.079 noise floor, the
1.9σ outlier cell, `P(decline | just helped) = 0.026`, the 10.4 → 7.0 per-differing-slot
decline, which I confirmed analytically using `E[Hamming | foil ≠ target]` = 2 and 4.058).

Three things now stand between this and a clean review, in order:

1. **Finding 4 does not survive a control that is already in its own results file.** Scored
   against the m=0 cell that `entrain2_modal.py` computes for exactly this purpose, the
   period-2 entrainment effect is **+0.002**, not +0.208. Details below; this is new since my
   first pass and it is the most important item in this report.
2. **The figures were not regenerated after the round-2 revision.** `phase_diagram.png` is
   timestamped 22:38, before the rewrite. Its panel A still reads *"11 measured cells, no
   fitted parameters, mean |error| = 0.029"* — the exact claim the revision retracted. The
   headline figure now contradicts the corrected text.
3. **`stance_gen.json` is still undisclosed, and the revision made it load-bearing.** The new
   "two limits" paragraph quotes `P(help | just declined) = 0.87` from the classifier the log
   flagged as artifact-prone, and the three-way re-check built to replace it returned 10%
   coverage and zero measurable transitions.

## Step 1 — numbers in the current draft

### Verified correct

| claim | source | value on disk |
| --- | --- | --- |
| 0.053 (1.5B) / 0.045 (7B) over six informative cells | `lsweep_qwen1.5b/7b.json` | 0.05269 / 0.04502 |
| 18 identity cells: 9 with W \| ℓ, 9 zero-write; all 18 pass | same | confirmed — 9 cells store the *identical float* as `full_peak`, 9 store exactly 0.0 |
| noise floor near 0.079 | same | mean SEM(R) over the six cells = 0.0790 (7B: 0.0802) |
| ℓ=3, W=4 undershoots at 1.9σ, the only visible departure | same | 1.86σ; next largest is 1.26σ |
| 96.9% vs 51.2% | `controls.json` | 0.96875 / 0.51250 (both at frac 0.5) |
| cosine 0.108 vs 0.026 expected in 1536 dims | `stance.json` | 0.10828; 1/√1536 = 0.02552; r² = 1.2% |
| P(decline \| just helped) = 0.026 | `stance.json` | 1/(1+38) = 0.0256 |
| +71.8 → +77.5 fixed-Hamming | `controls.json` | 71.745 → 77.452 |
| per-differing-slot 10.4 → 7.0 | `stance.json` | 20.729/2 = 10.36; 28.584/4.058 = 7.04 |
| +0.208 ± 0.045, t = 4.6 | `entrain2.json` | +0.2085 ± 0.0451, t = 4.62 |
| nulls 0.400 and 0.500; i.i.d. 0.470–0.528 vs 0.496–0.501 | `entrain2.json` | exact — and I reproduced all eight nulls in closed form, no Monte Carlo; they are right |
| +55.3 → −1.2 scramble | `convex.json` | 55.326 → −1.153 (frac 0.35) |
| +18.94 ± 2.40 vs +17.42 ± 2.81; 7B +16.76 vs +13.79 | `lsweep_*.json` | exact |
| σ₁ = 89% | `controls.json` | 0.89245 raw / 0.89376 unit rows |
| S = +3.6 ± 1.3 at W=4 | `convex.json` | 3.5611 ± 1.2740 |
| graded L1 −5.06, L2 −7.35, L4 +3.49, L5 +2.94; binary +15.0 vs graded +11.8; +11.8 → −16.2 | `graded.json` | exact |
| exec summary ≈ 526 words | — | 529 |

### Wrong or unsupported

**(a) "Δ per covered slot is flat (22.2–23.7 for language, 7.8–8.4 for intensity)" — survived
the revision and is still wrong.** From `wsweep.json`, every condition with coverage ≥ 2:

- language: **17.60** (W1\_m2), 23.02, 22.20, 23.65, 22.83, **18.63** (W4\_m1), **20.91**
  (W4\_m2), 23.41, 22.19, 22.19, 22.19 → true range **17.60–23.65**
- intensity: **7.07** (W1\_m2), 8.30, 7.83, **7.31** (W3\_m1), 7.82, 8.41, 8.05, 7.76, 7.78,
  7.78, 7.78 → true range **7.07–8.41**

Three conditions fall outside the quoted language range and two outside the intensity range.
There is no reading of "from two covered segments upward" that produces 22.2–23.7 — under
W ≥ 2 the minimum is 18.63, under coverage ≥ 2 it is 17.60. Note that `review_audit.md:126`
tabulates W1\_m2 as 17.60 and 7.07, so **the summary contradicts the audit document it
cites**. The retraction is unaffected — Δ still tracks coverage far better than width — so
this costs nothing to fix.

**(b) `Δ ∝ N^1.27` is imported from a different dataset than the result it retracts.** The
exponent is calibrated on the two endpoints of the **stance** W-sweep
(`review_audit.md:692`), which I reproduce: q = 1.2673. But the superadditivity being
retracted (S = +3.6 ± 1.3 at W=4) is measured on `convex.json`, whose own dose curve fits
**q = 1.125** (1.100 at frac 0.2, 1.111 at 0.5). The two are not interchangeable:

| N | S observed | S predicted, q = 1.125 | S predicted, q = 1.27 |
| --- | --- | --- | --- |
| 2 | +0.47 | +0.97 | +2.20 |
| 4 | +3.56 | +4.05 | +9.71 |
| 8 | +4.47 | +12.70 | +32.23 |

At q = 1.27 the dose law overshoots the observed superadditivity by 2.7–7×. "Reproduces the
curve" also does more work than the underlying analysis supports: on the stance sweep the
power law is calibrated on 2 of 4 points and the W=4 point misses by **−27.7% (−1.5σ)**,
which `review_audit.md` states plainly and the summary does not.

**The retraction itself is still correct**, and there is a cleaner argument for it that costs
one sentence: `S = Δ(N) − N·Δ₁` is, by definition, the deviation of the dose curve from
linearity, so "superadditive" and "q > 1" are the same statement — and the matched-coverage
contiguous-versus-scattered null is what shows it is dose curvature rather than a window
effect. Say that, cite q = 1.13 from the grid the S came from, and the point is unassailable.

**(c) "Constant-in-length is still a real property, and it is what we claim" directly follows
a reported 32% decline.** The same bullet says the normalisation "turns the staged refusal
sweep (+20.7 → +28.6) into a per-differing-slot *decline* of 10.4 → 7.0" — which I verified —
and then asserts constancy. Both facts are true of *different tasks*: flat on the
language/intensity family (fixed-Hamming, 71.7–80.8 with no trend for k ≥ 4), declining by a
third on stance. Scope the sentence to the family it holds for.

**(d) Smaller items.**

- *"flat (+71.8 → +77.5 across k = 2…10)"* quotes the endpoints; the interior maximum is
  **+80.8 at k=4**, so the range is 71.7–80.8 and k=2 sits 11% below k=4 (Welch t = 1.75,
  n = 32). Accurate phrasing: "flat from k=4 onward". The previous draft listed all five
  values, which was better.
- *"the 0.079 noise floor"* is the SEM of `obs` divided by Δ\_full, ignoring Δ\_full's own
  error. Propagating it gives **0.088** (7B: 0.090). The author picked the *smaller*, less
  flattering number — worth saying so in one clause, because it makes the 0.053 result look
  earned rather than lucky.
- *"Every arm peaks at the top of its dose grid"* is not literally true: the broadcast arms
  peak at 0.05–0.35 (`stance.json`, `graded.json`). True of every arm with a real effect.
- The graded list omits **L3 = −3.02**. Including it makes the sequence look *more* ordered,
  not less, so the omission works against the author's own gate.
- `round2.json` and `dict.json` do not exist anywhere in the repo, but `round2_modal.py` and
  `dict_modal.py` are both in the map table. The `N^1.27` analysis traces to
  `review_audit.md` reading `stance.json`, so it is reproducible — just not from the file the
  map implies.

## Step 2 — the claims

### Finding 4 does not survive its own m=0 control

This is the one that changes the document. `entrain2_modal.py:216-219` runs a **W=0 cell per
family** with `m = 0.0` — no steering at all — and scores all six slots against the profile.
The docstring calls it "the model's innate rate of matching the profile". It is the exact
baseline for "did steering the prefix make the model carry the pattern", and the summary does
not use it. Scored against it:

| family | unsteered (W=0) | tail acc at W=3 | vs persistence null | **vs unsteered** |
| --- | --- | --- | --- | --- |
| period-2 | 0.540 | 0.542 | **+0.208** | **+0.002** |
| alternating | 0.517 | 0.458 | +0.125 | −0.058 |
| i.i.d. | 0.413 | 0.470 | −0.027 | +0.057 |
| balanced | 0.476 | 0.428 | +0.026 | −0.049 |

The period-2 model with a three-sentence steered prefix scores 0.542 on the untouched tail.
With no steering at all it scores 0.540. The headline t = 4.6 measures the distance from a
*persistence* strategy, not from the model's own unsteered behaviour.

At W=4 it is worse. The persistence null there is **exactly 0.000** (a persister gets nothing
right, which I verified in closed form), so the reported "+0.221" *is* the raw accuracy — and
0.221 is **−0.319 below** the unsteered baseline of 0.540. The model got worse, and the
metric scored it as the largest excess in the experiment.

The general failure is that `acc − persistence_null` rewards *any* departure from
persistence, including randomness. A coin-flipper scores 0.5 at every cell, so its "excess"
is +0.167 at period-2 W=3 and **+0.500** at period-2 W=4 — larger than anything observed. The
sentence *"Profiles that are unpredictable by construction stay at their nulls at every
width, which is the check that this is inference rather than a scoring artefact"* is the
weakest link: for the i.i.d. family the persistence null **is** 0.5, so a coin-flipper scores
zero excess there by construction. The control family is precisely the family where the two
competing hypotheses make identical predictions, so it cannot discriminate between them.

*Fairness check.* W=0 scores all six slots while the W=3 tail scores slots 3–5, so the
comparison is not slot-matched. For +0.208 to be real, the unsteered model would have to
score ≈0.33 on slots 3–5 and therefore ≈0.75 on slots 0–2 — matching a randomly-phased
profile three-quarters of the time with no information about it. That is not credible.
Settling it properly needs per-episode labels, which `entrain2_modal.py` does not save; that
is a one-line change worth making.

**Recommendation: retract finding 4.** Its methodological yield — that a balanced profile
forces a 0.400 persistence null, so a naive 0.5 null makes a working model look broken —
is genuine and already sits in "What we corrected", which is where the whole thing should
live. Retracting a fourth claim on the strength of a control you built yourself is a better
look than the finding was ever going to be.

### Finding 1: the reframing is right, and the figure now contradicts it

Reframing the experiment as *a measurement of linearity* is the best decision in the
revision. It is honest about `R = mean_b |μ_b|` being algebra rather than physics, and it
converts "we predicted a phase diagram" into the sharper "the steering response is linear in
the schedule to within 0.053 against a 0.079 noise floor, in six independent cells". The
design defences added in "The experiments" — coverage pinned, equal injected norm per row,
normalisation by the same eval pairs dividing out extensivity — are exactly the three
objections a reviewer raises, answered pre-emptively.

**But `phase_diagram.png` (22:38) predates the rewrite and was never regenerated.** As
shipped:

- Panel A's title reads *"11 measured cells, no fitted parameters, mean |error| = 0.029"* —
  the retracted claim, in the figure the retraction is printed next to.
- Panel A hatches "fixed by construction" on the W=1 column and the zero cells only, because
  `scripts/plot_phase_diagram.py:34-40` returns `"identity"` only `if W == 1`. The five cells
  where W divides ℓ and W > 1 — (2,2), (3,3), (6,2), (6,3), (6,6) — are drawn as measurements
  though the body text now correctly calls them identities. One-line fix:
  `if ell % W == 0: return "identity"`.
- Panel C is captioned "6 fractional cells" and scatters **11** points, five of them stacked
  at exactly (1.00, 1.00).

A reviewer reads the figure before the prose. Regenerating it is the highest
value-per-minute action available and needs no GPU.

**One framing point survives from my first pass.** "The experiments" still presents
`R = RMS(μ_b)` as an "energy-matched alternative" that the data rules out. It is not a rival
hypothesis — both predictions come from the *same* formula `Δ = δW Σ_b c_b μ_b`, one with
`c = sign(μ)` and one with `c = μ/RMS(μ)`, and the experimenter chooses which to write. What
the data actually shows is better: one linear law predicting two different normalisations
without fitting (mean absolute error 0.064 for the cap arm against its own prediction, 0.080
for the energy arm against its own). Say that instead; a reviewer who works out that RMS is
not a rival will assume the framing was chosen to manufacture a model comparison.

### Finding 2: "peaks at W ≈ ℓ" is still overstated, and panel B still shows it

For W an odd multiple of ℓ, `R = 1/(2m+1)` and `W = (2m+1)ℓ`, so `R·W/k = ℓ/k` — **identical
for every odd multiple of ℓ**. The predicted efficiency curve has ties, not a unique peak:

- ℓ=1: W=1 and W=3 both predict 0.0833; observed 0.0833 vs 0.0622 → argmax at W=ℓ.
- ℓ=2: W=2 and W=6 both predict 0.1667; observed 0.1667 vs **0.1765** → argmax at W=3ℓ.
- ℓ=3, ℓ=6: W=ℓ is the unique maximum in the tested grid.

So the argmax claim is unambiguous in two of four rows, and where the theory predicts a tie
the noise breaks it once each way. Panel B is titled *"Control efficiency peaks at W ≈ ℓ"*
with its **ℓ=2 star sitting at W=6**, which a reviewer will notice in about four seconds.

The fix makes the finding stronger, because the exactly-true statement is sharper and is the
one a safety reader wants: **W = ℓ is the largest width that retains full fidelity (R = 1
iff W divides ℓ), so lossless control costs k/ℓ parameters instead of k.** That also repairs
"and no wider" / "collapses when it straddles them", which currently implies monotone decay —
efficiency actually returns to ℓ/k at every odd multiple and collapses only at even ones.

### Finding 3: the undisclosed harness, now load-bearing

The metric is sound and the write-up defends it well. I checked `controls_modal.py:334`:
`d = (s_marg - b_marg) * signs[t]` with `signs` alternating over a shuffled balanced profile,
so an arm that pushes everything one way scores ~50% — and the broadcast arm's 51.2% is the
empirical proof. The added sentence explaining *why* 51.2% is the expected value remains the
best writing in the document. Held-out bank split is genuine (`REFUSE[:6]` fit, `[6:]`
evaluated).

**What is missing.** `results/temporal_screen/stance_gen.json` is the output of the harness
the log says was built to replace the artifact-prone classifier. It contains:

- `precheck3`: `classifier_coverage = 0.101`, `n_transitions_from_refuse = 0`,
  `p_comply_after_refuse = 0.0`, one classified transition in the entire run. The three-way
  re-check **did not reproduce the 0.870 gate**; it produced no usable data.
- `menu` (generation-time, model picks between held-out candidates, chance 0.5):
  template@0.5 = **0.536 ± 0.016**, single@0.5 = **0.526 ± 0.013**, broadcast@0.5 = 0.500,
  unsteered = 0.500; at frac 0.35 every arm is exactly 0.500 ± 0.000.

The revision made this worse, not better. The new paragraph now states
`P(help | just declined) = 0.87` and `P(decline | just helped) = 0.026` as established facts
about the model's dynamics — both from `stance.json`'s `precheck`, i.e. the regex the log
describes as scoring "every *unmatched* sentence as comply", and whose corrected re-run
failed. Meanwhile Limitations says only that a harness "was rebuilt mid-sprint after its
classifier proved artifact-prone", which reads as though the rebuild succeeded.

Either reading obliges disclosure. If the harness works, the generation-time effect is +3.6
points over chance with single-segment at +2.6, and that belongs beside 96.9% vs 51.2%. If it
is broken — four of seven arms returning exactly 0.500 ± 0.000 alongside 10% coverage says it
probably is — then say so and drop the 0.87/0.026 sentence, because its only support is the
classifier that failed. **This is the one item a reviewer with repo access would call
selective reporting.**

Two smaller points that cost nothing: the single-segment arm's structural ceiling is
1/8 = 0.125 and it scored exactly 0.125, i.e. it succeeded on 100% of the one slot it wrote
(panel B draws the 1/8 line — good — but the annotation "1/8 = one slot of eight" is
**overprinted by the bold 0.125 label** and unreadable in the PNG); and per *written* slot
the single arm's shift is 1.363 against the template's 1.004, so the schedule wins on
coverage-with-correct-sign, not per-slot potency.

### Finding 5: the scramble is a linearity check, and the bullet says otherwise

The exec summary asserts *"The effect is the order, not the mass"* and leads with the
scramble collapse. "The experiments" section says the opposite, correctly: *"a linear
response predicts a scrambled schedule nets zero, so its collapse confirms the design rather
than revealing a mechanism."* The document argues against itself in two places, and the exec
summary is the one that is wrong.

The scramble is genuinely valuable — it holds coverage, contiguity and injected norm exactly
fixed and kills "any injected norm helps". But it is a *confirmed prediction of finding 1*,
not an independent result. Two figure notes: the W=1 bar labelled "100%" is the identity
(`convex_modal.py:194-197` shuffles a one-element list), so it is a sanity check and should
be annotated as one; and the negative scrambled values are frac-0.35 specific — at frac 0.5
they are +3.9 and +2.2.

### Two power notes

- *"contiguous versus scattered … are indistinguishable"* is a failure to reject, not
  equivalence: 95% CIs on the difference are [−5.7, +8.8] at 1.5B and [−2.5, +8.4] at 7B, so
  a 50–60% contiguity advantage is not excluded. Say "no detectable difference at n = 28",
  and give the 7B SEMs, which are omitted while the 1.5B ones are given.
- The document still reports no sample sizes. They are: lsweep n\_eval = 28 pairs, controls
  n\_eval = 32 and n\_stance = 20 (160 slots, clustered within 20 prompts), convex n\_eval =
  28, stance n\_eval = 32, graded n\_eval = 32, entrain2 n\_gen = 48. One parenthetical in
  "The experiments" would cover it.

## Step 3 — writing judgement

### Does the retraction material read as honest self-correction or damage control?

**Honest self-correction, decisively** — and it is now the most credibility-generating
material in the document. Three retractions, each naming the control that forced it and the
weaker claim that replaced it, plus a fourth claim scoped rather than retracted, is more
self-policing than most published work contains. The reframing of finding 1 around linearity
is the strongest single move: it *reduces* the claim and *increases* the credibility, which
is the trade a good reviewer is looking for evidence you are willing to make.

It reads as damage control in exactly three places, and in all three the summary is **less
honest than its own supporting documents**:

- the per-covered-slot range (`review_audit.md:126` has the numbers that contradict it);
- "reproduces the curve" for the power law (`review_audit.md:695` discloses the −1.5σ miss);
- the rebuilt generation harness (the log records why the original was suspect; the summary
  records neither that nor the rebuild's result).

The fix in each case is to copy a sentence that already exists one document over. That is a
20-minute job and it removes every instance of the pattern.

### Is finding 5 doing too much?

Yes — it is doing two unrelated jobs in one bullet, and the positive half is the half the body
text contradicts. Restructure:

- **Fold the scramble into finding 1.** Under linearity a scrambled schedule nets zero, so
  the collapse is finding 1's own prediction confirmed on a second task. That is where it is
  strongest and where it stops competing with the retractions for the reader's attention.
- **Make the retractions their own numbered finding**, kept in the executive summary, but
  *listed* rather than *argued*: one line each and a pointer to the section. A reviewer
  skimming only the exec summary should see them; a reviewer reading the whole thing should
  not meet the argument twice.

With finding 4 retracted, that gives four findings: linearity (with the scramble as its
check), `W* = ℓ` and the k/ℓ parameter count, declining/helping order, and the retractions.
Four findings, three figures, every finding doing one job.

### Smaller writing notes

- Panel A says "mean |error| = 0.029" and panel C says "0.053 / 0.045" in the same figure,
  with no explanation of why one figure carries two error rates. After regeneration, only the
  second should survive.
- `δW` and `c_b` appear in `Δ = δW Σ_b c_b μ_b` and are never defined anywhere in the
  document. Either gloss them or drop the formula and keep the prose, which is already clear.
- "About 40 template-generated requests" — the requests are combinatorial (32 verbs × 22
  objects); 40 is `n_train`, the direction-fitting budget. Reword.
- The map table advertises `round2_modal.py` and `dict_modal.py`, neither of which has an
  output file in `results/`. Mark them "written, not run" or cut the rows — in a document
  whose central scoping note is about dictionaries, an unrun `dict_modal.py` in the file list
  invites the wrong question.

## Step 4 — one experiment, one cut

### The experiment (CPU-only, since Modal is blocked)

**It is already done and the answer is above.** The decisive test of finding 4 needed no GPU:
the m=0 control was sitting in `entrain2.json` the whole time, and re-scoring against it takes
the period-2 effect from +0.208 to +0.002. I also reproduced all eight persistence nulls in
closed form — they are correct, the problem is the choice of baseline, not its computation.

The remaining CPU-only work, in order of value per minute:

1. **Regenerate `phase_diagram.png`** after the one-line `cell_kind` fix
   (`ell % W == 0`) and retitle panel A to the six-cell result. Right now the headline figure
   asserts a number the text retracts. Minutes, no GPU, and it is the first thing a reviewer
   looks at.
2. **Re-plot or drop `entrainment.png`.** If finding 4 is retracted, the figure goes with it;
   if it is kept in any form, it must plot raw tail accuracy with the unsteered baseline as a
   reference line, because the current y-axis (`acc − persistence null`) is what hid the
   result.
3. **Save what would settle the two open questions**: per-episode labels in
   `entrain2_modal.py` (slot-matched unsteered comparison) and per-item marginal deltas in
   `convex_modal.py` (a correctly-paired SEM for S — the current one uses the SEM of Δ(B)
   alone and ignores the subtracted term entirely, which inflates t from ~1.7 to 2.8; harmless
   now that S is retracted, but it will recur).

When GPU capacity returns, the one experiment that most strengthens the paper is unchanged:
**rerun the resolution family at k=24** with ℓ ∈ {1,2,3,4,6,8} and W ∈ {1,2,3,4,6,8,12}. k=12
with ℓ ∈ {1,2,3,6} is nearly the worst possible grid for this law — W and ℓ are almost always
in a divides / even-multiple relation, which is exactly when the theory risks nothing — so the
informative cell count rises from 6 to roughly 25, the predicted R values spread across many
distinct fractions instead of {1/3, 2/3}, and the **odd-multiple tie prediction**
(`R·W/k = ℓ/k` at W = ℓ and W = 3ℓ) gets tested twice, which is the fix for finding 2. Same
script, different arguments.

### The cut

**Finding 4.** Previously I recommended demoting it on grounds of taste; now it is on
evidence. Its methodological yield — the 0.400 balanced-profile null — is real and belongs in
"What we corrected", one paragraph, where it is already well written.

If more length is needed after that, the second candidate is **"What problem this is, and why
it is interesting"**, whose first two paragraphs restate the executive summary's opening. Its
third paragraph — the prior-context note about broadcast beating a schedule on days-of-the-week
— is new information and should be kept wherever the rest goes.
