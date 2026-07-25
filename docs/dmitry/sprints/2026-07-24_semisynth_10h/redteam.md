---
author: Claude redteam agent
date: 2026-07-24
tags:
  - results
  - complete
---

## Scope and verdict

Adversarial read of [[summary]] as of the 22:42 revision (commit `f4565908`), against the
raw JSONs in `results/temporal_screen/`, the experiment code in
`experiments/temporal_screen/trajectory_steering/`, and the figures in
`plots/2026-07-24_trajectory_steering/`.

**Verdict.** The core result is real and the document is unusually honest for a 10-hour
sprint — the cell audit in finding 1 and the fixed-Hamming retraction are the kind of
self-correction most write-ups omit. But the honesty is applied unevenly. Three things
would sink it with a hostile reviewer who has repo access:

1. The cell audit **undercounts by five**. Eighteen of the 24 cells are fixed by
   construction, not thirteen, so the advertised "11 measured cells, mean error 0.029" is
   still inflated and the figure's own hatching is wrong.
2. Finding 2's headline (`W* ≈ ℓ`) is **contradicted by panel B of its own figure**, which
   puts the ℓ=2 star at W=6.
3. `results/temporal_screen/stance_gen.json` — the harness built specifically to check
   finding 3 — contains a **null result and a failed pre-check that the summary does not
   mention**. This is the only item I would call an integrity problem rather than a
   presentation problem.

Numbers are otherwise in good shape: I checked 41 cited figures and 37 reproduce exactly.
Four do not (details below).

## Step 1 — every number, checked

Reproduce with `uv run python` against the JSONs named in each row.

### Verified correct

| claim in `summary.md` | source | value on disk |
| --- | --- | --- |
| mean abs error 0.029 / 0.025 over 11 cells | `lsweep_qwen1.5b/7b.json` | 0.02874 / 0.02456 |
| 0.053 / 0.045 over 6 fractional cells | same | 0.05269 / 0.04502 |
| cap predicts 0.667, RMS 0.816, measured 0.628 and 0.641 | same | 0.62774, 0.64062 |
| ℓ=6: W=6 → 1.00 beats W=4 → 0.64 | same | 1.0000, 0.64062 |
| σ₁ = 89% of the energy | `controls.json` | 0.89245 raw, 0.89376 unit rows |
| stance choice-shift 96.9% vs 51.2% | `controls.json` | 0.96875, 0.51250 (both at frac 0.5) |
| teacher-forced +20.7 → +28.6, constant ≈ 0 | `stance.json` | 20.729, 20.658, 23.890, 28.584; −0.16, −0.45, +0.16, +0.53 |
| random direction does nothing at any length | `stance.json` | 1.17±1.07, 0.78±1.38, 0.23±1.29, −0.15±1.24 |
| fixed-Hamming +71.8, +80.8, +79.6, +78.0, +77.5 | `controls.json` | 71.745, 80.843, 79.605, 77.976, 77.452 |
| scramble +29.0 → −2.3 (W=4), +55.3 → −1.2 (W=8) | `convex.json` | 28.987 → −2.251; 55.326 → −1.153 (frac 0.35) |
| contiguous +18.94 ± 2.40 vs scattered +17.42 ± 2.81 | `lsweep_qwen1.5b.json` | 18.9431 ± 2.3960, 17.4187 ± 2.8052 |
| 7B +16.76 vs +13.79 | `lsweep_qwen7b.json` | 16.7567, 13.7920 |
| S = +3.6 and +4.5 at W = 4, 8 | `convex.json` | 3.5611, 4.4750 (frac 0.35) |
| period-2 +0.208 ± 0.045 (t=4.6), +0.221 ± 0.058 at W=4 | `entrain2.json` | +0.2085 ± 0.0451 (t=4.62), +0.2213 ± 0.0584 (t=3.79) |
| unpredictable families \|t\| ≤ 1.3 | `entrain2.json` | iid max 0.97, balanced max 1.31 |
| i.i.d. 0.470–0.528 against nulls 0.496–0.501 | `entrain2.json` | exact |
| balanced null is 0.400 | `entrain2.json` | 0.400, 0.401, 0.401, 0.399 |
| graded L1 −5.06, L2 −7.35, L4 +3.49, L5 +2.94 | `graded.json` | −5.0628, −7.3536, +3.4884, +2.9367 |
| binary +15.0 versus graded +11.8; flipped −16.2 | `graded.json` | 15.0305, 11.8286, −16.2333 |

### Wrong, or not supported as stated

**(a) "13 are fixed by construction" — it is 18.** This is the most consequential error in
the document. The summary says only "4 where a width-1 block *is* the full template". In
fact the block-constant handle is bit-identical to the full per-segment template whenever
**W divides ℓ**, which is nine cells, not four: `(ℓ,W)` = (1,1), (2,1), (2,2), (3,1),
(3,3), (6,1), (6,2), (6,3), (6,6). In every one of them `sign(μ_b)` reproduces the ±1
template exactly, the same steering vector is written, and the saved `obs` is the *identical
float* to `full_peak.mean`. Verified: 9 of 24 cells have `obs == full_peak` bit-for-bit, in
both models.

So the census is 9 identity cells (R ≡ 1) + 9 structurally silent cells (R ≡ 0) + **6
at-risk cells** = 24. The clean statement of the partition is: *R = 1 exactly when W divides
ℓ; R = 0 exactly when 2ℓ divides W; only the remaining six cells can disagree with the
theory.*

Consequences:

- The "11 measured cells, mean \|error\| = 0.029" number — in the exec summary, in panel A's
  title, and in the log — averages 6 nonzero errors over 11 cells, 5 of which are exact
  algebraic identities contributing zero. It should be dropped entirely. The only defensible
  headline is **0.053 (1.5B) and 0.045 (7B) over 6 cells**, which the document already gives.
- `scripts/plot_phase_diagram.py:34-40` has the bug: `cell_kind` returns `"identity"` only
  `if W == 1`. The fix is one line — `if ell % W == 0: return "identity"`. Until then panel A
  hatches ℓ=6/W=1 but not ℓ=6/W=2, ℓ=6/W=3 or ℓ=6/W=6, all of which are the same
  construction; and panel C, captioned "6 fractional cells", actually scatters 11 points
  including five stacked at exactly (1.00, 1.00).

**(b) "Δ per covered slot is flat (22.2–23.7 for language, 7.8–8.4 for intensity)" — the
true ranges are 17.6–23.7 and 7.1–8.4.** From `wsweep.json`, Δ/coverage for every condition
with coverage ≥ 2:

- language: 17.60 (W1\_m2), 23.02, 22.20, 23.65, 22.83, **18.63 (W4\_m1)**, **20.91
  (W4\_m2)**, 23.41, 22.19, 22.19, 22.19
- intensity: 7.07 (W1\_m2), 8.30, 7.83, **7.31 (W3\_m1)**, 7.82, 8.41, 8.05, 7.76, 7.78,
  7.78, 7.78

The quoted range silently drops the three conditions that fall outside it. There is no
reading of "from two covered segments upward" that yields 22.2–23.7 — under W ≥ 2 the
minimum is 18.63, under coverage ≥ 2 it is 17.60. Also note that of the "thirteen
conditions", one (broadcast) has coverage 0 so Δ/coverage is undefined, and three share one
identical float (`full` = `W6_m2` = `W12_m1` = 266.23298297449946), leaving **10 distinct
measurements**. Fix the numbers; the retraction still stands, because Δ still tracks
coverage far better than width.

**(c) The superadditivity t-statistics are computed with the wrong standard error.**
`convex_modal.py:231-234` computes `S = ds.mean() - sum_marg` but stores `sem = ds.std/√n`
— the standard error of Δ(B) alone. The uncertainty in the subtracted Σ Δ_t is ignored
entirely. Treating the marginals as independent gives sem(S) = 2.06 and 2.80, so **t = 1.73
and 1.60**, not the quoted "t ≈ 2.6–2.8". The truth lies between those bounds because the
marginals come from the same 28 pairs and are positively correlated, but the per-item
marginal deltas are not saved, so it cannot be pinned down from the JSON. As written, the
limitation "a small positive effect" is stated with more confidence than the data supports;
under the independent bound S is not distinguishable from zero at either width. One-line
fix in the code: store `deltas` for the marginals and compute S per item.

**(d) There are four predicted reversals, not three.** The summary names ℓ=6, ℓ=1 and ℓ=2.
It omits **ℓ=3: W=2 (0.628) → W=3 (1.000)**, which is exactly the same type as the ℓ=6 one
it promotes. All four reversals are reproduced in both models, so this is a free point being
left on the table.

### Two claims that are correct but stated without their dose or their n

- The 96.9% / 51.2% pair is at **frac = 0.5**; at 0.2 and 0.35 the scheduled arm is 93.75%.
  The scramble collapse is at **frac = 0.35**; at frac 0.5 the scrambled arms are **+3.9 and
  +2.2**, i.e. still ≈0 but positive, not negative. Quoting the doses would cost six words
  and pre-empts the "which α did you pick?" question.
- The document reports no sample sizes anywhere. They are: lsweep n\_eval = 28 pairs,
  controls n\_eval = 32 and n\_stance = 20 (160 slots), convex n\_eval = 28, stance n\_eval
  = 32, graded n\_eval = 32, entrain2 n\_gen = 48. A reader cannot interpret a single ± in
  the document without these.

## Step 2 — the headline claims, attacked

### Finding 1: is "zero free parameters" honest?

**Yes, and this is the strongest thing in the sprint.** `R = mean_b |μ_b|` is computed from
the profile and the block structure alone; nothing is fitted. The obvious attack — that R is
normalised by Δ\_full measured on the same pairs at the same frac, so the agreement is partly
definitional — lands only on the W=1 column, where obs/Δ\_full is 1.0 by identical
arithmetic. It does not land on the six at-risk cells, where the numerator is a different
intervention from the denominator. The design also pins coverage at 12 in every cell, which
kills the coverage confound that ate the first centrepiece. Good.

The real weakness is not "definitional", it is **thin**: the law is tested on six cells, at
two predicted values (0.333 and 0.667), in each model. Per-cell relative error on those six
is 0.75, 1.06, 1.06, 0.94, 0.62, 0.96 (1.5B) — i.e. one cell (ℓ=3, W=4: predicted 0.333,
observed 0.206) is off by 38%, and it is the one cell the summary does not quote. The text
says the energy-matched alternative "is distinguishable at three cells" and then reports
two. Report all three: even including the bad one, cap beats RMS decisively (mean absolute
error 0.064 vs 0.188 across the three cells), so the omission buys nothing and costs
credibility.

**A second objection the write-up does not anticipate.** The framing "the energy-matched
alternative (`R = RMS(μ_b)`) is distinguishable at three cells and the cap form fits" presents
a *different experimental arm* as a *rival hypothesis*. Both predictions come from the same
formula `Δ = δW Σ_b c_b μ_b`: feed it `c = sign(μ)` and you get mean|μ|; feed it
`c = μ/RMS(μ)` and you get RMS(μ). The experimenter chooses which to write. So this is not a
horse race between two laws — it is one law predicting two arms, and it predicts both
(mean absolute error 0.064 for the cap arm against its own prediction, 0.080 for the energy
arm against its own). That is a *better* result than a horse race and should be stated that
way: **one formula, two normalisations, both predicted without fitting.** As written, a
reviewer who works out that RMS is not a rival will assume the framing was chosen to look
like a model comparison.

### Finding 2: is "fidelity per knob peaks at W = ℓ" supported?

**No, not as stated — and the figure shows it.** Panel B's title reads "Control efficiency
peaks at W ≈ ℓ" and its green star sits at **W = 6 for ℓ = 2**.

The underlying reason is a fact about the theory that the write-up has not noticed. For
W an odd multiple of ℓ, `R = 1/(2m+1)` and `W = (2m+1)ℓ`, so `R·W/k = ℓ/k` — **identical for
every odd multiple of ℓ**. The predicted efficiency curve therefore has *ties*, not a unique
peak:

- ℓ=1: W=1 and W=3 both predict 0.0833. Observed 0.0833 vs 0.0622 → argmax lands on W=ℓ.
- ℓ=2: W=2 and W=6 both predict 0.1667. Observed 0.1667 vs **0.1765** → argmax lands on W=3ℓ.
- ℓ=3: W=3 is the unique maximum in the tested grid (W=9 does not divide 12).
- ℓ=6: W=6 is the unique maximum (W=18 > k).

So the argmax claim is unambiguously supported in **two of four rows**, and in the two rows
where the theory predicts a tie the noise breaks it once each way. The claim as written is
overstated and the figure visibly contradicts it.

The fix is easy and makes the finding *stronger*, because the correct statement is sharper:
**W = ℓ is the largest width that retains full fidelity, so the minimum number of knobs for
lossless control is k/ℓ.** That is exactly true (R = 1 iff W | ℓ), it is what the safety
reader cares about, and it does not depend on an argmax over a tied curve. It also converts
"wider than that collapses" — which is false, since efficiency returns to ℓ/k at every odd
multiple — into "wider than that is never better, and is worthless at even multiples of ℓ".

### Finding 3: does the staged-refusal result depend on the banks, and is 96.9% a fair metric?

**The metric is fair, and the write-up now defends it well.** I checked
`controls_modal.py:334`: `d = (s_marg - b_marg) * signs[t]`, where `signs` alternates over a
shuffled balanced profile. An arm that pushes every slot one way therefore scores ~50%, not
~100%, and the broadcast arm's 51.2% is the empirical demonstration of exactly that. The
added sentence — "it is right on exactly the half of the slots that were meant to decline" —
is the single best piece of writing in the document. Direction fitting on `REFUSE[:6]`/
`COMPLY[:6]` and evaluation on `[6:]` is genuine held-out use, as claimed.

**The "single" arm asymmetry is real but the figure already handles it.** With
`coef = signs[0] if t == 0 else 0.0`, slots 1–7 get `vec = None`, so `d ≡ 0` and
`correct = float(d > 0) = 0`. The arm's structural ceiling is 1/8 = 0.125 and it scored
**exactly 0.125** — it succeeded on 100% of the one slot it wrote. Panel B draws the 1/8
line, which is the right call. Two follow-ups the text should make: the annotation
"1/8 = one slot of eight" is **overprinted by the bold 0.125 value label** in the rendered
PNG and is unreadable; and per *written* slot the single arm's mean shift is 1.363 against
the template's 1.004, so the schedule's advantage is coverage-with-correct-sign, not
per-slot potency. Saying so costs nothing and removes an easy jab.

**Three objections the write-up does not answer.**

- *Arms are not prefix-matched.* `controls_modal.py:337-340` appends the model's **steered**
  pick to `ids` and continues. By slot 4 the template arm is conditioning on a coherent
  alternating refuse/comply prefix while the broadcast arm is conditioning on an all-refuse
  prefix. Some of the 96.9% may be the prefix priming the alternation rather than the write
  at that slot. A teacher-forced fixed prefix, identical across arms, removes this.
- *Slots are clustered.* n = 160 slots come from 20 prompts, and the SEM is computed as if
  all 160 were independent. The 96.9 vs 51.2 gap survives any plausible clustering
  correction, but the reported ± on `mean_shift` is understated.
- *The task's premise is currently unverified.* See below — this is the serious one.

**The undisclosed null result.** `results/temporal_screen/stance_gen.json` is the output of
the harness the log says was built to replace the artifact-prone classifier. It contains:

- `precheck3.classifier_coverage = 0.101`, `n_transitions_from_refuse = 0`,
  `p_comply_after_refuse = 0.0`, with a single classified transition in the whole run. The
  three-way re-run therefore **did not reproduce the 0.870 gate** that licensed the staged-
  refusal design; it produced no usable data at all.
- `menu` (generation-time, model picks between held-out candidates, chance 0.5):
  template@0.5 = **0.536 ± 0.016**, single@0.5 = **0.526 ± 0.013**, broadcast@0.5 = 0.500,
  unsteered = 0.500. At frac 0.35 every arm is exactly 0.500 ± 0.000.

Either reading obliges disclosure. If the harness works, the *generation-time* behavioural
effect is +3.6 points over chance with the single-segment arm at +2.6 — a much weaker and
much less separated result than 96.9% vs 51.2%, and it belongs next to it. If the harness is
broken — and four of seven arms returning exactly 0.500 ± 0.000 alongside 10% classifier
coverage says it probably is — then the pre-registered gate for the whole task is currently
unverified and the limitation should say so. What the summary currently says is *"one of the
two generation harnesses had to be rebuilt mid-sprint after its classifier proved
artifact-prone"*, which reads as though the rebuild succeeded. **This is the one place where
a reviewer with repo access would accuse the document of selective reporting, and it is the
highest-priority fix in this report.**

On bank dependence specifically: the requests are combinatorial (32 verbs × 22 objects), not
"~40" as the text says — 40 is `n_train`, the direction-fitting budget. Reword. Bank
dependence itself is untested (one refuse bank, one comply bank, one model), which is fair
for a sprint but should be named in Limitations alongside "one layer, one language pair".

### Finding 4 (entrainment): the weakest item, and the figure oversells it

The numbers all check out. The interpretation does not.

**The raw tail accuracy does not rise at W=3 in either predictable family.** From
`entrain2.json`, unsteered tail accuracy is: period-2 → 0.578, 0.502, 0.542, **0.221**;
alternating → 0.440, 0.523, 0.458, 0.367. Nothing jumps. The entire "+0.208" at W=3 comes
from the analytic null falling from 0.500 to **0.333** at that width — and it falls to 0.333
for *both* predictable families at *exactly* W=3, which is also the only width at which
either family shows an excess. The null-subtraction may well be the right analysis (if
persistence stops paying, holding accuracy constant does mean the model is doing more than
persisting), but the figure plots only `acc − null`, so a reader cannot see that the raw
number is flat, and the word **"jumps"** in the summary is not true of anything the model did.

**The W=4 point is degenerate.** `analytic_persistence_null = 0.000` there, so the reported
"+0.221 ± 0.058 at W=4" *is* the raw accuracy, which has fallen from 0.542 to 0.221. Quoting
it immediately after the W=3 number implies the effect strengthens; in absolute terms it more
than halved. It should either be dropped or explicitly flagged as a zero-null cell.

**Both predictable families peak at W=3, but they have different ℓ.** W\* = ℓ+1 predicts
W\*=2 for the alternating family and W\*=3 for period-2. The alternating curve does not move
at W=2. The summary calls this "the honest blemish" — good — but the figure does not: a grey
band and a bold blue annotation sit at W=3 as if a shared threshold explained both curves.
And at W=4 the alternating family goes **significantly negative** (−0.133, t = −2.38), which
is evidence against the mechanism, not merely a blemish.

**Unreported:** realized coverage is 0.73–0.89, so at nominal W=4 only ~2.9 sentences were
actually steered on average; and `analytic_null` is a 20 000-trial Monte Carlo estimate, so
"analytic" is a misnomer.

With one family showing the predicted threshold, the other showing it at the wrong width and
then reversing, and the whole signal carried by a null that moves under the data, this is an
*inconclusive* result presented as a positive one. See Step 4.

### Finding 5: is the scramble control as decisive as claimed?

**No — and the summary knows better, because the log says so and the summary dropped it.**

`log.md` line 178: *"Note this confirms the additive/schedule account rather than proving
superadditivity: under additivity a random internal permutation matches ~half the slots and
nets ≈ 0."* That sentence is exactly right and it is **absent from summary.md**. What
survives is the heading *"The effect is the order, not the mass"* and a collapse from +29.0
to −2.3, with no statement of what the additive model predicts for the scrambled arm.

Under the additive account the scrambled write contributes `Σ_t π_{σ(t)}·π_t`, which has mean
zero — so ≈0 is the prediction of the *same* model that finding 1 already assumes. The
control is genuinely valuable: it kills "any injected norm in the right place helps", holding
coverage, contiguity and norm exactly fixed. But it does not discriminate between additivity
and anything else, and framing it as the decisive control invites a reviewer to point out
that it confirms the null model. Restore the log's sentence.

Two smaller points on the same figure: the W=1 bar labelled "100%" is the identity —
`convex_modal.py:194-197` shuffles a one-element list — so it is a sanity check, not a
result, and should be annotated as such. And the negative scrambled values are frac-0.35
specific; at frac 0.5 they are +3.9 and +2.2.

### Two more soft spots

- **"the curve is flat"** (fixed-Hamming). k=2 is 11% below k=4 (71.75 vs 80.84, Welch
  t = 1.75, n = 32 each); OLS slope over k=2…10 is +0.43 per k. The log's own phrasing —
  *"spread 3.4 over k=4…10"* — is accurate and the summary dropped the qualifier. Say "flat
  from k=4 onward, with the k=2 point 11% lower".
- **"contiguous versus scattered … are indistinguishable"** is a failure to reject, not an
  equivalence. 95% CIs on the difference are [−5.7, +8.8] at 1.5B and [−2.5, +8.4] at 7B —
  the upper bounds admit a 50–60% contiguity advantage. Say "no detectable difference at
  n = 28" and give the 7B SEMs, which are currently omitted while the 1.5B ones are given.

## Step 3 — writing quality

Judged against the brief: 2–5 findings each carried by one self-explanatory graph, executive
summary under ~600 words, positive phrasing, no unexplained jargon, honest about limitations,
figures legible to a fresh reader.

**What works.** The level/shape framing in the opening two sentences is excellent and does
real work — it tells a reader who has never heard of a temporal crosscoder why to care by the
third line. The permutation design is explained in one paragraph that a non-specialist can
follow. The "51.2% is its expected value rather than a malfunction" passage turns a weak-
looking baseline into the cleanest statement of the thesis in the document. Phrasing is
positive throughout; there is no hedging fog.

**Length.** The executive summary is **1252 words**, more than double the ~600 target. Finding
1 alone is 355 words and most of them are cell bookkeeping. The bookkeeping is *correct* to
include, but it belongs one level down: the exec summary should say "six cells can disagree
with the theory; mean error 0.053 and 0.045, nothing fitted" and push the 9/9/6 partition
into "The experiments". As it stands the reader meets a census before they meet the result.

**Five findings is one too many.** The brief allows up to five, but finding 4 is the weakest
by a wide margin (see above) and finding 5 is half retraction. Four would read better.

**Sentences that overstate.**

- *"Control efficiency peaks at W ≈ ℓ"* (panel B title) — contradicted by the panel's own
  ℓ=2 star.
- *"jumps at W=3"* (finding 4) — nothing in the raw accuracy jumps.
- *"the tail sits at its null through W=2 and jumps at W=3 … exactly the predicted threshold"*
  — true of one family; the other family's threshold prediction failed.
- *"Δ per covered slot is flat (22.2–23.7 …)"* — the true range is 17.6–23.7.
- *"the curve is flat"* — flat from k=4.
- *"are indistinguishable"* — underpowered, not equivalent.
- *"t ≈ 2.6–2.8"* — wrong standard error; 1.6–1.7 under the independent bound.
- *"11 cells that are genuine measurements"* — 6 are.

**Things a fresh reader cannot follow.**

- Panel A says "mean \|error\| = 0.029" and panel C says "0.053 / 0.045" in the same figure,
  with no explanation of why one figure reports two different error rates. Pick one (0.053 /
  0.045) and caption the other away.
- `R`, `μ_b`, `ℓ`, `W`, `k`, `δW`, `S(B)`, "fidelity per knob", "coverage", "Hamming",
  "teacher-forced margin", "block-constant handle" all appear without definition. `Δ = δW Σ_b
  c_b μ_b` is dropped into "The experiments" with `δ` and `c_b` never defined anywhere in the
  document. Either define them in one bracketed line each or drop the formula and keep the
  prose statement ("the handle retains the part of the profile that is constant across W").
- "the seed of the superadditivity result above" points *down* — superadditivity appears only
  in Limitations, below.
- **"Controls (finding 4)"** in "The experiments" is now an off-by-one: finding 4 is
  entrainment and controls are finding 5. There is also **no methods paragraph for the
  entrainment experiment at all** — the renumber left a hole.
- The map table lists `dict_modal.py` — "window-spanning vs per-token dictionaries at matched
  knob budget" — but there is no `dict*.json` in `results/temporal_screen/`. Advertising an
  experiment with no output, in a document whose central retraction is about a dictionary
  claim, is asking for trouble. Mark it "written, not run" or cut the row.
- The stance figure's "1/8 = one slot of eight" annotation is overprinted and illegible.

**Is the retraction section convincing, and is it in the right place?** Convincing — genuinely
so. Three retractions, each with the control that forced it and the surviving weaker claim
named explicitly, is the most credibility-generating material in the document. Two changes:

1. **Do not move it earlier.** A reviewer decides whether to read the code based on whether
   there is a result worth checking; leading with retractions inverts that. Its current
   position — after the findings, before Limitations — is right.
2. **But signal it earlier.** One clause in the exec summary — currently the retractions are
   mentioned only inside finding 5, where "Two claims we started with did not survive their
   own controls" is buried behind the scramble result. Promote it to its own sentence at the
   end of the exec summary: "Three of the claims we started the night with did not survive
   their own controls; all three, and what replaced them, are in *What we corrected*." That
   converts a defensive read into a confident one.

It does not read as damage control. It reads as damage control *only* in one place — the
Limitations line about the rebuilt generation harness, which describes a rebuild without
reporting that the rebuilt harness returned nothing. Fix that and the section is clean.

## Step 4 — the one experiment, and the one cut

### Run this

**Extend the resolution family to k=24.** The single biggest weakness in the write-up is that
its best result rests on six informative cells at two distinct predicted values. k=12 with
ℓ ∈ {1,2,3,6} is nearly the worst possible grid for this law, because W and ℓ are almost
always in a divides / even-multiple relationship, which is precisely when the theory makes no
risky prediction. At k=24 with ℓ ∈ {1,2,3,4,6,8} and W ∈ {1,2,3,4,6,8,12}, most cells fall in
neither degenerate class, and the number of at-risk cells rises from 6 to roughly 25 — with
predicted R values spread across many distinct fractions instead of just {1/3, 2/3}.

```bash
modal run experiments/temporal_screen/trajectory_steering/lsweep_modal.py \
  --k 24 --ells "1,2,3,4,6,8" --ws "1,2,3,4,6,8,12" --n-eval 28
```

It is the same script with different arguments, so the risk of it failing is near zero, and on
an A10G it is well inside the remaining budget. It buys three things at once:

- turns "mean error 0.053 over 6 cells" into a real law with a scatter plot that has points
  spread along the diagonal instead of clustered at two x-values;
- **directly tests the odd-multiple tie prediction** — `R·W/k = ℓ/k` for every odd multiple of
  ℓ — which is a sharp, falsifiable, currently-untested consequence of the theory, and the
  fix for finding 2's overstatement. At k=24 with ℓ=2 you get W=2 and W=6 predicted equal, and
  with ℓ=4 you get W=4 and W=12 predicted equal, so it is tested twice;
- removes the ℓ=4 exclusion, since 4 divides 24 with zero DC component, which closes the one
  "why is this run length missing?" question a reviewer will ask.

Runner-up, if the stance result matters more than the law: fix and rerun `stance_gen_modal.py`
so the generation-time metric either confirms or disconfirms the 96.9%. That is higher-value
if it works and higher-risk if it does not, and it can be run concurrently.

### Cut this

**Demote finding 4 (entrainment) to a paragraph in "What we corrected".** It is the only
finding whose headline is not supported by the raw measurements, its two predictable families
disagree with each other, one of them significantly anti-entrains at the widest prefix, and
its W=4 point is scored against a null of zero. The genuinely valuable part — the discovery
that a balanced profile forces a 0.400 persistence null, so a naive 0.5 null makes a working
model look broken — is *methodological* and already lives in the corrections section, where it
is one of the better paragraphs in the document. Keeping the positive framing costs the
document more credibility than the result adds, and cutting it brings the exec summary back
toward the word limit and the figure count to a clean 3-for-4.

If something must be cut for length rather than substance, the second candidate is **"What
problem this is, and why it is interesting"**, whose first two paragraphs restate the exec
summary's opening. Its third paragraph — the prior-context note about broadcast beating a
schedule on days-of-the-week — is new information and should be kept wherever the rest goes.
