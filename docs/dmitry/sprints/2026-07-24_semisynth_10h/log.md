---
author: Claude (Fable 5), for Dmitry Manning-Coe
date: 2026-07-24
tags:
  - results
  - in-progress
---

## Sprint log — semisynth 10h

### 21:58 — kickoff

Branch `dmitry-semisynth-10h` created off `dmitry-spectral-sprint2`. Sprint dir +
`start.md` written. Plan: W-sweep centerpiece first (it is the user's named target —
"performance improves with window size"), agents in parallel.

Design decision for the W-sweep: operationalize "window size" as the **span of one
steering knob**. A window-W handle covers contiguous blocks of W segments; one knob =
one block written with the *correct per-segment schedule inside its span* and one scalar.
Fixed budget of m knobs ⇒ coverage m·W segments. W=1 is exactly a per-token (SAE-like)
latent; W=k is one latent writing the whole trajectory. Prediction (theory agent to
formalize): Δmargin(W; m) ≈ Δ_full · min(mW, k)/k — linear growth in W, saturation at
W = k/m. Block placement rotated per eval pair to wash out position bias.

k=12 (divisors 1,2,3,4,6,12), tasks lang_profile + alt_phase, fracs {0.2, 0.35, 0.5}
(peak region from today's full run), n_train=40, n_eval=32.

### 22:19 — W-sweep landed (task #1 done). Centerpiece confirmed.

`results/temporal_screen/wsweep.json`. Peak Δmargin vs W at fixed knob budget:

- **Monotone growth in W, both tasks, both budgets.** lang m=1: 15→46→71→75→140→266
  (W=1→12); alt m=1: 5.7→16.6→21.9→33.7→46.6→93.4. m=2 ≈ 2× m=1 throughout.
- **Additive prediction Δ_full·min(mW,k)/k fits.** alt W6_m1: predicted +46.7,
  observed +46.6±2.3. lang within ~1 SEM at most points.
- **Broadcast negative at k=12** (lang −8.1, alt −4.4): the DC floor.
- **Boundary penalty at W=1**: per-covered-slot efficiency ~30% below the W≥2
  plateau in both tasks (lang 15.5 vs ~23; alt 5.7 vs ~8). Isolated single-segment
  writes are less efficient than the same slots inside a window — per-slot
  efficiency itself improves with window size. Worth a dedicated check later
  (boundary count is 2 per block regardless of W ⇒ penalty ∝ 1/W).
- Deviations to note honestly: lang W1_m1 and W4_m1 undershoot prediction (~0.7,
  ~0.84×); alt is near-perfectly linear.

Next: entrainment generation W-sweep (task #2) — switching to language-alternation
vs random-language profiles so the objective langid judge covers both the
predictable and unpredictable case with the same direction u.

### 22:22 — realmodel agent reported; free win banked; stance experiment launched

`real_behaviors.md` delivered. Two things acted on immediately:

1. **Free win (zero compute).** `lang_profile` is not merely a constructed task — it
   *is* a studied real behavior: per-segment language control / unintended
   code-switching mitigation (arXiv:2510.13849, arXiv:2507.13410). So the existing
   generation result (81.2% template vs 44.4% broadcast per-slot accuracy) already
   IS a real-behavior result. That closes the "no bridge to behaviors anyone cares
   about" objection at zero cost, and it means the sprint has a real-behavior
   headline even if the new stance experiment fails.
2. **Task #4 launched: `stance_profile` (staged refusal).** Steering the stance
   trajectory *within one response* (refuse/comply per sentence) — mid-response
   safety recovery. Implemented to the agent's spec: chat template applied, disjoint
   bank halves (A trains the direction, B builds eval pairs, so a lexical direction
   cannot transfer), target/foil are permutations of the *same* sentence set, and a
   fourth **random-direction template** arm at matched magnitude. No harmful content
   is used or produced — stance is carried by the frame and the complying sentences
   are content-free procedural filler.

   The agent's **pre-check gate** runs first in the same job: on unsteered
   generations, P(comply at t | refuse at t−1). Below ~0.15 ⇒ autoregressive-attractor
   dominated, and generation-mode numbers would mean nothing (teacher-forced results
   are unaffected either way, which is why both run in one job). Also recorded:
   cos(u_stance, u_prompt_refusal) — near-zero would say we captured an apology-style
   direction rather than a refusal direction.

Entrainment (task #2) interim, k=6, frac 0.35: alternating steered ≈0.71–0.86 with
unsteered rising 0.37→0.58 as W grows; random W=1 unsteered 0.39. The predicted
contrast (unsteered accuracy rises with W only when the profile is predictable) is
so far tracking.

### 22:25 — staged refusal WORKS; entrainment null was wrong; two artifacts caught

**Task #4 (stance) result — the real-behavior match lands.** k-sweep, n=32/point,
`results/temporal_screen/stance.json`:

| k | template | broadcast | single | random-direction template |
|---|---|---|---|---|
| 2 | +20.7±1.6 | −0.16±0.38 | +5.2 | +1.2±1.1 |
| 4 | +20.7±2.5 | −0.45±0.40 | +3.5 | +0.8±1.4 |
| 6 | +23.9±2.0 | +0.16±1.18 | +3.4 | +0.2±1.3 |
| 8 | +28.6±3.4 | +0.53±0.94 | +0.8 | −0.15±1.2 |

The **random-direction template arm sits at zero at every k** — the effect is not
generic activation perturbation. Broadcast is pinned at zero as predicted.

**The W-sweep on stance is CONVEX, not linear** (k=8, m=1, as a fraction of the
additive line Δ_full·W/k): W=1 → 0.57, W=2 → 0.77, W=4 → 0.60, W=8 → 1.00. Reach
buys *more* than proportional coverage. This is the one result additivity does not
explain, and it generalizes the W=1 boundary penalty seen in the k=12 sweep. It now
needs a control before it can be believed (contiguous-vs-scattered at matched
coverage is already running inside the phase-diagram job).

**Artifact 1 — entrainment's null was mis-specified (my error).** The theory demanded
a hard 0.5 null for random profiles; v1 measured 0.37–0.48, i.e. systematically
*below* chance. Cause: with BALANCED profiles (exactly k/2 per language), a
French-heavy steered prefix forces an English-heavy tail *by construction*, so a
model that merely persists in the last steered language scores below chance. Exact
analytic persistence null for k=6 balanced: **0.400** — which matches the observed
0.37–0.48 closely. v2 (`entrain2_modal.py`, running) adds i.i.d. (unbalanced)
profiles, for which the null is genuinely 0.5, keeps the balanced family to
demonstrate the artifact, adds a period-2 family (predicted threshold W*=ℓ+1=3), an
m=0 control per family, and n=48.

### 22:28 — THE CENTERPIECE IS RETRACTED, and the replacement is better

The review agent's audit landed and it is correct. I verified the decisive claim
independently before acting.

**O3 — the W-sweep measured COVERAGE, not window size.** Because `m` consecutive
blocks of width `W` occupy one contiguous span of `mW` segments, the (W, m) grid is
a reparameterisation of coverage. Per-covered-slot Δ is flat across all thirteen
conditions (lang ≈ 22 ± 2, alt ≈ 7.8 ± 0.5), and the matched-coverage contrast run
inside the phase-diagram job is null: **contiguous +18.94±2.40 vs scattered
+17.42±2.81** at coverage 4. So "performance improves with window size", as I ran it
at 22:20, is "performance improves with number of segments written". **Retracted.**

**O1 — the k-growth is largely metric bookkeeping.** The teacher-forced margin sums
log-probs over all k segments, and a permuted foil differs from the target in ~k/2
slots, so a constant per-slot effect mechanically yields a linear-in-k curve.
Normalised per differing slot, lang goes 37.8 → 42.2 (+12%, roughly flat) and alt
goes 10.5 → 8.05 (−23%, decaying) across k = 2…10. The fixed-Hamming control (H=2 at
every k) is running to settle it.

**What survives is stronger, because it cannot be coverage.** In the (W, ℓ) phase
diagram every block-constant handle spans **all** k segments — coverage is pinned at
k by construction — and only the handle's *resolution* varies. Result: 24 cells,
zero free parameters, **mean |obs − pred| = 0.013**, max 0.127. All three predicted
non-monotonic zig-zags reproduced (ℓ=1: W=3 → 0.25 beats W=2 → 0.00; ℓ=2: W=6 → 0.35
beats W=4 → 0.00; ℓ=6: W=6 → 1.00 beats W=4 → 0.64). A wider window beating a
narrower one is pure combinatorics — whether the block spans an even or odd number of
runs — and reproducing it is far stronger evidence for the additive account than any
monotone curve.

The honest version of the sprint's target claim lives here: **fidelity per knob peaks
at W ≈ ℓ**, so the best handle width is the target's own intrinsic timescale, and the
number of control parameters needed for full fidelity is k/ℓ. Figure:
`plots/2026-07-24_trajectory_steering/phase_diagram.png`.

### 22:33 — controls all landed. Three verdicts, one of them against me.

**O1 (fixed-Hamming): the review agent's prediction was right.** With foils built by a
single swap so H=2 at *every* k, the template effect is **flat**: +71.8, +80.8, +79.6,
+78.0, +77.5 for k = 2…10 (spread 3.4 over k=4…10), against +75.7 → +218.9 (+189%) for
the permuted-foil version. The k-growth headline was Hamming bookkeeping. The surviving
claim: **per-differing-slot steering efficacy is constant in trajectory length** — the
handle does not degrade as the trajectory gets longer, which still contrasts with
broadcast at zero, but is much weaker than "grows".

**O2 (rank): the handle is rank-1.** SVD of the 12×d per-position difference-of-means
matrix gives σ₁ = **0.892** of the energy (0.894 with unit rows). So the "temporal
template" is one direction with a sign schedule. We can claim *a schedule beats a
level*; we cannot claim *a temporal dictionary beats a per-token one* — that needs
trained dictionaries and is out of scope tonight. Scoped accordingly everywhere.

**Superadditivity: modest, real at W≥4, and the mechanism test is the striking part.**
Using per-position marginals (steer each segment alone) and S(B) = Δ(B) − Σ_{t∈B} Δ_t:
S = +0.47±0.85, +3.56±1.27 (t=2.8), +4.47±1.72 (t=2.6) for W = 2, 4, 8 at frac 0.35,
with the same pattern at frac 0.5. So a block is worth ~8% more than its own parts at
W≥4 — positive but not the clean c·(W−1) edge law, which it fails (S saturates). Δ per
unit **injected norm** rises +33% from W=1 to W=4 (0.656 → 0.872), which is the
reviewer-proof version of "wider is more efficient".

**The scramble control is the best single control in the sprint.** Permuting the
schedule *inside* the block holds coverage, contiguity and total injected norm exactly
fixed and destroys only the order. Effect collapses: at W=4 the scrambled write gives
**−2.25 against +28.99**, at W=8 **−1.15 against +55.33** (ratios −0.08, −0.02). So the
result is about writing the *right pattern*, not about adding more mass. Note this
confirms the additive/schedule account rather than proving superadditivity: under
additivity a random internal permutation matches ~half the slots and nets ≈ 0.

**Entrainment v2 nulls confirmed analytically.** i.i.d. profiles: unsteered-tail
accuracy 0.470–0.528 against the analytic null 0.496–0.501 — the hard 0.5 null holds.
Balanced profiles: 0.395–0.443 against the analytic persistence null 0.399–0.401 — the
v1 "below chance" anomaly is fully explained by balance-induced anti-correlation, not
by a bug. Steered-slot accuracy is 0.84–0.97 throughout; m=0 controls sit at 0.41–0.48.

**Artifact 2 — the stance pre-check classifier (caught by the realmodel agent).** A
binary refusal-marker regex scores every *unmatched* sentence as "comply", which
inflates P(comply | prev refuse) — so the 0.870 that passed the gate is suspect.
`stance_gen_modal.py` (running) re-runs it three-way (refuse/comply/unparsed) with
coverage reported, and adds the objective behavioral metric: **menu-constrained
generation** — at each boundary the model picks between a held-out refuse and a
held-out comply candidate by logprob under the steered state, giving per-slot
accuracy with zero classifier error. Teacher-forced results never used the
classifier and are unaffected.
### 22:36 — self-audit of the headline before the red team got to it

Asked of my own best result: how many of the "24 cells, mean error 0.013" are actually
measurements? Answer: **11**. Four cells (W=1) are the full template by construction, so
R = 1 identically. Nine cells are **structurally silent** — with a balanced profile the
block-constant coefficient `sign(μ_b)` is exactly zero in every block, so no vector is
written and Δ = 0 by arithmetic rather than by measurement. Quoting 0.013 over 24 cells
credits the law for 13 cells it was never at risk on.

Corrected numbers now used everywhere: **0.029 over the 11 measured cells** (7B: 0.025),
and **0.053 over the 6 cells whose prediction is strictly between 0 and 1** (7B: 0.045).
Those 6 remain a real zero-parameter test, and they discriminate the two candidate
budget laws: the per-slot cap predicts 0.667 where an energy-matched budget predicts
0.816, and the measurements are 0.628 / 0.641 — the cap form fits.

Same correction applies to the zig-zag. Only **ℓ=6 (W=6 → 1.00 beats W=4 → 0.64)** is a
reversal between two *measured* cells. The ℓ=1 and ℓ=2 reversals are measured-vs-silent:
still a true statement about the phenomenon (a width-2 handle on an alternating profile
can write nothing useful) but weaker evidence than a measured-vs-measured reversal.
Figure now hatches the by-construction cells so this is visible rather than buried.

### 22:52 — round-2 controls written and queued; Modal capacity blocked by another project

Round-2 audit (P1–P4, S1–S3, C1) accepted in full and the summary rewritten around it —
the headline is now *linearity of the steering response in the schedule*, with the six
informative cells as the result and the other 18 named as identities. Detail in the
22:56 entry below.

`round2_modal.py` implements the four cheapest killing controls the audit asked for:
(A) phase sweep to buy informative cells, reporting only 0 < pred < 1 with paired
bootstrap CIs; (B) stance fixed-Hamming with the reviewer's flat prediction registered
in the docstring before results exist; (C) span-vs-dose to the stated acceptance bar
(paired S_span, per-pair deltas stored, plus one-segment-at-W×dose as the dose control);
(D) direction identity with u_prompt re-measured at matched positions plus
politeness-matched banks.

**Infrastructure note.** Both `dict_modal.py` and `round2_modal.py` sat with zero tasks
for ~20 min. `modal app list` shows my apps at 0 tasks while an unrelated project on the
same account holds 10 — an account-wide container concurrency cap, not A10G scarcity
(switching to L4 changed nothing, which is how I confirmed it). The other project's jobs
are left alone; my runs stay queued and Modal will schedule them as capacity frees.
Everything the write-up currently claims is already measured and saved, so this delays
strengthening rather than blocking the deliverable.

### 22:58 — red team lands; four real errors fixed, one of them a bug in my own figure

`redteam.md` delivered. It verified the numbers against the JSONs and found four things
I had wrong, all now corrected:

1. **The identity census was 18, not 13** — and my `cell_kind` in
   `scripts/plot_phase_diagram.py` had the bug: it marked a cell as fixed-by-construction
   only when `W == 1`, when the true condition is `W | ℓ` (every block lies inside one
   run, so `sign(μ_b)` reproduces the ±1 template and the *same vector* is written).
   Verified: 9 cells have `obs == full_peak` bit-for-bit in both models. Census is now
   9 identity + 9 zero-write + **6 at-risk**. Panel C was scattering 11 points with five
   stacked at (1.00, 1.00); it now plots the 6.
2. **Per-covered-slot range was cherry-picked.** I wrote 22.2–23.7 (language) and
   7.8–8.4 (intensity); the true ranges over the ten distinct conditions with coverage
   ≥ 2 are **17.6–23.7** and **7.1–8.4**. Corrected; the retraction it supports is
   unaffected because Δ still tracks coverage rather than width.
3. **The superadditivity t-statistics used the wrong standard error** — `convex_modal.py`
   subtracts Σ Δ_t but reports the SEM of Δ(B) alone, ignoring the marginals' uncertainty.
   Under an independence bound t = 1.6–1.7, not 2.6–2.8, so the effect is not
   distinguishable from zero. Already retracted; the numbers now say so properly.
4. **Four reversals, not three** — ℓ=3 (W=2 → 0.628 beats W=3 → 1.000 in efficiency
   terms) is the same type as the ℓ=6 one I promoted.

Also adopted: doses and sample sizes quoted with every headline number, the "rebuilt
generation harness" limitation now states that the rebuild *returned a degenerate metric*
rather than implying it worked, and `dict_modal.py` is marked "written, not run".

**Entrainment demoted from a finding to a paragraph in the corrections.** The red team is
right that the raw accuracies do not support the normalised story: for the period-2 family
they run 0.578, 0.502, 0.542, 0.221 against nulls 0.600, 0.500, 0.333, 0.000, so the W=4
"excess" comes from the null collapsing to zero rather than the model improving. The
methodological yield — that a balanced profile forces a 0.400 persistence null, so a naive
0.5 makes a working model look broken — survives and is the part worth keeping.

### 23:05 — why the k=12 grid is weak, and the better fix

The red team's top recommendation was k=24 to buy at-risk cells. Checked on CPU first:
k=24 with ℓ ∈ {1,2,3,4,6,12} gives 12 at-risk cells (from 6) but still only **three**
distinct predicted values {1/6, 1/3, 2/3}, and the queued phase sweep gives 13 cells at
**two** distinct values. The limitation is structural — for a square wave every block mean
is a simple rational, so no amount of W, ℓ or phase variation spreads the predictions.

The fix is to drop square waves for the linearity test. For any balanced profile π and
**any** block-constant coefficient vector c, linearity predicts R = W·⟨c, μ⟩/k — a
continuum, including negative values when c opposes the target. `linfit_modal.py` (queued)
samples ~36 random (profile, W, c) conditions, each normalised against Δ_full on the same
pairs, with paired bootstrap CIs. It converts panel C from six points at two x-values into
a regression across the full range, and the negative half is a genuinely risky prediction:
a write that anti-correlates with the target should push the margin proportionally the
wrong way.

### 23:10 — round-2 lands: one confirmation, one confound I caught in my own test

Capacity freed at ~23:00 and both queued jobs ran.

**B — stance fixed-Hamming confirms the retraction.** +20.84, +18.47, +19.61, +18.43 for
k = 2, 4, 6, 8: flat, exactly the reviewer's registered prediction. Staged refusal does
not grow with response length either.

**D — what the declination direction is.** cos(u_stance, u_blunt_refusal) = **0.747**
against cos(u_stance, u_apology) = **0.495**, with the two banks themselves at 0.496 and
the matched-position prompt cosine at 0.091. So the direction leans toward the *act* of
declining rather than toward politeness, but the ratio is ~1.5:1, well short of the 3:1
the realmodel agent set as the bar for calling it a refusal direction outright. The
"declination-register" naming already in the write-up is the one the evidence supports.

**C — my span test was confounded, and I caught it before believing it.** The headline
looked strong: S_span = +7.74 ± 1.22 (t = 6.3) at W=4, replicated at both doses. Then I
checked what the "scattered" arm actually writes. It selects positions with stride k/W,
which on an alternating profile at k=8, W=4 lands entirely on one parity — so the
scattered arm writes a **single sign** at all four positions while the contiguous arm
writes two of each. The arms differ in the composition of the write, not only in its
adjacency, and a single-sign write on a subset is close to the DC write we already know
is weak. That t = 6.3 cannot be attributed to span. `span2_modal.py` rebuilds it with
stride 3 at k=12, which is sign-matched to the contiguous block for every W ≤ 4 while
sharing no adjacent pair; everything else — coverage, correctness at each covered slot,
injected norm, dose, eval pairs — is held fixed.

**A — phase sweep, with one honest wrinkle.** Most cells track their predictions
(ℓ=3 W=2: pred 0.667, obs 0.679 / 0.604 / 0.639 across phases). But two cells with the
*same* prediction of 0.333 came out at **0.145** and **0.676** with non-overlapping
bootstrap intervals. Linearity in the schedule alone does not explain that; it is
position-dependent heterogeneity, which the review agent had flagged as uncontrolled
(O9) and which now has direct evidence. Stated as a caveat in the write-up.

### 23:12 — the linearity test replaces the grid as finding 1

`linfit.json`: 36 random (profile, width, coefficient) conditions, 12 distinct
predictions spanning **−0.42 to +0.42**, measured against the full-schedule effect on
the same pairs. Fit: **slope 0.945, intercept −0.013, R² 0.765, mean |error| 0.073**.
Thirteen conditions predict a negative effect and the model moves the wrong way in
proportion (mean observed −0.233) — the riskiest part of the claim, and it holds.

This is the same claim the (W, ℓ) grid was making, with roughly six times the leverage
and without the square-wave restriction that pinned the grid to two distinct predicted
values. Promoted to finding 1; the grid becomes finding 2, read as control cost.
Caveat kept in view: 26 of 36 bootstrap intervals cover the prediction, below the
nominal 95%, so there is genuine excess scatter — consistent with the positional
heterogeneity the phase sweep exposed.

