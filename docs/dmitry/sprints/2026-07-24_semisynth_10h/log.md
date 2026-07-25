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

### 22:20 — W-sweep landed (task #1 done). Centerpiece confirmed.

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

### 22:40 — realmodel agent reported; free win banked; stance experiment launched

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

### 23:10 — staged refusal WORKS; entrainment null was wrong; two artifacts caught

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

### 23:00 — THE CENTERPIECE IS RETRACTED, and the replacement is better

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

### 23:40 — controls all landed. Three verdicts, one of them against me.

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
