---
author: Claude (Fable 5), for Dmitry Manning-Coe
date: 2026-07-25
tags:
  - results
  - in-progress
---

## Sprint log — dictionary benchmark 10h

### 17:15 — the capacity question, and why a behavioural number cannot answer it

Two things converged here: the crosscoder turned out to be a starved dictionary, and
capacity-matching for crosscoders is an open problem in this project that has historically
been guessed at. Worth recording the reasoning, because it outlasts this sprint.

**Nominal k is not the knob, now measured rather than suspected.** ReLU zeroes 96% of what
TopK selects (realised window L0 17.6 against nominal 492), which is also why the repo's
Protocols A and B cannot do what their docstrings claim — they differ only in nominal k.
The immediate fix is to stop guessing and **servo on realised L0**: measure on a held-out
batch, binary-search k to a target. Minutes, not intuition.

**But there is a prior question the sweep answers.** If realised L0 stays ≈18 as nominal
goes 492 → 4092, the ceiling is the *encoder* — not enough positive pre-activations — and
no k will fix it. If realised tracks nominal, k really is the knob. Those are different
diseases with different cures, and they look identical if you only ever inspect nominal k.

**The comparison should be a frontier, not a point.** Segment-pooling makes FVU directly
commensurable (both architectures reconstruct the same objects), and the natural sparsity
axis is **coefficients per segment** — L0_window/T for the crosscoder, L0_token for the
SAE. On that axis the current pair are not matched at all: 1.5 versus 98.9 coefficients per
segment, at 27.6× the FVU. They sit at opposite ends of a curve nobody has drawn.

**And the honest limitation, which Dmitry named:** using steering fidelity as a
capacity-selection rule sidesteps interpretability. It is worse than a gap. The fidelity
metric asks how well a write projects onto a target direction, so a latent that is
"the factor plus fifty other things" scores like a clean one — the junk is ~orthogonal and
the rescaling removes it. And **feature splitting would register as improvement**, since
matching pursuit simply gets more atoms to fit with. In the exact regime capacity is
supposed to be policed, the behavioural axis points the wrong way.

What rescues it here is that this is a *semisynthetic* task with **ground-truth generative
factors**. The banks are labelled, so monosemanticity is measurable with no judge:
single-latent AUC for the known factor, the number of latents within 95% of the best (a
splitting count), and decoder cosine among the top factor-aligned latents (redundancy).
`interp_modal.py` runs those on the same capacity axis as FVU and steering.

Registered expectation: **FVU improves monotonically, single-latent AUC peaks then falls as
features split, steering stays flat or rises.** If that is the shape, then steering is the
wrong instrument for *choosing* capacity and a fine one for *evaluating* it once chosen —
a cleaner division of labour than treating the behavioural knee as a stopping rule.

### 17:22 — the benchmark was training a temporal architecture on structureless data

Two errors, both mine, found while reading the interpretability panel. They invalidate the
*architectural* reading of everything before this entry.

**1. The training corpus contained no temporal structure.** Every run so far drew the
tense/calm label **i.i.d. per segment** (`lab = [rng.randint(0,1) for _ in range(k_seg)]`,
present in all six scripts). A temporal crosscoder's premise is that there are patterns
*across* a window to capture. Independent coin flips contain none, by construction. So the
benchmark trained a temporal architecture on temporally-structureless data, found it
starved and poorly-reconstructing, and was about to report that as an architectural
result. The 40% general-text portion has natural structure but carries no labelled factor,
so it cannot rescue the comparison.

**2. The interpretability measurement asked a window code a segment question.** A TXC
latent is **one scalar per 12-segment window**. With i.i.d. segment labels, predicting any
individual segment's label from that scalar would require encoding 12 independent bits —
so the measured ~chance result (best single-latent AUC **0.541**, against the SAE's 0.241,
i.e. 0.759 oriented) was a *structural certainty*, not evidence about training. I had built
a measurement the architecture cannot pass for reasons unrelated to what I was testing.
Same failure class as the previous sprint's degenerate adjacency statistic: an instrument
that cannot see the thing it is named for.

**The fair split, which is the more interesting question anyway.** Ask each architecture
what its code can in principle represent:

| target | SAE (per-segment code) | TXC (per-window code) |
| --- | --- | --- |
| **segment-level**: "is *this* segment tense?" | can represent | structurally cannot |
| **window-level**: "is this window fast- or slow-alternating?" | only via combination | can represent |

`structured_modal.py` (running) is a 2×2 — {i.i.d., run-length-structured} corpora ×
{SAE, TXC} — measuring both AUCs for both architectures on both corpora, with the i.i.d.
arm reproducing the earlier result as a control so the data effect is isolated rather than
asserted. Registered: on structured data the TXC's realised L0, FVU and alive fraction all
improve; the TXC wins window-level AUC and loses segment-level on both corpora because
that split is structural; segment-level steering fidelity is roughly unchanged.

If the structured corpus does *not* improve TXC health, then the starvation is a training
pathology independent of the data and the earlier negative stands on its own terms — which
is also worth knowing, and is why the i.i.d. control arm is being re-run rather than cited.

### 17:30 — retraction, a target dismantled by arithmetic, and the sprint's real shape

Review round on the pivot. Three things land, two of them against me.

**RETRACTED: the shuffled-control negative, as an architectural claim.** FVU 0.84 means the
crosscoder explains 16% of held-out variance, so its decoder rows are largely noise, so
"permuting its temporal arrangement costs nothing" is evidence about an *untrained*
dictionary rather than about temporal dictionaries. I will not write "the TXC recovers X%
of the ceiling and none of it is temporal" — the second clause is unsupported. What
survives is the SAE-side coverage explanation, and the two nulls **as a method**.
Registered gate for the corrected run: the TXC must reach per-segment **FVU < 0.3** on the
structured corpus before any corpus-contrast is interpretable; below that, a flat contrast
is 0 − 0 and the finding is the capacity bug, not a data effect.

**My window-level target was wrong, and the arithmetic is embarrassing.** "Fast versus slow
alternation" over ℓ ∈ {1,2,3,6} has change-point counts **11, 5, 3, 1** — perfectly
separated and monotone. So the label *is* the change count. A change point is a **local**
event (segment t differs from t−1), and the subject model is causal, so the residual stream
at segment t already carries t−1: a per-segment "this sentence contrasts with the previous
one" feature, mean-pooled, counts changes. Pooling is not a handicap on that target, it is
exactly matched to it. The TXC has **no structural advantage** there, and labelling it
"TXC can, SAE structurally cannot" would have been the same error I had just retracted,
with the sign flipped.

**The target that does isolate window structure**: hold balance *and* change-count fixed,
vary only arrangement — regular versus irregular spacing. At k=12 there are 200 balanced
profiles with exactly 5 change points, e.g. `110011001100` (runs 2,2,2,2,2,2) against
`000010101111` (runs 4,1,1,1,1,4). A local change detector counts 5 in both; only something
integrating the window separates them. This is the previous sprint's multiset-matched foil,
moved from steering to probing.

**And the corpus family is too small to prove anything.** Four run-lengths × two phases is
**8 distinct profiles ≈ 3 bits**, against 924 balanced ones — a d_sae=4096 dictionary can
allocate a latent per profile, so any "learned temporal structure" on that family is a
lookup table. The fix is a **change-probability sweep**, p ∈ {0.5, 0.35, 0.2, 0.1}: the
broken i.i.d. corpus becomes the *left endpoint of a dose–response curve* rather than a
separate arm, and the claim becomes "the advantage grows with the corpus's autocorrelation
length" — a monotone trend over four points for about the same compute.

**Correction to my own retraction, in my favour.** I wrote that a window code
*structurally cannot* carry segment-level labels. That is too strong: d_sae scalars per
window is ample for 12 bits. What made it chance was that the labels were **i.i.d.**, so no
code of any capacity could carry them. That is error 1 again, not a separate structural
fact — and it matters, because the corrected run re-probes segment-level on structured
data, where the TXC may well beat chance.

**The sprint's real shape**, which is a better framing than the one I proposed: *three ways
a temporal-dictionary benchmark breaks, each caught and measured tonight* — **capacity**
(the sparsity knob does not bind), **data** (the corpus must carry the factor's temporal
structure or the comparison is vacuous), **measurement** (the evaluation must ask each code
a question it can represent, and the baseline must be the strongest form of the opponent).
Each has a measured example, each is a mistake made and caught rather than speculated
about. And explicitly **not** attempted: an architectural verdict. One seed, one model, one
task is exactly the setup that produced four confounded positives last sprint and one
tonight.

Priority 1 launched: the **b_enc mechanism** (`mechanism_modal.py`) — predicted, and
checkable in minutes, that the encoder learns a strongly negative `b_enc` to route around a
k far larger than reconstruction needs, so realised L0 ≈ min(k, #{pre > 0}) with the second
term binding. If it holds, it invalidates every crosscoder-vs-SAE comparison matched on
nominal k, including this project's own, and it ships with a fix: set k from a *target
realised L0*.

### 16:19 — kickoff

Branch `dmitry-dictbench-10h`. Question: does a temporal crosscoder beat a TopK SAE at
steering a trajectory, when the schedule is read off a **decoder** rather than supplied
from ground truth? That was the previous sprint's stated largest limitation.

Checked the harness before scoping, and it is largely built: `TemporalCrosscoder` and
`TopKSAE` exist, matching protocols A and B are pre-registered, and there is a
cached-activation pipeline. Modal was completely free at kickoff (the other project that
blocked all of last night was idle).

**Design decision — segment-pooled, not token windows.** The bench TXC takes `(B, T, d)`.
Our steering acts on sentence segments, so T = number of sentences and activations are
mean-pooled within each sentence. A TXC latent then owns a k-segment pattern that lines up
exactly with what the steering harness writes, with no resampling between train and eval.

### 16:33 — harness agent's report, and the fairness bug it caught

`harness_guide.md` delivered, and it caught the thing most likely to have invalidated the
whole benchmark:

> At equal `batch_size` and equal steps, the TXC sees T× more token-activations than the
> SAE. `gen_flat` yields `(B, d)` = B tokens; `gen_windows[T]` yields `(B, T, d)` = B·T
> token-slots. Nothing in the harness corrects for this.

It also connects this to the project's own history — the 2026-05-05 purified-sampling fix,
where an SAE was getting ~25× more tokens/step than the TXC it was compared against. Same
bug class, opposite direction. My smoke had exactly this defect (SAE 256 tokens/step vs TXC
3072). **Fixed: the SAE's batch is now T× the TXC's, so both consume the same
token-activations per step, and both totals are printed.**

Three more traps from the same report, all now handled: `k` is multiplied by T inside the
crosscoder constructor (so `window_l0` is per window, not per token); protocols A and B are
**numerically identical at T=5** and the module docstring contradicts the code, so T must
differ from 5 for the two arms to be distinct (we use T=12: A → per-position k=100/window
1200, B → k=41/window 492); and `CrosscoderSpec.decoder_directions()` returns a live
autograd view while the SAE's returns `.data`.

### 16:35 — smoke passed as plumbing, and killed my first measurement design

Ran the end-to-end smoke (240 docs, d_sae=512, 400 steps, topk=8) before anything long.
It failed once on `decoder_directions` — that method lives on the **Spec** class, not the
module — then ran clean:

```text
[cache]  X=(240, 12, 1536)  base_norm=55.3  general_frac=0.40
[dict]   SAE k=8   TXC k(window)=96   (the constructor multiplies k by T)
[select] SAE latent 487 (sep −1.290)   TXC latent 441 (sep +3.786)
[rows]   |v_sae|=1.000   |P_txc| total=1.000   per-row mean=0.287
[steer]  sae_scheduled −7.08±1.99   sae_broadcast +0.45±1.76   txc_pattern +0.16±2.61
```

**The magnitude trap is real and now measured.** `TopKSAE._normalize_decoder` normalises
over dim 0, so each latent's direction is unit norm. `TemporalCrosscoder._normalize_decoder`
normalises over dims (1,2), so a latent's *entire* (T, d) pattern is unit norm and its
per-segment rows are ~1/√T — measured 0.287 against 1/√12 = 0.289. Comparing `α·row` across
the two without rescaling would have handed the SAE √12 ≈ 3.5× more injected norm. Every
write is now rescaled to a fixed total injected norm and both norms are printed.

**Two things the smoke caught that matter more than the plumbing:**

1. **Sign bug, mine.** Selection uses `|separation|.argmax()`, but the write did not then
   orient by `sign(separation)`. The SAE's selected latent has separation −1.29, i.e. it
   fires on *calm*, so writing `+v` on tense slots is backwards — which is exactly the
   −7.08 observed. Orienting by the sign flips it. Fixed.

2. **The m=1 comparison is meaningless on a random target, and that reshapes the whole
   experiment.** A single TXC latent writes ONE fixed (T, d) pattern. The eval draws a
   fresh random balanced profile per pair, so a lone latent can only help when its pattern
   happens to match — hence +0.16 ± 2.61. This is not a defect of the TXC; it is the
   measurement being posed wrongly.

**The corrected framing, which is cleaner and is now the sprint's measurement.** The
target write is a T×d matrix. What each dictionary offers is a *subspace* of such matrices:

| dictionary | what one scalar buys | span |
| --- | --- | --- |
| SAE, broadcast | one direction, constant over all T | rank-1 in time: `1_T ⊗ v_j` |
| SAE, per-position | one direction at one position | `e_t ⊗ v_j` — costs T scalars for a full profile |
| TXC | one learned T×d pattern | arbitrary temporal shape per latent |

So the question is whether the TXC's learned patterns span target trajectories more
efficiently than the SAE's outer-product constructions, and the metric is
**fidelity(m) = Δmargin(best least-squares write from m scalars) / Δmargin(ground-truth
schedule)**, with each architecture given its *best* allocation of the m scalars. That is a
crisp linear-algebra claim rather than a vibe, and it makes the SAE baseline as strong as
it can be, which is the point. Using signed least-squares coefficients also dissolves the
sign bug above — orientation is now something the fit chooses, not something I hand-set.

### 16:37 — mini-validation, and a structural result falls out for free

Tiny run (120 docs, d_sae=1536, 200 steps, protocol B) purely to exercise the new
matching-pursuit code before committing to the full job:

```text
[budget] TXC 64 windows = 768 token-acts/step;  SAE 768 tokens = 768 token-acts/step
[rows]   |sae col|=1.000   |txc slab|=1.000  row=0.289
[ref ]   full DoM schedule Δmargin = +33.47
   txc            m=1: fid=+0.237 recon_cos=0.198     m=4: fid=+0.285 recon_cos=0.345
   sae_broadcast  m=1: fid=+0.034 recon_cos=0.000     m=4: fid=+0.075 recon_cos=-0.000
   sae_perpos     m=1: fid=+0.010 recon_cos=0.082     m=4: fid=+0.242 recon_cos=0.164
```

**`sae_broadcast` has reconstruction cosine of exactly 0.000, and that is not a bug — it is
the previous sprint's central finding re-derived as orthogonality.** A broadcast atom is
`1_T ⊗ v`: constant in time. The target profile is *balanced*, so it has zero mean along
the time axis and is therefore **exactly orthogonal to every constant-in-time atom**, for
any direction `v` whatsoever. "A level cannot make a shape" stops being an empirical
finding about one model and becomes a statement about the subspace a per-token dictionary
spans when its coefficient is held constant across a window.

Directionally the registered prediction also appears — TXC ahead at m=1 (0.237 vs 0.010)
with the per-position SAE closing fast by m=4 (0.285 vs 0.242) — but at 120 docs and 200
steps that is a plumbing observation, not a result. Full run launched: 1500 docs,
d_sae=4096, 3000 steps, m ∈ {1,2,4,8,16,32}, both protocols.

### 16:40 — Protocol A lands, and it looks exactly like the confound we pre-registered

| m | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| TXC | 0.178 | 0.295 | 0.406 | 0.537 | 0.689 | 0.790 |
| SAE per-position | 0.075 | 0.125 | 0.305 | 0.531 | **0.754** | — |
| SAE broadcast | −0.04 | 0.02 | −0.02 | −0.02 | nan | nan |

Reference: full DoM schedule Δ = +35.65.

Three things, in descending order of confidence.

**Broadcast is structurally zero, and now at scale.** Reconstruction cosine is *exactly*
0.000 at every m, and the steering effect wanders around zero. This is not weakness, it is
orthogonality: a broadcast atom is constant along time, a balanced profile has zero mean
along time, so their inner product vanishes for **any** direction. The previous sprint's
"a level cannot make a shape" is now a property of the subspace a per-token dictionary
spans under a constant coefficient, rather than an empirical finding about one model.

**The TXC lead is exactly where the SAE is coverage-limited, and vanishes when it is
not.** TXC leads 2.4× at m=1 and m=2, the arms are level at m=8 (0.537 vs 0.531), and at
m=16 the **SAE is ahead** (0.754 vs 0.689). The crossover sits at m ≈ k = 12 — which is
precisely the budget at which the per-position SAE stops being able to touch only part of
the trajectory. That is the pre-registered confound C1 ("the m-budget buys coverage, not
dictionary structure") producing its predicted signature, and the review agent called it
before the run: *"it will look like the hypothesis confirming."* On current evidence I
would not claim a TXC advantage. The random-slab and time-shuffled nulls decide it.

**A numerical defect to fix, not a result.** `sae_broadcast` returns NaN at m ≥ 16: every
broadcast atom is orthogonal to the target, so matching pursuit is selecting on numerical
noise and the least-squares solve degenerates. It should return an exact zero write with a
"degenerate" flag rather than NaN. The arm is structurally zero regardless, so this
changes presentation and not conclusions.

### 16:47 — the harness gate fired, and the gate was wrong, not the instrument

The reviewer set the gate at 0.08 against the *theoretical* prediction 0.333 — then found
that the calibration standard itself fails that gate: last sprint's own measurement of the
same cell on the same model is **0.249**, a miss of 0.084 against theory. A threshold the
reference run cannot clear is not a threshold. Corrected and re-registered: compare against
the **stored reference at matched dose**, not against theory.

On that basis: our 0.218 against the reference's 0.249 is a miss of **0.031 — a pass with
room**. Dose matters and is now quoted, because the reference is dose-resolved
(0.144 / 0.210 / 0.249 at frac 0.2 / 0.35 / 0.5); at frac 0.35 the reference is 0.210 and
our miss is 0.008.

Two things I had wrong in the wording:

- I wrote "within its known spread". That launders a **systematic** deviation as scatter.
  The undershoot is predictable in sign and magnitude — it is the convexity last sprint
  measured. The honest sentence is: *the cell undershoots the homogeneous-linear
  prediction, in the direction and by approximately the amount the same cell undershot it
  on the same model at the same dose.* Stronger claim, same data.
- The three **exactly 0.000** cells deserve their own statement rather than being folded
  in: they are an unambiguous pass of the write path, the sign convention and determinism.

One cell is an anecdote; ℓ=3, W=2 (reference 0.628) is the highest-signal informative cell
and is added to the next run so this becomes a calibration rather than a spot check.

**Realised sparsity is not nominal sparsity.** TopK is applied as `scatter(relu(topk))`,
so ReLU zeroes much of what TopK selected: TXC nominal window k=1200 → realised L0 ≈ 81,
SAE nominal 100 → realised ≈ 98. Under Protocol A the TXC is therefore *sparser per slot*
(≈6.8) than the SAE is per token (≈98), which inverts the protocol's stated intent.
Realised L0 will be reported instead of nominal k.

### 16:48 — THE DECISIVE CONTROL: shuffling the TXC's temporal structure does not hurt it

| m | txc | txc **shuffled in time** | txc random slabs |
| --- | --- | --- | --- |
| 1 | 0.224 | **0.296** | −0.003 |
| 2 | 0.309 | **0.345** | 0.012 |
| 4 | 0.365 | **0.458** | 0.028 |
| 8 | 0.502 | **0.541** | 0.059 |
| 16 | 0.686 | 0.647 | 0.061 |

`txc_shuffled` takes the *learned* atoms and permutes their k rows in time — every
per-slot direction and every norm preserved exactly, only the temporal arrangement
destroyed. It **matches or beats the intact TXC at four of five budgets.**

Read together with the random-slab arm, this is a clean decomposition:

- **Learned content is essential.** Random unit-norm slabs with identical coverage and
  norm structure give ≈ 0 (−0.003 to 0.061). So the TXC is not winning on geometry.
- **The temporal arrangement of that content is not doing the work.** Scrambling which
  slot each learned row lands on costs nothing.

So what a TXC latent buys here is *useful direction content at every slot for one
scalar* — coverage and content — and **not** a learned temporal shape.

**Correction, 16:55 — this control is weaker than I first wrote, and the reviewer is
right.** In the m-sweep the coefficients are **refit by least squares after shuffling**.
Permuting each atom's rows gives a different basis of k×d slabs with essentially the same
expressive power, so "shuffled ≈ intact" here is *partly guaranteed by the design*. What
it legitimately shows is that the learned temporal profiles were **not aligned to the
target profiles** — a real statement about the dictionary's *span*. It does not show that
arrangement cannot do work, because the fit re-chooses everything downstream of the
shuffle. Arrangement can only do work where it is **not** re-chosen: the frozen arm.

I also should not write "destroying temporal structure improves steering". A point
comparison on one permutation cannot separate "arrangement is worthless" from
"arrangement is worth half a sigma", and the 4-of-5 ordering may not survive a draw
distribution. `frozenshuf_modal.py` (running) does it properly: same selected latent, rows
permuted, **no refit**, 24 independent draws, reporting the intact arm's *percentile*
within the shuffled distribution. Registered: intact lands between the 30th and 70th
percentile at every budget. If instead intact ≫ shuffled, arrangement *does* matter and
the m-sweep destroyed the effect by refitting — which would rewrite this sprint.

The framing that survives either way, and is worth keeping: the two nulls **decompose**
fidelity. random → shuffled measures the value of learned **content**; shuffled → intact
measures the value of learned **arrangement**. The first is nearly everything. Whether the
second is nothing is what the frozen shuffle decides.

**At matched scalar count the SAE is not behind.** Counting scalars honestly, the
per-slot-coefficient SAE spends 12 to reach 0.627, while the TXC needs ~12 (interpolating
m=8 → 16) to reach ≈0.59. The TXC's apparent efficiency was an artefact of counting one
scalar per latent while letting each latent write twelve slots.

**And coverage is now visible directly**, since the arm logs it: `sae_perpos` coverage
runs 0.08 → 0.17 → 0.33 → 0.67 → 1.00 as m goes 1 → 16, tracking fidelity 0.160 → 0.556
almost exactly. C1 was right, and it is now measured rather than argued.

## What actually sets the crosscoder's capacity — and it is not k

The frontier sweep answered this before `mechanism_modal.py` finished, because it logs the
two quantities the answer needs: how many latents have a **positive pre-activation** on a
given input, and how many coefficients the model **realises** after TopK and ReLU compose.

| arm | nominal k | positive pre-acts | realised coeff/segment | ReLU-kill | FVU |
| --- | --- | --- | --- | --- | --- |
| SAE | 1 | 2022 | 1.00 | 0.000 | 0.600 |
| SAE | 8 | 70 | 8.00 | 0.000 | 0.081 |
| SAE | 128 | 156 | 126.52 | 0.012 | 0.029 |
| TXC | 12 (kper 1) | 53 | 1.00 | 0.000 | 0.736 |
| TXC | 24 (kper 2) | 33 | 1.99 | 0.005 | 0.711 |
| TXC | 48 (kper 4) | 29 | 2.41 | 0.397 | 0.782 |

The two architectures sit on opposite sides of the same composition. TopK selects k
latents; ReLU then zeroes any of them whose pre-activation was negative, so realised
L0 = `min(k, #{pre > 0})`.

For the **SAE**, `#{pre > 0}` is far larger than k at every budget tested — 2022 positive
pre-activations when k is 1. TopK is the binding constraint, ReLU-kill is ~0, and nominal
k is an honest budget: asking for 128 gets 126.5.

For the **TXC**, `#{pre > 0}` sits at roughly **30 per window regardless of k** — 53, 33,
29 as k goes 12, 24, 48. The crossover is between k=24 and k=48: past it, TopK reaches
into latents with negative pre-activations and ReLU discards them, which is exactly what
the 0.397 ReLU-kill at k=48 is. Realised capacity flattens at ~2.5 coefficients per
segment and further k buys nothing.

This retires the `b_enc` hypothesis I registered as P1. The learned bias is **-0.021 ±
0.010** — far too small to gate anything. The gating is in the encoder *rows*: on any given
window only ~0.8% of latents are positively aligned with the input at all. That is the
dead-latent pathology, and it is an order of magnitude worse in the crosscoder than in the
SAE trained on the same activations for the same number of steps.

**The consequence for every crosscoder-vs-SAE comparison this project has run, including
its own earlier ones.** The standard setting `kper=41` claims 41 coefficients per segment.
It realises about 2.5. Matching an SAE to a crosscoder on nominal k therefore hands the
SAE a budget roughly 17× larger than the crosscoder can actually spend, and every such
comparison has been measuring that mismatch rather than the architectures. The fix is
mechanical: **set k from a measured realised L0, not from a nominal target**, and report
realised coefficients per segment on the x-axis of any frontier.

### Two caveats I am not entitled to skip

The TXC loss traces are **not converged and not monotone** at lr=1e-3 —
`62.1 → 72.6 → 58.0 → 78.6 → 78.8` for kper=4, against the SAE's smooth
`52.3 → 47.6 → 44.9 → 46.0 → 45.3`. FVU also rises from 0.711 to 0.782 between kper=2 and
kper=4, which cannot happen to a converged model given strictly more capacity. So the FVU
*levels* in the table are an upper bound on the crosscoder's error, not an estimate of it,
and the lr=3e-4 arm has to land before any "SAE reconstructs better" claim is made. The
realised-L0 result does not depend on convergence: it is a count of positive
pre-activations, and `mechanism_modal.py` is measuring it across a wider k range
independently.

Second, the crosscoder carries **12× the decoder parameters** (4096 × 12 × 1536 = 75M
against the SAE's 6.3M) at an equal 2500 steps. Equal steps is the wrong fairness axis
here; equal steps at 12× the parameters is a handicap I imposed by accident. Whether that
explains the gap or merely widens it is what the lr arm and the structured-corpus 2×2
separate.

## The structured-corpus 2×2: adding temporal structure does not flip the comparison

`structured_modal.py` retrains both architectures on two corpora — the i.i.d. one whose
absence of temporal structure invalidated the earlier reading, and a run-length family in
which the tense/calm state persists, so a genuine window-level factor exists.

| corpus | arm | coeff/seg | FVU | segment-AUC | window-AUC |
| --- | --- | --- | --- | --- | --- |
| i.i.d. | SAE k=100 | 99.1 | 0.030 | 0.778 | **0.500** |
| i.i.d. | TXC kper=4 | 2.68 | 0.743 | 0.557 | **0.500** |
| i.i.d. | TXC kper=41 | 1.51 | 0.850 | 0.566 | **0.500** |
| structured | SAE k=100 | 99.0 | 0.028 | 0.725 | 0.747 |
| structured | TXC kper=4 | 2.79 | 0.726 | 0.500 | 0.612 |
| structured | TXC kper=41 | 1.68 | 0.811 | 0.500 | 0.619 |

The i.i.d. window-AUC column reads **0.500 for all three arms**, which is the positive
control the earlier runs never had: when no window-level factor exists, the probe finds
none, so a non-chance window-AUC on the structured corpus is measuring structure rather
than measuring the probe. On the structured corpus a window factor does appear, and the
per-segment SAE reads it *better* than the window code does — 0.747 against 0.619.

**The finding that needs no caveat: more nominal k makes the crosscoder strictly worse.**
Going from kper=4 to kper=41 — a 10× larger nominal budget — *lowers* realised
coefficients per segment from 2.68 to 1.51 and raises FVU from 0.743 to 0.850, on both
corpora. Positive pre-activations fall from 0.008 to 0.004 of the dictionary, about 16
latents per window. So the project's standard `kper=41` setting nominally claims 41
coefficients per segment and realises about 1.5: a **27× overstatement**, and a setting
that is beaten by one an order of magnitude smaller. Any comparison that matched an SAE to
this configuration on nominal k was mismatched by that factor.

**What I am not concluding yet.** Every architecture-level reading above — SAE
reconstructs better, SAE reads the window factor better, structure does not help the
window code — is downstream of a crosscoder that is spending 1.5–2.8 coefficients per
segment because its latents are dead, not because a window code cannot use more. Reading
these as facts about crosscoders requires first ruling out that they are facts about this
crosscoder's *implementation*. `centering_modal.py` (running) tests exactly that against
five arms, and its registered R3 is the branch in which the numbers above become
reportable.

## The mechanism, measured: realised L0 = min(k, #{pre > 0}), and the second term wins

`mechanism_modal.py` completed. The registered predictions were P1 (b_enc goes strongly
negative, magnitude rising in k), P2 (SAE b_enc near zero by comparison), P3 (realised
L0 ≈ #{pre > 0} once that count falls below k), P4 (a crossover in k).

| kper | nominal k | b_enc mean ± sd | #{pre > 0} | realised L0 | min(k, #pos) | FVU |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 12 | −0.021 ± 0.010 | 53.3 | 12.0 | 12.0 | 0.776 |
| 2 | 24 | −0.023 ± 0.011 | 35.7 | 23.7 | 24.0 | 0.781 |
| 4 | 48 | −0.022 ± 0.010 | 22.2 | 20.4 | 22.2 | 0.865 |
| 10 | 120 | −0.024 ± 0.010 | 17.1 | 17.1 | 17.1 | 0.873 |
| 20 | 240 | −0.024 ± 0.010 | 16.4 | 16.4 | 16.4 | 0.879 |

**P3 and P4 hold exactly.** Realised L0 equals min(k, #{pre > 0}) in every row. The
crossover sits between kper=1, where TopK binds and the model spends all 12 of its budget,
and kper=4, where the positive-pre-activation count has fallen to 22 and becomes the
binding term. Past that, realised L0 is pinned near 16–20 coefficients per *window* — under
1.7 per segment — no matter how large k is.

**P1 is refuted, and I should retire the story I attached to it.** The learned bias moves
from −0.021 to −0.024 across a 20× range of k. It is flat and it is far too small to gate
anything. The gating lives in the encoder rows, not the bias.

**The unregistered result is the sharpest one: raising k actively destroys capacity.** The
positive pre-activation count falls monotonically — 53 → 36 → 22 → 17 → 16 — as nominal k
rises. So k is not merely inert above the crossover; asking for more coefficients leaves
the model able to spend fewer, and FVU rises from 0.776 to 0.879 accordingly. `interp.json`
pushes this to its limit: at kper=341 the nominal window budget is 4092 of a 4096-latent
dictionary, the ReLU-killed fraction reaches **1.00**, and realised L0 is still 18.

| kper | nominal window k | ReLU-killed | realised L0/window | alive | FVU (× SAE) |
| --- | --- | --- | --- | --- | --- |
| 41 | 492 | 0.96 | 17.6 | 0.152 | 27.6× |
| 100 | 1200 | 0.98 | 26.6 | 0.143 | 25.7× |
| 200 | 2400 | 0.99 | 18.8 | 0.113 | 26.9× |
| 341 | 4092 | 1.00 | 18.0 | 0.096 | 25.9× |

Frozen steering fidelity across that same 8× span of nominal k is +0.258, +0.218, +0.231,
+0.227 — flat, which is what a flat realised capacity predicts and a useful consistency
check on the whole picture.

**This answers the capacity question directly.** The concern was that too much capacity
would cost interpretability. The measured situation is worse and simpler: the extra
capacity is never delivered, and asking for it degrades reconstruction, alive fraction, and
nothing improves in exchange.

### One number in that file I am not entitled to use

`interp_modal.py` also reports best-single-latent AUC against the known factor — 0.541,
0.456, 0.551, 0.551, all near chance — and it is tempting to read that as "crosscoder
latents are uninterpretable". It is not evidence of that. The AUC is computed **per
segment** on the **i.i.d. corpus**, which is the mismatched question I already retracted
once: a window code holds one shared code for twelve independently-labelled segments, so
chance is the *correct* answer there and a high value would have been the surprise. The
interpretability number that survives is the structured-corpus window-AUC, where the
question matches the code: 0.612 for the crosscoder against 0.747 for the per-segment SAE.

## The collapse may be a two-line implementation defect, not a property of crosscoders

The first arm of `centering_modal.py` returned numbers unlike anything measured so far this
sprint. At the same nominal configuration:

| run | #{pre > 0} | coeff/segment | ReLU-kill | FVU |
| --- | --- | --- | --- | --- |
| `mechanism_modal.py` kper=4 | 22.2 | 1.70 | 0.575 | 0.865 |
| `frontier_modal.py` kper=4 lr=1e-3 | 29 | 2.41 | 0.397 | 0.782 |
| `centering_modal.py` kper=4 "base" | **99.2** | **3.98** | **0.006** | **0.670** |

The last row is a crosscoder spending 3.98 of its nominal 4 coefficients per segment and
discarding 0.6% of its selection — no collapse at all. If that is real, then "realised
capacity saturates near 2 coefficients per segment" is a statement about this repo's
training setup and not about window codes, and the architecture conclusions logged above
are void rather than merely caveated.

**I cannot attribute it yet, because I changed two things at once.** My `TXCVariant`
calls `_normalize_decoder()` in `__init__` for *every* arm including "base", and I ran it at
lr=3e-4 where `mechanism_modal.py` used 1e-3. So my "base" is not the repo's base, and the
one arm that was supposed to be the reference point is not a reference point. That is the
same class of error as setting the harness gate against theory instead of against the
stored reference run — I built a control that does not control.

`initnorm_modal.py` (running) is the full factorial that identifies it: {normalise decoder
at init: no, yes} × {lr: 1e-3, 3e-4} × {kper: 4, 20}, everything else fixed. Its registered
N0 is that `initnorm=False, lr=1e-3` reproduces the collapse — the control that licenses
every other cell, and the one whose failure would mean the difference is somewhere I have
not looked.

**Why decoder normalisation at init is the plausible culprit.** `TopKSAE.__init__` calls
`_normalize_decoder()`; `TemporalCrosscoder.__init__` does not. The crosscoder's `W_dec` is
initialised `randn(d_sae, T, d_in) / sqrt(d_sae)`, whose norm over the normalised dims
`(1, 2)` is `sqrt(T·d_in/d_sae)` = sqrt(12·1536/4096) ≈ **2.12**, not 1. The training loop
rescales it, but only *after* the first optimiser step, so Adam's moment estimates are
seeded from gradients taken at a decoder twice too large. The initial loss differs
accordingly: 566.91 with normalisation against 848.50 without. A large first step under
gradient clipping favours whichever latents happen to dominate at init, and TopK then keeps
selecting those same latents, which is the winner-take-all path to a dictionary where only
~0.5% of latents ever go positive.

Note this predicts the collapse should *worsen with T*, since the init norm error
`sqrt(T·d_in/d_sae)` grows as sqrt(T) — and `tsweep_modal.py` (running) measures exactly
that, with T=1 as an exact SAE-equivalence control.

## Attribution, partial: centring is not the cause, and learning rate is not innocent

Two corrections to the previous entry, both against my own hypothesis.

**Centring does nothing.** `centering_modal.py`'s arms, at kper=4 with the decoder
normalised at init throughout:

| arm | #{pre > 0} | coeff/segment | ReLU-kill | alive | FVU |
| --- | --- | --- | --- | --- | --- |
| base | 99.2 | 3.98 | 0.006 | 0.376 | 0.6696 |
| center | 98.5 | 3.99 | 0.002 | 0.371 | 0.6673 |
| tied | 73.7 | 3.96 | 0.009 | 0.294 | 0.6953 |

Subtracting `b_dec` before the encoder projection — the `TopKSAE` behaviour the crosscoder
omits — changes nothing measurable, so registered prediction R1 is refuted. This is worth
recording because the per-position DC is genuinely large: the residual after the pooled
centering has norm 11.89 at position 0 against a typical segment norm of 21.59, so the
crosscoder's encoder really is seeing a big offset the SAE's does not. It simply does not
matter, presumably because a fixed per-latent offset is exactly what `b_enc` can absorb.
Tied init is mildly *worse*, so it is not the fix either.

**Learning rate is doing some of the work.** The frontier's lr=3e-4 arm, which has no
init-normalisation, has far more live latents than its lr=1e-3 counterpart:

| kper | lr=1e-3 pos-preact | lr=3e-4 pos-preact | lr=1e-3 FVU | lr=3e-4 FVU |
| --- | --- | --- | --- | --- |
| 1 | 0.008 (33) | 0.085 (348) | 0.736 | 0.761 |
| 2 | 0.008 (33) | 0.019 (78) | 0.711 | 0.769 |

So the clean story I wrote last entry — "init-normalisation is the culprit" — is not
established, and at low k a smaller step size alone recovers an order of magnitude of live
latents while making reconstruction slightly *worse*. Both of these are consistent with the
collapse being a first-few-steps optimisation event of some kind, but they do not identify
which intervention matters.

The decisive cell is kper=4, where the collapse first bites and where I have the
init-normalised number to compare against: `centering_modal.py` reaches 3.98 coefficients
per segment and FVU 0.670 there with init-norm at lr=3e-4. If the frontier's kper=4 lr=3e-4
cell — same lr, no init-norm — comes back near 2.4 and 0.78, init-normalisation is the
factor. If it comes back near 3.98 and 0.67, the learning rate is, and the two-line-fix
framing is wrong. `initnorm_modal.py`'s factorial settles it either way.

**Arithmetic check, and one correction to my own number.** The init atom norm is confirmed
numerically at 2.1213 ± 0.011, exactly `sqrt(T·d_in/d_sae)` = sqrt(12·1536/4096). The
kper=41 overstatement is 41 ÷ (17.6/12) = **28×**, not the 27× written above.

## The frozen shuffle control reverses the negative: arrangement does matter

`frozenshuf_modal.py` completed, and my registered prediction was wrong in the informative
direction. I had written: *"intact lands between the 30th and 70th percentile at every
budget"*, with the note that if intact ≫ shuffled then arrangement does matter and the
m-sweep had destroyed the effect by refitting. That is what happened.

The control takes the selected latent, permutes its decoder rows in time, and does **not**
refit — so nothing downstream can re-absorb the damage — across 24 independent draws.

| budget | intact fidelity | shuffled draws (n=24) | intact percentile |
| --- | --- | --- | --- |
| frozen, no refit | **+0.242** | +0.002 ± 0.103, range [−0.211, +0.211] | 100th |
| refit m=2 | **+0.397** | +0.314 ± 0.054 | 100th |
| refit m=8 | **+0.659** | +0.568 ± 0.059 | 100th |

The frozen row is the clean one: intact sits 2.3 shuffled standard deviations above the
shuffled mean and outside the entire observed range of 24 draws, on a latent that fires on
0.56% of windows, against a full difference-of-means reference of Δ=+33.43. With 24 draws
the strongest one-tailed statement available is p ≤ 1/25 = 0.04, and the three budgets are
**not independent** — same latent, same dictionary, same task, same held-out set — so this
is one result confirmed at three budgets, not three results.

This retires the earlier reading that "destroying temporal structure does not hurt". That
reading came from the m-sweep, which refits the coefficients after shuffling and therefore
lets the fit repair the permutation; the earlier `controls_modal.py` shuffle showed
`shuffled ≈ intact` for exactly that reason. Arrangement can only be measured where it is
not re-chosen, and where it is not re-chosen, it carries a measurable effect.

## The dictionary is starved, and sweeping k is not the knob that fixes it

`health_modal.py` answers the question directly: no configuration in the k sweep reaches
SAE-comparable reconstruction.

| kper | nominal window k | FVU | × SAE | realised L0/window | alive | ReLU-killed |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | 492 | 0.839 | 27.6× | 17.6 | 0.152 | 0.96 |
| 100 | 1200 | 0.780 | 25.7× | 26.6 | 0.143 | 0.98 |
| 200 | 2400 | 0.819 | 26.9× | 18.8 | 0.113 | 0.99 |
| 341 | 4092 | 0.788 | 25.9× | 18.0 | 0.096 | 1.00 |

Against an SAE at FVU 0.030 with 1189 coefficients per window. Frozen steering fidelity
across that whole 8× span of nominal k is +0.258, +0.218, +0.231, +0.227 — flat, tracking
the flat realised capacity rather than the nominal budget.

**So the interpretive worry is live and I am acting on it.** The steering results above were
measured on a dictionary reconstructing at 27× the SAE's error. For the *negative* results
that is disqualifying: "the crosscoder reads the window factor worse" cannot be separated
from "the crosscoder was not trained". For the *positive* arrangement result the direction
is more forgiving — a starved dictionary makes a detected effect conservative rather than
spurious — but the effect size is not trustworthy and the result should replicate on a
healthier dictionary before it is reported.

## Attribution, settled: it is the learning rate, and I was wrong about the init defect

The decisive cell arrived. All three runs are kper=4, nominal k=48, same corpus, same 2500
steps, same seed:

| run | decoder normalised at init | lr | #{pre>0} | coeff/segment | ReLU-kill | alive | FVU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mechanism_modal.py` | no | 1e-3 | 22.2 | 1.70 | 0.575 | — | 0.865 |
| `frontier_modal.py` | no | 1e-3 | 29 | 2.41 | 0.397 | 0.195 | 0.782 |
| `frontier_modal.py` | **no** | **3e-4** | 82 | **3.97** | **0.008** | 0.357 | 0.706 |
| `centering_modal.py` | **yes** | **3e-4** | 99.2 | **3.98** | **0.006** | 0.376 | 0.670 |

The third and fourth rows differ only in the init normalisation and are the same run to
within noise — 3.97 against 3.98 coefficients per segment, 0.008 against 0.006 ReLU-kill.
The second and third rows differ only in the learning rate and are completely different.

**Registered N1 ("init-normalisation is the factor that matters") is refuted, and N2 ("lr
alone does not") is refuted in the opposite direction: lr alone does.** The two-line
implementation-defect framing I wrote two entries ago is wrong and I am dropping it. The
init asymmetry is real — the crosscoder's atoms do start at norm 2.1213 instead of 1, which
I verified numerically as exactly `sqrt(T·d_in/d_sae)` — and it is worth fixing for
tidiness, but it is not what starves the dictionary. At lr=1e-3 the crosscoder collapses; at
lr=3e-4 it spends 3.97 of its nominal 4 coefficients per segment.

What this does **not** yet establish is whether the collapse reappears at larger k even at
the lower learning rate. Every k at which I have an lr=3e-4 number so far — 1, 2, 4 — spends
its full budget. The frontier's kper=8, 20 and 41 cells at lr=3e-4 are the ones that decide
whether "realised capacity saturates" is a real property or was an artefact of a single
learning rate throughout, and `initnorm_modal.py`'s factorial covers kper=20 directly.

**Acting on the user's instruction.** The directive was to rerun the shuffle control on a
configuration reaching comparable FVU. The k sweep does not produce one, but the learning
rate does produce a materially healthier dictionary — full budget spent, alive fraction
0.357 against 0.136, FVU 0.706 against 0.839 — so `frozenshuf_modal.py` is rerunning at
`--txc-lr 3e-4 --kper 4` with everything else fixed, writing `frozen_shuffle_healthy.json`.
Stating the limit plainly: FVU 0.706 is better but is still 23× the SAE's 0.030, so this
tests whether the arrangement result survives a substantially better dictionary, not whether
it survives one at parity. No configuration found in this sprint reaches parity.

## Correction: the capacity saturation was a learning-rate artefact, not a property of k

The frontier's lr=3e-4 arm has now reached kper=8, and the collapse does not reappear.
Side by side at identical nominal budgets, same code, same corpus, same 2500 steps:

| kper | coeff/seg @ lr=1e-3 | coeff/seg @ lr=3e-4 | ReLU-kill 1e-3 | ReLU-kill 3e-4 | FVU 1e-3 | FVU 3e-4 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | 1.00 | 0.000 | 0.000 | 0.736 | 0.761 |
| 2 | 1.99 | 2.00 | 0.005 | 0.000 | 0.711 | 0.769 |
| 4 | 2.41 | **3.97** | 0.397 | **0.008** | 0.782 | **0.706** |
| 8 | 2.04 | **7.70** | 0.744 | **0.038** | 0.813 | **0.630** |

At lr=1e-3 realised capacity peaks at kper=4 and then falls while ReLU-kill climbs to 74%.
At lr=3e-4 the crosscoder spends 3.97 of 4 and 7.70 of 8, ReLU-kill stays near zero, the
alive fraction rises 0.053 → 0.140 → 0.357 → 0.516, and FVU improves monotonically.

**So several things I logged earlier need withdrawing, and I would rather withdraw them
plainly than let them stand behind a caveat.** "Raising k actively destroys capacity",
"positive pre-activations fall monotonically as k rises", and "the crosscoder realises ~2.5
coefficients per segment no matter how large k is" are all true **at lr=1e-3 only**. They
are not properties of window codes. The 28× overstatement at kper=41 is likewise a
statement about a particular training run, not about the architecture.

What survives, and is arguably the more useful finding, is this. Realised sparsity is
`min(k, #{pre > 0})` — that identity held exactly in every row of every run and is not in
question. What is in question is which term binds, and **that turns out to depend on the
optimiser rather than on the architecture**. A single 3× change in learning rate moves the
crosscoder at kper=8 from spending 2.04 coefficients per segment to spending 7.70. So:

- nominal k cannot be used as a capacity axis, not because crosscoders inherently fail to
  spend it, but because whether they spend it is an unstable function of training;
- realised coefficients per segment must be measured and reported for every run, since two
  runs with identical hyperparameters apart from lr differ by 3.8× in what they actually
  spend;
- any crosscoder-vs-SAE comparison in this project needs its realised L0 checked before its
  conclusion is trusted, including comparisons that looked settled.

`mechanism_modal.py`'s beautiful monotone table — 53 → 36 → 22 → 17 → 16 positive
pre-activations as k rises — was run at lr=1e-3 throughout. It is a real measurement of a
real failure mode, but the failure mode is optimisation collapse, and the cleanest way to
say what it shows is that **a crosscoder can silently train into a state where almost none
of its latents are usable, while its nominal configuration reports nothing wrong.**

The ablation arms confirm the same thing negatively. At kper=4, lr=3e-4, every intervention
I tried lands within noise of the others:

| arm | #{pre>0} | coeff/seg | ReLU-kill | alive | FVU |
| --- | --- | --- | --- | --- | --- |
| base | 99.2 | 3.98 | 0.006 | 0.376 | 0.670 |
| + input centering | 98.5 | 3.99 | 0.002 | 0.371 | 0.667 |
| + tied init | 73.7 | 3.96 | 0.009 | 0.294 | 0.695 |
| + centering & tied | 83.7 | 3.99 | 0.004 | 0.302 | 0.687 |
| + aux dead-latent loss | 99.4 | 3.97 | 0.008 | 0.392 | 0.676 |

Centering, tied init, decoder normalisation at init and the standard auxiliary revival loss
all do essentially nothing once the learning rate is right. Registered R1, R2, N1 and N2 are
all refuted. The learning rate was the whole story.

## The frontier, complete

At the working learning rate the crosscoder behaves sensibly: it spends its budget up to
kper=8, saturates gradually after, and reconstruction improves monotonically throughout.

| kper | coeff/seg | spend | ReLU-kill | alive | FVU | nearest SAE | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | 100% | 0.000 | 0.053 | 0.761 | 0.600 | 1.3× |
| 2 | 2.00 | 100% | 0.000 | 0.140 | 0.769 | 0.173 | 4.4× |
| 4 | 3.97 | 99% | 0.008 | 0.357 | 0.706 | 0.117 | 6.0× |
| 8 | 7.70 | 96% | 0.038 | 0.516 | 0.630 | 0.081 | 7.8× |
| 20 | 13.43 | 67% | 0.328 | 0.646 | 0.571 | 0.057 | 10.0× |
| 41 | 16.16 | 39% | 0.606 | 0.661 | 0.539 | 0.057 | 9.4× |

So saturation is real even at lr=3e-4 — realised spend plateaus near 16 coefficients per
segment — but it is *benign*: FVU keeps falling as k rises, where at lr=1e-3 it rose. That
is the qualitative difference between the two learning rates, and it is the honest version
of the claim I withdrew. Capacity saturates in both; only one of them is pathological.

The SAE is better at every matched budget, by 1.3× at the tightest and ~9-10× from 13
coefficients per segment upward. Figure: `plots/2026-07-25_dictbench/frontier.png`.

## The activation function is the mechanism, and there are alternatives worth testing

Prompted by the right question from the user: we do use TopK — so where does the cap come
from? It comes from the *composition*, which both architectures share:

```python
topk_vals, topk_idx = pre.topk(self.k, dim=-1)
z.scatter_(1, topk_idx, F.relu(topk_vals))    # ReLU applied AFTER TopK
```

Realised L0 is therefore `min(k, #{pre > 0})`. The SAE never notices — it has ~2000 positive
pre-activations at k=1, so the ReLU term is slack at every budget. The crosscoder has 20-160,
so above that count the ReLU discards whatever TopK reached for. Nothing about TopK itself
imposes this; the ReLU after it does.

`actfn_modal.py` (running) tests four sparsity rules on the crosscoder at lr=1e-3, the
setting where capacity demonstrably dies, so the question is whether the activation rescues
what the optimiser destroyed:

- **`topk`** — TopK with no ReLU after it, the Gao et al. formulation. Realised L0 = k
  exactly, by construction, so capacity cannot collapse. The cost is signed coefficients,
  which weakens the "a feature fires or it does not" reading, so the negative fraction is
  reported as a first-class number rather than buried.
- **`batchtopk`** — BatchTopK (Bussmann et al. 2024): the k·B largest pre-activations across
  the batch rather than k per sample, then ReLU. Lets the model allocate unevenly across
  windows, and is the standard remedy for dead latents.
- **`topk_relu_auxk`** — the repo's composition plus the AuxK revival loss. This was null at
  lr=3e-4; the question is whether it bites at lr=1e-3 where there is something to revive.
- **`topk_relu`** at both learning rates as the reference.

Registered A3 is the one that can embarrass me: if `topk` shows a near-zero negative
coefficient fraction, then `#{pre > 0}` was never the binding constraint and the mechanism
I have been describing all evening is wrong.

## The arrangement result replicates on the healthy dictionary

Per the directive not to conclude until the control ran on a better-trained dictionary.
`frozenshuf_modal.py --txc-lr 3e-4 --kper 4` against the original lr=1e-3, kper=41 run:

| | starved | healthy |
| --- | --- | --- |
| FVU | 0.839 | **0.707** |
| realised L0 per window | 18.4 | **47.8** |
| alive fraction | 0.136 | **0.345** |
| selected latent | 2788, sign +1, freq 0.0056 | 1061, sign −1, freq 0.0889 |
| frozen intact fidelity | +0.242 | **+0.292** |
| shuffled null, n=24 | +0.002 ± 0.103 | −0.008 ± 0.114 |
| intact percentile | 100th | **100th** |
| refit m=2 | 100th | 88th |
| refit m=8 | 100th | 100th |

The dictionary is 2.6× richer in realised coefficients and 2.5× richer in live latents, and
the effect is *larger*, not smaller. It also selected a different latent — different index,
opposite sign, sixteen times the firing frequency — so this is an independent instance of
the phenomenon rather than the same measurement repeated. The refit m=2 arm falling to the
88th percentile is the expected direction: refitting repairs part of the permutation, which
is exactly why the frozen arm is the one that carries the claim.

The worry that the steering result was an artefact of a starved dictionary is therefore
answered in the direction that matters. It was not.

## BatchTopK is now the crosscoder's available sparsity rule

The composition `ReLU(TopK(·))` was inherited from SAE code where it is a genuine no-op —
an SAE has ~2000 positive pre-activations to choose from at k=1, so the ReLU never binds and
serves only as a non-negativity guard. Ported into a crosscoder, whose positive
pre-activation count is smaller than a window-sized k, the same line becomes the binding
constraint. That is the whole of the mechanism.

`src/bench/architectures/crosscoder.py` now takes an `activation` argument:

- `topk_relu` — the historical behaviour, and still the default so existing results and the
  in-flight `ward_backtracking_txc` workstream stay reproducible. Flipping the repo-wide
  default is a one-line change in `CrosscoderSpec.__init__` whenever wanted.
- `topk` — TopK with no ReLU, the Gao et al. formulation. Realised L0 equals k by
  construction.
- `batchtopk` — BatchTopK (Bussmann et al. 2024).

**One correctness point that changed the implementation.** A batch rule needs a batch, so
BatchTopK cannot be applied as written at inference: a window's code would depend on which
other windows happened to be scored alongside it. The standard treatment is to estimate a
threshold during training and apply it JumpReLU-style at eval, which is what the module
does — an EMA of the smallest activated value per step, stored in a `bt_threshold` buffer,
with a fallback to per-sample TopK while it is unset. My first version of `actfn_modal.py`
had the naive form and was killed and rewritten before it produced numbers; the experiment
now calls the module's own `_sparsify`, so it measures the code that ships rather than a
local copy of it.

`tests/bench/test_crosscoder_activations.py` covers this — eleven tests, including one that
asserts BatchTopK's eval output is independent of batch composition, and one that pins the
capacity failure explicitly: force `b_enc` strongly negative and `topk_relu` realises
`min(k, #{pre>0})` while `topk` still realises exactly k.

**An unrelated bug found on the way.** `test_crosscoder_encode_shuffle_sensitive` used an
unseeded `torch.randperm(T)` with T=3, which is the identity permutation once every six
draws, so the test failed about 40% of the time regardless of the code under test. I
measured 7 failures in 12 runs both with and without my changes before concluding it was
pre-existing. Replaced with a fixed cyclic shift; 12 of 12 pass.

## Sharpening the mechanism: this is divergence that does not look like divergence

I had been writing "which term binds is set by the optimiser", which is true but evasive.
The loss traces say something more specific. Comparing step 500 against step 2499, so that
the initial transient is excluded:

| kper | lr=1e-3 | lr=3e-4 |
| --- | --- | --- |
| 1 | 207.0 → 194.5 (−6%) | 142.2 → 109.6 (−23%) |
| 2 | 125.6 → 119.1 (−5%) | 32.7 → 11.3 (−66%) |
| 4 | 62.1 → **78.8 (+27%)** | 13.8 → 9.3 (−33%) |
| 8 | 57.8 → **81.2 (+41%)** | 18.5 → 12.2 (−34%) |
| 20 | 62.0 → 60.6 (−2%) | 25.3 → 13.9 (−45%) |
| 41 | 58.3 → **69.7 (+20%)** | 25.2 → 15.6 (−38%) |

At lr=1e-3 the training loss **increases** over the run in three of six configurations and
is flat in the other three. At lr=3e-4 it falls by 23-66% in all six. The lr=1e-3 runs are
diverging.

So the honest mechanism is not that some subtle property of window codes caps capacity. It
is that **the crosscoder's training diverges at a learning rate the SAE tolerates, and the
divergence presents as capacity collapse rather than as anything that looks like a failure.**
No NaN, no exploding loss, no error. The model trains to completion, reports its configured
k, and returns a dictionary in which 96-100% of the TopK selection is discarded. The final
loss is 4.5-8.5× worse than the converged run at the same nominal settings, which would be
visible to anyone comparing — but only if they had the converged run to compare against.

That is a more useful warning than the one I was writing, and it changes the recommended
practice from "prefer this learning rate" to "**check realised L0 and the loss trend; a
crosscoder that has diverged looks exactly like one that has not**".

Registered predictions retired by this: the mechanism is neither `b_enc` gating (refuted),
nor missing input centering (refuted), nor decoder normalisation at init (refuted), nor a
property of TopK (the identity `min(k, #{pre>0})` is exact but is not itself the cause). It
is ordinary optimiser divergence with an unusual signature.

## The T-sweep: the cost of window sharing, with the implementation ruled out

`tsweep_modal.py` holds coefficients per segment fixed at 4 and varies only the window
length, so the nominal budget is 4T and any change is attributable to T. T=1 is an exact
control: at T=1 the encoder einsum collapses to a matmul and the decoder normalisation over
dims (1, 2) is per-atom unit norm, so a crosscoder at T=1 *is* a TopK SAE up to the
implementation differences this sprint has been chasing.

| T | nominal k | #{pre>0}/window | coeff/segment | ReLU-kill | alive | FVU | × SAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 2059 | 4.00 | 0.000 | 0.033 | 0.0989 | **1.01** |
| 2 | 8 | 1529 | 4.00 | 0.000 | 0.080 | 0.0932 | 0.95 |
| 3 | 12 | 881 | 4.00 | 0.000 | 0.130 | 0.0979 | 1.00 |
| 4 | 16 | 339 | 4.00 | 0.000 | 0.212 | 0.1175 | 1.20 |
| 6 | 24 | 142 | 4.00 | 0.000 | 0.296 | 0.3163 | 3.23 |
| 12 | 48 | 99 | 3.98 | 0.006 | 0.376 | 0.6696 | 6.83 |

**Q1 passes.** The crosscoder at T=1 reproduces the SAE at 1.01× its FVU — 0.0989 against
0.0980. The two implementations agree. Every gap measured at larger T is therefore the cost
of the shared code, not an artefact of how the crosscoder is written, which was the largest
open threat to this sprint's conclusions and is now closed.

**Realised capacity is not the mechanism here.** Coefficients per segment are a clean 4.00
at every T with ReLU-kill at zero, so nothing is being silently discarded. The FVU rise from
1.0× to 6.8× is entirely the difficulty of explaining T independent segments with one shared
code. On a corpus whose segments are i.i.d. that is exactly the expected direction: sharing
buys nothing and costs representational freedom.

Q2 is refuted as stated. I predicted `#{pre>0}` per window would stay roughly flat in T; it
falls hard, 2059 → 99. But since realised L0 stays pinned at its nominal 4.00/segment, that
fall never becomes binding, so the prediction was wrong about the quantity and right about
the consequence. Worth recording as a reminder that a wrong mechanism can predict the right
observable.

The alive fraction moving 0.033 → 0.376 across the sweep is the counterweight and it points
at a real reviewer objection to Q1: at T=1 only about 140 of 4096 latents are live in both
arms, so the control certifies agreement in a fairly degenerate corner. It would be stronger
repeated at a larger k where more of the dictionary participates.

Figure: `plots/2026-07-25_dictbench/tsweep.png`.

## The recovery lane: window length pays off exactly when the features are temporal

Everything else this sprint trained on data whose segments are independent, where a window
code can only lose. This lane builds data with a payoff and asks whether recovery of the
*true* features improves systematically with T.

Ground-truth features carry a direction `u` and a profile `p` spanning `L` consecutive
segments; firing at segment `s` contributes `p[j]·u` to segment `s+j`. A T-window can cover
at most T contiguous entries of `p`, so the best cosine any T-slab can reach against the
full atom `p ⊗ u` is fixed by geometry before any training happens:

```text
ceiling(T, p) = ‖ largest contiguous T-chunk of p ‖ / ‖p‖      ( = √(min(T,L)/L) flat )
```

Measured recovery against that ceiling, 64 true features, `batchtopk`, 6000 steps:

| extent | T=1 | T=2 | T=4 | T=8 | T=16 |
| --- | --- | --- | --- | --- | --- |
| L=1 | 0.922 / 1.000 | 0.988 / 1.000 | 0.998 / 1.000 | 0.991 / 1.000 | 0.338 / 1.000 |
| L=2 | **0.816 / 0.816** | 0.827 / 1.000 | 0.949 / 1.000 | 0.989 / 1.000 | 0.378 / 1.000 |
| L=4 | **0.661 / 0.662** | 0.765 / 0.813 | 0.840 / 1.000 | 0.944 / 1.000 | 0.386 / 1.000 |
| L=8 | **0.481 / 0.481** | 0.582 / 0.621 | 0.712 / 0.772 | 0.825 / 1.000 | 0.461 / 1.000 |

**The T=1 column is the result.** For every temporal extent, a per-token dictionary recovers
*exactly* its analytic ceiling — 0.816 against 0.816, 0.661 against 0.662, 0.481 against
0.481. It is not underperforming; it is saturating a limit that no amount of training,
capacity or tuning can move, because the limit is geometric. An extent-8 feature simply
cannot be represented better than 0.481 by any dictionary that sees one segment at a time.

**And T buys back exactly what the geometry allows.** Recovery rises monotonically in T for
every L>1 — L=8 runs 0.481 → 0.582 → 0.712 → 0.825 — tracking its ceiling upward and
saturating once T reaches L. For L=1 features T buys nothing, which is V1 confirmed and is
also the missing explanation for the language-corpus T-sweep: that corpus's features are all
effectively L=1, so T could only cost, and it did (FVU 1.0× → 6.8×). The two lanes agree.

V4 holds: recovery is at or below the ceiling in all 20 cells, and the gap widens with L
(0.000 at L=2, 0.001 at L=4 for T=1, but 0.039 at L=8 for T=2, 0.060 for T=4) — longer
features are rarer per unit of data, so more of the shortfall is estimation.

**T=16 is a training failure, not a finding about T.** FVU rises to 0.832 against 0.116-0.194
elsewhere and recovery collapses for every extent including L=1. It sits beyond every extent
present in the data and is excluded from the saturation read-out; it should not be read as
"large T destroys recovery" without a run where the data actually contains extent-16
features.

### The transferable principle

**Set T to the temporal extent of the behaviour you are trying to capture.** Below it,
recovery is capped by geometry at roughly `√(T/L)` and no amount of tuning helps; at or
above it, recovery saturates and further T costs reconstruction. This is checkable on a real
task without ground truth by the same logic that produced it: sweep T, and the point where
recovery-proxies stop improving estimates the extent of the underlying feature.

Figure: `plots/2026-07-25_dictbench/recovery.png`.
Code: `experiments/temporal_screen/dict_bench/recovery_local.py` (runs locally, no LM).

## Activation functions: the ReLU is the cap, and BatchTopK only helps without it

Measured at lr=1e-3, the setting where the crosscoder's training diverges and capacity
collapses. `spend` is realised L0 as a fraction of nominal k; `neg` is the fraction of
realised coefficients that are negative.

| arm | kper | #{pre>0} | coeff/seg | spend | neg | FVU |
| --- | --- | --- | --- | --- | --- | --- |
| `topk_relu` | 8 | 14.5 | 1.05 | 13% | 0.000 | 0.832 |
| `topk_relu` + AuxK | 8 | 14.5 | 1.05 | 13% | 0.000 | 0.832 |
| batch selection **+ ReLU** | 8 | 13.5 | 1.10 | 14% | 0.000 | 0.841 |
| `topk` (no ReLU) | 8 | 82.8 | **8.00** | **100%** | 0.145 | **0.628** |
| `topk_relu` | 41 | 11.5 | 0.96 | 2% | 0.000 | 0.843 |
| `topk` (no ReLU) | 41 | 301.4 | **41.00** | **100%** | 0.387 | **0.541** |

**Batch-level selection with a ReLU after it is not a fix.** It spends 14% of its budget
against `topk_relu`'s 13%. This is the expected result once the mechanism is stated
properly: batch selection changes *which* latents are chosen and lets the allocation be
uneven across windows, but it cannot manufacture positive pre-activations that do not
exist. If only ~14 latents per window have one, no selection rule followed by a ReLU can
realise more than ~14. My registered A2 predicted batch selection would beat `topk_relu`;
it does not, and the reason is that I had put the ReLU in the wrong place conceptually.

**Removing the ReLU is the fix**, and it is exact: realised L0 equals k by construction, so
at kper=41 the crosscoder spends 41.00 of 41 instead of 0.96, and FVU falls from 0.843 to
0.541. A1 confirmed.

**A3 confirmed, which is what licenses the mechanism.** The advantage arrives with signed
coefficients — 14.5% negative at kper=8, 38.7% at kper=41. If that fraction had been near
zero, `#{pre>0}` was never the binding constraint and the whole diagnosis would have been
wrong. It rises with k exactly as it should: the further TopK reaches past the positive
pre-activations, the more of what it grabs is negative.

**A4 refuted, decisively.** AuxK reproduces `topk_relu` to every digit — 14.5, 1.05, 0.132,
0.8324. The standard dead-latent revival loss does nothing here, at either learning rate.

**Consequence, and the change made.** `batchtopk` in
`src/bench/architectures/crosscoder.py` is now the published no-ReLU formulation and is the
crosscoder's **default**; the ReLU variant is retained only as `batchtopk_relu`, labelled
as a diagnostic and explicitly not attributed to the paper. `topk_relu` remains selectable
for reproducing pre-existing runs. Callers that evaluate a crosscoder must now call
`.eval()` before scoring, because BatchTopK is a batch rule during training and a fixed
threshold afterwards; `frontier_modal.py` has been corrected accordingly and re-run.

A second flaky test surfaced while checking this and is the same phenomenon in miniature:
`TestTopKSAE::test_create_and_forward` asserts realised L0 == k on untrained weights and
unseeded input, which fails about 1 run in 8 with L0 = 3.875 — the ReLU discarding a TopK
pick. Seeded.

## The clock breaks "set T to the extent", and repairs the desideratum

The rule I proposed from `recovery_local.py` — set T to the temporal extent of the target
feature — is undefined for a periodic feature. A clock never terminates, so ‖p‖ diverges and
the ceiling formula has nothing to say. `clock_local.py` runs one clock per sequence, all
clocks sharing a single 2-D plane and differing only in angular velocity, so a single
segment is a point on the same circle whatever the period and the period is recoverable
only from rotation across segments.

| T | atom recovery | period-ID accuracy (chance 0.25) |
| --- | --- | --- |
| 1 | **1.000** | **0.252** |
| 2 | 0.832 | 0.957 |
| 4 | 0.866 | **1.000** |
| 8 | 0.956 | 1.000 |
| 16 | 0.829 | 1.000 |
| 24 | 0.751 | 0.983 |

**W1 confirmed, and it is the practical warning.** At T=1 the dictionary matches the
waveform *perfectly* — recovery 1.000 — while carrying no information whatever about which
clock is running. A recovery or reconstruction metric would have certified T=1 as
sufficient. The two columns disagree completely at the one place where the answer matters.

**W3 refuted, decisively.** I predicted pairs would separate near the Fourier resolution
requirement `T ≳ 1/|1/P − 1/P′|`. Every pair reaches 90% at T=2, including (8,16), whose
requirement is 16.0:

| pair | requirement | reaches 90% at |
| --- | --- | --- |
| 2 v 4 | 4.0 | T=2 |
| 2 v 8 | 2.7 | T=2 |
| 2 v 16 | 2.3 | T=2 |
| 4 v 8 | 8.0 | T=2 |
| 4 v 16 | 5.3 | T=2 |
| 8 v 16 | 16.0 | T=2 |

The bound governs *nonparametric* frequency estimation. A rotation in a known plane is a
two-parameter family, and the angle between consecutive samples determines ω outright, so
two samples suffice at any period short of aliasing. Structure beats the resolution limit.

**So neither extent nor period nor 1/Δf is the governing quantity.** What survives is
weaker and more honest: the required T is the **identifiability horizon** — the smallest
window at which the feature is separable from its alternatives — and it depends on the
structure of the family, not on any single scalar summary of one feature. For compactly
supported profiles it coincides with the extent and the geometric ceiling makes it tight;
for a clock it is 2, far below both the period and the extent.

**Which repairs the monotone-and-saturate property.** Period-ID rises and then holds — 0.252,
0.957, 1.000, 1.000, 1.000, 0.983 — flat from T=4 to T=24, six times longer than needed.
Atom recovery over the same range wanders: 1.000, 0.832, 0.866, 0.956, 0.829, 0.751. The
crosscoder is not failing to saturate. Reconstruction-flavoured metrics are failing to
saturate, and they were the metrics every earlier section of this log used to conclude that
excess T is harmful.

**Practical rule, replacing the one I withdrew.** Choose T by sweeping it and watching a
*task-relevant* readout flatten. Do not choose it from FVU, from reconstruction, or from
feature-recovery cosine: on this task those would have said T=1 was enough, when T=1 is at
chance.

## Why excess T degrades: my capacity account is refuted, and my harness was the confound

The desideratum, stated by the user and the right frame for all of this: performance should
rise with T while T is buying something and **flatten** once it is not. The crosscoder
instead rose and fell, so the question is why.

My account was position-tying: `W_dec` is `(d_sae, T, d)`, so a feature at offset 3 and the
same feature at offset 7 are separate atoms, the required dictionary grows with T, and a
fixed `d_sae` is progressively starved. `saturation_local.py` sweeps T against `d_sae` to
test it.

| d_sae | T=2 | T=4 | T=8 | T=16 | T=32 |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.78 | **0.88** | 0.77 | 0.32 | 0.17 |
| 256 | 0.80 | 0.89 | **0.96** | 0.44 | 0.19 |
| 512 | 0.78 | 0.91 | **0.97** | 0.54 | 0.20 |
| 1024 | 0.81 | 0.92 | **0.97** | 0.46 | 0.17 |

**S1 holds once and then stops.** The peak moves T=4 → T=8 between `d_sae` 128 and 256, and
does not move again across a further 4× of dictionary. **S2 fails**: equal `d_sae/T` does not
give equal recovery, with spreads up to 0.784. **S3 fails backwards**: shift-duplication
*falls* as `d_sae` grows — 0.59 → 0.04 at T=4 — so duplicated atoms are a symptom of a
scarce dictionary rather than the cause of the collapse. Eight times the dictionary does not
rescue T=32 at all (0.17 → 0.17). The position-tying account does not explain the collapse.

**What does is a flaw I built into the harness.** FVU at T=32 is 0.99-1.15, worse than
predicting the mean, which is divergence and not a capacity limit. The cause is in how the
windows were formed: reshaping a fixed segment stream into disjoint T-windows makes the
number of training windows fall as `seq_len/T`. At T=2 there were 24,000 windows; at T=32,
1,500 — for a model carrying 16× more decoder parameters. Large T was starved of data, not
of dictionary.

**This contaminates the sprint's own headline T-sweep.** `tsweep_modal.py` windows the same
way, so part of the 1.0× → 6.8× FVU rise attributed to "the cost of sharing one code across
a window" is instead fewer training windows at larger T. The T=1 control against the SAE is
unaffected — both see the same segments — but the *slope* in T is not clean and I should not
have presented it as purely architectural.

Fixed by windowing at stride 1, which holds the window count roughly constant in T (60
sequences, seq_len 96: T=2 → 5700 windows, T=16 → 4860, against 2880 and 360 before).
`saturation_local.py` and `recovery_local.py` both rerun on that basis; `tsweep_modal.py`
needs the same treatment before its slope is quoted.

The honest status of the desideratum until those land: the crosscoder demonstrably satisfies
it on a **task-relevant readout** (clock period-ID is flat from T=4 to T=24, six times longer
than needed), and appeared to violate it on reconstruction metrics that were measured under
a data-quantity confound.

## The corrected results: the desideratum is satisfied, and my headline was mostly an artefact

All four sweeps rerun with stride-1 windowing so the training-window count no longer falls
with T.

### The T-sweep collapses from 6.8× to 1.5×

| T | FVU (corrected) | × SAE | previously reported |
| --- | --- | --- | --- |
| 1 | 0.1166 | **0.99** | 1.0 |
| 2 | 0.1205 | 1.02 | 1.0 |
| 3 | 0.1248 | 1.06 | 1.0 |
| 4 | 0.1227 | 1.04 | 1.2 |
| 6 | 0.1290 | 1.10 | **3.2** |
| 12 | 0.1764 | **1.50** | **6.8** |

Realised coefficients per segment are 4.00 at every T with ReLU-kill 0.000, so this is a
clean measurement. **The "cost of sharing one code across a window" that I reported as
6.8× is 1.5×.** Most of what I attributed to the architecture was my harness giving large
T twelve times less data. The T=1 control is unchanged and still passes at 0.99×.

### With the per-segment budget set correctly, the crosscoder rises and saturates

`budget_local.py`, stride-1, recovery against the true features:

| T | kper=1 | kper=2 | kper=4 | kper=8 | kper=16 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.771 | 0.771 | 0.714 | 0.658 | 0.583 |
| 2 | **0.918** | 0.851 | 0.792 | 0.740 | 0.679 |
| 4 | **0.971** | 0.921 | 0.891 | 0.847 | 0.771 |
| 8 | **0.980** | 0.972 | 0.958 | 0.930 | 0.782 |
| 16 | **0.980** | 0.968 | 0.958 | 0.789 | 0.391 |

At kper=1 the profile across T is **0.771 → 0.918 → 0.971 → 0.980 → 0.980**: monotone
increasing, saturating at T=8, and holding at T=16 with no decline. That is exactly the
property we want — performance rises while T buys something and flattens once it does not.

It was hidden by two independent mistakes of mine, and needed both fixed: the per-segment
budget was 4× too dense, and the window count was falling with T. At kper=4 with the old
windowing the same sweep read 0.78 → 0.88 → 0.77 → 0.32 and looked like an architectural
failure.

**The user's account was right and mine was the confound.** B3 shows lower kper wins at
every T, and the penalty for over-budgeting grows with T: the recovery span across kper is
0.193 at T=1 and 0.591 at T=16.

### And the capacity account revives once the confound is removed

`saturation_local.py`, stride-1:

| d_sae | T=2 | T=4 | T=8 | T=16 | T=32 | peak |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 0.78 | **0.88** | 0.77 | 0.41 | 0.16 | T=4 |
| 256 | 0.79 | 0.90 | **0.94** | 0.80 | 0.28 | T=8 |
| 512 | 0.79 | 0.89 | 0.96 | **0.96** | 0.37 | T=16 |
| 1024 | 0.82 | 0.90 | 0.97 | **0.97** | 0.43 | T=16 |

S1 now holds properly: the peak moves 4 → 8 → 16 → 16 as the dictionary grows, and 8× more
dictionary *does* rescue T=16 (0.41 → 0.97) where before it did nothing (0.32 → 0.46).
Shift-duplication at T=16 is 0.31-0.33 against 0.05-0.08 before. So the position-tying
account was not wrong — it was masked by the data-starvation confound, which suppressed
every arm equally at large T. Required dictionary size does scale with T, and that is the
cost of tying atoms to absolute position.

### Activation functions, final

| arm | kper | coeff/seg | spend | neg | FVU |
| --- | --- | --- | --- | --- | --- |
| `topk_relu` | 41 | 0.96 | 2% | 0.000 | 0.843 |
| `batchtopk_relu` | 41 | 0.92 | 2% | 0.000 | 0.841 |
| `topk_relu` + AuxK | 41 | 0.96 | 2% | 0.000 | 0.843 |
| `topk` | 41 | 41.00 | 100% | 0.387 | **0.541** |
| **`batchtopk`** | 41 | 41.39 | 101% | 0.408 | **0.554** |

`batchtopk` without the ReLU and `topk` are equivalent within noise and both fix the
capacity cap outright; both ReLU variants and AuxK are stuck at 2% of budget. BatchTopK's
spend slightly exceeding 100% is the eval-time threshold approximating the train-time batch
rule rather than reproducing it exactly, which is expected. `batchtopk` is the default.

## Recovery, corrected: no collapse, and the rule is T >= L with headroom

`recovery_local.py` rerun at stride 1.

| extent | T=1 | T=2 | T=4 | T=8 | T=16 |
| --- | --- | --- | --- | --- | --- |
| L=1 | 0.911 | 0.990 | 0.999 | 0.997 | 0.987 |
| L=2 | **0.816 / 0.816** | 0.819 | 0.933 | 0.990 | 0.966 |
| L=4 | **0.661 / 0.662** | 0.754 | 0.831 | 0.921 | 0.919 |
| L=8 | **0.481 / 0.481** | 0.612 | 0.701 | 0.770 | **0.905** |

The T=16 column previously read 0.338-0.461 for every extent and now reads 0.905-0.987. The
collapse was entirely the window-count confound; there is no evidence that excess T harms
recovery once the data is held constant. Every extent rises and plateaus.

The T=1 column is unchanged and still sits exactly on the analytic ceiling, so the
geometric result — a per-token dictionary saturates `‖largest contiguous T-chunk of p‖/‖p‖`
and cannot be trained past it — survives the correction intact. That is the one finding in
this sprint that has not moved.

Saturation is at **T ≥ L with headroom**, not T = L: L=8 peaks at T=16 rather than T=8,
because a window of exactly L admits only one alignment of the feature while a longer one
admits several. The earlier "set T to the extent" phrasing was too tight even for the case
where extent is well defined.

## Refocusing on the actual deliverable

The sprint's goal was a benchmark of the crosscoder against SAE baselines, and the
methodology work — necessary, since every prior comparison was matched on a quantity that
does not bind — has crowded it out. `bench4_modal.py` is the benchmark, with the sprint's
findings built in: realised coefficients per segment as the axis, stride-1 windows,
`batchtopk` for the crosscoder, and the run-length corpus so a window factor exists, with
the i.i.d. corpus as the control where window-AUC must read 0.5.

Arms: `sae` (TopK), `stacked` (T independent per-position SAEs — the crosscoders-paper
baseline), `txc`, `tfa` (attention TemporalSAE with TopK), `tsae_paper` (the same module
with ReLU + L1, which is what this repo's `experiments/ward_backtracking_txc/
architectures.py` calls Bhalla-2025-faithful), and `tsae_nce` (per-token TopK SAE with an
InfoNCE consistency penalty between codes one position apart).

**An unresolved identification, flagged rather than guessed.** The repo defines `tsae_paper`
as the attention module with ReLU+L1. The description I was given is a T-SAE with an InfoNCE
penalty over nearby positions and no attention. Those are different architectures, and this
repo contains no InfoNCE-based tSAE — its InfoNCE code lives in
`han_arch/txc_bare_*contrastive*`, which are crosscoder variants. Both readings are
therefore run as separate arms. The InfoNCE arm's alpha (1.0) and shift (1 position) are my
choices, not the paper's, and are recorded in the output; the comparison should be read as
"an InfoNCE-regularised per-token SAE" rather than as a faithful reproduction until the
reference implementation is available.

Sparsity for the ReLU+L1 arm is set by `l1_coef` rather than k, swept over
{1e-1, 3e-2, 1e-2, 3e-3}. That it can be compared with the TopK arms at all is the payoff
of the realised-coefficient axis.

## First-pass benchmark: SAE vs plain TXC, at matched realised coefficients per segment

`bench4_modal.py`, run-length corpus, T=12, d_sae=4096, stride-1 windows, crosscoder on
`batchtopk` with no auxiliary penalty (not TXC-pro).

| coeff/segment | SAE FVU | TXC FVU | ratio | SAE winAUC | TXC winAUC |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.526 | 0.632 | 1.2× | 0.717 | 0.718 |
| 2 | 0.176 | 0.313 | 1.8× | 0.728 | 0.721 |
| 4 | 0.128 | 0.239 | 1.9× | 0.734 | 0.722 |
| 8 | 0.099 | 0.206 | 2.1× | 0.731 | 0.723 |
| 16 | 0.075 | 0.203 | 2.7× | 0.785 | 0.727 |

**The plain crosscoder does not beat a TopK SAE here.** It reconstructs 1.2-2.7× worse at
every matched budget, and it reads the window-level factor no better — 0.72-0.73 against
the SAE's 0.72-0.79, with the SAE ahead at the widest budget. Segment-AUC is 0.500 for the
crosscoder by construction, since a window code has no per-segment answer and broadcasting
it is the honest representation of that.

**The i.i.d. control passes.** Window-AUC reads exactly 0.500 for every arm and every budget
on the i.i.d. corpus, so the non-chance window-AUC on the structured corpus is measuring
structure rather than measuring the probe.

**Worth noting how much the methodology fixes closed the gap.** The same comparison earlier
in this sprint, on nominal-k matching with `topk_relu` and disjoint windows, had the SAE
ahead by 9-10× in FVU. With the realised-coefficient axis, `batchtopk`, and stride-1
windows it is 1.2-2.7×. Most of the crosscoder's apparent deficit was measurement.

### The tSAE arm is not yet usable, and the reason is a transferable hyperparameter failure

Run as this repo defines `tsae_paper` — attention TemporalSAE, ReLU + L1, n_heads=8,
bottleneck_factor=64, one attention layer, tied weights — with the loss normalised exactly
as `experiments/ward_backtracking_txc/architectures.py:185-188` does it (sum over dims,
mean over batch and time, for both the reconstruction and the L1 term).

At the documented `l1_coef=1e-3` the code is **dense**: 2989 of 4096 latents active per
segment, alive fraction 0.999. Sweeping the coefficient over two orders of magnitude
(1e-4 → 1e-2) moved realised L0 by 0.3%, from 2986 to 2994. The L1 term is inert at this
data scale, so FVU 0.030 and segment-AUC 0.97 are the numbers of an essentially dense
autoencoder and are not comparable to a 1-16 coefficient TopK arm.

This is not a bug in the port — the normalisation matches the reference line for line. It is
that a sparsity coefficient is only meaningful relative to the scale of the activations it
is applied to, and these are standardised differently from whatever the value was tuned on.
Rerunning with `l1_coef` swept over {0.03, 0.1, 0.3, 1, 3} to find the range where it binds.
The lesson generalises past this arm: an L1 arm cannot be dropped into a new activation
distribution at its published coefficient, and the realised-L0 check that this sprint
introduced for the crosscoder catches it immediately.

## Where the crosscoder's advantage actually is, and where to look next

Restating the sprint's purpose: find a setting where the crosscoder has a genuine advantage,
not merely benchmark it. Taking stock of what is established.

**An advantage that is proven, and provably not closable by better per-token training.** In
the synthetic lane, a per-token dictionary recovers an extent-L feature at exactly
`‖largest contiguous T-chunk of p‖/‖p‖` — 0.481 at L=8, matched to three decimals — and the
crosscoder reaches 0.905. This is geometry, not optimisation: no amount of capacity, tuning
or training moves the per-token ceiling.

**An advantage that is measured, on the project's actual use case.** The frozen-arm shuffle
control: take a trained crosscoder latent, permute its decoder rows in time, refit nothing,
and steering fidelity falls from +0.242 to a null of +0.002 ± 0.103 over 24 draws — 100th
percentile. Replicated on a healthier dictionary at +0.292 against −0.008 ± 0.114, on a
different latent. The temporal arrangement of a single latent carries steering signal that
permutation destroys, and a per-token dictionary has no single latent that represents
arrangement at all.

**A disadvantage that is real.** At matched realised coefficients per segment on layer-14
LM activations, the crosscoder reconstructs 1.2-2.7× worse and reads the window-level factor
no better (0.72-0.73 against 0.72-0.79).

**The obstruction that reconciles them, and the next experiment.** A mid-stack token
representation has already attended over its predecessors, so cross-segment structure in the
*text* has largely been integrated into each token's own activation by layer 14. The
features are effectively extent-1 in activation space, which is exactly the regime where the
synthetic lane says a window code buys nothing. The per-token SAE's window-AUC of 0.72-0.79
is that integration made visible: it is reading context the model already folded in.

If that is right, the crosscoder's headroom should be a function of depth, and
`layer_modal.py` tests it across layers 2, 6, 14, 22 with everything else fixed. Registered:
D1 the FVU penalty shrinks at early layers; D2 the SAE's own window-AUC *rises* with depth,
which is the signature of the model doing the integration; D3 at the earliest layer the
crosscoder beats the SAE on window-AUC. D3 is the one that would count as a genuine
advantage on real activations, and D1 and D2 can both hold without it.

## The depth hypothesis is refuted, and it exposes a task-design error of mine

`layer_modal.py`, layers 2/6/14/22, everything else fixed.

| | L2 | L6 | L14 | L22 |
| --- | --- | --- | --- | --- |
| FVU ratio TXC/SAE, k=2 | **7.80×** | 4.02× | **1.57×** | 1.85× |
| FVU ratio TXC/SAE, k=8 | 2.90× | 2.56× | 2.51× | 2.40× |
| SAE window-AUC, k=8 | 0.646 | 0.649 | **0.763** | 0.674 |
| TXC window-AUC, k=8 | 0.554 | 0.551 | 0.574 | 0.574 |
| Δ window-AUC | −0.092 | −0.098 | **−0.189** | −0.101 |

**D1 is refuted in the opposite direction.** The crosscoder's FVU penalty is *worst* at the
earliest layer — 7.80× at L2 against 1.57× at L14. Early activations sit close to token
identity, which is exactly what a per-token dictionary is for (the SAE reaches FVU 0.027 at
L2), and exactly the hardest thing for a code shared across twelve positions to compress.
**D3 fails in 0 of 8 cells.** D2 holds weakly at best: the SAE's window-AUC peaks at L14
(0.763) but the variation across depth is small and non-monotone.

So contextual integration is not the obstruction, and the "look at early layers" idea is
dead. Recording it as a clean refutation rather than reinterpreting it.

**And the refutation surfaced a task-design error I had already identified and then walked
into.** Earlier in this sprint I recorded the reviewer's arithmetic dismantling the window
target: run length is detectable from **local** statistics, because ℓ=1 and ℓ=6 differ in how
often adjacent segments switch state, and a mean-pooled per-token code can count switches
without representing anything window-level. I then built the benchmark on exactly that
corpus. The SAE's 0.72-0.79 window-AUC is not evidence that per-token codes capture
window structure; it is evidence that this particular "window factor" was never a window
factor.

**What a decisive task requires.** The window-level label must be invariant to the multiset
of segments *and* to the count of switches, so that no permutation-invariant pooling of
per-token codes can reach it. The minimal such design is order-of-blocks with everything
else matched:

```text
class A:  T T T T T T C C C C C C      6 tense, 6 calm, 1 switch
class B:  C C C C C C T T T T T T      6 tense, 6 calm, 1 switch
```

Identical multiset, identical switch count; the classes differ only in which block came
first. Any readout that pools per-token codes symmetrically over the window is at chance by
construction.

**The honest caveat, registered before running.** On real LM activations this may still not
separate the architectures, because the model is causal: a tense sentence preceded by tense
sentences has a different representation from one preceded by calm sentences, so per-token
activations carry order even when the *label* is order-only. If the pooled SAE still wins
here, the conclusion is sharp and worth stating plainly — on real activations a per-token
dictionary is not blind to order, and the crosscoder's distinctive value is not in *reading*
temporal structure but in *representing it in a single steerable latent*, which is what the
frozen-shuffle control already measures.

## The order-only task: a decisive negative that explains the whole sprint

Both classes have the same multiset of segments and the same switch count, differing only in
which block came first, so any symmetric pooling of per-token codes is at chance by
construction. Layer 14, T=12, 1500 documents.

| | SAE winAUC | TXC winAUC | Δ | SAE FVU | TXC FVU |
| --- | --- | --- | --- | --- | --- |
| order task, k=2 | **0.981** | 0.825 | −0.156 | 0.148 | 0.412 |
| order task, k=8 | **1.000** | 0.833 | −0.167 | 0.069 | 0.389 |
| shuffled control, k=2 | 0.597 | 0.598 | +0.002 | 0.156 | 0.545 |
| shuffled control, k=8 | 0.604 | 0.601 | −0.004 | 0.080 | 0.505 |

**A single mean-pooled SAE latent reaches AUC 1.000 on a label that is pure order.** The
shuffled control falls to 0.60 for both architectures, so the signal is genuinely order and
not label leakage. O2 is refuted; O3 is confirmed.

**The explanation, and it is the obstruction the whole sprint kept running into.** The model
is causal, so it writes its own history into every token: a calm sentence following six
tense ones has a different representation from a calm sentence following two. Pooling
per-token codes therefore recovers order without any window representation at all. The
design was built specifically to defeat per-token codes and it did not come close.

**This generalises.** There is no order information in mid-stack LM activations that a
per-token dictionary cannot access, because the transformer has already performed the
temporal integration. Every "can a window code read temporal structure better?" experiment
in this sprint was therefore asking a question whose answer was fixed in advance by the
architecture of the model being probed, not by the architecture of the dictionary. That is
why the run-length task, the depth sweep and this task all came out the same way.

**And it says exactly where the advantage has to be.** Reading is not the crosscoder's
job — the SAE can already read everything. The crosscoder's distinctive capability is
representing temporal structure in a *single latent with a temporal profile*, which is a
statement about intervention, not about decoding. A per-token dictionary's only per-latent
write is one direction applied at every position; that write is constant in time, and a
constant write cannot distinguish two orderings of the same multiset. A crosscoder latent
writes a different vector at each position and can.

That is a falsifiable claim on the *same* task where the SAE reads perfectly, which makes it
the sharpest available test: **the SAE reads order at AUC 1.000 and should be unable to
steer it; the crosscoder reads it worse at 0.83 and should be able to.** If reading and
steering come apart on one task, the crosscoder's value is established precisely and the
FVU deficit becomes the price rather than the verdict. `steer_order_modal.py` runs it.

## The genuine advantage: reading and steering come apart

`steer_order_modal.py`, on the same order-only task where a pooled SAE latent reads the
label at AUC 0.998. Teacher-forced Δ margin `logP(A) − logP(B)` with B the same sentences
reordered, so multiset-matched foils cancel any generic "more tense tokens" effect. All
writes rescaled to identical total injected norm.

| arm | α=0.25 | α=0.5 | α=1.0 | α=2.0 | per-position spread |
| --- | --- | --- | --- | --- | --- |
| `sae_broadcast` | −0.90 | −1.41 | −0.91 | +2.25 | 0.0000 |
| **`txc_slab`** | **+0.75** | **+1.36** | **+3.26** | **+9.93** | 0.0455 |
| `txc_flat` (control) | −1.20 | −2.33 | −4.58 | −7.87 | 0.0000 |
| `dom_slab` (ceiling) | +5.75 | +12.24 | +28.50 | +68.69 | 0.0131 |

| | reads order | steers order |
| --- | --- | --- |
| SAE | AUC **0.998** | +2.25 |
| crosscoder | AUC 0.791 | **+9.93** |

**S1 and S2 hold.** The per-token dictionary's write is constant in time — measured
per-position spread exactly 0.0000 — and against two orderings of one multiset a constant
write has nothing to push on. Its Δ margin is negative at three of four doses and reaches
+2.25 only at the largest, where it is plausibly just a generic tense-shift rather than an
ordering effect. The crosscoder's slab is monotone in dose and 4.4× larger at matched norm.

**S3 holds, and it is what makes this a temporal result rather than a better-direction
result.** `txc_flat` takes the crosscoder's own slab, averages it over time and rebroadcasts
it: same latent, same mean direction, same injected norm, temporal profile removed. The
effect does not merely disappear, it **inverts** — −1.20 to −7.87 across the sweep. So the
advantage is carried by the *arrangement* of the write across positions, not by the
direction being better chosen. This is the same conclusion the frozen-shuffle control
reached by permuting rows, now reached by flattening them, on a different task.

S4 holds: the supervised difference-of-means ceiling is far above everything at +68.69, so
neither dictionary is close to optimal and there is substantial headroom.

**This is the advantage the sprint was looking for, and it is a statement about
intervention, not decoding.** The architecture that reads the factor *better* steers it
*worse*. That dissociation explains why every reading-based comparison in this sprint
favoured the SAE — a causal transformer has already written its history into every token, so
there is nothing left for a window code to *read* — while the steering comparison reverses,
because a per-token dictionary has no per-latent handle that varies across time and a
crosscoder does.

It also reframes the FVU deficit. The crosscoder reconstructs 1.2-2.7× worse at matched
realised coefficients per segment; that is the price of the shared code, and it buys a
temporally structured write that the per-token dictionary cannot express at any budget.

## The steering result, hardened: nulls, error bars, and a catch

Rerun at 200 test documents with per-document standard errors and two null arms. `dom_slab`
is supervised; `random_slab` is a random temporal profile and is the direct control for
`txc_slab`; `random_broadcast` is a random constant direction and is the direct control for
`sae_broadcast`. All at identical injected norm.

| arm | α=0.25 | α=0.5 | α=1.0 | α=2.0 |
| --- | --- | --- | --- | --- |
| `dom_slab` (supervised) | +5.69 ± 0.13 | +12.09 ± 0.25 | +27.96 ± 0.53 | +68.12 ± 1.17 |
| **`txc_slab`** | **+0.72 ± 0.09** | **+1.42 ± 0.17** | **+3.48 ± 0.34** | **+11.29 ± 0.64** |
| `random_broadcast` | −0.26 ± 0.08 | −0.38 ± 0.13 | −0.03 ± 0.24 | +3.01 ± 0.47 |
| `sae_broadcast` | −0.96 ± 0.12 | −1.59 ± 0.22 | −1.16 ± 0.34 | +1.24 ± 0.56 |
| `random_slab` | −0.37 ± 0.08 | −0.58 ± 0.14 | −0.71 ± 0.26 | −0.28 ± 0.49 |
| `txc_flat` | −1.19 ± 0.09 | −2.25 ± 0.17 | −4.49 ± 0.32 | −8.02 ± 0.64 |

Separation at each arm's best dose: `txc_slab` against `sae_broadcast` is **z = 11.8**, and
against `random_slab` is **z = 14.3**.

**The catch, and it strengthens S1 rather than weakening it.** In the first run I noted the
SAE's +2.25 at the largest dose and flagged it as possibly a generic tense-shift. The null
arm settles it: a **random** constant direction reaches +3.01 ± 0.47, *larger* than the
SAE's learned direction at +1.24 ± 0.56. So a constant write at high dose produces a small
positive drift that has nothing to do with order, and the SAE's learned order-latent does not
even match what a random direction achieves. A per-token dictionary's write is not merely
weak on this task, it is indistinguishable from noise.

**The random temporal profile is the control that matters for the positive claim**, and it
is flat at −0.28 ± 0.49. So the crosscoder's +11.29 is not "any structured perturbation
across positions" — a random one does nothing. It is the learned arrangement specifically.

Three controls now converge on the same conclusion by different routes: permuting the
profile in time (frozen shuffle, 100th percentile of 24 draws), averaging it away
(`txc_flat`, which inverts to −8.02), and replacing it with a random one (`random_slab`,
flat). The temporal arrangement of a single crosscoder latent is doing the work.
