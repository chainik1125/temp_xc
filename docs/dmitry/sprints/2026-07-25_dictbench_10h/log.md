---
author: Claude (Fable 5), for Dmitry Manning-Coe
date: 2026-07-25
tags:
  - results
  - in-progress
---

## Sprint log — dictionary benchmark 10h

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

