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
it can be, which is the point.
