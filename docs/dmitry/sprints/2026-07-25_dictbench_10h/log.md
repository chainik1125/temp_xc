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
it can be, which is the point. Using signed least-squares coefficients also dissolves the
sign bug above — orientation is now something the fit chooses, not something I hand-set.

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
