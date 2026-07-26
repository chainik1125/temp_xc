---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - proposal
  - todo
---

## What I would run first with another ten hours

**The sprint's central claim is an optimisation claim wearing architectural clothes, and we
never tested the obvious fix for it.** Every win this sprint is a discovery result — the
crosscoder found a write unsupervised that a per-token dictionary could express but did not
learn — and every loss is the same statement in reverse: on `recency_var`, `rot_m6` and the
phase ladder the supervised write existed and training failed to find it. But "training failed
to find it" is a claim about a *search*, and the search we ran gave each architecture **one
dictionary init and one best-of-4096 latent selection**. We already know that lottery is
high-variance, and the size of it depends on the recipe. Across three inits of `phase5` the
crosscoder's peak margin spans 1.56 / 15.70 / 11.48 at the old recipe (lr 3e-4, 2000 steps,
one-sided doses) and 3.64 / 1.47 / 4.63 at the current one (lr 1e-3, symmetric doses) — a 10×
spread shrinking to **3.1× at peak and 1.4× at matched dose** once the recipe is fixed. Quote
the 3.1×: it is the variance under the protocol we actually report, and the shrinkage is itself
evidence that much of the apparent init instability was under-training. The sharper fact is
qualitative and survives at the current recipe: on the order task the crosscoder's selected
latent **reverses direction** between inits — rising to the right at init 0 and to the left at
init 1 — while `txc_flat` and the SAE hold orientation in both. **We never gave
the SAE the same number of tickets.** So the first experiment is to scale seeds — twenty inits
per architecture on `recency`, `evidence` and `rot_m12`, reporting the distribution of each
arm's best latent rather than one draw. If the SAE's best-of-twenty closes the gap, the sprint's
headline becomes "the crosscoder finds this write more *often*", which is a weaker and more
honest claim than the one currently written; if it does not close, the discovery claim is
enormously strengthened, because it survives the most obvious attack on it. Either outcome is
worth more than a new task, and it is cheap — `dict_seed` is already a flag and no new corpus
is needed.

Two things follow it, both enabled by infrastructure that did not exist this morning.

- **Run the geometry screen at scale.** `geometry_modal.py` measures `c`, `r1`, the unsteered
  baseline and shared-write retention with **no dictionary training at all**, in about two
  minutes per task. Every claim about `c` currently rests on seven points from one model at one
  layer. The screen is cheap enough to map the `c` landscape across dozens of candidate tasks,
  every layer, and several models — turning "`c` classifies 6 of 7" into a real empirical
  regularity or killing it. This is the highest ratio of information to compute available.
- **Replicate the 3B transfer cell.** It is the thinnest load-bearing result in the sprint: two
  inits, `win = True` in one and `False` in the other, and the verdict flips between matched and
  peak dose. The negatives at 0.5B and SmolLM2 have three inits each and `win = False` in all six.
  Three or more inits at 3B, plus a fourth model, is what would let "scale is not the axis" stand
  on evidence rather than on one split cell.
- **Explain the SmolLM2 discovery failure.** The factor is present and strongly reachable there — a
  gradient write moves it at every one of six depths, +13.38 at L6, larger than in the model where
  we succeed, and a plain constant write reaches +3.87 — and every learned arm misses it. **Three
  mechanisms were proposed and all three refuted** (`cos(P_dom, Ḡ)`; dictionary-tracks-`P_dom`; a
  `cos(v_sae, u₁(·))` account that passed a pre-registered test and failed on a borrowed baseline).
  Normalised, SmolLM2 is a quantitatively worse cell rather than a mechanistically different one:
  the constant-write ceiling is the same share of the optimum in both models (0.29 vs 0.25) and only
  the learned arms differ (0.07 vs 0.24–0.30). **A fourth candidate should be registered with its
  falsifier before it is tested**, given the base rate.
- **Test the baseline-sign observation, which is registered and not believed.** The sign of the
  unsteered `score(A) − score(B)` splits the two working models from the two failing ones, 4 of 4 —
  Qwen2.5-1.5B −2.54 and 3B −10.49 against SmolLM2 +2.19 and Qwen2.5-0.5B +1.50. **This is n = 4
  with one degree of freedom and is recorded as a thing to test, not a thing observed.** Three
  patterns of exactly this shape were proposed and dissolved during the sprint.

What I would **not** do is add tasks. Nine were run and the marginal one taught less than any
of the audits did; the binding constraint is coverage per cell, not the number of behaviours tried. That constraint
has partly lifted — instruction position now has eight dictionary inits and the three decomposition
cells have three each — but the transfer set, the four sparse `broadcast_optimal` cells and every
cross-model claim still rest on one or two.
