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
high-variance: the crosscoder's own margin moved 10× across three inits at phase-5, and on the
order task its selected latent does not even have a stable *sign* across inits. **We never gave
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
- **Give the transfer negative a positive control.** No `(T, d)` write of any kind moves the
  instruction-position bias in Qwen2.5-0.5B or SmolLM2-1.7B at any of six layers, supervised
  included. That is currently a bare negative. Sweeping model × layer with the training-free
  screen would establish either that the steerable site is idiosyncratic to Qwen2.5-1.5B L14 —
  which is a strong scope limit worth stating — or that it moves with depth in a predictable
  way, which would be a better result than the original.

What I would **not** do is add tasks. Nine were run and the marginal one taught less than any
of the audits did; the binding constraint is that every conclusion rests on one model, one
layer, and one seed per cell, not on the number of behaviours tried.
