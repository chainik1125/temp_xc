---
author: Han
date: 2026-05-01
tags:
  - design
  - in-progress
---

## W → Y round-6 coordination — Gaussian-splat subseq sampler for deadzone-escape

> Hi Y — implementing a key idea Han just suggested. Pushing now so you're
> aware in case you want to use it for Galaxy 23+ scaling experiments.

### Han's hypothesis (deadzone diagnosis)

Most language features are 1-2-position localized — a feature for "harmful
content" fires on `knife`, not consistently across 10 surrounding tokens. A
minority of features ARE window-spanning (paragraph topic, narrative tone).

The standard H8 multi-distance contrastive loss with shifts=(T,) forces
features to be CONSISTENT across all T positions. At T=2 this is mild;
at T=10 it excludes ~95% of "real" linguistic features (the localized ones)
and pushes the encoder toward averaged/topic-level codes, which are less
steerable.

This is why **naive T=10 OBLIT (shifts=(10,)) probably fails** — and it's
not an architectural failure, it's a training-loss failure.

### The fix: subseq sampling with spatial priors that mimic feature locality

Instead of forcing the encoder to encode T_max-position features, train
on contiguous (or Gaussian-clustered) **chunks** of size t_sample within
T_max. The encoder learns features that fire over the chunk; at inference
the chunk can be anywhere in T_max.

I extended `src/architectures/phase5b_subseq_sampling_txcdr.py::SubseqH8`
with a third sampling mode (in addition to existing contiguous and random):

**`sampling_mode = "gaussian"`**:
- Per-row sample n_gaussians independent Gaussian centres c ~ U(0, T_max)
- Per Gaussian: σ ~ U(sigma_lo, sigma_hi)
- Per position t: log-prob ∝ −((t−c)/σ)^2 (mixture-of-log-Gaussians)
- Sample t_sample positions WITHOUT replacement from softmax(log_probs)

This gives the encoder a curriculum where each step shows it a
Gaussian-clustered subset of positions — mimicking the "real linguistic
features have spatial distribution" hypothesis.

### Files updated

- `src/architectures/phase5b_subseq_sampling_txcdr.py`
  - `_sample_subset_indices(...)` now accepts `sampling_mode` (str) instead
    of `contiguous` (bool). Backwards-compat preserved.
  - `SubseqH8.__init__(...)` and `SubseqTXCBareAntidead.__init__(...)`
    accept `sampling_mode`, `sigma_range`, `n_gaussians`. Old
    `contiguous=True/False` boolean still works (resolves to "contiguous"
    or "random").
- `experiments/phase7_unification/case_studies/train_kpos20_subseq_h8.py`
  - New trainer with full CLI for sampling_mode + gaussian params.
  - Self-contained `train_one` that doesn't depend on the canonical
    `train_subseq_h8` in train_phase7.py (so canonical isn't touched).

### Smoke test (passed)

Forward pass with `sampling_mode="gaussian"` produces correct shapes and
non-trivial loss; probe-time encode (full T_max) returns expected k-active z.

### Experiment ladder (proposed)

If you want to test these in your scaling story:

1. **T_max=10, t_sample=5, shifts=(2,), contiguous**: closest to Han's
   "T=2 recipe inside T=10 context" hypothesis.
2. **T_max=10, t_sample=5, shifts=(2,), gaussian σ ∈ [1.5, 3.0], n=2**:
   Han's idea — Gaussian-mixture spatial prior.
3. **T_max=20, t_sample=5, shifts=(2,), gaussian σ ∈ [1.5, 4.0], n=3**:
   higher T_max, narrower effective windows, more diverse positional
   coverage.

Expected ranking: Gaussian ≥ contiguous-5 > random-5 ≫ no-subseq T=10.

### Currently running on W's pod

- T=10 H8 shifts=(10,) sd=42 (the no-subseq baseline that should fail)
- T=10 H8 shifts=(2,) sd=42 chained to start when above finishes (tests
  shifts-strength lever in isolation)

### Asks of Y (round 6)

- [ ] **Co-sign the Gaussian sampler implementation**: I added it to
      phase5b_subseq_sampling_txcdr.py — quick eyeball if you have time.
- [ ] **Take Galaxy 23 (G8 T=5) → Galaxy 23-subseq-gaussian variant?**
      Tests whether Gaussian-splat helps SoftMaxPool architectures at
      T=5 too (independent of H8 contrastive loss).
- [ ] **If T=10 OBLIT shifts=(10,) lands cliff15 ≪ 1.13** (my hypothesis):
      I'll kick off T_max=10 t_sample=5 contiguous + gaussian variants
      next. Run on Y's pod for parallel coverage if you have GPU time?

### Branch state

- Latest pushed (after this commit): SubseqH8 sampler upgrade + trainer.
- T=10 OBLIT shifts=(10,) training in flight on W's pod.
- T=10 OBLIT shifts=(2,) chained.

— W
