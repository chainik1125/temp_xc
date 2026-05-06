---
author: Han
date: 2026-04-26
tags:
  - results
  - complete
---

## Bug: tsae_paper group_sizes drift between trainer and probing loader

### Summary

`run_probing_phase7._load_phase7_model`'s `TemporalMatryoshkaBatchTopKSAE`
branch was building the model with a **different `group_sizes` than
the trainer**. Sum of the loader's group_sizes was 34560, but the
trainer's was 18432 (= `d_sae`). The class's `__init__` asserts
`sum(group_sizes) == dict_size`, so every probe call on a `tsae_paper_*`
ckpt would have crashed.

### Discovery

Caught by Agent C during scope review, before any probe call had
been made on a `tsae_paper_*` ckpt. The bug was latent: the seed=42
batch successfully trained `tsae_paper_k500__seed42` and `tsae_paper_k20__seed42`
and pushed both to `han1823123123/txcdr-base`. The probing pass would
have crashed on first contact.

### Root cause

Two-source-of-truth drift:

| | trainer (`train_phase7.train_tsae_paper`) | loader (`_load_phase7_model`) |
|---|---|---|
| convention | `group_fractions = [0.2, 0.8]` | `[d_sae//8, d_sae//4, d_sae//2, d_sae]` |
| computed group_sizes (d_sae=18432) | `[3686, 14746]` | `[2304, 4608, 9216, 18432]` |
| sum | **18432 ✓** | **34560 ✗** |

The trainer's value comes from
`experiments/phase6_qualitative_latents/train_tsae_paper.py:115`
(Phase 6 convention, 2-group split) which was the canonical reference
when porting. The loader was written from scratch in
`run_probing_phase7.py` using a different (4-group) convention and
nobody noticed because the probing pass hadn't run yet.

### Fix (commit `0ffaa5e`)

Two changes to ensure this can't recur:

1. **Loader uses trainer's convention** AND prefers reading
   `group_sizes` from meta if present:
   ```python
   if meta.get("group_sizes"):
       group_sizes = list(meta["group_sizes"])
   else:
       group_sizes = [int(0.2 * d_sae), d_sae - int(0.2 * d_sae)]
   assert sum(group_sizes) == d_sae, ...
   ```

2. **Trainer bakes `group_sizes` into meta** for `tsae_paper_*` rows
   in `_meta_from_arch`. Future ckpts have the value in
   `training_index.jsonl`; the loader is then independent of the
   trainer's formula. Two-source-of-truth bugs of this exact shape
   stop being possible.

Existing `tsae_paper_k500__seed42` ckpt verified to load cleanly
under the patched loader (85M params, k=500, dict_size=18432).

### Lessons / convention going forward

- For any arch where the model class takes parameters NOT derivable
  from `(d_in, d_sae, k, T)` (e.g., `group_sizes`, `prefix_lens`,
  `latent_splits`), bake those parameters into `_meta_from_arch` so
  the meta dict is the canonical record.
- Cross-reference: Phase 6's training script
  (`experiments/phase6_qualitative_latents/train_tsae_paper.py:115`)
  is the canonical port of T-SAE's group convention. Future Phase 7
  refactors of `train_tsae_paper` should keep the [0.2, 0.8] split
  unless explicitly justified.
- Probe loader should never silently disagree with the trainer; the
  `assert sum(group_sizes) == d_sae` line in the loader is now a
  hard guardrail.

### Related: similar audit needed for other archs?

Quick check of the other 11 src_class trainers vs loaders:

- `TopKSAE`: `(d_in, d_sae, k)` — derivable from meta. ✓
- `MultiLayerCrosscoder`: `(d_in, d_sae, n_layers, k)` — `n_layers`
  is in meta. ✓
- `MLCContrastive`, `MLCContrastiveMultiscale`: same plus `h` (Matryoshka
  head size). Trainer hardcodes `h = int(d_sae * 0.2)`; loader does
  the same. **Drift-resistant only by coincidence**; should bake `h`
  into meta defensively. (Tracked as low-priority follow-up; not
  blocking since both trainer and loader use the same formula today.)
- `TemporalSAE` (TFA): `(dimin, width, n_heads, sae_diff_type, kval_topk,
  ...)` with hard-coded `n_heads=4`, `bottleneck_factor=4`. Both
  trainer and loader use these values. Same coincidence-only situation
  as MLC.
- `MatryoshkaTXCDRContrastiveMultiscale`: `n_contr_scales` and `gamma`
  read from arch meta. ✓
- `TXCBareAntidead`: `(d_in, d_sae, T, k)` only. ✓
- `SubseqTXCBareAntidead`, `SubseqH8`: `(d_in, d_sae, T_max, k, t_sample)`
  + auto-derived shifts. Both trainer and loader compute shifts from
  T_max identically. Drift-resistant only by coincidence.
- `TemporalCrosscoder`: `(d_in, d_sae, T, k)` only. ✓
- `TXCBareMultiDistanceContrastiveAntidead`: shifts are in arch meta. ✓

**Conclusion**: only `tsae_paper` had a real bug. The other 5
"coincidence-only" archs (MLC family, TFA, Subseq family) should bake
their derived hyperparameters (`h`, `n_heads`, `bottleneck_factor`,
shifts) into meta as a follow-up. Not blocking the seed=42 probing
pass — but worth doing before seed=1 starts to make those ckpts
self-describing too.
