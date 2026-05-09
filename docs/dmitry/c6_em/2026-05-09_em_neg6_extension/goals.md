---
author: Dmitry Manning-Coe
date: 2026-05-09
tags:
  - design
  - in-progress
---

## Negative + extended steering range — α ∈ {-6, ..., +6}

Re-run of the medical, finance, sports replications (`2026-05-08_em_repl_finance_sports`) over a wider, **symmetric** steering grid:

```
α ∈ {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6}    # 13 integer points
```

Previous grid topped out at α = 3 and only went to α = 0 (full ablation). The new grid:

- **Doubles the positive range** to α = 6 (catches saturation behaviour at high steering).
- **Adds a negative arm** down to α = -6, which:
  - For the **additive** recipe `act += (α-1)·f·W_dec`, α<0 means `(α-1) ∈ [-7, -1]` — a *strong* anti-feature push (the previous min was (α-1) = -1 at α=0).
  - For the **QK→QK** recipe `features[idx] *= α`, α<0 *flips the feature sign*, so α=-1 is "do the opposite of α=+1" and α=-6 is a 6× anti-amplification.
  - For **OV recipes** (write through `W_V`), same sign-flip semantics applies to the projected feature direction.

This lets us look for asymmetry between "amplifying the EM-direction" and "ablating + opposing it" — the previous range only covered the positive (amplification) side beyond α = 1.

## What's new

1. **9 streams** (3 domains × 3 eval seeds) re-run end-to-end on the extended α grid using the validated fast path (batched generation + per-prompt seeds + parallel judge).
2. **Single SCP/judge/combine/plot pipeline** in `fra_proj/scripts/run_postprocess_neg6.sh` so the new data can be merged + visualised in one command.
3. **Re-rendered figures** (1×3 panel, 3-bar headline, 3 seed-grids) under a `_neg6` suffix so they don't overwrite the 2026-05-08 originals.

## Compute

Per stream: 13 α × (5 SAE additive + 3 FRA recipes) + 1 baseline ≈ 105 cell-batches × ~11 s ≈ 20 min compute, plus 2 model loads + ranking ≈ 7 min, ≈ **27 min wall per stream**. 9 streams in 2 waves on 6 free H100 GPUs ⇒ **~55 min wall** for compute, +~10 min for judge/combine/plot.

`h100_5_em` is occupied training the new L24 ln1 SAE (separate task, not a candidate for eval streams in this run).

## Outputs

- `temp_xc/plots/2026-05-09_em_neg6/streams/{em}_{seed}/qualitative_*.json` (additive + FRA per stream, flat layout)
- `temp_xc/plots/2026-05-09_em_neg6/streams/gpt4o_combined_*.json` (per-(sae, em) cross-seed summaries — same schema as the 2026-05-08 run)
- `fra_proj/figures/em_figures/phase1_{*}_neg6.{png,pdf}` — re-rendered figures with `_neg6` suffix
- `fra_proj_tex/figures/phase1_{*}_neg6.{png,pdf}` — paper-ready PNG + PDF copies

## Pass criteria

Sanity:
- α = 1 cell aligns with the 2026-05-08 run within judge noise (~5 pts on `mean_alignment`) — same recipe, same RNG conventions.
- Hook no-op check at α = 1 still passes byte-identical (we didn't change the steering math).

Headline:
- Does QK→QK extend monotonically as α grows from 1 → 6, or does it saturate / collapse?
- Does the negative arm produce a symmetric-but-opposite effect on alignment, or asymmetric?
- Are there α regions where the additive recipe outperforms FRA — and where in α-space?
