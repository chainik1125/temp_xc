---
author: Dmitry
date: 2026-04-28
tags:
  - results
  - in-progress
---

## Wang causal screen identifies different features than firing-difference ranking

### Summary

The Wang et al. 2025 attribution procedure ([arXiv:2506.19823](https://arxiv.org/abs/2506.19823)) ranks SAE/Han latents by **causal effect on alignment when steered**, not just by *firing difference* between base and bad-medical models. Running the first stage of that procedure (top-100 features by Δz̄ → screen each by steering at α=±1) on our v2 100k checkpoints reveals that **the highest-Δz̄ feature is not the most causally important** in either architecture.

For Han H8 v2 100k, the encoder-side ranking elects feature 688 (Δz̄ = +1.208, by far the largest firing increase). The causal screen demotes it to rank 10. The actual causal champion is feature 693 (Δz̄ = +0.112 — 10× smaller firing increase), which produces a 23.14-point alignment swing between α=+1 and α=−1 steering, vs feature 688's 13.57 points.

For SAE arditi v2 100k, the analogous causal champion is feature 29650 (Δz̄ = +0.106, screen score +26.98), which is rank 31 by Δz̄ alone.

### Why this matters

Many SAE/Han features fire more strongly in a bad-medical fine-tune *as a side effect* of the broader distribution shift — not because they *cause* the misaligned behavior. Δz̄ ranking conflates correlation with causation. The Wang causal screen, which steers each candidate and watches whether alignment actually shifts, isolates the latents whose decoder direction is genuinely on the misalignment axis.

This explains why our earlier `k=10` cosine-bundle and `k=1` encoder-Δz̄ steering peaks (Δalign ≈ +6 to +12) were modest: we were largely steering with features that don't actually move the alignment axis. The causal-screen champions show ~25-point swings on a *single* feature with no bundling — substantially larger than anything we saw with bundled or Δz̄-ranked steering.

### Top causal-screen features (Wang stage 2 only — stages 3-4 blocked by Gemini quota)

**SAE arditi v2 (k=128, d_sae=32k, T=1) at step 100k — top-5 by causal screen:**

| screen rank | feature_id | Δz̄ | screen score | α=+1 align | α=−1 align |
|---:|---:|---:|---:|---:|---:|
| 1 | 29650 | +0.106 | +26.98 | 33.7 | 60.7 |
| 2 | 16095 | +0.073 | +23.67 | 30.7 | 54.3 |
| 3 | 16663 | +0.044 | +20.33 | 32.7 | 53.0 |
| 4 | 7303 | +0.043 | +18.67 | 35.7 | 54.3 |
| 5 | 14988 | +0.199 | +17.24 | 31.3 | 48.6 |

**Han H8 champion v2 (k=128, d_sae=32k, T=5) at step 100k — top-5 + the high-Δz̄ outlier:**

| screen rank | feature_id | Δz̄ | screen score | α=+1 align | α=−1 align |
|---:|---:|---:|---:|---:|---:|
| 1 | 693 | +0.112 | +23.14 | 39.7 | 62.9 |
| 2 | 4875 | +0.077 | +17.67 | 33.7 | 51.3 |
| 3 | 2444 | +0.106 | +17.33 | 31.3 | 48.7 |
| 4 | 3415 | +0.168 | +17.22 | 32.1 | 49.3 |
| 5 | 1442 | +0.138 | +15.90 | 38.3 | 54.2 |
| **10** | **688** | **+1.208** | **+13.57** | **45.7** | **59.3** |

(`screen score` = align(α=−1) − align(α=+1). Each measurement uses 8 EM questions × 2 rollouts = 16 generations, judged by gemini-3.1-flash-lite-preview at temp=0.5.)

### Method

For each candidate feature `i` in the top-100 by Δz̄ on the medical-prompts dataset (1000 prompts):

1. extract the decoder direction (SAE: `W_dec[i, :]`; Han: `W_dec[i, -1, :]` — last temporal slot)
2. generate 16 outputs at α = +1 with steerer = α × W_dec[i] applied at layer 15 of the bad-medical Qwen
3. generate 16 outputs at α = −1 (same direction, opposite sign)
4. judge each generation on alignment (Gemini, temperature 0.5, deterministic seed)
5. screen score = mean_align(α = −1) − mean_align(α = +1)
6. rank features by screen score, take top-K

Wang report this in their Sec. 3.1 / Appendix D.2 as the gating step that takes their candidate list from ~1000 features down to ~40, then a per-feature coherence-aware strength sweep narrows further before single-feature steering.

### Comparison vs prior procedures we tried

| procedure | best Δalign vs α=0 baseline | peak coherence | best peak α |
|---|---:|---:|---:|
| cosine k=10 bundle (SAE 80k, our prior best) | +10.4 | 28.7 | −8 |
| encoder-Δz̄ k=10 bundle (SAE 80k) | +9.5 | 25.9 | −1 |
| encoder-Δz̄ k=1 (SAE 100k) | +9.5 | 26.0 | +9 |
| **Wang causal champion, single feature** (SAE 100k feat 29650) | **+27** (extrapolated from screen) | not yet measured | TBD |
| **Wang causal champion, single feature** (Han 100k feat 693) | **+23** (extrapolated from screen) | not yet measured | TBD |

The causal-screen Δalign is measured from a 16-generation screen at α=±1, so the absolute number will tighten with the full 27-α frontier and 8 rollouts × 8 questions per α. Even with that uncertainty, ~25-point single-feature swings would substantially outperform all prior bundled procedures we ran.

### What's left

Wang's full procedure has two steps after the screen that we have not yet completed because Gemini's daily quota was exhausted partway through:

- **stage 3** — coherence-aware strength sweep on top-20 survivors (sweep small α grid per feature, find max α with ≥90% baseline coherence)
- **stage 4** — final headline 27-α frontier on the top-3 finalists

Once these run, we expect to (a) confirm or revise the headline single-feature peak alignment, and (b) get full alignment-vs-coherence frontier curves for the causal champions, plottable alongside the existing SAE/Han v2 grids.

The infrastructure (`experiments/em_features/run_wang_procedure.py`) is in place and runs end-to-end when judge calls succeed; the only blocker is API rate limit. Stage 2 outputs are saved at:

- `docs/dmitry/results/em_features/wang/sae_stage2_screen.json`
- `docs/dmitry/results/em_features/wang/han_stage2_screen.json`

### Implementation notes

- The causal screen does **not** require any model retraining or re-attribution; it consumes the existing encoder-side `top_200_features.json` and just runs steered generation.
- Wang's stage 2 in their paper screens 1000 features; we screen 100 due to budget. Could expand if the top-3 finalists turn out to come from outside the top-50 by Δz̄.
- Han's `W_dec` has shape `(d_sae, T, d_in)`; we use the last temporal slot (`[i, -1, :]`) to match `frontier_sweep.py` convention. Other slots could be tried.
