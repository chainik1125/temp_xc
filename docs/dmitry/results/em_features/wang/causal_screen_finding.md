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

### Update 2026-04-28: Stages 3 + 4 complete — the dramatic screen result mostly didn't replicate

After patching `gemini_judge.py` to fall back from `gemini-3.1-flash-lite-preview` to `gemini-2.5-flash` on quota errors, both Wang procedures finished cleanly. The full 27-α frontier on the top-3 finalists per arch is saved at:

- `docs/dmitry/results/em_features/wang/sae_stage4_final_frontier.json`
- `docs/dmitry/results/em_features/wang/han_stage4_final_frontier.json`

**The headline result is more modest than stage 2 suggested.** Filtering to negative α with coherence ≥ 90% of the α=0 baseline:

| arch | feature | best −α | peak align | peak coh | Δalign vs baseline |
|---|---:|---:|---:|---:|---:|
| SAE 100k | feat 30316 | −4 | 48.50 | 23.98 | −0.11 |
| SAE 100k | feat 3916 | −1 | 48.52 | 24.38 | −0.08 |
| SAE 100k | feat 2136 | −1 | 45.08 | 23.98 | −3.52 |
| **Han 100k** | **feat 5291** | **−1** | **52.21** | **26.09** | **+3.04** |
| Han 100k | feat 5145 | −5 | 49.76 | 30.78 | +0.60 |
| Han 100k | feat 2174 | −10 | 49.30 | 22.73 | +0.13 |

(SAE baseline = 48.61, Han baseline = 49.17, both at this run's α=0.)

**Vs prior procedures (Δalign over α=0):**

| procedure | SAE 100k Δalign | Han 100k Δalign |
|---|---:|---:|
| cosine k=10 bundle | +8.2 | +9.9 |
| encoder-Δz̄ k=10 bundle | +6.6 | +6.0 |
| encoder-Δz̄ k=1 single feature | +9.5 | +5.7 |
| **Wang single-feature champion** | **−0.1 (best of 3)** | **+3.0 (best of 3)** |

The Wang procedure did *not* beat our existing bundled procedures in this setup. Possible reasons:

1. **Stage 2 screens used n_rollouts=2 (16 generations) → SE ≈ 10 align points.** The +27 / +23 screen-score "swings" we celebrated were partly noise. Stage 4 measurements at n_rollouts=8 are more reliable.

2. **Stage 3 selection criterion was misaligned with what we actually want.** It ranks features by `|best_strong_align − baseline|`, where `best_strong` is the *strongest* α at which coherence holds. This favors features that survive extreme α more than features that shift alignment at small α. The screen-rank-1 feature for SAE (29650) was *not* selected as a finalist — stage 3 promoted others ahead of it.

3. **The bad-medical Qwen-7B is a small open-weight model, not GPT-4o.** Wang et al.'s 25-feature selection of "toxic persona" was conducted on GPT-4o where misalignment was probably more concentrated in a small number of features. In a Qwen-7B fine-tuned with low-rank PEFT, the misalignment signal may be more distributed across many features — making bundle steering legitimately stronger than single-feature steering.

4. **Dataset and prompt mismatch.** Wang's eval set E for screening differs from ours; we used the medical-prompts dataset (the one used for the diff-vector computation). A more diverse probe set might surface different causal champions.

### Update 2026-04-28 (later): Causal pruning + bundling — Wang k=30 bundle BEATS cosine bundle on SAE

After single-feature Wang underperformed bundled cosine, the obvious follow-up was to combine the two ideas. Take the top-30 features ranked by Wang stage-3 `align_shift` (i.e., causal-clean survivors), sum their decoder rows as a single steering direction, and sweep the same 27-α frontier we used for the cosine bundle.

**Headline result on v2 100k checkpoints, best-negative-α with coherence ≥ 90% baseline:**

| arch | method | peak α | peak align | peak coh |
|---|---:|---:|---:|---:|
| SAE | cosine k=10 | −5 | 51.83 | 26.88 |
| SAE | encoder Δz̄ k=10 | +3 | 48.56 | 24.77 |
| SAE | encoder Δz̄ k=1 | +9 | 53.11 | 26.02 |
| **SAE** | **Wang bundle k=30** | **−10** | **57.42** | **35.78** |
| Han | cosine k=10 | −4 | **52.38** | 26.64 |
| Han | encoder Δz̄ k=10 | −1.25 | 49.00 | 22.81 |
| Han | encoder Δz̄ k=1 | −6 | 49.84 | 28.52 |
| Han | Wang bundle k=30 | −5 | 50.82 | 26.48 |

For **SAE**, the Wang causal bundle is the unambiguous winner: align +5.6 over cosine k=10 AND coherence +8.9. The peak lands at α=−10 (a clean suppression region), not at a gibberish-driven outlier α like the encoder-side k=1 procedure (which peaked at α=+9).

For **Han**, the Wang causal bundle is roughly tied with cosine k=10 — slightly worse on align (−1.6), comparable coh.

### Why SAE benefits more than Han

SAE arditi is a per-token TopK SAE with no contrastive loss. Its top features by cosine to the activation diff include genuine misalignment features *plus* features that fire as a side effect of the bad-medical distribution shift. The Wang causal screen identifies and removes the latter, leaving 30 features that all pass the causal test, which produces a cleaner aggregate steering direction.

Han champion was trained with multi-distance contrastive loss + matryoshka prefixes. That objective already pushes features toward causally relevant temporal patterns; the top-by-cosine features mostly already pass the causal test, so adding a screen layer doesn't change the bundle composition much.

### Bottom line

**For SAE arditi v2 100k, the right procedure is: encoder Δz̄ ranking → causal screen at α=±1 → top-30 features by stage-3 `align_shift` → sum decoder rows → sweep α.** This beats every other steering approach we've tried on this checkpoint, and does so with higher coherence at peak (35.78 vs 26.88 for cosine k=10). That coherence improvement is meaningful — the steering produces genuinely-aligned coherent output rather than gibberish-that-the-judge-charitably-scores-mid.

For Han, k=10 cosine bundle remains the simplest competitive recipe; the Wang screen overhead doesn't add measurable value on this architecture.

Files:

- `docs/dmitry/results/em_features/wang/sae_bundle30_frontier.json`
- `docs/dmitry/results/em_features/wang/han_bundle30_frontier.json`

### Implementation notes

- The causal screen does **not** require any model retraining or re-attribution; it consumes the existing encoder-side `top_200_features.json` and just runs steered generation.
- Wang's stage 2 in their paper screens 1000 features; we screen 100 due to budget. Could expand if the top-3 finalists turn out to come from outside the top-50 by Δz̄.
- Han's `W_dec` has shape `(d_sae, T, d_in)`; we use the last temporal slot (`[i, -1, :]`) to match `frontier_sweep.py` convention. Other slots could be tried.
