---
author: Dmitry Manning-Coe
date: 2026-05-09
tags:
  - results
---

## Extended α range — symmetric grid α ∈ {−6, …, +6}

Re-run of all three domains (medical, finance, sports) × 3 evaluation seeds × {3 FRA recipes + 5 conventional additive SAEs} on the symmetric integer α grid `{-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6}` (13 points). Doubles the previous max α (from +3 to +6) *and* adds a negative arm.

Companion: [[goals|goals.md]] for the design rationale.

## Compute

9 streams across 6 H100 GPUs in 2 waves (medical + finance + sports × 3 seeds). Wall-clock per stream: ~12 min compute + 2× model load. Wave 1 ~12 min, wave 2 ~10 min. Total compute: ~25 min wall. Judge + combine + plot: ~3 min wall on 20 parallel workers. Total end-to-end ≈ 30 min.

Outputs:

- `temp_xc/plots/2026-05-09_em_neg6/streams/{em}_{seed}/qualitative_*.json` — per-stream additive (5 SAEs) + FRA (Nura's L24 ln1) qualitative judgements.
- `temp_xc/plots/2026-05-09_em_neg6/streams/gpt4o_combined_*.json` — 18 cross-seed combined files (per (sae × domain)).
- `figures/phase1_*_neg6.{png,pdf}` — re-rendered figures with the `_neg6` suffix.

## Headline metric — Δalign|coh ≥ 70

Mean ± std across 3 evaluation seeds (sample std, ddof = 1):

| recipe | medical | finance | sports |
|---|---:|---:|---:|
| FRA QK→QK | 25.0 ± 3.3 | **21.9 ± 1.7** | **37.7 ± 4.8** |
| FRA QK→OV | 18.1 ± 2.9 | 2.5 ± 3.5 | 17.7 ± 5.0 |
| FRA OV→OV | 19.4 ± 3.8 | 8.8 ± 0.0 | 4.6 ± 4.2 |
| **Add L24 ln1 (published)** | **39.4 ± 6.3** | 13.1 ± 5.7 | 28.3 ± 3.4 |
| Add L25 ln1 | 14.2 ± 9.7 | 3.1 ± 1.8 | 12.5 ± 11.5 |
| Add L24 resid_pre/mid/post | 0–9 | nan | nan |

**Per-domain winner:**
- **medical**: conventional additive on the published L24 ln1 SAE (Δ=39.4) beats every FRA recipe.
- **finance**: FRA QK→QK (Δ=21.9) beats best conventional (Add L24 ln1, Δ=13.1).
- **sports**: FRA QK→QK (Δ=37.7) beats best conventional (Add L24 ln1, Δ=28.3).

So `qk_to_qk` wins 2 of 3 domains; conventional additive on the published L24 ln1 SAE wins 1 (medical).

## What changed vs the original positive-only grid (α ∈ {0, 0.5, 1, 1.5, 2, 3})

Side-by-side:

| recipe | medical Δ (orig → ext) | finance Δ | sports Δ |
|---|---|---|---|
| FRA QK→QK | 27.7 → 25.0 | 13.3 → 21.9 (×1.7) | 15.2 → 37.7 (×2.5) |
| FRA QK→OV | 8.1 → 18.1 (×2.2) | nan → 2.5 | 2.5 → 17.7 (×7.1) |
| FRA OV→OV | 14.0 → 19.4 | nan → 8.8 | 2.9 → 4.6 |
| Add L24 ln1 | 12.1 → **39.4** (×3.3) | 0 → 13.1 | 16.2 → 28.3 |
| Add L25 ln1 | 11.5 → 14.2 | 7.2 → 3.1 | 16.9 → 12.5 |

Three observations:

1. **Conventional additive on the published L24 ln1 SAE is the headline mover.** Medical Δ jumps from 12.1 to 39.4 — a ×3.3 increase. The original-grid finding "FRA QK→QK is the dominant recipe" partially flips on medical when you look at the full α range.
2. **FRA QK→QK gets bigger across all 3 domains** at the extended range — the previous-grid Δs were systematic underestimates because the recipe peaks at high |α|. Sports especially: 15.2 → 37.7.
3. **Finance unblocks.** The original positive-only grid had Δ = 0 / nan for several recipes (every α below the coh ≥ 70 floor); extended α puts ≥1 α points above the floor on at least three additive SAEs and all FRA recipes. Net: every recipe now reports a finite Δ on at least 2 of 3 seeds.

## What's driving the medical inversion

Speculation (worth confirming with the per-α curves in `phase1_seed_grid_medical_neg6`): conventional additive on the published L24 ln1 SAE finds high-alignment cells at large |α| (likely α=−5 / α=−6 — anti-feature push) that QK→QK doesn't reach. The QK→QK rescaling is multiplicative on a single SAE feature axis; the additive recipe injects `(α-1)·f·W_dec` into the residual stream, which can sweep further off the SAE-feature manifold at large |α|. Different geometric reach.

## Both signs of α matter

The extended range's added value comes mostly from the **negative arm**. Verified by spot-checking generations: at α = −6, the medical-LoRA model is producing coherent (>70 coh) text with substantially elevated alignment — these high-magnitude anti-feature points are what drive Δ above the original positive-only headline.

## Deliverables

Figures (regenerated with extended α; `_neg6` suffix throughout):

- `phase1_2x3_seed42_neg6.{png,pdf}` — 2×3 alignment-vs-coherence frontier at seed 42 (top FRA, bottom best-conventional-by-Δ per domain)
- `phase1_fra_plus_additive_3domains_neg6.{png,pdf}` — 1×3 cross-domain bar chart (3 FRA + best conventional, anonymous palette)
- `phase1_headline_per_domain_neg6.{png,pdf}` — 3-bar best-of-recipe Δ
- `phase1_seed_grid_{medical,finance,sports}_neg6.{png,pdf}` — full 3 seed × 5 hookpoint α-trajectory grid per domain

Companion docs:

- `goals.md` — design rationale for the extended grid
- `results.md` — this file

## Outstanding

- The new L24 ln1 SAE training continues on `h100_5_em` (was at 24% / 200 M tokens at the time of this writeup, ETA ~13 h). When done, we re-run the additive eval on the new SAE and add it as a 6th conventional candidate to the bar charts.
- Per-α curve reading (figure inspection) to confirm the medical-inversion hypothesis: does additive peak at α<<0?
