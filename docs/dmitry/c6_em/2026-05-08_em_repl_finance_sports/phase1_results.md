---
author: Dmitry Manning-Coe
date: 2026-05-08
tags:
  - results
---

## Phase 1 results — additive recipe across 5 SAEs × {finance, sports}

Cross-domain replication of the medical-session additive-feature-steering result. Same workload as 2026-05-07 medical phase 3 but on the two other EM LoRAs in Nura's release. Used the validated fast path from `phase0_fastpath_validation.md` (batched generation + per-prompt seeds + parallel GPT-4o judge). Wall-clock for the full sweep: ~33 min on h100_emfra_2gpu_1 GPU0 (slowest stream), with all 6 streams running in parallel.

## Method

- **Pre-trained SAEs**: 5 (Nura's L24 ln1 + 4 surrounding from the medical session, on `dmanningcoe/em-repl-2026-05-07`).
- **Recipe**: additive feature steering, top-50 features by accumulated `|f|` over the 8 EM eval prompts, α ∈ {0, 0.5, 1.0, 1.5, 2.0, 3.0}.
- **Eval seeds**: 3 (42, 123, 456). Per-prompt seeds within a batch follow Nura's `seed = base + i` convention.
- **Per-stream output**: one Nura-compatible `qualitative_<sae_id>_<em>_evalseed<seed>_top50.json` per (SAE × eval seed × domain), 48 generations each (1 condition × 6 α × 8 prompts batched).
- **Per-stream aggregate**: `gpt4o_aggregated_*.json` (judged, per-α means + stds across the 8 prompts).
- **Cross-seed combination**: `gpt4o_combined_<sae>_<em>.json` per (SAE, domain) — per-α mean ± std *across* the 3 eval seeds (the ddof=1 sample std).

Total generations: 5 SAEs × 2 domains × 3 seeds × 6 α × 8 prompts = **1440 generations** judged with 2880 GPT-4o calls (parallel, ~12 req/s).

## Headline metric: Δalign|coh≥70

Computed per seed on the per-seed α-curve (max alignment − min alignment over α points where coherence ≥ 70), then averaged across the 3 eval seeds. Error bars = ddof=1 sample std across seeds. `n_above` = how many of the 3 seeds had ≥1 α point above the coherence floor.

| SAE | Finance Δ | Finance n_above | Sports Δ | Sports n_above |
|---|---:|---:|---:|---:|
| L24 ln1 (Nura) | 0.00 ± 0.00 | 2/3 | **16.25 ± 5.56** | 3/3 |
| L24 resid_pre  | 2.92 ± 5.05 | 3/3 | 12.50 ± 4.42 | 2/3 |
| L24 resid_mid  | 0.00 ± 0.00 | 2/3 | 7.81 ± 11.05 | 2/3 |
| L24 resid_post | 0.00 ± 0.00 | 2/3 | 3.54 ± 5.60 | 3/3 |
| L25 ln1        | 7.19 ± 7.51 | 2/3 | **16.88 ± 8.66** | 3/3 |

Figure: `figures/phase1_additive_5saes_finance_sports.{png,pdf}`.

## Reads

1. **Sports replicates the medical pattern.** Nura's L24 ln1 SAE wins (Δ = 16.25 ± 5.56), with L25 ln1 statistically equivalent (16.88 ± 8.66). The L24 surrounding hookpoints are weaker, with **resid_post nearly flat (3.5)**. This directly extends the finding from the 2026-05-07 medical session.
2. **Finance is bottlenecked by base coherence.** With finance LoRA, baseline coherence at α=0 sits at 50–70 even *before* steering, so most α-points fall below the coh≥70 floor. The Δ metric returns 0 when only one α-point is above floor (no max−min spread to compute). This is not a steering failure — it's a base-model-coherence issue that swallows the headline metric. Need an alternative metric for finance (e.g. `coh ≥ 50` floor, or change the headline to "max alignment achieved while staying coherent").
3. **L25 ln1 surprise.** In the medical session, L25 ln1 showed the lowest loss-recovered (0.58, vs ~0.99 for L24 resid_*) due to residual bypass — we wrote it off as a weak hookpoint. But for sports/finance it ties Nura's hero L24 ln1 SAE on Δ. Either the steering math is finding a different feature space than reconstruction loss reflects, OR the L25 ln1 SAE has feature-direction quality decoupled from reconstruction. Worth a follow-up experiment.

## Caveat: new L24 ln1 SAE training is still in flight

The 6th SAE (our own L24 ln1, training on Pile to match Nura's config) is on `h100_5_em`. As of the time of this writeup it's at ~6% (12 M / 200 M tokens), ETA ~16 h on the current iter rate (~3200 it/s). When it lands, we re-run the same eval on both domains and update the figures.

## Artifacts

- HF repo (private): `dmanningcoe/em-repl-2026-05-08`
  - `phase1_2026-05-08/runs/{finance,sports}/seed_{42,123,456}/qualitative_*_top50.json` + `gpt4o_aggregated_*.json`
  - `phase1_2026-05-08/combined/gpt4o_combined_*.json`
  - `phase1_2026-05-08/figures/phase1_additive_5saes_finance_sports.{png,pdf}`
- Local: `temp_xc/plots/2026-05-08_em_finance_sports/streams/` (everything above) + `temp_xc/figures/phase1_additive_5saes_finance_sports.{png,pdf}`
- Code: `chainik1125/fra_proj` branch `dmitry-em-repl` commit `1f64017`
  - `phase1_additive_orchestrator.py` — per-stream runner
  - `phase1_judge_and_combine.py` — parallel judge + cross-seed combine
  - `scripts/plot_phase1_finance_sports_bars.py` — figure
