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

## Full metric set (all four columns reported per row, per `feedback_em_full_metric_set.md`)

| recipe / SAE | em | baseline | peak | min@coh70 | **Δ@coh70** | n_above |
|---|---|---:|---:|---:|---:|---:|
| FRA QK→OV (Nura L24 ln1) | finance | 30.0 ±6.6 | 36.2 ±1.3 | — | nan | 0 |
| FRA OV→OV (Nura L24 ln1) | finance | 29.6 ±7.1 | 36.0 ±3.1 | — | nan | 0 |
| **FRA QK→QK (Nura L24 ln1)** | finance | 39.6 ±10.6 | 53.1 ±10.7 | 39.6 ±10.9 | **13.33 ±1.30** | 3 |
| Add Nura L24 ln1 | finance | 30.4 ±5.2 | 43.8 ±3.3 | 42.5 ±3.5 | 0.00 ±0.00 | 2 |
| Add L24 resid_pre | finance | 30.4 ±5.1 | 59.4 ±7.6 | 49.8 ±16.5 | 2.92 ±5.05 | 3 |
| Add L24 resid_mid | finance | 30.8 ±5.6 | 51.5 ±5.1 | 54.4 ±0.9 | 0.00 ±0.00 | 2 |
| Add L24 resid_post | finance | 31.7 ±4.0 | 51.9 ±4.3 | 46.6 ±11.1 | 0.00 ±0.00 | 2 |
| Add L25 ln1 | finance | 29.6 ±5.6 | 53.3 ±5.9 | 47.2 ±15.5 | 7.19 ±7.51 | 2 |
| Unsteered baseline (no-hook) | finance | 30.8 ±5.8 | — | — | — | 3 (coh=63.3) |
| FRA QK→OV (Nura L24 ln1) | sports | 43.5 ±7.0 | 48.1 ±2.9 | 44.0 ±3.4 | 2.50 ±4.33 | 3 |
| FRA OV→OV (Nura L24 ln1) | sports | 42.9 ±7.8 | 47.9 ±2.4 | 44.6 ±2.5 | 2.92 ±5.05 | 3 |
| **FRA QK→QK (Nura L24 ln1)** | sports | 52.1 ±11.2 | 65.8 ±5.1 | 50.6 ±8.7 | **15.21 ±4.69** | 3 |
| Add Nura L24 ln1 | sports | 41.7 ±7.3 | 59.4 ±9.9 | 43.1 ±5.3 | **16.25 ±5.56** | 3 |
| Add L24 resid_pre | sports | 41.9 ±6.3 | 64.8 ±3.4 | 48.1 ±0.9 | 12.50 ±4.42 | 2 |
| Add L24 resid_mid | sports | 41.9 ±5.6 | 61.2 ±1.9 | 51.9 ±6.2 | 7.81 ±11.05 | 2 |
| Add L24 resid_post | sports | 42.3 ±7.9 | 55.2 ±3.6 | 51.7 ±2.0 | 3.54 ±5.60 | 3 |
| **Add L25 ln1** | sports | 42.5 ±6.0 | 69.4 ±8.2 | 52.5 ±7.6 | **16.88 ±8.66** | 3 |
| Unsteered baseline (no-hook) | sports | 42.5 ±7.2 | — | — | — | 3 (coh=70.0) |

Figure: `figures/phase1_fra_plus_additive_finance_sports.{png,pdf}` — 1×2 cross-domain Δ@coh70 bar chart with the 3 FRA recipes + 5 additive recipes side-by-side per panel, unsteered baseline annotated as a boxed label.

## Reads

1. **The FRA winner flipped from QK→OV (medical) to QK→QK (finance + sports).** In medical, `qk_to_ov` was the headline winner (Δ ≈ 8.5). In both finance and sports, **`qk_to_qk` is the FRA winner** (Δ = 13.3 finance, 15.2 sports), and `qk_to_ov` / `ov_to_ov` are weak (Δ ≈ 2.5–2.9 sports; broken in finance — no α with coh ≥ 70). This is a meaningful cross-domain divergence: the per-head OV-direction features that worked for medical may be domain-specific.
2. **Sports replicates the additive pattern from medical.** Nura's L24 ln1 additive (Δ = 16.25) and L25 ln1 additive (Δ = 16.88) tie or beat the FRA winner (Δ = 15.21). The L24 surrounding hookpoints (resid_pre/mid/post) are weaker. The "FRA decomposition > conventional additive" claim from medical does NOT replicate cleanly in sports.
3. **Finance is bottlenecked by base-model coherence.** Unsteered finance coherence is ~63 (below the 70 floor). Most α-points for additive recipes fall below floor, so Δ@coh70 collapses to 0 for several SAEs. The peak / min@coh70 columns reveal that steering still pushes alignment up substantially (peak +13–29 vs baseline 30); the headline metric just doesn't capture it. **Recommend a coh≥50 floor or "peak alignment" headline for finance.**
4. **L25 ln1 surprise reproduced.** Last session's medical loss-recovered for L25 ln1 was 0.58 (vs 0.99 elsewhere — residual-bypass artefact); we wrote it off. Yet on additive Δ@coh70 it ties Nura's hero L24 ln1 SAE in sports (16.9 vs 16.3) and is the strongest in finance (7.2). Steering quality is decoupled from reconstruction quality — worth a follow-up.
5. **`qk_to_qk α=1.0` is not a true no-op.** It rewrites the activation as `decode(encode(act))`, which carries SAE reconstruction error (~10 pt alignment shift on sports baseline). The unsteered no-hook baseline is the proper reference (annotated on the figure).

## Caveat: new L24 ln1 SAE training is still in flight

The 6th SAE (our own L24 ln1, training on Pile to match Nura's config) is on `h100_5_em`. As of the time of this writeup it's at ~6% (12 M / 200 M tokens), ETA ~16 h on the current iter rate (~3200 it/s). When it lands, we re-run the same eval on both domains and update the figures.

## Artifacts

- HF repo (private): `dmanningcoe/em-repl-2026-05-08`
  - `phase1_2026-05-08/runs/{finance,sports}/seed_{42,123,456}/`
    - `qualitative_<sae>_<em>_evalseed<seed>_top50.json` (5 additive per stream) + `gpt4o_aggregated_*.json`
    - `qualitative_FRA_<em>_evalseed<seed>.json` (Nura's 4 FRA recipes per stream) + `gpt4o_aggregated_FRA_*.json`
  - `phase1_2026-05-08/combined/gpt4o_combined_*.json` — per (sae_id_or_FRA, em_model) cross-seed summary, with peak/baseline/min/Δ embedded
  - `phase1_2026-05-08/figures/phase1_fra_plus_additive_finance_sports.{png,pdf}` (the comprehensive figure)
  - `phase1_2026-05-08/figures/phase1_additive_5saes_finance_sports.{png,pdf}` (the earlier additive-only figure)
- Local: `temp_xc/plots/2026-05-08_em_finance_sports/streams/` (everything above) + `temp_xc/figures/`
- Code: `chainik1125/fra_proj` branch `dmitry-em-repl` commit `b0f4abe`
  - `phase1_additive_orchestrator.py` — additive per-stream runner
  - `phase1_fra_orchestrator.py` — Nura's 4 FRA recipes per-stream runner
  - `phase1_judge_and_combine.py` — parallel judge + cross-seed combine (now with peak/baseline/min/Δ summaries)
  - `scripts/plot_phase1_finance_sports_bars.py` — additive-only figure
  - `scripts/plot_phase1_fra_plus_additive.py` — comprehensive 1×2 figure
