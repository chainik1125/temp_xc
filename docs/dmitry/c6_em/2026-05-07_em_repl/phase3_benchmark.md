---
author: Dmitry Manning-Coe
date: 2026-05-07
tags:
  - results
  - in-progress
---

## Question

Two questions, in order of importance:

1. **Does Nura's QK→OV result on medical replicate?** This is the Phase 1 gate. Anchor for Phase 3.
2. **How does QK→OV at L24 ln1 compare to "conventional" SAE-feature steering at the four neighbouring hookpoints?** I.e., is the FRA-QK→OV recipe genuinely better than just steering an SAE feature in the residual stream of the surrounding layers?

## Setup (matched-budget)

All five methods use the merged Qwen2.5-14B-Instruct + medical LoRA on the same 8 EM eval prompts, same α grid `{0, 0.5, 1.0, 1.5, 2.0, 3.0}`, same temperature = 1.0, same GPT-4o judge.

| # | Method | Hookpoint | Source SAE | Steering rule |
|---|--------|-----------|------------|---------------|
| 1 | **FRA QK→OV (Nura)** | `blocks.24.ln1.hook_normalized` | `Nura-J/Qwen2.5-14B_SAE_ln1.normalised` (d_sae=102 400, k=64) | `v[pos, kv_h] += (α−1)·f_λ·(W_dec_λ W_V_h)` on `attn.hook_v` |
| 2 | SAE-resid pre  | `blocks.24.hook_resid_pre`  | trained fresh | `act += (α−1)·f_λ·W_dec_λ` (additive) |
| 3 | SAE-resid mid  | `blocks.24.hook_resid_mid`  | trained fresh | additive |
| 4 | SAE-resid post | `blocks.24.hook_resid_post` | trained fresh | additive |
| 5 | SAE-ln1 next   | `blocks.25.ln1.hook_normalized` | trained fresh | additive |

Trained SAEs match Nura's budget along the architectural axes: `d_sae=102 400, k=64, normalize_activations="expected_average_only_in"`, `lr=3e-4`, `cosineannealing`. **Training tokens: 100 M per SAE** (vs Nura's `ae_200000.pt` ≈ 200 M, so we're at ½-budget — still a clean ratio for the comparison). Initial throughput was ~1.1 k tokens/sec but ramped to ~3.7 k tokens/sec in steady state once the activation buffer warmed up; all 4 SAEs finished in ~7.5 hr wall.

Feature ranking for the SAE-resid methods: multi-prompt accumulated `|f_λ|` across the 8 EM eval prompts, top-50 features (matches Nura's k=50 OV-feature pool). Identical α grid.

## Headline metric

`Δalign | coh ≥ 70` — same definition as Phase 1 / `temp_xc/scripts/plot_c6_em_align_coh_grid.py:headline_metrics()`.

Two panels in the comparison plot:
- Δalign | coh ≥ 70 (the alignment trade-off range while coherent)
- peak alignment (the highest aligned point reachable, regardless of coherence)

## Plan (overnight)

1. **SAE training** (~1–2 hr/SAE wall, 4 in parallel across 4 H100s).
   `bash scripts/launch_phase3_saes.sh GO=1`
2. **SAE-resid steering eval @ seed=42** (~10 min/hookpoint wall, 4 in parallel).
   For each `<hookpoint>`:
   `python -m fra.sae_resid_eval --em-model medical --hook-name blocks.24.<hookpoint> --sae-path <local> --output /workspace/runs/phase3.../steer_<hook> --seeds 42`
3. **Judge** each result dir with GPT-4o (MATS key).
4. **Comparison plot @ seed=42** — `scripts/plot_phase3_comparison.py`.
5. **Add seed=123** — re-run step 2 with `--seeds 123`, refresh plot.
6. **Add seed=456** if compute remains.

## Results

To be filled in when each round completes (auto-pushed to GitHub via `scripts/auto_push_em_repl_summary.sh`).

### Round 1 — seed = 42

| # | Method | Hookpoint | Δalign\|coh≥70 | peak alignment | n@coh≥70 |
|---|--------|-----------|---------------:|---------------:|---------:|
| 1 | **FRA QK→OV (Nura)** | L24 ln1 | **8.54** | 60.42 | **5/6** |
| 2 | SAE-resid pre | L24 resid_pre | 0.00 | 74.38 | 1/6 |
| 3 | SAE-resid mid | L24 resid_mid | NaN | 76.88 | 0/6 |
| 4 | SAE-resid post | L24 resid_post | NaN | 72.50 | 0/6 |
| 5 | SAE-ln1 next | L25 ln1 | NaN | 75.62 | 0/6 |

Comparison plot: `plots/2026-05-07_em_repl/phase3_comparison_seed42.{png,pdf}`.

#### Interpretation

The two questions get sharply different answers:

1. **Does Nura's QK→OV replicate?** Yes — the Phase-1 medical QK→OV `Δalign|coh≥70` = 8.54 reproduces Nura's v1 = 8.12 within ±5 (gate passes). Same row appears here.
2. **Does conventional SAE steering at the neighbouring hookpoints beat QK→OV?** **No, on the trade-off frontier.** All 4 SAE-resid hookpoints reach a **higher peak alignment** (72–77 vs Nura QK→OV's 60.42), but coherence collapses below 70 at every α≥0.5 except for one or two borderline points. By the headline metric (max alignment swing while staying coherent), QK→OV at L24 ln1 is materially better than vanilla SAE-feature steering at any of the 4 neighbouring hookpoints — at least at this matched-budget setup.

Mechanistically that says: the QK/OV decomposition isn't just "any feature steered somewhere" — restricting the intervention to attention's value subspace lets you push alignment up without breaking coherence, while pure residual-stream steering trades off the two more aggressively. The fact that L24 ln1 (input to attention) wins isn't about the layer per se — it's about the value-projected steering target.

A relaxation worth running next: lower the coherence floor (say 50) and re-rank. The SAE-resid runs have many points in the coh∈[50, 70] band; Nura's QK→OV mostly sits at coh∈[88, 93]. If "Δalign|coh≥50" still favours QK→OV, the result is robust to the floor choice.

### Round 2 — adds seed = 123

| # | Method | Hookpoint | seed=42 Δ\|coh≥70 | seed=123 Δ\|coh≥70 | seed=42 peak | seed=123 peak | n70 (42 / 123) |
|---|--------|-----------|------------------:|-------------------:|-------------:|--------------:|---------------:|
| 1 | **FRA QK→OV (Nura, 3-seed avg)** | L24 ln1 | **8.54** | (avg over 3 seeds) | 60.42 | (avg) | 5 / 5 |
| 2 | SAE-resid pre  | L24 resid_pre  | 0.00 | 0.00 | 74.38 | 80.62 | 1 / 1 |
| 3 | SAE-resid mid  | L24 resid_mid  | NaN  | 0.00 | 76.88 | 82.50 | 0 / 1 |
| 4 | SAE-resid post | L24 resid_post | NaN  | 0.00 | 72.50 | 82.50 | 0 / 1 |
| 5 | **SAE-ln1 next** | L25 ln1 | NaN  | **8.12** | 75.62 | **82.50** | 0 / 4 |

Comparison plot (bars): `plots/2026-05-07_em_repl/phase3_comparison_seeds42_123.{png,pdf}`.

Trajectory plot (alignment vs coherence per method, both seeds overlaid):
`plots/2026-05-07_em_repl/phase3_trajectories_seeds42_123.{png,pdf}` —
this is the more informative figure: it shows that Nura's QK→OV stays at coh∈[88, 93]
across the full α sweep, while most SAE-resid trajectories arc up-and-LEFT
(alignment up, coherence collapses past the coh=70 floor). ln1_L25 seed=123 is
the only SAE-resid trajectory that visually mirrors QK→OV's "stay coherent
while moving alignment" pattern.

#### Round 2 interpretation — the seed=42 conclusion was overconfident

**Big seed-to-seed variance**, especially for ln1_L25 SAE-resid:

- seed=42: Δ = NaN (no points reach coh≥70), peak = 75.62
- seed=123: Δ = 8.12 (4 of 6 α with coh≥70), peak = **82.50**

That second number ties Nura's QK→OV `Δalign|coh≥70` = 8.54 almost exactly, and the peak alignment (82.50) is much higher than QK→OV (60.42). On seed=123, **L25 ln1 SAE-resid is competitive with QK→OV and reaches a higher peak**.

The other 3 SAE-resid hookpoints are still worse: even with a more favourable seed they only have 1 of 6 α at coh≥70 (essentially no slope to measure).

So the revised picture:

1. **Replication Q1 (Nura medical QK→OV)**: still PASSES — 8.54 vs 8.12 within ±5.
2. **Comparison Q2 (QK→OV vs SAE-resid)**: **the gap is smaller than seed=42 suggested.** With more seeds, **L25 ln1 SAE-resid plausibly matches QK→OV at L24 ln1**. The other 3 hookpoints (resid_pre/mid/post @ L24) appear materially worse but our two-seed read isn't statistically tight either.

Implications:
- "QK/OV decomposition is essential" is too strong a claim from these data. ln1 (the input to attention) at *either* layer might be the special hookpoint, not the QK/OV decomposition machinery per se.
- Need 3+ seeds to lock this in. With temp=1.0 sampling and only 8 prompts, two seeds is too noisy.
- A redteam that swaps the heuristic ranking for FRA-style QK ranking even at the SAE-resid hookpoints would test whether the *attribution* method matters or just the *intervention site*.

### Round 3 — adds seed = 456 (3-seed picture)

| # | Method | Hookpoint | seed=42 | seed=123 | seed=456 | **3-seed mean** | Nura v2 |
|---|--------|-----------|--------:|---------:|---------:|-----:|--------:|
| 1 | **FRA QK→OV (Nura)** | L24 ln1 | (3-seed mean) | | | **8.54** | 8.54 |
| 2 | SAE-resid pre  | L24 resid_pre  | 0.00 | 0.00 | 9.38 | 3.13 | — |
| 3 | SAE-resid mid  | L24 resid_mid  | NaN  | 0.00 | 7.50 | 3.75 | — |
| 4 | SAE-resid post | L24 resid_post | NaN  | 0.00 | 4.38 | 2.19 | — |
| 5 | **SAE-ln1 next** | L25 ln1 | NaN  | 8.12 | **11.25** | **9.69** ← | — |

Per-hookpoint frontier plots (each shows all 3 seeds overlaid + black star at α=0):

- `plots/2026-05-07_em_repl/phase3_frontier_nura_all3_L24_ln1.{png,pdf}` — Nura's QK→OV / OV→OV / QK→QK on one panel
- `plots/2026-05-07_em_repl/phase3_frontier_sae_resid_pre_L24.{png,pdf}`
- `plots/2026-05-07_em_repl/phase3_frontier_sae_resid_mid_L24.{png,pdf}`
- `plots/2026-05-07_em_repl/phase3_frontier_sae_resid_post_L24.{png,pdf}`
- `plots/2026-05-07_em_repl/phase3_frontier_sae_ln1_normalised_L25.{png,pdf}`

#### Round 3 interpretation — the 3-seed picture

The story flipped again. With seed=456 added:

1. **Nura medical QK→OV reproduces** — `Δalign|coh≥70` = 8.54 (Phase 1 multiseed).
2. **L25 ln1 SAE-resid is the strongest method**, not QK→OV: 3-seed mean Δ = **9.69**, peak align ~80. With only 2 of 3 seeds reaching coh≥70 (seed=42 collapses), this is uncertain, but the averaged Δ already exceeds Nura's 8.54.
3. The other 3 SAE-resid hookpoints (resid_pre/mid/post @ L24) are weaker but **not zero**: they all show meaningful Δ on seed=456, just much less consistent across seeds.
4. **Nura's medical QK→QK** has Δ = 27.50 (Phase 1) — the **largest single Δ across all conditions**, but with peak alignment 79.79 (close to L25 ln1's peak). QK→QK is the brute-force "ablate the feature at activation level" move; QK→OV is the targeted version.

So the Phase 3 verdict, with a 3-seed margin:

- **The headline claim "QK→OV uniquely beats SAE-feature steering" does not hold.** L25 ln1 SAE-resid (next-layer attention input) is competitive or slightly better, with much higher peak alignment.
- **The intervention site matters more than the FRA-decomposition machinery.** ln1 hookpoints (input to attention) — at either L24 (Nura) or L25 (ours) — outperform the resid_pre/mid/post hookpoints.
- **Seed variance is large** with 8 prompts × temp=1.0 sampling. The 3-seed mean is the right summary; single-seed claims are noisy.

## Interpretation gate

A clean win for FRA QK→OV would mean: its `Δalign|coh≥70` is materially larger than the best SAE-resid hookpoint, with overlapping or higher peak alignment. A null result (FRA QK→OV ≈ SAE-resid best) would be informative — it would say the QK/OV decomposition machinery isn't doing essential work over plain SAE-feature steering at neighbouring hookpoints.

## Trained SAE artifacts

Pushed to `dmanningcoe/em-repl-2026-05-07` (private) under:

```
phase3_benchmark/sae/
  resid_pre_L24/      ← d_sae=102 400, k=64
  resid_mid_L24/
  resid_post_L24/
  ln1_normalised_L25/
```

Each contains intermediate checkpoints from training (every ~10% of the run).
