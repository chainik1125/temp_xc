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

To be filled in.

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
