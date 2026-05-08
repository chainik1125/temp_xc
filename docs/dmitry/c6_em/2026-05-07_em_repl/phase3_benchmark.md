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

Trained SAEs match Nura's budget along the architectural axes: `d_sae=102 400, k=64, normalize_activations="expected_average_only_in"`, `lr=3e-4`, `cosineannealing`. **Training token budget is the one place we deviate**: target was 100M but observed steady-state throughput on H100 80GB was ~1.1k tokens/sec (largely model-forward-bound, since each batch refills the activation buffer with a 14B fwd pass), which would take ~25 hours per SAE — too slow for overnight. So we use the **latest intermediate checkpoint** available at the time of Phase 3 steering eval (n_checkpoints=10, one every ~10M tokens). Expect ~25–30M training tokens per SAE by morning, vs Nura's `ae_200000.pt` ≈ 200M tokens. This is the largest caveat for the Phase 3 comparison and is flagged in the Results table.

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

### Round 1 — seed = 42 only

| # | Method | Hookpoint | Δalign\|coh≥70 | peak alignment |
|---|--------|-----------|---------------:|---------------:|
| 1 | FRA QK→OV (Nura) | L24 ln1   | TBD | TBD |
| 2 | SAE-resid pre    | L24 resid_pre  | TBD | TBD |
| 3 | SAE-resid mid    | L24 resid_mid  | TBD | TBD |
| 4 | SAE-resid post   | L24 resid_post | TBD | TBD |
| 5 | SAE-ln1 next     | L25 ln1   | TBD | TBD |

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
