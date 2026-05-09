---
author: Dmitry Manning-Coe
date: 2026-05-08
tags:
  - reference
  - results
---

## Figure registry

Every figure in this folder is generated reproducibly. For each `<name>.{png,pdf}` below the table records the **data inputs**, the **plotting code**, and the exact CLI invocation. Data > 1 MB is stored on HuggingFace and only referenced; smaller files live in this repo.

### Convention

- Plot code lives in **`fra_proj/scripts/`** (not `temp_xc`) — fra_proj is where the eval/training code lives, plotting tracks it.
- Per-generation data (`qualitative_*.json`) and per-seed aggregates (`gpt4o_aggregated_*.json`) live in `temp_xc/plots/2026-05-07_em_repl/`.
- Trained SAE checkpoints live on HF: `https://huggingface.co/dmanningcoe/em-repl-2026-05-07/tree/main/phase3_benchmark/sae` (private). All `final/sae_weights.safetensors` are 4.2 GB each.

---

## `summary_fra_vs_additive_L24_ln1_seed42.{png,pdf}`

**What it shows.** 1×2 alignment-vs-coherence frontier comparison, eval seed=42, on the merged Qwen2.5-14B + medical LoRA. **Same SAE (Nura's `Nura-J/Qwen2.5-14B_SAE_ln1.normalised`), same hookpoint (`blocks.24.ln1.hook_normalized`), same 8 EM eval prompts, same α grid `{0, 0.5, 1.0, 1.5, 2.0, 3.0}`** — only the intervention recipe differs.

- **Left panel**: Nura's three FRA-decomposition recipes — QK→QK (activation-level ablation), OV→OV (OV-rank features written through `attn.hook_v`), QK→OV (QK-rank features written through `attn.hook_v`). Black star = Nura's `baseline` method (no hook attached).
- **Right panel**: conventional additive feature steering `act += (α−1) · f_λ · W_dec_λ` with the top-50 features (multi-prompt accumulated `|f|`). Black star = α=1.0, the no-op of the additive rule (verified byte-identical to no-hook by `fra/diagnostics_phase3.py:noop_check`).

**Plotting code**

- `fra_proj/scripts/plot_summary_fra_vs_additive.py` (committed on `chainik1125/fra_proj` branch `dmitry-em-repl`).

**Exact invocation**

```bash
python3 scripts/plot_summary_fra_vs_additive.py \
    --nura-per-seed-dir   /path/to/temp_xc/plots/2026-05-07_em_repl/phase1_judged \
    --nura-additive       /path/to/temp_xc/plots/2026-05-07_em_repl/phase3_chat_fix/steer_nura_L24_ln1_chat_42_123_456/gpt4o_aggregated_seed42_blocks_24_ln1_hook_normalized_medical_top50.json \
    --seed 42 \
    --out /path/to/temp_xc/figures/summary_fra_vs_additive_L24_ln1_seed42
```

**Data inputs**

| input | path | size | source |
|---|---|---:|---|
| Nura per-seed Phase-1 aggregates (qk_to_ov / ov_to_ov / qk_to_qk / baseline) | `plots/2026-05-07_em_repl/phase1_judged/aggregated_seed42_medical.json` | ~10 KB | `fra_proj/scripts/reaggregate_nura_per_seed.py` re-aggregating `qualitative_medical_L24_H38_k50.json` |
| Nura SAE + additive (this run) | `plots/2026-05-07_em_repl/phase3_chat_fix/steer_nura_L24_ln1_chat_42_123_456/gpt4o_aggregated_seed42_blocks_24_ln1_hook_normalized_medical_top50.json` | ~2 KB | `fra/sae_resid_eval.py --sae-source nura …` then `judge_multiseed.py` then re-aggregated per-seed by inline script |
| Source qualitative (per-generation) for the additive run | `plots/2026-05-07_em_repl/phase3_chat_fix/steer_nura_L24_ln1_chat_42_123_456/qualitative_blocks_24_ln1_hook_normalized_medical_top50.json` | ~600 KB | `fra/sae_resid_eval.py` + `judge_multiseed.py` |
| Source qualitative for Nura's frontier_multiseed | `plots/2026-05-07_em_repl/phase1_judged/qualitative_medical_L24_H38_k50.json` | ~270 KB | `run_experiments.py --task frontier_multiseed --em-model medical --head 38 --seeds 42 123 456 …` then `judge_multiseed.py` |

**SAE used by both panels** (private HF, > 1 MB, NOT committed locally):

- `Nura-J/Qwen2.5-14B_SAE_ln1.normalised` (Nura's published SAE, `ae_200000.pt`, top-k=64, d_sae=102 400)
