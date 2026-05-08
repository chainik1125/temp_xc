---
author: Dmitry Manning-Coe
date: 2026-05-07
tags:
  - results
  - todo
---

## Trigger

This document is **only** filled in if Phase 1 reproduction misses Nura's `Δalign|coh≥70` for medical QK→OV outside seed CI. Per the overnight directive, on failure all overnight effort goes here instead of Phase 3.

## Reproduction summary

Filled at end of Phase 1.

| Quantity | Nura's reported | Ours (this run) | Δ | Within 3-seed CI? |
|----------|-----------------|------------------|---|---|
| medical QK→OV `Δalign|coh≥70` | TBD | TBD | TBD | TBD |
| medical QK→OV peak alignment | TBD | TBD | TBD | TBD |
| medical OV→OV `Δalign|coh≥70` | TBD | TBD | TBD | TBD |
| medical QK→QK `Δalign|coh≥70` | TBD | TBD | TBD | TBD |
| finance QK→OV `Δalign|coh≥70` | TBD | TBD | TBD | TBD |
| sports  QK→OV `Δalign|coh≥70` | TBD | TBD | TBD | TBD |

## Possible root causes (ordered by likelihood)

For each, list: how to test, expected fingerprint, current evidence.

### 1. Model-load divergence

- **LoRA path/revision**: did we use `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` at the same commit as Nura? (Confirm via `huggingface_hub.hf_hub_download` revision vs CODEBASE.md.)
- **Merge ordering**: `PeftModel.from_pretrained → merge_and_unload → from_pretrained_no_processing`. If sequence differs, weights diverge silently.
- **Test**: load both Nura's and our merged-state-dict mean-norm, log a few weight tensors, diff. Look for layer-24 W_V drift.

### 2. SAE-load divergence

- We load `Nura-J/Qwen2.5-14B_SAE_ln1.normalised/ae_200000.pt`. Confirm hash + d_in / d_sae / k match (smoke test confirmed `d_in=5120, d_sae=102400, top_k=64`).
- **Test**: log a single forward of one prompt through `sae.encode` and compare top-10 active features index list vs an artifact from Nura's repo if available.

### 3. GQA head mapping

- Qwen2.5-14B has 40 Q heads ↔ 8 KV heads (5:1). Head 38 maps to KV head `38 // 5 = 7`.
- **Test**: in `fra/ov_steering.py:run_ov_steering`, log `kv_head` chosen for `head=38`. Should be 7. If it is 6 or 8, that's a one-off bug.

### 4. RMSNorm / RoPE convention

- `from_pretrained_no_processing` is meant to skip TL's LayerNorm folding. Verify with one prompt that pre-attention activations match the HF reference within 1e-3 rel-error.
- RoPE θ for Qwen 2.5 differs from Llama; ensure TL applies the right one (via `model_from_pretrained_kwargs` if needed).

### 5. Judge variance

- GPT-4o judge defaults can drift between runs. Re-judge a held-out subset with Claude (Sonnet-4.6 via `ANTHROPIC_API_KEY_MATS`) to estimate judge-induced variance.
- If Claude vs GPT-4o ranking flips, the headline depends on judge identity → not a clean reproduction failure.

### 6. Prompt drift

- The 8 EM eval prompts are hardcoded in both `run_experiments.py:101–112` and `fra/em_evaluation.py:22–31`. Diff the two and Nura's CODEBASE.md ordering.

### 7. α sweep / sign of `(α − 1)`

- At α=1.0 the OV hook should be a no-op. Confirm by re-running with hooks vs without and diffing logits — if non-zero, the steering arithmetic is wrong.

### 8. Seed handling

- `frontier_multiseed` uses `--seeds 42 123 456` per `run_all_multiseed.sh`. Ensure each subprocess actually re-seeds (`torch.manual_seed`, `numpy.random.seed`, sampler state).

## Fix proposals (filled per-cause)

To be written based on which root cause(s) match the failure fingerprint.

## What changed gets us to the right number

The decisive code/config diff that we believe will close the gap. To be filled.
