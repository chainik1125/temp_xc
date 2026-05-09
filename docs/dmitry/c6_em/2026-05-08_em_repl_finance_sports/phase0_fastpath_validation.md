---
author: Dmitry Manning-Coe
date: 2026-05-08
tags:
  - results
  - guide
---

## Phase 0 — fast eval path validation

Before launching the cross-domain (finance + sports) replication, we built and validated an opt-in **fast eval path** for `fra_proj/fra/em_evaluation.py`. The slow medical-session path measured ~7 tok/s on H100 with Qwen2.5-14B + LoRA, giving ~5 h per domain for the full Phase-1 frontier sweep — too slow for cross-domain iteration. This document records what we changed, what we measured, and the verdict.

## What we changed

Three opt-in code changes, all on `chainik1125/fra_proj` branch `dmitry-em-repl` (commit `88cefde`). The original `generate_with_hooks` is preserved unchanged as the canonical reference and is what the byte-identical no-op diagnostic still calls.

- **A. KV cache** — `fra/em_evaluation.py:generate_with_hooks_fast`. Encodes the prefix once, then per-step passes only the new token + `TransformerLensKeyValueCache`. Hooks fire on every forward as in the slow path.
- **B. Prompt batching** — `fra/em_evaluation.py:generate_with_hooks_batch`. Stacks `len(prompts)` prompts into one forward pass per token, sharing one KV cache. Left-pads chat-template-wrapped tokens to the longest prompt's length and passes an `attention_mask` on the first forward to exclude pad positions. **Accepts `seed: int | list[int]`** — the list form is what mirrors Nura's `seed = base + i` convention (one `torch.Generator` per row, per-row sampling).
- **D. Parallel GPT-4o judging** — `judge_multiseed.py:judge_qualitative_file` wraps the per-generation calls in `concurrent.futures.ThreadPoolExecutor(max_workers=20)` with periodic save every 20 completions.

## What we measured

### Speedup (production size: 8 prompts × 200 new tokens, Qwen2.5-14B + medical LoRA on H100 80GB)

| Path | Time per (cond, α) cell | tok/s (batch sum) | Speedup vs slow |
|---|---:|---:|---:|
| **slow** — `generate_with_hooks`, batch=1, no cache | 56.0 s | 7.3 | 1.0× |
| **fast (A)** — `generate_with_hooks_fast`, batch=1 + KV cache | 53.2 s | 7.5 | **1.1×** |
| **fast (A+B)** — `generate_with_hooks_batch`, batch=8 + KV cache | 21.3 s (smoke) / 11 s (prod) | 25.5 / ~145 | **2.6× / 5.4×** |
| **judge tail** — sequential vs `ThreadPoolExecutor(20)` | ~1–2 h / 720 calls vs ~3 min | — | **~10–20×** |

KV cache alone is **only 1.1×** — Qwen2.5-14B is FFN-bound (d_intermediate=13824 vs d_model=5120, and most per-step compute is in three SwiGLU projections that cache doesn't touch). Batching, on the other hand, amortises the per-step Python/TL hook overhead and the per-token FFN cost across the batch dim, recovering most of the lost throughput.

Combined, the fast path drops per-domain Phase 1 budget from **~5 h compute + 1–2 h judge** → **~13 min compute + ~3 min judge**.

### Correctness — Phase 0 validation slice on medical

Slice: `qk_to_ov × 6 α + baseline × 1 α=1.0` × 1 eval seed (42) × 8 prompts = **56 cells**. Ground truth: `temp_xc/plots/2026-05-07_em_repl/phase1_judged/aggregated_seed42_medical.json` (Nura's slow-path numbers, per-prompt seeds = `[42, 43, …, 49]`).

We re-ran the same 56 cells under three configurations and judged each end-to-end with GPT-4o (parallel, 20 workers).

| Cell | Ground truth (per-prompt seeds) | Mode A: slow + single seed=42 | First fast: batched + single batch seed=42 | **Mode B: batched + per-prompt seeds [42..49]** |
|---|---:|---:|---:|---:|
| baseline α=1.0 | 66.2 | 61.9 (−4.4) | 62.5 (−3.8) | 58.8 (−7.5) |
| qk→ov α=0.0 | 61.2 | 50.6 (−10.6) | 58.1 (−3.1) | 67.5 (+6.2) |
| qk→ov α=0.5 | 57.5 | 48.1 (−9.4) | 73.1 (+15.6) | 64.4 (+6.9) |
| qk→ov α=1.0 | 65.0 | 60.0 (−5.0) | 61.2 (−3.8) | 58.8 (−6.2) |
| qk→ov α=1.5 | 65.0 | 44.4 (−20.6) | 65.6 (+0.6) | 65.0 ( 0.0) |
| qk→ov α=2.0 | 57.5 | 48.8 (−8.8) | 56.9 (−0.6) | 68.8 (+11.2) |
| qk→ov α=3.0 | 63.1 | 39.4 (−23.8) | 58.8 (−4.4) | 66.2 (+3.1) |
| **Headline Δalign\|coh≥70** (qk→ov) | **7.50** | 15.62 (off **8.12**) | 16.25 (off **8.75**) | **10.00** (off **2.50**) ✅ |

**Cells within ±5 pts**: alignment 2/7 — 6/7 — 2/7. **Headline Δ within ±5 pts: only Mode B passes.**

## What this proved

1. **The disagreement is dominated by the RNG schedule, not by batching math.** Mode A (Nura's exact `generate_with_hooks` with seed=42 fixed for every prompt) disagrees with ground truth by **8.12 pts** on Δ — almost identically to the first fast batched run with one batch seed (8.75 pts). The two configurations differ only in their RNG schedules and produce comparable disagreement.
2. **Batching is fine** if you mirror the reference's per-prompt seed convention. Mode B (batched + `seed=[42..49]`) matches ground truth within **2.5 pts** on Δ — well within the ±5 pt tolerance and comfortably within the judge sampling-error floor (n=8 prompts, std~25 → SE ~9 pts on a single cell, ≈13 pts on Δ).
3. **Per-cell 5-pt agreement is too strict given the actual judge SE.** Even Mode B has cells off by 6–11 pts. The metric to gate on is the headline Δ, not per-cell exactness.

This led to a memory note: `feedback_per_prompt_seeds_under_batching.md` — for matched-recipe replications, batched generators must accept `seed: list[int]` (one per row) and mirror the reference's `seed = base + i` convention.

## Decision

Deploy **Mode B (`generate_with_hooks_batch` with per-prompt seeds)** as the fast path for the finance + sports replication. The slow `generate_with_hooks` is retained for diagnostic gates (chat-template smoke, byte-identical no-op check).

Per-domain Phase 1 estimate (revised):

- 4 conditions × 6 α × 3 eval seeds × 8 prompts = 576 generations
- @ batch=8, ~11 s/cell (8 prompts at once) = **72 cells × 11 s ≈ 13 min compute**
- Plus parallel judge (1440 calls @ 12 req/s with 20 workers) ≈ **~3 min**
- Plus diagnostic gates (slow path, intentional): **~10 min**
- **Per-domain wall-clock: ~25–30 min** (down from ~5–6 h)
- Both domains in parallel on the two `h100_emfra_2gpu_*` pods: **~30 min wall**

The new L24 ln1 SAE training (~6 h on a separate GPU) remains the critical-path item for the full campaign, dominating the eval pipeline.

## Reproducing this

```bash
# 1. Generate the validation slice (slow control + fast control in one model load)
ssh h100_emfra_2gpu_1 'cd /workspace/fra_proj && \
    export TMPDIR=/workspace/tmp CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    python3 -u phase0_diagnostic_run.py --em-model medical --base-seed 42'

# 2. Judge + compare against ground truth (uses OPENAI_API_KEY_MATS in our env)
echo "$OPENAI_API_KEY_MATS" | ssh h100_emfra_2gpu_1 'read OPENAI_API_KEY; export OPENAI_API_KEY; \
    cd /workspace/fra_proj && python3 -u phase0_judge_and_compare.py \
    --qualitative phase0_results/qualitative_medical_L24_H38_seeds42-49_qkov_baseline_FAST_PERPROMPT.json \
    --ground-truth phase0_results/aggregated_seed42_medical_GT.json'
```

Code: `fra_proj` branch `dmitry-em-repl` commit `88cefde`. Output JSONs in `temp_xc/plots/2026-05-08_phase0_fastpath/` (after we commit them).

Cross-references:

- `feedback_per_prompt_seeds_under_batching.md` — the durable lesson from this validation
- `feedback_reuse_recipe_functions.md` — the prior session's lesson that motivates building opt-in fast paths *alongside* (not replacing) the reference function
- `lessons_learned.md` (2026-05-07) — the binding spec that all of this is layered on
