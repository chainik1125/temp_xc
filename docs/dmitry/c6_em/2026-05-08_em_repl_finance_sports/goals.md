---
author: Dmitry Manning-Coe
date: 2026-05-08
tags:
  - design
  - in-progress
---

## Cross-domain EM-FRA replication: finance + sports

Following the [[2026-05-07_em_repl/goals|2026-05-07 medical replication]], we extend the same end-to-end measurement to Nura's two other emergent-misalignment LoRAs:

- **finance** — `ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice`
- **sports** — `ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports`

The medical replication produced a strikingly favourable headline (FRA-decomposition recipes show a wider Δalign|coh≥70 frontier than conventional additive feature steering). The user is sceptical of how clean those numbers were and wants a *cross-domain* check before drawing broad conclusions about FRA vs additive steering. If the same ordering replicates across all three domains, the medical headline is robust; if it doesn't, we know the medical result was domain-specific or noise.

## What we keep fixed

- **SAE at L24 ln1**: Nura's `Nura-J/Qwen2.5-14B_SAE_ln1.normalised` (top-k=64, d_sae=102,400, normalize=expected_average_only_in). Single SAE across all 3 domains — `run_experiments.py:90` hardcodes this regardless of `--em-model`.
- **Surrounding-hookpoint SAEs**: the 4 we trained for medical (`resid_pre L24`, `resid_mid L24`, `resid_post L24`, `ln1 L25`) on `dmanningcoe/em-repl-2026-05-07`. Trained on Pile (domain-agnostic) — reusable.
- **8 EM eval prompts**: `run_experiments.py:101–112` — same set across all domains.
- **Hookpoint, α grid, eval seeds**: `blocks.24.ln1.hook_normalized`, α ∈ {0, 0.5, 1.0, 1.5, 2.0, 3.0}, seeds {42, 123, 456}, top-k=50 features.
- **Generator**: `fra/em_evaluation.py:generate_with_hooks` (chat-template correct, lesson §1).
- **Judge**: `judge_multiseed.py --variant {finance,medical,sports}`.

## What we change vs the medical run

- **+1 new SAE**: train one L24 ln1 SAE matching Nura's config (top-k=64, d_sae=102,400, ~200M tokens on Pile, normalize=expected_average_only_in, sae-lens 6.43, float32 SAE + bf16 model). Disentangles "Nura's specific SAE" from "the recipe" at the hero hookpoint.
- **Aggressive eval path** (gated on Phase 0 validation): KV cache + prompt batching in `generate_with_hooks*`, parallel GPT-4o judging via `ThreadPoolExecutor`. Targets ~30× speedup over the medical-session slow path.

## Three streams in parallel

1. **finance** evaluation — Phase 1 (Nura `frontier_multiseed`) + additive on 5 SAEs (Nura's L24 ln1 + 4 surrounding) + new L24 ln1 SAE once trained.
2. **sports** evaluation — same composition.
3. **new L24 ln1 SAE training** on Pile (~6h on 1 H100), runs in parallel with eval streams 1 & 2.

## Headline metric

Same as medical: `Δalign|coh≥70` per (domain, recipe). Reported as a 1×3 cross-domain bar chart with eval-seed std error bars.

## Pass criteria

- Per-domain Phase 1 baseline (Nura `baseline` method, no hook) `mean_alignment` agrees within ~5 pts with our additive α=1.0 (no-op of additive recipe). Lesson §3 gate. If fail: stop and debug.
- Hook no-op diagnostic: byte-identical tokens at α=1.0 with vs without our additive hook on a sample of 3 prompts × 3 seeds.
- Headline `Δalign|coh≥70` for `qk_to_ov` agrees with Nura's published `gpt4o_aggregated_{finance,sports}_L24_H38_k50.json` within ~5 pts.

## Companion docs

- [[plan|plan.md]] — phased implementation plan
- [[phase0_fastpath_validation|phase0_fastpath_validation.md]] — Phase 0 validation of the aggressive eval path
- [[phase1_finance|phase1_finance.md]] — finance results
- [[phase1_sports|phase1_sports.md]] — sports results
- [[summary|summary.md]] — cross-domain headline writeup

The lessons doc from 2026-05-07 (`docs/dmitry/c6_em/2026-05-07_em_repl/lessons_learned.md`) is the binding spec.
