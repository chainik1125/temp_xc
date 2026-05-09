---
author: Dmitry Manning-Coe
date: 2026-05-08
tags:
  - design
  - in-progress
---

## Plan: cross-domain EM-FRA replication on finance + sports

Phased plan with explicit go/no-go gates. The cross-cutting constraint: every diagnostic from `2026-05-07_em_repl/lessons_learned.md` runs before headline numbers are accepted.

## Phase 0 — Validate the aggressive eval path ✅ PASSED

Full writeup: [[phase0_fastpath_validation|phase0_fastpath_validation.md]].

**Verdict**: deploy `generate_with_hooks_batch(seed=list[int])` (batched + per-prompt seeds) as the production fast path. Headline `Δalign|coh≥70` matches ground truth within 2.5 pts (≤5 pt threshold). 5.4× speedup on generation, ~10× on judge tail.

**Key finding**: the disagreement between fast-path and slow-path numbers is dominated by **RNG schedule** (single seed vs per-prompt seeds), not by batching math. Saved to memory: `feedback_per_prompt_seeds_under_batching.md`.

**Original goal (kept for record)**: confirm batching + KV cache + parallel judge produces medical-Phase-1 numbers that agree within judge noise (~5 pts on per-cell `mean_alignment`, ~5 pts on `Δalign|coh≥70`).

### Code changes (`chainik1125/fra_proj` branch `dmitry-em-repl`)

- `fra/em_evaluation.py:generate_with_hooks_fast` — KV cache, single prompt. New function alongside the original `generate_with_hooks` (which stays as the slow-path reference for byte-identical diagnostics).
- `fra/em_evaluation.py:generate_with_hooks_batch` — KV cache + batched prompts. Left-pads chat-template-wrapped tokens, runs `run_with_hooks` with `[B, L]` input + `TransformerLensKeyValueCache(batch_size=B)`. Single `torch.multinomial` per step over `[B, V]`.
- `judge_multiseed.py:judge_qualitative_file` — `ThreadPoolExecutor(max_workers=20)` parallelism, periodic save every 20 completed.

### Validation slice on medical (~17% of Phase 1)

- 2 conditions: `qk_to_ov` (headline winner) + `baseline` (no-hook)
- All 6 α values
- 1 eval seed = 42
- All 8 EM prompts
- = **96 generations**

### Gates

1. **Smoke test** — verify the fast generators load + run + produce nontrivial outputs. ✅ ran (smoke_fastpath.log).
2. **Speedup measurement** — actual `tok/s` for slow vs fast vs batch=8 paths at production size (max_new_tokens=200).
3. **Per-cell agreement** — fast-path `mean_alignment` within ±5 pts of slow-path on ≥11/12 cells.
4. **Headline agreement** — fast-path `Δalign|coh≥70` for `qk_to_ov` within ±5 pts of slow-path.

### Decision tree

- All 4 gates pass → proceed to Phase 1 with the fast path.
- Speedup < 2× → drop B (batching), keep just D (parallel judge). Re-validate.
- Per-cell agreement fails → debug RNG / hook semantics; fall back to slow path with parallel judge only (~1.5× total speedup).

## Phase 1 — Finance + sports evaluation

Gated on Phase 0 passing.

### Per-domain workload

| Component | Cells | Slow-path | Fast-path target |
|---|---|---:|---:|
| Phase 1 `frontier_multiseed` (Nura recipes: 4 conditions × 6 α × 3 seeds × 8 prompts) | 576 gens | 5 h | 1–2 h |
| Additive on Nura L24 ln1 SAE (1 × 6 × 3 × 8) | 144 gens | 1.2 h | 15–25 min |
| Additive on each of 4 surrounding SAEs | 144 gens × 4 | 4.8 h | 1–1.6 h |
| Additive on new L24 ln1 SAE (post-training) | 144 gens | 1.2 h | 15–25 min |
| Diagnostic gates (smoke + baseline-agreement + no-op) | small | 30 min | 30 min (slow path on purpose) |
| Judge (parallel) | 720 calls | 1–2 h | 5–10 min |

### GPU allocation (10 H100s free)

| Pod | GPU | Stream |
|---|---|---|
| `h100_emfra_2gpu_1` | GPU0 | finance: Phase 1 + additive on 5 existing SAEs |
| `h100_emfra_2gpu_1` | GPU1 | train new L24 ln1 SAE on Pile (~6h) |
| `h100_emfra_2gpu_2` | GPU0 | sports: Phase 1 + additive on 5 existing SAEs |
| `h100_emfra_2gpu_2` | GPU1 | spare / diagnostics |
| Single H100s × 6 | reserve | spare; activate only if Phase 0 narrowed the fast path |

### Diagnostic gates (per domain, before launching α-sweep)

1. Smoke generate for the new LoRA (finance / sports) at no-hook through `generate_with_hooks` — confirm normal-looking assistant reply (not multi-speaker hallucination). Catches the chat-template class of bug.
2. Baseline-agreement: Nura `baseline` method `mean_alignment`/`mean_coherence` within ~5 pts of our additive α=1.0 (math no-op) on the same prompts + seeds. Lessons §3.
3. Hook no-op: byte-identical tokens at α=1.0 with vs without our additive hook on 3 prompts × 3 seeds. Lessons §6.

If any gate fails on a domain → STOP that domain's pipeline, debug. Don't waste compute on a broken pipeline.

## Phase 2 — Headline figures + dashboard

- 1×3 cross-domain headline bar chart: `Δalign|coh≥70` for medical / finance / sports across {QK→QK, OV→OV, QK→OV, conventional additive on Nura SAE, conventional additive on new our-SAE, conventional additive on each of 4 surrounding SAEs}. Std error bars across 3 eval seeds.
- Per-domain frontier figure (`summary_fra_vs_additive_L24_ln1_seed42_{finance,sports}.{png,pdf}`) using the existing paper-quality recipe (see `feedback_paper_figure_style.md` in memory).
- Dashboard v2: extend `temp_xc/scripts/build_em_dashboard.py` to add `finance` and `sports` tabs alongside `medical`.

## Phase 3 — Verification (before commit/push)

1. Per-domain Nura `baseline` agrees with Nura's published `gpt4o_aggregated_{finance,sports}_L24_H38_k50.json` within ±5 pts.
2. Hook no-op diagnostic JSONs `diag_noop_{finance,sports}.json` show 100% byte-identical tokens.
3. Open `figures/headline_bar_delta_at_coh70_all_three_domains.png` — does the FRA-decomposition vs conventional ordering match medical, or surprise us?
4. Open `figures/dashboard_v2.html` locally, switch tabs, spot-check α=2.0 / α=3.0 generations for off-aligned + still-coherent text.
5. `git push` to `chainik1125/temp_xc` branch `dmitry-em-repl`.
