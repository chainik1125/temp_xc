---
author: Dmitry Manning-Coe
date: 2026-05-08
tags:
  - reference
  - results
---

## Phase 0 fast-path validation outputs

Three runs of the same 56-cell medical slice (qk_to_ov × 6 α + baseline × 1, seed=42, 8 prompts) under different generator + RNG configurations. Used to validate the opt-in fast eval path in `fra_proj/fra/em_evaluation.py` (commit `88cefde`).

Full writeup: [[../../docs/dmitry/c6_em/2026-05-08_em_repl_finance_sports/phase0_fastpath_validation|phase0_fastpath_validation.md]].

| File | Generator | RNG schedule | Δ off ground truth |
|---|---|---|---:|
| `*_FAST.json` | `generate_with_hooks_batch`, batch=8 | single batch seed=42 | 8.75 pts |
| `*_SLOW_SINGLE.json` | `generate_with_hooks` (Nura's reference), batch=1 | seed=42 fixed for all 8 prompts | 8.12 pts |
| `*_FAST_PERPROMPT.json` | `generate_with_hooks_batch`, batch=8 | per-prompt seeds [42..49] | **2.50 pts ✅** |

Ground truth: `temp_xc/plots/2026-05-07_em_repl/phase1_judged/aggregated_seed42_medical.json` (Nura's slow-path numbers, per-prompt seeds).

`qualitative_*.json` — per-generation responses (56 entries each).
`gpt4o_aggregated_*.json` — per-(condition, α) aggregates (means + stds).

Verdict: deploy `generate_with_hooks_batch(seed=list[int])` as the fast eval path for the finance + sports replication. The disagreement of single-seed runs is dominated by the RNG schedule, not by batching math.
