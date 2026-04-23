---
author: Aniket
date: 2026-04-23
tags:
  - results
  - in-progress
---

## venhoff_scrappy agentic log

Append-only research log. Pattern follows Han's `docs/han/research_logs/phase5_downstream_utility/2026-04-21-agentic-log.md` — cycles appended linearly, never rewritten.

Each cycle section has the same shape: **hypothesis → code edits → command → result row → takeaway → next hypothesis**. The ledger of raw Δ numbers lives at `experiments/venhoff_scrappy/results/autoresearch_index.jsonl`; this doc is the prose layer the autoresearch agent (and humans) reason from.

## Reference

- Full-budget plan: [[docs/aniket/experiments/venhoff_eval/plan|venhoff_eval plan]]
- Paper: Venhoff et al. 2025, arXiv:2510.07364
- Paper baseline Gap Recovery (MATH500, Llama-3.1-8B ↔ DeepSeek-R1-Distill-Llama-8B cell): **3.5%**
- Our scrappy-slice noise floor (n=20): ~±10 pp stderr — Δ needs to be >+10 pp to clear noise.

## Scaffold state

- `experiments/venhoff_scrappy/` directory scaffolded 2026-04-23.
- `run_cycle.py` is a **scaffold** — it accepts the merged config, writes a placeholder `grade_results.json`, and does not yet dispatch to `src/bench/venhoff/`. Real dispatch lands on first use (when we're ready to actually iterate post full-budget run).
- Until then, `run_autoresearch.sh` is end-to-end runnable in *smoke-test* mode: it exercises the orchestrator → summariser → ledger → commit chain with placeholder numbers, so when the real dispatch lands we're not debugging infra.

## Curated batch plan (post-web-claude review 2026-04-23)

Before the autoresearch agent takes over, 4 hand-curated cycles establish sanity + catch a fatal interpretability bug:

| Cycle | Candidate | Hypothesis / Purpose | Action if fails |
|---|---|---|---|
| 00 | `baseline_sae` | Reference point (Δ = 0 by definition). Establishes absolute GR on the scrappy slice. | If GR is outside [0.10, 0.30] → abort, re-check judge, model ids, slice selection. |
| 01 | `baseline_sae_shuffled` | TFA-confound check. Shuffle activations within-sequence before Phase 2. Expected: GR_shuffled ≪ GR_baseline_sae. | If GR_shuffled ≈ GR_baseline_sae → STOP. Pipeline has dense-channel confound; conclusions about "which arch wins" become uninterpretable. |
| 02 | `baseline_tempxc` | Scrappy slice preserves TempXC > SAE ordering from full-budget run? | If TempXC ≤ SAE → slice is too small / wrong; bump to n=50 and revisit. |
| 03 | `baseline_mlc` | MLC viable at scrappy budget? | If MLC fails to train in 5 iters → expected; not blocking, just note. |

Then: hand off to agent for screening phase (§6.3 of plan — 3×3×2 fractional factorial, 18 cycles) + exploit phase.

## Cycle 00 — baseline_sae (placeholder)

**Status**: scaffold only — pending run_cycle.py wiring (see plan §5).

**Command**: `bash experiments/venhoff_scrappy/run_autoresearch.sh baseline_sae`

**Expected**: absolute Gap Recovery on scrappy slice, base for all Δ.

## Cycle 01 — baseline_sae_shuffled (placeholder, CRITICAL)

**Status**: scaffold only.

**Command**: `bash experiments/venhoff_scrappy/run_autoresearch.sh baseline_sae_shuffled`

**Critical decision point**: if Δ ≈ 0 (shuffle didn't hurt), the entire scrappy loop's conclusions about arch comparisons are suspect. Do not proceed to the agentic screening phase until this check passes.
