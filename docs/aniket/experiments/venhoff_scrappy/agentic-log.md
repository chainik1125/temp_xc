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

## Cycle 00 — baseline_sae (placeholder)

**Hypothesis**: none — pure reference point.

**Status**: scaffold only.

**Arch**: `sae` with Venhoff's shipped optimized vectors (no Phase 2 training).

**Command**: `bash experiments/venhoff_scrappy/run_autoresearch.sh baseline_sae`

**Result**: pending first real run.

**Next**: once full-budget run lands numbers, revisit this cycle with real Phase 3 grading on the scrappy slice.
