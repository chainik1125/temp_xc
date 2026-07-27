---
author: Dmitry
date: 2026-07-27
tags:
  - design
  - in-progress
---

## Sprint kickoff — Stacked SAE at paper protocol (10h overnight)

I want to do an experiment that gives you a chance to be creative and rigorous.

Trial a 10h unsupervised sprint in the temporal-crosscoders reviewer-response
workstream. 10h is *wall-clock*: start 2026-07-27 05:11 UTC, hard stop
~15:15 UTC, writing reserved from ~14:15 UTC.

## Context

ICML reviewer 1 (borderline accept): the paper's own claimed key baseline —
Stacked SAE, which isolates temporal aggregation from cross-position weight
sharing — has no reported results on any real-world case study, despite App A
claiming it was "used in C1, C2, and C7". Audit (2026-07-26 session) found:
stacked results exist only on synthetic benches, plus two pre-protocol runs
(phase-5 gemma probing; ward stage-B C7 pilot, both on HF), plus a full 7-arch
C7 Δgc sweep *including* stacked on `origin/temp-bench`
(`experiments/c7_backtracking/results.json`: stacked 0.328 vs txc_base 0.426)
whose run generation differs from the paper's final numbers.

Full audit + protocol targets: `~/.claude/plans/ok-can-you-make-synthetic-journal.md`
(to be mirrored into this dir as `plan.md`).

## Goal

Produce Stacked SAE results **matched to the headline TXC cell, at locked paper
protocol**, for all four real-world case studies — C7 backtracking (Table 2
detection row + Fig 4 Δgc arm), C3 sparse probing (AUC-bar line, 3 seeds),
C6 emergent misalignment (detection; Wang steering if time), HH-RLHF
(decomposition row) — so the reviewer-1 response can cite paper-currency
numbers rather than pilot evidence. Rebuttal wave (C7+C3) is the priority;
C6/RLHF are the camera-ready wave, started tonight as far as compute allows.

## Key files / where the relevant code lives

- Branch: `dmitry-stacked-arxiv` (this worktree, off `origin/arxiv` @ 4bcd2b70;
  never push to arxiv itself; runner refuses dirty trees — commit before runs).
- Generic runner: `run.py` → `src/temp_bench/core/runner.py::run_experiment`;
  arch registry `configs/archs.yaml` (stacked_sae / stacked_batchtopk /
  stacked_batchtopk_btkonly already registered).
- Known blockers to fix (never edit `temp_bench/core/`):
  `evals/probing.py` + `evals/rlhf.py` squeeze(1) shape bugs for (B,T,d_sae);
  correct reductions to copy from
  `experiments/ward_backtracking_txc/architectures.py` (main checkout).
- C7 canonical driver: `origin/temp-bench:experiments/c7_backtracking/run.py`
  (7 locked archs incl. stacked_sae, protocol 2.0.0).
- C6 steering (camera-ready): `origin/final:purified/` WangFull
  (`case_studies/em.py`; decoder_row + encode_mean need stacked fixes;
  **no √T rescale for stacked**).
- Caches: HF `han1823123123/temp-bench-data` (probing anchor
  `act_cache/e4916bcae1881963`, c6 Qwen L24, c7 Llama L10),
  `txcdr-base-data` (RLHF stream), cohort/eval caches per
  `$TEMP_BENCH_EM_COHORT_DIR` / `$TEMP_BENCH_HH_RLHF_DIR`.
- Matched cells: C7 stacked T=5 k20 d32768 seed42 · C3 T=5 k20 d18432
  seeds{1,2,42} · C6 d32768 k25 seeds{42,1} medical-7B first · RLHF d18432
  k_pos100 (k_win=500 convention) seed42.

## How I'll judge it

Follow the interesting results and make them rigorous; tell a coherent story.
At the end review `summary.md`; at least one hour iterating on the writing
alone, red-team/blue-team pass included. Code only gets read if the write-up
sounds interesting and correct.

Keep a log (`log.md`, hourly wall-clock checks) including dead ends.

## Autonomy & compute

Fully autonomous for 10h; find alternatives around permission blocks.

Compute: fresh RunPod pods via `runpodctl` (authenticated on this Mac).
Parallelize generously — enough GPUs that training, eval, and debugging never
serialize; the constraint is wall-clock, not pod-hours. Two collaborator pods
(`backtracking-two`, `sparse_probing_rlhf_ablations`) are RUNNING on the shared
account — do not touch, flag in summary. Judge spend: Sonnet-4.6 C7 (~$3–6) +
Haiku C6 (~$10/cell) under the $200 cap, key `ANTHROPIC_API_KEY`
(fallback `ANTHROPIC_API_KEY_MATS`).

Push the code branch before any pod bootstrap (lessons_learned:
code-not-pushed-before-bootstrap). Stage explicit paths only, never
`git add -A`.

## Verification gates (pre-registered)

1. Reproduce one existing arch cell (topk_sae probing seed 42) through the new
   code path; diff against its leaderboard row before trusting any stacked number.
2. `realized_l0` ≈ 20/token (C3/C7), 25 (C6), 100 (RLHF) — measured, not assumed.
3. Contract tests `tests/test_v2_interfaces.py` green (new archs auto-covered).
4. Stacked steering vector: unit-norm decoder rows, **no √T rescale**.
5. Pooling choice (amax vs mean over T) fixed once, documented, identical in
   mining and eval.

### Formatting

1. Executive summary = 2–5 key findings, each backed by one self-explanatory
   graph (fresh-agent test).
2. Write positively, not negatively.
