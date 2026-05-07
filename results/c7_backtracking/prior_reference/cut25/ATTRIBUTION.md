---
title: Prior reference outputs (cut25 protocol) — attribution
component: c7
status: read-only reference (NOT our results)
ported_by: [pipeline]
ported_on: 2026-05-03
source_branch: origin/case-backtracking
source_commit: a62175ee7e99528fe833ce45e62255103bb2bac5
source_path: results/ward_backtracking_txc/b3_math500_cut25/
---

These are **the prior wasteland Stage B outputs** under the cut25
protocol with his hill-climbed TXC (k=16, T=6, window L0=96 — NOT our
locked TXC-base / TXC-pro). They are kept here only as:

1. **Cohort qid lookup** — `flip_matrix.parquet` lists the 31
   truly-wrong + 30 originally-correct qids the prior author used. We can read
   these once instead of re-running R1-Distill-Llama on all 500
   MATH-500 questions to discover the cohort.
2. **Reference numbers** — `summary.json` has the prior author's per-magnitude
   rescue rates; `inducement_summary.csv` has his per-(arch, mag) Δgc
   table including the headline +1.574 peak. Useful as a sanity check
   on our `compute_delta_gc` implementation: if we re-judge the prior author's
   transcripts with our judge-output persistence pipeline, we should
   recover his numbers within Sonnet stochasticity.
3. **McNemar reference** — `mcnemar_table.csv` for cross-checking the
   p-value computation.

**These files MUST NOT be reported as our paper results.** Our
locked-arch (TXC-base + TXC-pro + 5 baselines) re-run is what
populates the AUTO-RESULTS block of `docs/components/c7.md`. See
PROTOCOL.md § 5 (two-TXC discipline) and the c7.md "Reference numbers
(wasteland — for context only)" section.

## Files

| File | Bytes | Use |
|---|---:|---|
| `summary.json` | 1,165 | Per-magnitude rescue-rate breakdown for the prior author's TXC headline arch. |
| `flip_matrix.parquet` | 12,262 | (arch × qid × magnitude) flip rows for all 6 arches × 25 mags × 61 qids = 9150. |
| `inducement.parquet` | 68,317 | Per-(arch, qid, mag) Δgc + Δkw inducement metrics. |
| `inducement_summary.csv` | 13,916 | Aggregated peak Δgc / stability per arch. |
| `mcnemar_table.csv` | 377 | McNemar p-values for net-rescue at peak magnitude per arch. |
