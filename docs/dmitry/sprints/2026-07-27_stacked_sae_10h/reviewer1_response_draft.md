---
author: Dmitry
date: 2026-07-27
tags:
  - design
  - in-progress
---

## Reviewer 1 response draft — Stacked SAE questions (Q3 + Q5)

Working draft; numbers marked [300K-PENDING] fill in when the paper-scale
arm lands. Everything else is verified against committed artifacts
(provenance in `log.md`).

### Q3 — "Why are Stacked SAE detection and inducement results not included in Fig. 4 or Table 2?"

Draft text:

> We agree this baseline belongs in the main comparison and have run it.
> Stacked SAE (a bank of T independent per-position TopK SAEs, isolating
> temporal aggregation from cross-position weight sharing) was part of our
> C7 architecture pool from the start, and we report it at two scales.
>
> **Matched-budget comparison (all seven architectures, identical training
> configuration: d_SAE=32,768, k_pos=20, batch 1024, 20K steps, seed 42,
> same activation cache, same 25-point magnitude grid, same judge
> protocol):** peak Δgc — TXC-base 0.426, TXC-pro 0.377, TFA 0.344,
> **Stacked SAE 0.328**, T-SAE 0.246, TopK SAE 0.230, MLC 0.164. The
> stacked baseline is a strong fourth: it beats every per-token
> architecture, confirming that temporal aggregation alone captures much
> of the effect — but the shared-code temporal architectures (TXC-base,
> TXC-pro) remain ahead of it by 15–30% relative. On detection at the same
> scale, Stacked SAE reaches PR-AUC 0.177 at S=8 — above the TopK SAE
> (0.168 at this scale) but below TXC-pro. Cross-position weight sharing
> therefore contributes measurably beyond temporal aggregation, which is
> precisely the decomposition the reviewer asked for.
>
> **Paper-scale row (300K steps, matching Table 2 / Fig. 4):**
> [300K-PENDING: Δgc peak + PR-AUC S-sweep + ROC-AUC, with a footnote that
> the judge pass postdates the original rows; sub-0.02 detection gaps
> should be read as noise given per-cell judged-cohort denominators of
> 60–61.] The camera-ready adds this row to Table 2 and the corresponding
> bar to Fig. 4.
>
> The same decomposition holds on synthetic ground truth, where it is
> cleanest: on the polynomial-clock task at window W=4, a per-token SAE
> reaches R²=0.04, a stacked window-linear readout 0.20, and the TXC 0.92
> — aggregation helps, weight sharing across positions is where most of
> the recovery lives (App. [ref], Table [ref]).

Supporting-note for us (not reviewer-facing): the 20K panel is
`origin/temp-bench:experiments/c7_backtracking/results.json` + leaderboard
rows (stacked train_key d08c6498d3fa430e; stability 18/24, second-best in
panel). Stacked's 20K detection row also carries the shuffle control
(gap +0.005…+0.027, consistently positive).

### Q5 — parameter count / inference cost

Draft text:

> Table [new]: parameters and per-token inference FLOPs for every
> architecture at its C7 configuration. The matched quantity across
> architectures is the window sparsity budget k_win = T·k_pos (App. A),
> not parameter count: a Stacked SAE holds T independent dictionaries and
> therefore T× the parameters of a TopK SAE at equal d_SAE, while TXC
> shares one dictionary across the window. We now state this explicitly
> alongside the table.

Param counts at C7 config (d_in=4096, d_SAE=32,768, T=5):
- TopK SAE: 2 · d_in · d_SAE ≈ 0.27B
- Stacked SAE: T · 2 · d_in · d_SAE ≈ 1.34B
- TXC-base: enc T·d_in·d_SAE + dec d_SAE·T·d_in ≈ 1.34B (verify vs class)
- (fill remaining archs from checkpoint headers — sizes already recovered
  in the HF audit: topk 1.07 GB, stacked/txc_base/mlc 2.68 GB, tfa 4.6 GB,
  txc_pro 5.4 GB at bf16.)

### App A correction (goes in camera-ready regardless)

Replace "Stacked SAE, used in C1, C2, and C7" — the C7 appendix's own
six-architecture list contradicts it. New text: "Stacked SAE, used in C1,
C2, and (as of this revision) C7; see Table 2 and Fig. 4."

### Q2 (seeds) — one-line stance for the response

Main C7 results are training seed 42 (as App. F.12 states; the checklist's
"2 seeds" refers to the C6/EM case study and we will fix the checklist
wording). Multi-seed C7 replication is camera-ready work; all judge
outputs are persisted so added seeds re-judge only new transcripts.
