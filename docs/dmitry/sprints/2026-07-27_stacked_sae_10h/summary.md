---
author: Dmitry
date: 2026-07-28
tags:
  - results
  - complete
---

## Executive summary

This sprint answered reviewer 1's central objection — "the paper never
shows that cross-position weight sharing, not just temporal aggregation,
drives the gains; the Stacked SAE comparison is missing" — by producing
the Stacked SAE column across all four real-world case studies at the
paper's own protocols, one training seed (42), window T=5.

1. **Weight sharing is most of the causal story.** At the published C7
   scale (d_SAE=32,768, 300K steps, 25-mag grid, Sonnet-4.6 judge), the
   Stacked SAE peaks at Δgc **0.246 @ m=−12** — the same peak magnitude
   as TXC-base, at **45%** of its 0.541. Aggregation alone reaches
   mid-pack (above T-SAE 0.164, tied with MLC 0.246); sharing doubles it.

2. **Aggregation alone does not explain the static results either.**
   On sparse probing the stacked cell scores mean AUC **0.8694**, below
   the per-token TopK (0.885–0.889) and the TXC family (0.89–0.90). On
   HH-RLHF it lands at preference AUC **0.602** vs the 0.61 headline,
   with 1/20 length-spurious top features and realized L0 within 7% of
   the 500 nominal.

3. **One genuine surprise: stacked wins EM detection at T=5** — PR-AUC
   @S16 **0.6516** vs the 0.54 TXC headline — but the result carries a
   panel-wide caveat: realized eval L0 runs far above nominal for every
   architecture (JumpReLU thresholds miscalibrate under train→rollout
   distribution shift; references 6–10× nominal, stacked ~32×), so the
   EM comparison is directional until the panel is re-thresholded.
   Notably the same cell at T=4 scored 0.512 — the win is T-sensitive.

4. **A forensic result that matters beyond this sprint: the paper's C7
   numbers are a two-generation story.** The printed Fig 4/Table 2
   values come from a 300K-step generation whose checkpoints are lost
   from git and HF (they reproduce, 6 of 7, from `origin/300k-tfa`);
   a complete, internally-consistent 20K seven-arch panel *including
   stacked* already existed unreported on `origin/temp-bench` (stacked
   0.328 vs TXC-base 0.426). Δgc peaks are max-over-grid, so only
   same-grid (25-mag) comparisons are valid — the planned 41-mag eval
   would have been comparable to nothing.

## The stacked column (T=5, seed 42, raw values)

| Task | Metric | Stacked | Headline | Untrained floor |
| --- | --- | --- | --- | --- |
| C7 inducement | Δgc peak, 25-mag | 0.246 @ −12 | 0.541 (TXC-base) | n/a (not run) |
| C7 detection | PR-AUC @ S=8 | 0.158 | 0.242 (TXC-pro) | n/a (not run) |
| C3 probing | mean AUC, 38 tasks | 0.8694 | 0.89 (TXC) | 0.8026 |
| C6 EM detection | PR-AUC @ S=16 | 0.6516† | 0.54 (TXC) | 0.3442 |
| RLHF decomposition | pref AUC @ k=20 | 0.602 | 0.61 | **0.6174** |

† see finding 3. Floor note: the untrained RLHF floor (0.6174)
*exceeds* both the trained stacked cell (0.602) and the 0.61 headline —
the preference-AUC metric is largely training-insensitive, reinforcing
the paper's reading of HH-RLHF as a length-confounded negative control.
The EM floor (0.344) sits at the prevalence baseline, so the trained EM
signal is real relative to floor; the probing floor (0.803) shows most
probing AUC comes from random features plus the probe, with training
adding +0.07 for stacked. Chance-normalized variants and the reviewer-facing
table live in `docs/dmitry/reviewer_responses/reviewer_responses.md`
(normalization for the steering-headline rows still under discussion).

## Where everything lives

- **Numbers**: leaderboard rows per pod, mirrored to HF
  `dmanningcoe/stacked-sae-rebuttal-2026-07` (`*/new_leaderboard_rows.jsonl`).
- **Checkpoints** (7): C7-300K stacked, C3 gate TopK + stacked, C6
  T=4-s42/T=4-s1/T=5-s42, RLHF T=5 — same HF repo, with manifests.
- **Judge transcripts**: 1,525 Sonnet-4.6 calls for the C7 Δgc row
  (`c7_300k/runs/.../judge_outputs.jsonl`); the poisoned first pass is
  archived beside it.
- **EM cohort cache** (hs16 + sidecars): `em_medical_cohort/` in the
  same repo — rebuilt deterministically from `origin/final` judge
  outputs via `phase4_em_depth`.
- **Code**: branch `dmitry-stacked-arxiv` (pooled adapters
  `stacked_sae_pooled` / `stacked_btkonly_pooled`, registry with
  per-section T=5 blocks, gate tests); branch `dmitry-stacked-c7-300k`
  (C7 driver pinned to 300K/25-mag/protocol-1.0.0).
- **Log with full incident timeline**: `log.md` in this directory.

## Process notes (what cost time, so the next sprint doesn't pay twice)

Three infrastructure facts caused ~9 of the sprint's first 10 hours of
GPU idling, all now fixed at source and recorded in memory: RunPod env
vars do not reach SSH sessions (tokens must be pushed as files); the
account's default ANTHROPIC_API_KEY has no credit (judge calls fail as
−1 sentinels that *look* like completed evals — the fake Δgc=0.0 was
caught only by reviewing a "done" result); and section entry points are
silent no-ops under `python -m` (dispatch through `run.py <section>`).
Two config traps: the pooled arch inherited T=4 from its parent
registry entry (C6 first trained off-target; caught by the L0
window/token ratio), and direct-CLI cells bypass cell-table overrides
(RLHF first trained at k_win=80 instead of 500). Every result above
postdates the fixes; the mislabeled artifacts are kept on HF for
provenance. Total spend: ≈$95 in pods, ≈$8 in judge calls (of a $200
cap).
