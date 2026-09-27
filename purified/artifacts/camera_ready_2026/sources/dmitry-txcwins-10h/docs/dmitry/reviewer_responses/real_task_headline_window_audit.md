---
author: Dmitry
date: 2026-07-27
tags:
  - results
  - complete
---

## Real-model headline and window-size audit

This note collates the live paper headline numbers for the four real-model
evaluations and the TXC window-size results that currently exist. The live task
sections and Figures 5--8 in `../temp_xc_tex/main.tex` are treated as
authoritative. Figure 1's rose-diagram sidecar is stale in two places, documented
below.

The distinction between *paper-protocol* and *auxiliary* results is
load-bearing. The reviewer table uses the seed-42 paper TXC checkpoint as both
the headline denominator and the ordered `T=5` cell. It never substitutes a
TXC-pro result or an independently trained auxiliary checkpoint. Consequently,
the ordered `T=5` entry is exactly `100%` by construction.

## Reviewer-ready window matrix

The first row for each task is the ordered result; the immediately following
row is the position-shuffled result for the *same metric*. The headline column
shows the raw seed-42 reference value. Every window cell is
`100 * window / headline` and is therefore a percentage of the paper-protocol
headline. An em dash means that the cell was not evaluated. All displayed
percentages are rounded to the nearest whole number; raw headline metrics
remain at two decimal places.

| Task/control | Window-control metric | Headline, raw | `T=1` | `T=2` | `T=4` | `T=5` | `T=6` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sparse probing | Mean ROC-AUC, 20-feature probe | 0.89 | 101% | 101% | 101% | 100% | 100% |
| ↳ shuffled | Same metric | — | 101% | 101% | 99% | 99% | 98% |
| Backtracking | Detection PR-AUC, `S=32` | 0.25 | in progress | in progress | in progress | 100% | in progress |
| ↳ shuffled | Same metric | — | in progress | in progress | in progress | in progress | in progress |
| Medical EM | Detection PR-AUC, `S=16` | 0.54 | in progress | in progress | in progress | 100% | in progress |
| ↳ shuffled | Same metric | — | in progress | in progress | in progress | 111% | in progress |
| HH-RLHF | Preference ROC-AUC, top-20 projection | in progress | in progress | in progress | in progress | in progress | in progress |
| ↳ shuffled | Same metric | — | in progress | in progress | in progress | in progress | in progress |

Protocol notes:

- Sparse probing `T={1,2,4,6}` uses Han's paper-composition control grid at a
  20-feature probe budget. `T=5` uses the paper-matched checkpoint evaluation.
- Backtracking uses the 300k seed-42 TXC paper cell (`0.249917` at `S=32`) at
  `T=5`. Aniket's exact-recipe window reruns are in progress. The earlier 20k
  sweep is retained below as auxiliary evidence only.
- Medical EM uses the seed-42 v1 TXC paper cell at `T=5`. Exact v1 reruns that
  alter only `T` are in progress at `T={1,2,4,6}`; the earlier v2 extensions
  are excluded.
- HH-RLHF's previously collated seed-42 preference result (`0.609647`) came
  from `agentic_txc_02`, whose recorded class is
  `MatryoshkaTXCDRContrastiveMultiscale`; it is therefore TXC-pro under the
  manuscript's architecture definitions and is excluded. This is distinct
  from Han's 2026-07-27 window sweep, which trains the base-family
  `txc_batchtopk_post_btkonly` architecture (and its ReLU-mix twin
  `txc_batchtopk_post`) without Matryoshka or contrastive losses.

## Coverage

| Task | Live paper headline | TXC window-size coverage | Missing cells |
| --- | --- | --- | --- |
| Sparse probing | Mean ROC-AUC over the log-spaced top-feature sweep | Original paper `T={5,10,20}`; paired control at `T={1,2,4,5,6}` | Complete for seed 42 |
| Backtracking | Peak genuine-backtracking lift and PR-AUC | Exact TXC `T=5`; exact 300k window reruns in progress | `T={1,2,4,6}` and shuffled cells pending |
| Medical emergent misalignment | Alignment dynamic range and PR-AUC at `S=16` | Exact v1 TXC `T=5`; exact v1 `T={1,2,4,6}` reruns in progress | Reruns pending |
| HH-RLHF preference decomposition | Semantic and length-spurious features among top 20 | Base-TXC window sweep running; archived paper row was TXC-pro | Final base-TXC window cells pending |

## Sparse probing

The manuscript's three-seed aggregate is `0.90` summary ROC-AUC for TXC-base at
`T=5`; it remains `0.90` at `T=10` and `T=20` when rounded to two decimal
places. The reviewer table instead follows the seed-42 convention and uses the
paired 20-feature probe endpoint:

| `T` | Ordered ROC-AUC | Shuffled ROC-AUC |
| ---: | ---: | ---: |
| 1 | 0.90 | 0.90 |
| 2 | 0.90 | 0.90 |
| 4 | 0.90 | 0.88 |
| 5 | 0.89 | 0.88 |
| 6 | 0.89 | 0.87 |

The `T={1,2,4,6}` cells come from Han's paper-composition control grid on
`origin/arxiv`; `T=5` comes from the paper-matched checkpoint evaluation.
Shuffling has no effect at `T=1` and reaches an approximately `2.5%` relative
effect at `T=6`.

Sources: `origin/final:purified/experiments/c3_probing/results.json` and
`origin/arxiv:results/leaderboard.jsonl`.

## Backtracking

### Paper headline at 300k steps

The reviewer headline is always the seed-42 TXC-base paper cell at `T=5`.

| Axis | Paper-leading cell | Headline value |
| --- | --- | ---: |
| Inducement | TXC-base, batch 1,024 | peak `Delta gc=0.541` at magnitude `-12`; raw `gc=1.20` versus `0.66` unsteered |
| Detection | TXC-base, batch 1,024 | PR-AUC `0.201` at `S=8`; `0.249917` at `S=32` |

The exact 300k TXC-base artifact is available in the
[reviewer-results repository](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/c7_headline/seed42_published_eval.json).

### Aniket's latest auxiliary 20k detection window sweep

The following is a separate 20k-step TXC-base experiment. The reviewer table
uses the publication sweep's 32-feature detector and, by reporting convention,
seed 42.

| `T` | Ordered PR-AUC | Shuffled PR-AUC |
| ---: | ---: | ---: |
| 1 | 0.22 | 0.22 |
| 2 | 0.22 | 0.22 |
| 4 | 0.24 | 0.23 |
| 5 | 0.26 | 0.24 |
| 6 | 0.26 | 0.25 |

Ordered performance rises by about `18%` from `T=1` to its peak at `T=5`,
where the order-dependent relative gap is about `6.2%`. The seed-42 `T=6`
result remains about `17%` above `T=1`, with a `3.5%` order gap. This is
stronger evidence for window aggregation than for dependence on exact
within-window order.

Exact cells and provenance are in Aniket's commit-pinned
[publication CSV](https://github.com/chainik1125/temp_xc/blob/73b6d0bf09723619c043bde1b55ea695141ee499/purified/results/neurips_rebuttal/backtracking_window_sweep/full/publication/window_sweep_seed_metrics.csv).

**Empty:** there is no 300k paper-match sweep over `T`, and no window sweep of
the causal inducement metric. The 20k table must not be described as a
multi-window replication of the 300k headline.

## Medical emergent misalignment

The paper uses Qwen-2.5-7B-Instruct with the bad-medical-advice LoRA. The
manuscript headline is a two-seed mean, while the reviewer table follows the
seed-42 convention.

| Architecture | Window | Alignment range at coherence at least 70 | PR-AUC at `S=16` |
| --- | ---: | ---: | ---: |
| T-SAE, task best | N/A | 25.86 | 0.71 |
| TXC-base, manuscript mean | 5 | 19.80 | 0.55 |

The exact-paper-recipe seed-42 position-control cells are:

| `T` | Ordered PR-AUC | Shuffled PR-AUC |
| ---: | ---: | ---: |
| 1 | in progress | in progress |
| 2 | in progress | in progress |
| 4 | in progress | in progress |
| 5 | 0.54 | 0.60 |
| 6 | in progress | in progress |

At the paper window `T=5`, shuffling improves the Medical metric substantially.
Claims across window sizes must wait for the exact v1 reruns.

Paper sources:
`../temp_xc_tex/scripts/make_c6_em_figures.py` and the live Figure 7 prose in
`../temp_xc_tex/main.tex`. The published seed-1 and seed-42 detection artifacts
are in the
[Medical EM folder](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/tree/main/reviewer_seed_audit_2026-07-27/medical_em).
The excluded v2 `T=4` and `T=6` extension rows are on
`origin/dmitry-btk-txc-sprint:results/leaderboard.jsonl`; they changed the
training sampler and therefore are not paper-recipe cells.

**In progress:** seed-42 v1 TXC cells at `T={1,2,4,6}`. There is no window
sweep for the steering endpoint.

## HH-RLHF preference decomposition

The live Figure 8 headline is a feature-decomposition metric, not the
preference ROC-AUC used in the later seed audit. The archived HH-RLHF result
labelled `TXC` used `agentic_txc_02`, whose training metadata records
`MatryoshkaTXCDRContrastiveMultiscale`, three Matryoshka scales, and
contrastive shifts `{1,2,3}`. Under the manuscript's definitions this is
TXC-pro, so neither its feature counts nor its later preference ROC-AUC belong
in a TXC-base headline.

| Architecture | Window/budget | Semantic top-20 | Length-spurious top-20 |
| --- | --- | ---: | ---: |
| Paper-faithful T-SAE | `k=20` | 14/20 | 0/20 |
| Archived `agentic_txc_02` (TXC-pro; excluded) | `T=5`, `k_win=500` | 7/20 | 3/20 |

Han's checkpoint repository already contains the matched reconstruction-only
base architecture, `txc_bare_antidead_t5`, for seeds `1`, `2`, and `42`.
Its training log identifies `TXCBareAntidead` at `T=5`, `d_sae=18432`,
`k_pos=100`, and `k_win=500`, with no Matryoshka or contrastive term. Thus no
new training is required to evaluate that historical base architecture.

Separately, Han's 2026-07-27 window experiment trains
`txc_batchtopk_post_btkonly` across window sizes, with a matched
`txc_batchtopk_post` ReLU-mix arm. Both are registered as category `txc`; the
registry begins the TXC-pro loss-dissection entries separately. This sweep is
therefore base TXC rather than TXC-pro, although it is a newer BatchTopK-post
recipe rather than the historical `txc_bare_antidead_t5` checkpoint family.

The TXC-base preference-classification grid is therefore:

| `T` | Ordered ROC-AUC | Shuffled ROC-AUC |
| ---: | ---: | ---: |
| 1 | in progress | in progress |
| 2 | in progress | in progress |
| 4 | in progress | in progress |
| 5 | in progress | in progress |
| 6 | in progress | in progress |

The earlier `0.609647` ordered and `0.597529` shuffled values are retained in
the artifact audit as excluded TXC-pro results, not as headline TXC values.

**In progress:** finish and collate Han's base-TXC window sweep. The
historical `txc_bare_antidead_t5` checkpoints remain available if an exact
checkpoint-family comparison with the old paper row is also required.

## Manuscript consistency flags

- Figure 1's rose sidecar reports Backtracking TXC-base peak
  `Delta gc=0.426`, while the current Figure 6/macros report `0.541`.
- Figure 1's axis named `RLHF` is populated from
  `purified/experiments/c5_steering/results.json` using peak steering grade at a
  coherence floor. The live HH-RLHF section instead headlines semantic and
  length-spurious feature counts. These are different tasks/endpoints.
- The live HH-RLHF prose calls `agentic_txc_02` “TXC,” but its recorded class
  and objectives identify it as TXC-pro. This does not apply to Han's
  2026-07-27 window sweep, which uses a base TXC architecture.
- The draft reviewer-response claim that every task becomes uncompetitive under
  shuffling is not supported. The completed controls show task-dependent
  effects: Sparse Probing has a modest positive gap, Backtracking has a
  positive gap at longer windows, Medical improves under shuffling, and
  HH-RLHF is mixed.

## Copy-ready conservative summary

The paper-matched seed-42 grid is still in progress. Sparse probing reaches a
`3`-point ordered--shuffled gap at `T=6`, while Medical improves under
shuffling at the paper window. HH-RLHF's archived result used TXC-pro and is
excluded, while Han's replacement base-TXC window sweep is in progress. We
will summarize cross-window effects only after exact TXC runs replace every
auxiliary or pro checkpoint.
