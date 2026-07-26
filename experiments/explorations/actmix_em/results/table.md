# ACTMIX P2 — EM btk-only: the exhibit table

Datasource `qwen_2_5_7b_instruct_medical_l15` (BASE-forward train, organism cohort detect, L15). Arm: **btk-only** (arch-name suffix carries the arm; relu_mode hashes into train_key). Primary pr_auc_S16; positive-rate floor 0.323. Nominal budget 20/token (txc: 20·T per window).

## seed 42

| T | TXC | TXC-shuffled | gap | SAE | TSAE | TXC l0/tok | untrained TXC |
|---|---|---|---|---|---|---|---|
| 1 | — | — | — | — | — | — | — |
| 2 | — | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — | — |
| 8 | — | — | — | — | — | — | — |
| 16 | — | — | — | — | — | — | — |

SAE realized l0/tok —; TSAE —; SAE/TSAE are per-token (T-invariant bands; no within-window shuffle exists at T = 1 — order-invariance holds by construction).

## seed 1

| T | TXC | TXC-shuffled | gap | SAE | TSAE | TXC l0/tok | untrained TXC |
|---|---|---|---|---|---|---|---|
| 1 | — | — | — | — | — | — | — |
| 2 | — | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — | — |
| 8 | — | — | — | — | — | — | — |
| 16 | — | — | — | — | — | — | — |

SAE realized l0/tok —; TSAE —; SAE/TSAE are per-token (T-invariant bands; no within-window shuffle exists at T = 1 — order-invariance holds by construction).

## seed mean

| T | TXC | TXC-shuffled | gap | SAE | TSAE | TXC l0/tok | untrained TXC |
|---|---|---|---|---|---|---|---|
| 1 | — | — | — | — | — | — | — |
| 2 | — | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — | — |
| 8 | — | — | — | — | — | — | — |
| 16 | — | — | — | — | — | — | — |

SAE realized l0/tok —; TSAE —; SAE/TSAE are per-token (T-invariant bands; no within-window shuffle exists at T = 1 — order-invariance holds by construction).

## Side-by-side with the paper's published § 5.3 negative

| cell | pr_auc_S16 s42 | s1 | shuffle_gap s42 | s1 |
|---|---|---|---|---|
| paper sae_arditi (128/tok) | 0.690 | 0.745 | — | — |
| paper txc_base (k25, T5) | 0.542 | 0.560 | -0.059 | -0.002 |

- (a) paper cells probed PER-CELL Wang cohorts (n_sent 79k-107k, base rates 0.32-0.47) vs this run's ONE fixed 1728-rollout cohort (0.323); PR-AUC is base-rate sensitive - cross-design deltas are context, not measurements
- (b) budgets differ (arditi 128/token vs panel 20/token; txc_base k_pos 25, T 5 paper knobs)
- (c) composition differs BY DESIGN - that is the ablation

## Mechanical scoring (CARD § 4)

```json
{
 "scored_at_cells": 0,
 "K3_cohort_integrity": {
  "violations": [],
  "pass": true
 },
 "K1_sae_falsifier": {
  "sae_pr_auc_S16": [],
  "pass": null
 },
 "K2_under_realization": {
  "cells": {},
  "pass": null
 },
 "E1_negative_persists": {
  "per_T": {},
  "holds": null
 },
 "E2_shuffle_below_bar": {
  "per_T": {},
  "holds": null
 },
 "E4_t1_limit": {
  "txc_T1": null,
  "sae": null,
  "delta": null,
  "holds": null
 },
 "E5_untrained_floors": {}
}
```
