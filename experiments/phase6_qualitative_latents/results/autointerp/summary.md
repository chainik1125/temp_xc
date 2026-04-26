# Phase 6.1 autointerp summary (upgraded metric)

N_TOP_FEATURES = 32  (from 8).  Multi-judge = 2 Haiku prompts.

## Per-cell metrics

| arch | seed | concat | /N sem | cov k/P | cov entropy | judge disagree | pdvar ∩var |
|---|---|---|---|---|---|---|---|
| phase57_partB_h8_bare_multidistance | 2 | A | 17/32 | 3/3 | 1.06 | 0.00 | 17 |
| phase57_partB_h8_bare_multidistance | 2 | B | 19/32 | 4/4 | 1.10 | 0.03 | 25 |
| phase57_partB_h8_bare_multidistance | 2 | random | 2/32 | 6/7 | 1.57 | 0.16 | 21 |

## Means ± stderr across seeds (per arch × concat)

| arch | concat | n_seeds | mean /N sem | mean cov k/P | mean cov entropy |
|---|---|---|---|---|---|
| phase57_partB_h8_bare_multidistance | A | 1 | 17.0 ± 0.00 | 3.0 ± 0.00 | 1.06 ± 0.00 |
| phase57_partB_h8_bare_multidistance | B | 1 | 19.0 ± 0.00 | 4.0 ± 0.00 | 1.10 ± 0.00 |
| phase57_partB_h8_bare_multidistance | random | 1 | 2.0 ± 0.00 | 6.0 ± 0.00 | 1.57 ± 0.00 |
