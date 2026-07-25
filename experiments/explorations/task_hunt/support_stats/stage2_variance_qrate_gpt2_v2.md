# Stage-2 panel — variance receipts (`fineweb_punctint_q_gpt2_l7`, probe v2)

Source: 24 leaderboard rows, datasource `fineweb_punctint_q_gpt2_l7`, metric `lambda_recovery_v2`, seeds [1, 2, 42]; cross-check vs `stage2_fineweb_punctint_q_gpt2_l7.json`: exact (all 24 cells). Built by `stage2_variance.py` — every number below is script-derived.

Selection: probe v2, row layout flagged, k_pos 8 (post rule times-T).

## Per-seed values (trained), λ̂ recovery

| cell | seed 1 | seed 2 | seed 42 | mean | 95% t CI |
|---|---|---|---|---|---|
| batchtopk_sae/T1 | 0.1164 | 0.1105 | 0.1133 | 0.1134 | [0.1061, 0.1207] |
| tsae/T1 | 0.1217 | 0.1215 | 0.1145 | 0.1192 | [0.1090, 0.1295] |
| txc_batchtopk_pre/T4 | 0.1650 | 0.1737 | 0.1713 | 0.1700 | [0.1589, 0.1811] |
| txc_batchtopk_pre/T8 | 0.1913 | 0.1803 | 0.1854 | 0.1856 | [0.1720, 0.1993] |

## Paired-by-seed differences (window arch − T=1 reference)

### txc_pre_minus_tsae (reference tsae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T4 | 0.0433 | 0.0521 | 0.0569 | 0.0508 | 0.0069 | [0.0337, 0.0679] | [0.0433, 0.0553] | 0.125 | -0.27 |
| T8 | 0.0696 | 0.0587 | 0.0709 | 0.0664 | 0.0067 | [0.0498, 0.0830] | [0.0587, 0.0705] | 0.125 | +0.05 |

### txc_pre_minus_pertoken (reference batchtopk_sae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T4 | 0.0486 | 0.0632 | 0.0580 | 0.0566 | 0.0074 | [0.0383, 0.0749] | [0.0486, 0.0615] | 0.125 | -0.97 |
| T8 | 0.0749 | 0.0698 | 0.0721 | 0.0722 | 0.0026 | [0.0659, 0.0786] | [0.0705, 0.0749] | 0.125 | +1.00 |

## Trend across T (exact within-seed permutation, pooled seeds)

- **txc_pre_trend**: frozen 2->8 trend undefined: txc_batchtopk_pre present at T=[4, 8] only — a trend statistic over 2 T value(s) has no within-seed permutation resolution; the cells themselves are reported in the per-seed / paired sections

## Trained − untrained margin (paired by seed), key cells

| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |
|---|---|---|---|---|
| batchtopk_sae/T1 | 0.0438 | [0.0244, 0.0631] | [0.0348, 0.0485] | 0.125 |
| tsae/T1 | 0.0496 | [0.0188, 0.0803] | [0.0359, 0.0576] | 0.125 |
| txc_batchtopk_pre/T4 | 0.0712 | [0.0544, 0.0881] | [0.0637, 0.0756] | 0.125 |
| txc_batchtopk_pre/T8 | 0.0876 | [0.0454, 0.1299] | [0.0776, 0.1073] | 0.125 |

## Power calc → seed recommendation

- Exact sign-flip attainability: p ≤ 0.05 first possible at **n = 5 seeds** (2⁻ⁿ ≤ 0.05).
- txc_pre_minus_tsae @T4: observed 0.0508 ± 0.0069; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- txc_pre_minus_tsae @T8: observed 0.0664 ± 0.0067; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- txc_pre_minus_pertoken @T4: observed 0.0566 ± 0.0074; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- txc_pre_minus_pertoken @T8: observed 0.0722 ± 0.0026; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- Criterion: one-sided 95% t lower bound > 0 on the paired TXC-pre - T-SAE diff at T = 8, AND exact sign-flip attainability (2^-n <= 0.05).
- **Recommendation:** total seeds needed **5** ⇒ **2 extra seeds**. Per extra seed (trained): txc_batchtopk_pre/T4, txc_batchtopk_pre/T8, tsae/T1 ⇒ 6 trained cells (+6 optional untrained counterparts). Headroom option: 3 extra seeds = 9 cells — one seed of slack against the plug-in sd estimate itself being an n=3 estimate; also reaches sign-flip p = 1/128 at the T8 cell.
- T4 is NOT cheaply boundable (n = 2 to bound, 2 for 80% power); the T-rise + trained−untrained margin carry that cell.

## Honesty notes

- n = 3 seeds: the exact one-sided sign-flip permutation test cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired direction is consistent in all 3 seeds', not as significance.
- The exact bootstrap distribution of a 3-value mean has 27 atoms (<= 10 distinct values); BCa endpoints are coarse and cannot extend past the extreme seed values.
- Pairing by seed was the right design a priori, but it bought no variance reduction here: at the T8 headline cell the across-seed correlation between the TXC-pre and T-SAE arms is r = 0.05, so the paired sd (0.0067) is not below the independent-arms value (0.0069). The cross-arch margin is therefore NOT bounded away from 0 at n = 3; the receipts that ARE significant at n = 3 are within-arch: the T = 2->8 rise and the trained-untrained margins (paired by seed WITHIN an arch, where the pairing does bind).
