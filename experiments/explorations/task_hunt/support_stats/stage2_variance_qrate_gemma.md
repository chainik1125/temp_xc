# Stage-2 panel — variance receipts (`fineweb_punctint_q_gemma2_l14`, probe v1)

Source: 84 leaderboard rows, datasource `fineweb_punctint_q_gemma2_l14`, metric `lambda_recovery`, seeds [1, 2, 42]; cross-check vs `stage2_fineweb_punctint_q_gemma2_l14.json`: exact (all 84 cells). Built by `stage2_variance.py` — every number below is script-derived.

Selection: probe v1, row layout paired, k_pos 8 (post rule times-T).

## Per-seed values (trained), λ̂ recovery

| cell | seed 1 | seed 2 | seed 42 | mean | 95% t CI |
|---|---|---|---|---|---|
| batchtopk_sae/T1 | 0.1751 | 0.2597 | 0.1522 | 0.1957 | [0.0550, 0.3363] |
| stacked_batchtopk/T2 | 0.2291 | 0.2726 | 0.2244 | 0.2421 | [0.1760, 0.3081] |
| stacked_batchtopk/T4 | 0.2239 | 0.2106 | 0.2483 | 0.2276 | [0.1801, 0.2750] |
| stacked_batchtopk/T8 | 0.1965 | 0.1732 | 0.1186 | 0.1628 | [0.0634, 0.2621] |
| stacked_batchtopk/T16 | 0.1096 | 0.0973 | 0.1106 | 0.1058 | [0.0874, 0.1242] |
| tsae/T1 | 0.2481 | 0.1429 | 0.1457 | 0.1789 | [0.0301, 0.3277] |
| txc_batchtopk_post/T2 | 0.2347 | 0.2777 | 0.2714 | 0.2613 | [0.2036, 0.3189] |
| txc_batchtopk_post/T4 | 0.2540 | 0.2463 | 0.2951 | 0.2652 | [0.2000, 0.3303] |
| txc_batchtopk_post/T8 | 0.1894 | 0.2868 | 0.2076 | 0.2279 | [0.0992, 0.3566] |
| txc_batchtopk_post/T16 | 0.1944 | 0.0484 | 0.3009 | 0.1812 | [-0.1337, 0.4962] |
| txc_batchtopk_pre/T2 | 0.2176 | 0.2595 | 0.1333 | 0.2035 | [0.0438, 0.3631] |
| txc_batchtopk_pre/T4 | 0.2597 | 0.2130 | 0.2293 | 0.2340 | [0.1751, 0.2929] |
| txc_batchtopk_pre/T8 | 0.2693 | 0.2580 | 0.2221 | 0.2498 | [0.1886, 0.3110] |
| txc_batchtopk_pre/T16 | 0.2407 | 0.1566 | 0.1930 | 0.1968 | [0.0920, 0.3015] |

## Paired-by-seed differences (window arch − T=1 reference)

### txc_pre_minus_tsae (reference tsae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | -0.0305 | 0.1166 | -0.0124 | 0.0246 | 0.0802 | [-0.1747, 0.2238] | [-0.0245, 0.1166] | 0.500 | +0.17 |
| T4 | 0.0117 | 0.0700 | 0.0836 | 0.0551 | 0.0382 | [-0.0399, 0.1501] | [0.0117, 0.0791] | 0.125 | +0.95 |
| T8 | 0.0213 | 0.1151 | 0.0764 | 0.0709 | 0.0471 | [-0.0462, 0.1880] | [0.0213, 0.1151] | 0.125 | +0.67 |
| T16 | -0.0074 | 0.0137 | 0.0473 | 0.0179 | 0.0276 | [-0.0507, 0.0864] | [-0.0003, 0.0473] | 0.250 | +0.91 |

### txc_pre_minus_pertoken (reference batchtopk_sae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.0424 | -0.0002 | -0.0189 | 0.0078 | 0.0314 | [-0.0703, 0.0859] | [-0.0127, 0.0424] | 0.500 | +0.87 |
| T4 | 0.0846 | -0.0467 | 0.0771 | 0.0383 | 0.0738 | [-0.1449, 0.2216] | [-0.0467, 0.0821] | 0.250 | -0.62 |
| T8 | 0.0942 | -0.0017 | 0.0699 | 0.0541 | 0.0499 | [-0.0697, 0.1780] | [-0.0017, 0.0861] | 0.250 | +0.47 |
| T16 | 0.0656 | -0.1031 | 0.0408 | 0.0011 | 0.0911 | [-0.2251, 0.2274] | [-0.1031, 0.0573] | 0.500 | -0.69 |

## Trend across T (exact within-seed permutation, pooled seeds)

| test | Ts | Σ slopes (per log₂T) | per-seed slopes | p (1-sided) | perms |
|---|---|---|---|---|---|
| txc_pre_trained_2to8 | [2, 4, 8] | 0.0695 | 0.0259, -0.0008, 0.0444 | 0.0787 | 216 |
| txc_pre_margin_2to8 | [2, 4, 8] | 0.0601 | 0.0316, -0.0150, 0.0435 | 0.1389 | 216 |
| txc_pre_trained_2to16_secondary | [2, 4, 8, 16] | -0.0013 | 0.0079, -0.0264, 0.0172 | 0.5156 | 13824 |

## Trained − untrained margin (paired by seed), key cells

| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |
|---|---|---|---|---|
| batchtopk_sae/T1 | 0.1549 | [-0.0128, 0.3225] | [0.1125, 0.2315] | 0.125 |
| stacked_batchtopk/T2 | 0.2075 | [0.1731, 0.2419] | [0.1990, 0.2234] | 0.125 |
| stacked_batchtopk/T4 | 0.1933 | [0.1036, 0.2830] | [0.1697, 0.2330] | 0.125 |
| stacked_batchtopk/T8 | 0.1258 | [0.0061, 0.2456] | [0.0713, 0.1563] | 0.125 |
| stacked_batchtopk/T16 | 0.0593 | [0.0227, 0.0960] | [0.0434, 0.0690] | 0.125 |
| tsae/T1 | 0.1381 | [-0.0004, 0.2767] | [0.1035, 0.2018] | 0.125 |
| txc_batchtopk_post/T2 | 0.2112 | [0.1754, 0.2469] | [0.1947, 0.2210] | 0.125 |
| txc_batchtopk_post/T4 | 0.2139 | [0.1660, 0.2618] | [0.2018, 0.2358] | 0.125 |
| txc_batchtopk_post/T8 | 0.1732 | [0.0278, 0.3185] | [0.1103, 0.2117] | 0.125 |
| txc_batchtopk_post/T16 | 0.1481 | [-0.1501, 0.4463] | [0.0256, 0.2655] | 0.125 |
| txc_batchtopk_pre/T2 | 0.1642 | [-0.0193, 0.3477] | [0.0887, 0.2134] | 0.125 |
| txc_batchtopk_pre/T4 | 0.1991 | [0.1494, 0.2487] | [0.1860, 0.2209] | 0.125 |
| txc_batchtopk_pre/T8 | 0.2043 | [0.1358, 0.2728] | [0.1758, 0.2308] | 0.125 |
| txc_batchtopk_pre/T16 | 0.1554 | [0.0475, 0.2633] | [0.1278, 0.2044] | 0.125 |

## Power calc → seed recommendation

- Exact sign-flip attainability: p ≤ 0.05 first possible at **n = 5 seeds** (2⁻ⁿ ≤ 0.05).
- txc_pre_minus_tsae @T4: observed 0.0551 ± 0.0382; n for 95% lower bound > 0: **4**; n for 80% power (one-sided t, α=0.05): **5**.
- txc_pre_minus_tsae @T8: observed 0.0709 ± 0.0471; n for 95% lower bound > 0: **4**; n for 80% power (one-sided t, α=0.05): **5**.
- txc_pre_minus_pertoken @T4: observed 0.0383 ± 0.0738; n for 95% lower bound > 0: **12**; n for 80% power (one-sided t, α=0.05): **25**.
- txc_pre_minus_pertoken @T8: observed 0.0541 ± 0.0499; n for 95% lower bound > 0: **5**; n for 80% power (one-sided t, α=0.05): **7**.
- Criterion: one-sided 95% t lower bound > 0 on the paired TXC-pre - T-SAE diff at T = 8, AND exact sign-flip attainability (2^-n <= 0.05).
- **Recommendation:** total seeds needed **5** ⇒ **2 extra seeds**. Per extra seed (trained): txc_batchtopk_pre/T4, txc_batchtopk_pre/T8, tsae/T1 ⇒ 6 trained cells (+6 optional untrained counterparts). Headroom option: 3 extra seeds = 9 cells — one seed of slack against the plug-in sd estimate itself being an n=3 estimate; also reaches sign-flip p = 1/128 at the T8 cell.
- T4 is NOT cheaply boundable (n = 4 to bound, 5 for 80% power); the T-rise + trained−untrained margin carry that cell.

## Honesty notes

- n = 3 seeds: the exact one-sided sign-flip permutation test cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired direction is consistent in all 3 seeds', not as significance.
- The exact bootstrap distribution of a 3-value mean has 27 atoms (<= 10 distinct values); BCa endpoints are coarse and cannot extend past the extreme seed values.
- Pairing by seed was the right design a priori, but it bought no variance reduction here: at the T8 headline cell the across-seed correlation between the TXC-pre and T-SAE arms is r = 0.67, so the paired sd (0.0471) is not below the independent-arms value (0.0648). The cross-arch margin is therefore NOT bounded away from 0 at n = 3; the receipts that ARE significant at n = 3 are within-arch: the T = 2->8 rise and the trained-untrained margins (paired by seed WITHIN an arch, where the pairing does bind).
- The T = 2->8 trend test is exact with 216 relabelings (min p = 1/216), so it carries real resolution at n = 3.
