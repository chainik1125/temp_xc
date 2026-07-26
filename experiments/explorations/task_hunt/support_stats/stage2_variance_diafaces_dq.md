# Stage-2 panel — variance receipts (`dial_real_dqgap_llama31_8b_l14`, probe v1)

Source: 102 leaderboard rows, datasource `dial_real_dqgap_llama31_8b_l14`, metric `lambda_recovery`, seeds [1, 2, 42]; cross-check vs `stage2_dial_real_dqgap_llama31_8b_l14.json`: exact (all 102 cells). Built by `stage2_variance.py` — every number below is script-derived.

Selection: probe v1, row layout paired, k_pos 8 (post rule fixed).

## Per-seed values (trained), λ̂ recovery

| cell | seed 1 | seed 2 | seed 42 | mean | 95% t CI |
|---|---|---|---|---|---|
| batchtopk_sae/T1 | 0.2366 | 0.2202 | 0.2285 | 0.2284 | [0.2080, 0.2488] |
| stacked_batchtopk/T2 | 0.2550 | 0.2547 | 0.2338 | 0.2478 | [0.2175, 0.2782] |
| stacked_batchtopk/T4 | 0.2916 | 0.2928 | 0.2885 | 0.2910 | [0.2854, 0.2965] |
| stacked_batchtopk/T8 | 0.3048 | 0.3017 | 0.3036 | 0.3034 | [0.2995, 0.3073] |
| stacked_batchtopk/T16 | 0.2012 | 0.2509 | 0.2167 | 0.2229 | [0.1598, 0.2861] |
| stacked_batchtopk/T32 | 0.2719 | 0.2765 | 0.3131 | 0.2872 | [0.2311, 0.3433] |
| tsae/T1 | 0.2562 | 0.2552 | 0.2379 | 0.2498 | [0.2242, 0.2754] |
| txc_batchtopk_post/T2 | 0.2427 | 0.2293 | 0.2497 | 0.2406 | [0.2148, 0.2663] |
| txc_batchtopk_post/T4 | 0.2888 | 0.2786 | 0.2831 | 0.2835 | [0.2709, 0.2961] |
| txc_batchtopk_post/T8 | 0.2949 | 0.2656 | 0.2853 | 0.2820 | [0.2448, 0.3191] |
| txc_batchtopk_post/T16 | 0.3109 | 0.3028 | 0.3050 | 0.3062 | [0.2958, 0.3166] |
| txc_batchtopk_post/T32 | 0.2858 | 0.2783 | 0.3113 | 0.2918 | [0.2489, 0.3347] |
| txc_batchtopk_pre/T2 | 0.2775 | 0.2923 | 0.2836 | 0.2844 | [0.2660, 0.3028] |
| txc_batchtopk_pre/T4 | 0.3767 | 0.3907 | 0.3722 | 0.3798 | [0.3559, 0.4038] |
| txc_batchtopk_pre/T8 | 0.4123 | 0.3978 | 0.4042 | 0.4048 | [0.3867, 0.4228] |
| txc_batchtopk_pre/T16 | 0.4104 | 0.3907 | 0.4072 | 0.4028 | [0.3765, 0.4290] |
| txc_batchtopk_pre/T32 | 0.2300 | 0.2238 | 0.2433 | 0.2324 | [0.2076, 0.2571] |

## Paired-by-seed differences (window arch − T=1 reference)

### txc_pre_minus_tsae (reference tsae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.0213 | 0.0370 | 0.0456 | 0.0346 | 0.0124 | [0.0040, 0.0653] | [0.0213, 0.0428] | 0.125 | +0.05 |
| T4 | 0.1204 | 0.1354 | 0.1343 | 0.1300 | 0.0084 | [0.1093, 0.1508] | [0.1204, 0.1351] | 0.125 | +0.65 |
| T8 | 0.1560 | 0.1426 | 0.1662 | 0.1550 | 0.0119 | [0.1255, 0.1844] | [0.1426, 0.1662] | 0.125 | +0.12 |
| T16 | 0.1541 | 0.1355 | 0.1693 | 0.1530 | 0.0169 | [0.1109, 0.1950] | [0.1355, 0.1693] | 0.125 | -0.32 |
| T32 | -0.0263 | -0.0314 | 0.0054 | -0.0174 | 0.0199 | [-0.0669, 0.0321] | [-0.0297, 0.0054] | 0.875 | -0.93 |

### txc_pre_minus_pertoken (reference batchtopk_sae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.0409 | 0.0721 | 0.0551 | 0.0560 | 0.0156 | [0.0173, 0.0948] | [0.0409, 0.0721] | 0.125 | -1.00 |
| T4 | 0.1400 | 0.1705 | 0.1437 | 0.1514 | 0.0166 | [0.1102, 0.1927] | [0.1413, 0.1705] | 0.125 | -0.73 |
| T8 | 0.1757 | 0.1776 | 0.1757 | 0.1763 | 0.0011 | [0.1735, 0.1791] | [0.1757, 0.1776] | 0.125 | +1.00 |
| T16 | 0.1737 | 0.1705 | 0.1787 | 0.1743 | 0.0042 | [0.1640, 0.1846] | [0.1716, 0.1787] | 0.125 | +0.93 |
| T32 | -0.0067 | 0.0036 | 0.0148 | 0.0039 | 0.0108 | [-0.0228, 0.0306] | [-0.0067, 0.0148] | 0.375 | +0.31 |

## Trend across T (exact within-seed permutation, pooled seeds)

| test | Ts | Σ slopes (per log₂T) | per-seed slopes | p (1-sided) | perms |
|---|---|---|---|---|---|
| txc_pre_trained_2to8 | [2, 4, 8] | 0.1805 | 0.0674, 0.0528, 0.0603 | 0.0046 | 216 |
| txc_pre_margin_2to8 | [2, 4, 8] | 0.2342 | 0.0905, 0.0681, 0.0756 | 0.0046 | 216 |

- **txc_pre_trained_2to32_secondary**: secondary full-ladder trend over T=[2, 4, 8, 16, 32] not computed: exact enumeration too large; reduce Ts/seeds — the frozen 2->8 primary carries the trend receipt; per-cell values are all reported

## Trained − untrained margin (paired by seed), key cells

| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |
|---|---|---|---|---|
| batchtopk_sae/T1 | 0.1287 | [0.0760, 0.1814] | [0.1073, 0.1497] | 0.125 |
| stacked_batchtopk/T2 | 0.1118 | [0.0679, 0.1557] | [0.0915, 0.1225] | 0.125 |
| stacked_batchtopk/T4 | 0.1966 | [0.1478, 0.2454] | [0.1836, 0.2176] | 0.125 |
| stacked_batchtopk/T8 | 0.1532 | [0.1014, 0.2051] | [0.1404, 0.1771] | 0.125 |
| stacked_batchtopk/T16 | 0.0284 | [-0.0276, 0.0845] | [0.0030, 0.0428] | 0.125 |
| stacked_batchtopk/T32 | 0.1061 | [0.0118, 0.2004] | [0.0656, 0.1312] | 0.125 |
| tsae/T1 | 0.1500 | [0.1083, 0.1918] | [0.1397, 0.1693] | 0.125 |
| txc_batchtopk_post/T2 | 0.1455 | [0.1322, 0.1589] | [0.1403, 0.1511] | 0.125 |
| txc_batchtopk_post/T4 | 0.2142 | [0.1631, 0.2653] | [0.1905, 0.2265] | 0.125 |
| txc_batchtopk_post/T8 | 0.2440 | [0.2293, 0.2588] | [0.2372, 0.2475] | 0.125 |
| txc_batchtopk_post/T16 | 0.2977 | [0.2731, 0.3224] | [0.2863, 0.3038] | 0.125 |
| txc_batchtopk_post/T32 | 0.2860 | [0.2183, 0.3537] | [0.2692, 0.3172] | 0.125 |
| txc_batchtopk_pre/T2 | 0.1626 | [0.1464, 0.1788] | [0.1560, 0.1691] | 0.125 |
| txc_batchtopk_pre/T4 | 0.2602 | [0.2099, 0.3105] | [0.2476, 0.2833] | 0.125 |
| txc_batchtopk_pre/T8 | 0.3187 | [0.2780, 0.3595] | [0.3082, 0.3370] | 0.125 |
| txc_batchtopk_pre/T16 | 0.3910 | [0.3550, 0.4271] | [0.3815, 0.4067] | 0.125 |
| txc_batchtopk_pre/T32 | 0.1387 | [0.0327, 0.2446] | [0.0928, 0.1668] | 0.125 |

## Power calc → seed recommendation

- Exact sign-flip attainability: p ≤ 0.05 first possible at **n = 5 seeds** (2⁻ⁿ ≤ 0.05).
- txc_pre_minus_tsae @T4: observed 0.1300 ± 0.0084; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- txc_pre_minus_tsae @T8: observed 0.1550 ± 0.0119; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- txc_pre_minus_pertoken @T4: observed 0.1514 ± 0.0166; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- txc_pre_minus_pertoken @T8: observed 0.1763 ± 0.0011; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- Criterion: one-sided 95% t lower bound > 0 on the paired TXC-pre - T-SAE diff at T = 8, AND exact sign-flip attainability (2^-n <= 0.05).
- **Recommendation:** total seeds needed **5** ⇒ **2 extra seeds**. Per extra seed (trained): txc_batchtopk_pre/T4, txc_batchtopk_pre/T8, tsae/T1 ⇒ 6 trained cells (+6 optional untrained counterparts). Headroom option: 3 extra seeds = 9 cells — one seed of slack against the plug-in sd estimate itself being an n=3 estimate; also reaches sign-flip p = 1/128 at the T8 cell.
- T4 is NOT cheaply boundable (n = 2 to bound, 2 for 80% power); the T-rise + trained−untrained margin carry that cell.

## Honesty notes

- n = 3 seeds: the exact one-sided sign-flip permutation test cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired direction is consistent in all 3 seeds', not as significance.
- The exact bootstrap distribution of a 3-value mean has 27 atoms (<= 10 distinct values); BCa endpoints are coarse and cannot extend past the extreme seed values.
- Pairing by seed was the right design a priori, but it bought no variance reduction here: at the T8 headline cell the across-seed correlation between the TXC-pre and T-SAE arms is r = 0.12, so the paired sd (0.0119) is not below the independent-arms value (0.0126). The cross-arch margin is therefore NOT bounded away from 0 at n = 3; the receipts that ARE significant at n = 3 are within-arch: the T = 2->8 rise and the trained-untrained margins (paired by seed WITHIN an arch, where the pairing does bind).
- The T = 2->8 trend test is exact with 216 relabelings (min p = 1/216), so it carries real resolution at n = 3.
