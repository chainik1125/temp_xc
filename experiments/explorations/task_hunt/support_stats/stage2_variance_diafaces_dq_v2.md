# Stage-2 panel — variance receipts (`dial_real_dqgap_llama31_8b_l14`, probe v2)

Source: 102 leaderboard rows, datasource `dial_real_dqgap_llama31_8b_l14`, metric `lambda_recovery_v2`, seeds [1, 2, 42]; cross-check vs `stage2_dial_real_dqgap_llama31_8b_l14.json`: exact (all 102 cells). Built by `stage2_variance.py` — every number below is script-derived.

Selection: probe v2, row layout flagged, k_pos 8 (post rule fixed).

## Per-seed values (trained), λ̂ recovery

| cell | seed 1 | seed 2 | seed 42 | mean | 95% t CI |
|---|---|---|---|---|---|
| batchtopk_sae/T1 | 0.2376 | 0.2330 | 0.2388 | 0.2365 | [0.2288, 0.2441] |
| stacked_batchtopk/T2 | 0.2888 | 0.2964 | 0.2881 | 0.2911 | [0.2797, 0.3025] |
| stacked_batchtopk/T4 | 0.3501 | 0.3603 | 0.3566 | 0.3557 | [0.3429, 0.3684] |
| stacked_batchtopk/T8 | 0.4135 | 0.4236 | 0.4209 | 0.4193 | [0.4063, 0.4324] |
| stacked_batchtopk/T16 | 0.4813 | 0.4869 | 0.4844 | 0.4842 | [0.4772, 0.4911] |
| stacked_batchtopk/T32 | 0.5387 | 0.5361 | 0.5359 | 0.5369 | [0.5330, 0.5408] |
| tsae/T1 | 0.2842 | 0.2866 | 0.2796 | 0.2835 | [0.2747, 0.2922] |
| txc_batchtopk_post/T2 | 0.2759 | 0.2771 | 0.2849 | 0.2793 | [0.2672, 0.2915] |
| txc_batchtopk_post/T4 | 0.3039 | 0.2995 | 0.3076 | 0.3036 | [0.2936, 0.3137] |
| txc_batchtopk_post/T8 | 0.3350 | 0.3035 | 0.3404 | 0.3263 | [0.2769, 0.3758] |
| txc_batchtopk_post/T16 | 0.3647 | 0.3719 | 0.3691 | 0.3686 | [0.3596, 0.3776] |
| txc_batchtopk_post/T32 | 0.3049 | 0.2753 | 0.3123 | 0.2975 | [0.2488, 0.3462] |
| txc_batchtopk_pre/T2 | 0.3174 | 0.3254 | 0.3169 | 0.3199 | [0.3080, 0.3317] |
| txc_batchtopk_pre/T4 | 0.4241 | 0.4116 | 0.4138 | 0.4165 | [0.4000, 0.4331] |
| txc_batchtopk_pre/T8 | 0.4717 | 0.4698 | 0.4738 | 0.4718 | [0.4667, 0.4768] |
| txc_batchtopk_pre/T16 | 0.5186 | 0.5315 | 0.5212 | 0.5237 | [0.5068, 0.5407] |
| txc_batchtopk_pre/T32 | 0.5620 | 0.5720 | 0.5600 | 0.5647 | [0.5488, 0.5805] |

## Paired-by-seed differences (window arch − T=1 reference)

### txc_pre_minus_tsae (reference tsae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.0332 | 0.0388 | 0.0372 | 0.0364 | 0.0029 | [0.0293, 0.0436] | [0.0332, 0.0383] | 0.125 | +0.80 |
| T4 | 0.1400 | 0.1251 | 0.1342 | 0.1331 | 0.0075 | [0.1144, 0.1517] | [0.1251, 0.1380] | 0.125 | +0.01 |
| T8 | 0.1876 | 0.1832 | 0.1942 | 0.1883 | 0.0055 | [0.1746, 0.2021] | [0.1832, 0.1942] | 0.125 | -0.99 |
| T16 | 0.2344 | 0.2449 | 0.2415 | 0.2403 | 0.0053 | [0.2270, 0.2535] | [0.2344, 0.2438] | 0.125 | +0.63 |
| T32 | 0.2778 | 0.2854 | 0.2804 | 0.2812 | 0.0038 | [0.2717, 0.2907] | [0.2787, 0.2854] | 0.125 | +0.86 |

### txc_pre_minus_pertoken (reference batchtopk_sae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.0798 | 0.0924 | 0.0781 | 0.0834 | 0.0078 | [0.0640, 0.1029] | [0.0787, 0.0924] | 0.125 | -0.99 |
| T4 | 0.1865 | 0.1787 | 0.1751 | 0.1801 | 0.0058 | [0.1656, 0.1946] | [0.1751, 0.1865] | 0.125 | +0.48 |
| T8 | 0.2341 | 0.2368 | 0.2351 | 0.2353 | 0.0014 | [0.2319, 0.2387] | [0.2344, 0.2368] | 0.125 | +0.94 |
| T16 | 0.2810 | 0.2985 | 0.2824 | 0.2873 | 0.0097 | [0.2631, 0.3115] | [0.2810, 0.2985] | 0.125 | -0.93 |
| T32 | 0.3243 | 0.3390 | 0.3213 | 0.3282 | 0.0095 | [0.3047, 0.3517] | [0.3223, 0.3390] | 0.125 | -1.00 |

## Trend across T (exact within-seed permutation, pooled seeds)

| test | Ts | Σ slopes (per log₂T) | per-seed slopes | p (1-sided) | perms |
|---|---|---|---|---|---|
| txc_pre_trained_2to8 | [2, 4, 8] | 0.2278 | 0.0772, 0.0722, 0.0785 | 0.0046 | 216 |
| txc_pre_margin_2to8 | [2, 4, 8] | 0.1769 | 0.0581, 0.0528, 0.0660 | 0.0046 | 216 |

- **txc_pre_trained_2to32_secondary**: secondary full-ladder trend over T=[2, 4, 8, 16, 32] not computed: exact enumeration too large; reduce Ts/seeds — the frozen 2->8 primary carries the trend receipt; per-cell values are all reported

## Trained − untrained margin (paired by seed), key cells

| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |
|---|---|---|---|---|
| batchtopk_sae/T1 | 0.0797 | [0.0449, 0.1145] | [0.0662, 0.0942] | 0.125 |
| stacked_batchtopk/T2 | 0.0827 | [0.0627, 0.1026] | [0.0776, 0.0917] | 0.125 |
| stacked_batchtopk/T4 | 0.1006 | [0.0768, 0.1243] | [0.0898, 0.1066] | 0.125 |
| stacked_batchtopk/T8 | 0.1170 | [0.0916, 0.1424] | [0.1108, 0.1287] | 0.125 |
| stacked_batchtopk/T16 | 0.1575 | [0.1384, 0.1765] | [0.1529, 0.1663] | 0.125 |
| stacked_batchtopk/T32 | 0.2058 | [0.1963, 0.2153] | [0.2035, 0.2102] | 0.125 |
| tsae/T1 | 0.1267 | [0.0966, 0.1568] | [0.1197, 0.1407] | 0.125 |
| txc_batchtopk_post/T2 | 0.1113 | [0.0945, 0.1282] | [0.1036, 0.1156] | 0.125 |
| txc_batchtopk_post/T4 | 0.1358 | [0.1002, 0.1713] | [0.1200, 0.1451] | 0.125 |
| txc_batchtopk_post/T8 | 0.1794 | [0.1450, 0.2139] | [0.1635, 0.1878] | 0.125 |
| txc_batchtopk_post/T16 | 0.2764 | [0.2526, 0.3001] | [0.2706, 0.2874] | 0.125 |
| txc_batchtopk_post/T32 | 0.2490 | [0.2099, 0.2881] | [0.2388, 0.2665] | 0.125 |
| txc_batchtopk_pre/T2 | 0.1146 | [0.0971, 0.1320] | [0.1080, 0.1220] | 0.125 |
| txc_batchtopk_pre/T4 | 0.1852 | [0.1560, 0.2143] | [0.1780, 0.1987] | 0.125 |
| txc_batchtopk_pre/T8 | 0.2325 | [0.2160, 0.2490] | [0.2284, 0.2401] | 0.125 |
| txc_batchtopk_pre/T16 | 0.2999 | [0.2506, 0.3492] | [0.2776, 0.3126] | 0.125 |
| txc_batchtopk_pre/T32 | 0.3714 | [0.3495, 0.3933] | [0.3623, 0.3798] | 0.125 |

## Power calc → seed recommendation

- Exact sign-flip attainability: p ≤ 0.05 first possible at **n = 5 seeds** (2⁻ⁿ ≤ 0.05).
- txc_pre_minus_tsae @T4: observed 0.1331 ± 0.0075; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- txc_pre_minus_tsae @T8: observed 0.1883 ± 0.0055; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- txc_pre_minus_pertoken @T4: observed 0.1801 ± 0.0058; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- txc_pre_minus_pertoken @T8: observed 0.2353 ± 0.0014; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- Criterion: one-sided 95% t lower bound > 0 on the paired TXC-pre - T-SAE diff at T = 8, AND exact sign-flip attainability (2^-n <= 0.05).
- **Recommendation:** total seeds needed **5** ⇒ **2 extra seeds**. Per extra seed (trained): txc_batchtopk_pre/T4, txc_batchtopk_pre/T8, tsae/T1 ⇒ 6 trained cells (+6 optional untrained counterparts). Headroom option: 3 extra seeds = 9 cells — one seed of slack against the plug-in sd estimate itself being an n=3 estimate; also reaches sign-flip p = 1/128 at the T8 cell.
- T4 is NOT cheaply boundable (n = 2 to bound, 2 for 80% power); the T-rise + trained−untrained margin carry that cell.

## Honesty notes

- n = 3 seeds: the exact one-sided sign-flip permutation test cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired direction is consistent in all 3 seeds', not as significance.
- The exact bootstrap distribution of a 3-value mean has 27 atoms (<= 10 distinct values); BCa endpoints are coarse and cannot extend past the extreme seed values.
- Pairing by seed was the right design a priori, but it bought no variance reduction here: at the T8 headline cell the across-seed correlation between the TXC-pre and T-SAE arms is r = -0.99, so the paired sd (0.0055) is not below the independent-arms value (0.0041). The cross-arch margin is therefore NOT bounded away from 0 at n = 3; the receipts that ARE significant at n = 3 are within-arch: the T = 2->8 rise and the trained-untrained margins (paired by seed WITHIN an arch, where the pairing does bind).
- The T = 2->8 trend test is exact with 216 relabelings (min p = 1/216), so it carries real resolution at n = 3.
