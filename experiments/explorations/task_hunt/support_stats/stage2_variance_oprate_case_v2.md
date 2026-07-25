# Stage-2 panel — variance receipts (`ward_real_oprate_case_base_l12`, probe v2)

Source: 84 leaderboard rows, datasource `ward_real_oprate_case_base_l12`, metric `lambda_recovery_v2`, seeds [1, 2, 42]; cross-check vs `stage2_ward_real_oprate_case_base_l12.json`: exact (all 84 cells). Built by `stage2_variance.py` — every number below is script-derived.

Selection: probe v2, row layout flagged, k_pos 8 (post rule times-T).

## Per-seed values (trained), λ̂ recovery

| cell | seed 1 | seed 2 | seed 42 | mean | 95% t CI |
|---|---|---|---|---|---|
| batchtopk_sae/T1 | 0.0979 | 0.0975 | 0.1272 | 0.1076 | [0.0652, 0.1499] |
| stacked_batchtopk/T2 | 0.1315 | 0.1138 | 0.1127 | 0.1193 | [0.0932, 0.1455] |
| stacked_batchtopk/T4 | 0.1667 | 0.1795 | 0.1388 | 0.1617 | [0.1099, 0.2135] |
| stacked_batchtopk/T8 | 0.2051 | 0.1782 | 0.1994 | 0.1942 | [0.1590, 0.2295] |
| stacked_batchtopk/T16 | 0.2401 | 0.2369 | 0.2213 | 0.2328 | [0.2078, 0.2578] |
| tsae/T1 | 0.0862 | 0.0456 | 0.0661 | 0.0660 | [0.0156, 0.1164] |
| txc_batchtopk_post/T2 | 0.1737 | 0.1862 | 0.1203 | 0.1601 | [0.0730, 0.2471] |
| txc_batchtopk_post/T4 | 0.2036 | 0.1980 | 0.1667 | 0.1894 | [0.1400, 0.2388] |
| txc_batchtopk_post/T8 | 0.1814 | 0.2371 | 0.1948 | 0.2044 | [0.1322, 0.2766] |
| txc_batchtopk_post/T16 | 0.2323 | 0.2381 | 0.2364 | 0.2356 | [0.2282, 0.2429] |
| txc_batchtopk_pre/T2 | 0.2028 | 0.1290 | 0.1405 | 0.1575 | [0.0588, 0.2561] |
| txc_batchtopk_pre/T4 | 0.1522 | 0.2008 | 0.1966 | 0.1832 | [0.1163, 0.2501] |
| txc_batchtopk_pre/T8 | 0.2192 | 0.2148 | 0.1975 | 0.2105 | [0.1819, 0.2390] |
| txc_batchtopk_pre/T16 | 0.2818 | 0.2768 | 0.2249 | 0.2611 | [0.1828, 0.3394] |

## Paired-by-seed differences (window arch − T=1 reference)

### txc_pre_minus_tsae (reference tsae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.1167 | 0.0834 | 0.0744 | 0.0915 | 0.0223 | [0.0362, 0.1468] | [0.0774, 0.1167] | 0.125 | +0.93 |
| T4 | 0.0660 | 0.1552 | 0.1305 | 0.1172 | 0.0460 | [0.0029, 0.2316] | [0.0660, 0.1469] | 0.125 | -0.90 |
| T8 | 0.1330 | 0.1692 | 0.1314 | 0.1445 | 0.0214 | [0.0913, 0.1977] | [0.1319, 0.1692] | 0.125 | +0.18 |
| T16 | 0.1956 | 0.2312 | 0.1588 | 0.1952 | 0.0362 | [0.1052, 0.2852] | [0.1588, 0.2312] | 0.125 | +0.07 |

### txc_pre_minus_pertoken (reference batchtopk_sae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.1049 | 0.0315 | 0.0133 | 0.0499 | 0.0485 | [-0.0706, 0.1704] | [0.0193, 0.1049] | 0.125 | -0.36 |
| T4 | 0.0542 | 0.1033 | 0.0693 | 0.0756 | 0.0251 | [0.0132, 0.1380] | [0.0593, 0.1033] | 0.125 | +0.42 |
| T8 | 0.1212 | 0.1174 | 0.0702 | 0.1029 | 0.0284 | [0.0324, 0.1735] | [0.0702, 0.1212] | 0.125 | -0.98 |
| T16 | 0.1838 | 0.1793 | 0.0976 | 0.1536 | 0.0485 | [0.0330, 0.2741] | [0.0976, 0.1823] | 0.125 | -1.00 |

## Trend across T (exact within-seed permutation, pooled seeds)

| test | Ts | Σ slopes (per log₂T) | per-seed slopes | p (1-sided) | perms |
|---|---|---|---|---|---|
| txc_pre_trained_2to8 | [2, 4, 8] | 0.0796 | 0.0082, 0.0429, 0.0285 | 0.0417 | 216 |
| txc_pre_margin_2to8 | [2, 4, 8] | 0.0581 | 0.0021, 0.0342, 0.0218 | 0.0880 | 216 |
| txc_pre_trained_2to16_secondary | [2, 4, 8, 16] | 0.1015 | 0.0304, 0.0457, 0.0254 | 0.0011 | 13824 |

## Trained − untrained margin (paired by seed), key cells

| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |
|---|---|---|---|---|
| batchtopk_sae/T1 | 0.0785 | [0.0389, 0.1180] | [0.0631, 0.0949] | 0.125 |
| stacked_batchtopk/T2 | 0.0795 | [0.0272, 0.1318] | [0.0656, 0.1020] | 0.125 |
| stacked_batchtopk/T4 | 0.1098 | [0.0255, 0.1941] | [0.0707, 0.1313] | 0.125 |
| stacked_batchtopk/T8 | 0.1244 | [0.1096, 0.1393] | [0.1188, 0.1307] | 0.125 |
| stacked_batchtopk/T16 | 0.1455 | [0.0761, 0.2149] | [0.1158, 0.1640] | 0.125 |
| tsae/T1 | 0.0369 | [-0.0311, 0.1048] | [0.0187, 0.0656] | 0.125 |
| txc_batchtopk_post/T2 | 0.1196 | [0.0158, 0.2234] | [0.0727, 0.1463] | 0.125 |
| txc_batchtopk_post/T4 | 0.1361 | [0.0835, 0.1887] | [0.1116, 0.1489] | 0.125 |
| txc_batchtopk_post/T8 | 0.1267 | [0.0713, 0.1820] | [0.1129, 0.1521] | 0.125 |
| txc_batchtopk_post/T16 | 0.1491 | [0.0540, 0.2441] | [0.1101, 0.1866] | 0.125 |
| txc_batchtopk_pre/T2 | 0.1260 | [0.0201, 0.2319] | [0.1014, 0.1752] | 0.125 |
| txc_batchtopk_pre/T4 | 0.1347 | [0.0807, 0.1887] | [0.1099, 0.1482] | 0.125 |
| txc_batchtopk_pre/T8 | 0.1647 | [0.1203, 0.2091] | [0.1448, 0.1763] | 0.125 |
| txc_batchtopk_pre/T16 | 0.2121 | [0.1064, 0.3178] | [0.1666, 0.2402] | 0.125 |

## Power calc → seed recommendation

- Exact sign-flip attainability: p ≤ 0.05 first possible at **n = 5 seeds** (2⁻ⁿ ≤ 0.05).
- txc_pre_minus_tsae @T4: observed 0.1172 ± 0.0460; n for 95% lower bound > 0: **3**; n for 80% power (one-sided t, α=0.05): **3**.
- txc_pre_minus_tsae @T8: observed 0.1445 ± 0.0214; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- txc_pre_minus_pertoken @T4: observed 0.0756 ± 0.0251; n for 95% lower bound > 0: **3**; n for 80% power (one-sided t, α=0.05): **3**.
- txc_pre_minus_pertoken @T8: observed 0.1029 ± 0.0284; n for 95% lower bound > 0: **3**; n for 80% power (one-sided t, α=0.05): **3**.
- Criterion: one-sided 95% t lower bound > 0 on the paired TXC-pre - T-SAE diff at T = 8, AND exact sign-flip attainability (2^-n <= 0.05).
- **Recommendation:** total seeds needed **5** ⇒ **2 extra seeds**. Per extra seed (trained): txc_batchtopk_pre/T4, txc_batchtopk_pre/T8, tsae/T1 ⇒ 6 trained cells (+6 optional untrained counterparts). Headroom option: 3 extra seeds = 9 cells — one seed of slack against the plug-in sd estimate itself being an n=3 estimate; also reaches sign-flip p = 1/128 at the T8 cell.
- T4 is NOT cheaply boundable (n = 3 to bound, 3 for 80% power); the T-rise + trained−untrained margin carry that cell.

## Honesty notes

- n = 3 seeds: the exact one-sided sign-flip permutation test cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired direction is consistent in all 3 seeds', not as significance.
- The exact bootstrap distribution of a 3-value mean has 27 atoms (<= 10 distinct values); BCa endpoints are coarse and cannot extend past the extreme seed values.
- Pairing by seed was the right design a priori, but it bought no variance reduction here: at the T8 headline cell the across-seed correlation between the TXC-pre and T-SAE arms is r = 0.18, so the paired sd (0.0214) is not below the independent-arms value (0.0233). The cross-arch margin is therefore NOT bounded away from 0 at n = 3; the receipts that ARE significant at n = 3 are within-arch: the T = 2->8 rise and the trained-untrained margins (paired by seed WITHIN an arch, where the pairing does bind).
- The T = 2->8 trend test is exact with 216 relabelings (min p = 1/216), so it carries real resolution at n = 3.
