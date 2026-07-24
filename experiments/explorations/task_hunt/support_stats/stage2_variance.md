# Stage-2 λ̂ panel — variance receipts (runpod-b, item 1 of `briefings/hunt-support-stats.md`)

Source: 84 leaderboard rows, datasource `ward_real_lambda_base_l12`, metric `lambda_recovery`, seeds [1, 2, 42]; cross-check vs `stage2_ward_real_lambda_base_l12.json`: exact (all 84 cells). Built by `stage2_variance.py` — every number below is script-derived.

## Per-seed values (trained), λ̂ recovery

| cell | seed 1 | seed 2 | seed 42 | mean | 95% t CI |
|---|---|---|---|---|---|
| batchtopk_sae/T1 | 0.1368 | 0.1179 | 0.0841 | 0.1130 | [0.0467, 0.1792] |
| stacked_batchtopk/T2 | 0.1001 | 0.1047 | 0.1222 | 0.1090 | [0.0801, 0.1379] |
| stacked_batchtopk/T4 | 0.1595 | 0.1435 | 0.1267 | 0.1432 | [0.1025, 0.1840] |
| stacked_batchtopk/T8 | 0.1295 | 0.0938 | 0.1509 | 0.1247 | [0.0531, 0.1963] |
| stacked_batchtopk/T16 | 0.0706 | 0.1046 | 0.1066 | 0.0940 | [0.0438, 0.1442] |
| tsae/T1 | 0.1850 | 0.1747 | 0.1025 | 0.1541 | [0.0424, 0.2657] |
| txc_batchtopk_post/T2 | 0.1509 | 0.1291 | 0.1089 | 0.1296 | [0.0776, 0.1817] |
| txc_batchtopk_post/T4 | 0.1816 | 0.1428 | 0.1578 | 0.1607 | [0.1122, 0.2093] |
| txc_batchtopk_post/T8 | 0.1818 | 0.1565 | 0.2161 | 0.1848 | [0.1105, 0.2591] |
| txc_batchtopk_post/T16 | 0.3190 | 0.2391 | 0.2063 | 0.2548 | [0.1108, 0.3988] |
| txc_batchtopk_pre/T2 | 0.1036 | 0.1187 | 0.1724 | 0.1316 | [0.0418, 0.2214] |
| txc_batchtopk_pre/T4 | 0.1924 | 0.1630 | 0.2201 | 0.1918 | [0.1209, 0.2628] |
| txc_batchtopk_pre/T8 | 0.2249 | 0.1786 | 0.2153 | 0.2063 | [0.1455, 0.2670] |
| txc_batchtopk_pre/T16 | 0.1635 | 0.1334 | 0.1169 | 0.1379 | [0.0792, 0.1966] |

## Paired-by-seed differences (window arch − T=1 reference)

### txc_pre_minus_tsae (reference tsae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | -0.0814 | -0.0560 | 0.0699 | -0.0225 | 0.0810 | [-0.2237, 0.1787] | [-0.0729, 0.0699] | 0.750 | -1.00 |
| T4 | 0.0074 | -0.0117 | 0.1176 | 0.0378 | 0.0698 | [-0.1356, 0.2111] | [-0.0053, 0.1176] | 0.375 | -0.79 |
| T8 | 0.0399 | 0.0039 | 0.1128 | 0.0522 | 0.0555 | [-0.0856, 0.1900] | [0.0159, 0.1128] | 0.125 | -0.21 |
| T16 | -0.0215 | -0.0413 | 0.0143 | -0.0162 | 0.0282 | [-0.0862, 0.0539] | [-0.0347, 0.0143] | 0.875 | +0.84 |

### txc_pre_minus_pertoken (reference batchtopk_sae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | -0.0332 | 0.0008 | 0.0883 | 0.0186 | 0.0627 | [-0.1370, 0.1743] | [-0.0219, 0.0883] | 0.375 | -0.99 |
| T4 | 0.0556 | 0.0451 | 0.1360 | 0.0789 | 0.0497 | [-0.0446, 0.2024] | [0.0451, 0.1360] | 0.125 | -0.62 |
| T8 | 0.0881 | 0.0607 | 0.1312 | 0.0933 | 0.0355 | [0.0050, 0.1816] | [0.0698, 0.1312] | 0.125 | +0.04 |
| T16 | 0.0267 | 0.0155 | 0.0327 | 0.0250 | 0.0087 | [0.0032, 0.0467] | [0.0155, 0.0307] | 0.125 | +0.95 |

## Trend across T (exact within-seed permutation, pooled seeds)

| test | Ts | Σ slopes (per log₂T) | per-seed slopes | p (1-sided) | perms |
|---|---|---|---|---|---|
| txc_pre_trained_2to8 | [2, 4, 8] | 0.1120 | 0.0606, 0.0299, 0.0214 | 0.0093 | 216 |
| txc_pre_margin_2to8 | [2, 4, 8] | 0.1651 | 0.0809, 0.0493, 0.0349 | 0.0046 | 216 |
| txc_pre_trained_2to16_secondary | [2, 4, 8, 16] | 0.0100 | 0.0212, 0.0060, -0.0171 | 0.3901 | 13824 |

## Trained − untrained margin (paired by seed), key cells

| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |
|---|---|---|---|---|
| batchtopk_sae/T1 | 0.0195 | [-0.0594, 0.0984] | [-0.0171, 0.0382] | 0.250 |
| stacked_batchtopk/T2 | 0.0169 | [-0.0061, 0.0400] | [0.0110, 0.0273] | 0.125 |
| stacked_batchtopk/T4 | 0.0634 | [0.0097, 0.1171] | [0.0385, 0.0776] | 0.125 |
| stacked_batchtopk/T8 | 0.0089 | [-0.0353, 0.0531] | [-0.0115, 0.0197] | 0.250 |
| stacked_batchtopk/T16 | -0.0767 | [-0.1521, -0.0014] | [-0.1110, -0.0576] | 1.000 |
| tsae/T1 | 0.0606 | [-0.0678, 0.1891] | [0.0012, 0.0921] | 0.125 |
| txc_batchtopk_post/T2 | 0.0472 | [0.0079, 0.0865] | [0.0371, 0.0650] | 0.125 |
| txc_batchtopk_post/T4 | 0.1008 | [0.0822, 0.1194] | [0.0960, 0.1092] | 0.125 |
| txc_batchtopk_post/T8 | 0.1692 | [0.1010, 0.2374] | [0.1416, 0.1965] | 0.125 |
| txc_batchtopk_post/T16 | 0.2262 | [0.0818, 0.3706] | [0.1655, 0.2648] | 0.125 |
| txc_batchtopk_pre/T2 | 0.0403 | [-0.0484, 0.1290] | [0.0180, 0.0810] | 0.125 |
| txc_batchtopk_pre/T4 | 0.1039 | [0.0600, 0.1478] | [0.0889, 0.1234] | 0.125 |
| txc_batchtopk_pre/T8 | 0.1503 | [0.0859, 0.2147] | [0.1241, 0.1676] | 0.125 |
| txc_batchtopk_pre/T16 | 0.1253 | [0.0106, 0.2400] | [0.0736, 0.1549] | 0.125 |

## Power calc → seed recommendation

- Exact sign-flip attainability: p ≤ 0.05 first possible at **n = 5 seeds** (2⁻ⁿ ≤ 0.05).
- txc_pre_minus_tsae @T4: observed 0.0378 ± 0.0698; n for 95% lower bound > 0: **12**; n for 80% power (one-sided t, α=0.05): **23**.
- txc_pre_minus_tsae @T8: observed 0.0522 ± 0.0555; n for 95% lower bound > 0: **6**; n for 80% power (one-sided t, α=0.05): **9**.
- txc_pre_minus_pertoken @T4: observed 0.0789 ± 0.0497; n for 95% lower bound > 0: **4**; n for 80% power (one-sided t, α=0.05): **5**.
- txc_pre_minus_pertoken @T8: observed 0.0933 ± 0.0355; n for 95% lower bound > 0: **3**; n for 80% power (one-sided t, α=0.05): **3**.
- Criterion: one-sided 95% t lower bound > 0 on the paired TXC-pre - T-SAE diff at T = 8, AND exact sign-flip attainability (2^-n <= 0.05).
- **Recommendation:** total seeds needed **6** ⇒ **3 extra seeds**. Per extra seed (trained): txc_batchtopk_pre/T4, txc_batchtopk_pre/T8, tsae/T1 ⇒ 9 trained cells (+9 optional untrained counterparts). Headroom option: 4 extra seeds = 12 cells — one seed of slack against the plug-in sd estimate itself being an n=3 estimate; also reaches sign-flip p = 1/128 at the T8 cell.
- T4 is NOT cheaply boundable (n = 12 to bound, 23 for 80% power); the T-rise + trained−untrained margin carry that cell.

## Honesty notes

- n = 3 seeds: the exact one-sided sign-flip permutation test cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired direction is consistent in all 3 seeds', not as significance.
- The exact bootstrap distribution of a 3-value mean has 27 atoms (<= 10 distinct values); BCa endpoints are coarse and cannot extend past the extreme seed values.
- Pairing by seed was the right design a priori, but it bought no variance reduction here: at the T8 headline cell the across-seed correlation between the TXC-pre and T-SAE arms is r = -0.21, so the paired sd (0.0555) is not below the independent-arms value (0.0512). The cross-arch margin is therefore NOT bounded away from 0 at n = 3; the receipts that ARE significant at n = 3 are within-arch: the T = 2->8 rise and the trained-untrained margins (paired by seed WITHIN an arch, where the pairing does bind).
- The T = 2->8 trend test is exact with 216 relabelings (min p = 1/216), so it carries real resolution at n = 3.
