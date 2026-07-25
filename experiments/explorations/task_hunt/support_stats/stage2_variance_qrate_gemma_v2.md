# Stage-2 panel — variance receipts (`fineweb_punctint_q_gemma2_l14`, probe v2)

Source: 84 leaderboard rows, datasource `fineweb_punctint_q_gemma2_l14`, metric `lambda_recovery_v2`, seeds [1, 2, 42]; cross-check vs `stage2_fineweb_punctint_q_gemma2_l14.json`: exact (all 84 cells). Built by `stage2_variance.py` — every number below is script-derived.

Selection: probe v2, row layout flagged, k_pos 8 (post rule times-T).

## Per-seed values (trained), λ̂ recovery

| cell | seed 1 | seed 2 | seed 42 | mean | 95% t CI |
|---|---|---|---|---|---|
| batchtopk_sae/T1 | 0.1831 | 0.2300 | 0.1558 | 0.1896 | [0.0964, 0.2829] |
| stacked_batchtopk/T2 | 0.2441 | 0.2639 | 0.2513 | 0.2531 | [0.2282, 0.2780] |
| stacked_batchtopk/T4 | 0.2630 | 0.2557 | 0.2682 | 0.2623 | [0.2466, 0.2779] |
| stacked_batchtopk/T8 | 0.2934 | 0.2678 | 0.2768 | 0.2793 | [0.2471, 0.3116] |
| stacked_batchtopk/T16 | 0.2421 | 0.2627 | 0.2639 | 0.2562 | [0.2258, 0.2866] |
| tsae/T1 | 0.2199 | 0.1740 | 0.1800 | 0.1913 | [0.1293, 0.2533] |
| txc_batchtopk_post/T2 | 0.2628 | 0.2628 | 0.2648 | 0.2635 | [0.2607, 0.2663] |
| txc_batchtopk_post/T4 | 0.2835 | 0.2728 | 0.3093 | 0.2885 | [0.2420, 0.3351] |
| txc_batchtopk_post/T8 | 0.2922 | 0.3441 | 0.3172 | 0.3178 | [0.2534, 0.3823] |
| txc_batchtopk_post/T16 | 0.3308 | 0.3482 | 0.4025 | 0.3605 | [0.2675, 0.4535] |
| txc_batchtopk_pre/T2 | 0.2481 | 0.2525 | 0.1810 | 0.2272 | [0.1276, 0.3268] |
| txc_batchtopk_pre/T4 | 0.2650 | 0.2336 | 0.2505 | 0.2497 | [0.2107, 0.2886] |
| txc_batchtopk_pre/T8 | 0.3126 | 0.3001 | 0.2618 | 0.2915 | [0.2258, 0.3573] |
| txc_batchtopk_pre/T16 | 0.3684 | 0.2829 | 0.3111 | 0.3208 | [0.2126, 0.4290] |

## Paired-by-seed differences (window arch − T=1 reference)

### txc_pre_minus_tsae (reference tsae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.0282 | 0.0785 | 0.0010 | 0.0359 | 0.0393 | [-0.0618, 0.1336] | [0.0101, 0.0785] | 0.125 | +0.34 |
| T4 | 0.0450 | 0.0596 | 0.0705 | 0.0584 | 0.0128 | [0.0267, 0.0901] | [0.0450, 0.0705] | 0.125 | +0.90 |
| T8 | 0.0927 | 0.1261 | 0.0818 | 0.1002 | 0.0231 | [0.0429, 0.1575] | [0.0855, 0.1261] | 0.125 | +0.60 |
| T16 | 0.1485 | 0.1088 | 0.1312 | 0.1295 | 0.0199 | [0.0801, 0.1788] | [0.1088, 0.1485] | 0.125 | +0.98 |

### txc_pre_minus_pertoken (reference batchtopk_sae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.0650 | 0.0225 | 0.0252 | 0.0376 | 0.0238 | [-0.0215, 0.0966] | [0.0234, 0.0650] | 0.125 | +0.81 |
| T4 | 0.0818 | 0.0036 | 0.0947 | 0.0600 | 0.0493 | [-0.0624, 0.1824] | [0.0036, 0.0904] | 0.125 | -0.66 |
| T8 | 0.1294 | 0.0702 | 0.1060 | 0.1019 | 0.0299 | [0.0277, 0.1761] | [0.0702, 0.1216] | 0.125 | +0.61 |
| T16 | 0.1852 | 0.0529 | 0.1554 | 0.1312 | 0.0694 | [-0.0412, 0.3036] | [0.0529, 0.1753] | 0.125 | -0.46 |

## Trend across T (exact within-seed permutation, pooled seeds)

| test | Ts | Σ slopes (per log₂T) | per-seed slopes | p (1-sided) | perms |
|---|---|---|---|---|---|
| txc_pre_trained_2to8 | [2, 4, 8] | 0.0965 | 0.0322, 0.0238, 0.0404 | 0.0185 | 216 |
| txc_pre_margin_2to8 | [2, 4, 8] | 0.0667 | 0.0247, 0.0144, 0.0277 | 0.0324 | 216 |
| txc_pre_trained_2to16_secondary | [2, 4, 8, 16] | 0.0968 | 0.0408, 0.0158, 0.0402 | 0.0009 | 13824 |

## Trained − untrained margin (paired by seed), key cells

| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |
|---|---|---|---|---|
| batchtopk_sae/T1 | 0.1257 | [0.0171, 0.2344] | [0.0969, 0.1725] | 0.125 |
| stacked_batchtopk/T2 | 0.1621 | [0.1413, 0.1830] | [0.1536, 0.1704] | 0.125 |
| stacked_batchtopk/T4 | 0.1520 | [0.1300, 0.1740] | [0.1463, 0.1619] | 0.125 |
| stacked_batchtopk/T8 | 0.1459 | [0.1024, 0.1894] | [0.1347, 0.1656] | 0.125 |
| stacked_batchtopk/T16 | 0.0960 | [0.0591, 0.1329] | [0.0792, 0.1053] | 0.125 |
| tsae/T1 | 0.1274 | [0.0659, 0.1889] | [0.1100, 0.1557] | 0.125 |
| txc_batchtopk_post/T2 | 0.1689 | [0.1661, 0.1717] | [0.1682, 0.1702] | 0.125 |
| txc_batchtopk_post/T4 | 0.1728 | [0.1503, 0.1953] | [0.1675, 0.1833] | 0.125 |
| txc_batchtopk_post/T8 | 0.1748 | [0.1133, 0.2363] | [0.1499, 0.1994] | 0.125 |
| txc_batchtopk_post/T16 | 0.1863 | [0.0809, 0.2916] | [0.1595, 0.2343] | 0.125 |
| txc_batchtopk_pre/T2 | 0.1415 | [0.0449, 0.2381] | [0.0967, 0.1648] | 0.125 |
| txc_batchtopk_pre/T4 | 0.1538 | [0.1208, 0.1869] | [0.1397, 0.1626] | 0.125 |
| txc_batchtopk_pre/T8 | 0.1860 | [0.1106, 0.2614] | [0.1521, 0.2055] | 0.125 |
| txc_batchtopk_pre/T16 | 0.2100 | [0.1141, 0.3059] | [0.1846, 0.2516] | 0.125 |

## Power calc → seed recommendation

- Exact sign-flip attainability: p ≤ 0.05 first possible at **n = 5 seeds** (2⁻ⁿ ≤ 0.05).
- txc_pre_minus_tsae @T4: observed 0.0584 ± 0.0128; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **3**.
- txc_pre_minus_tsae @T8: observed 0.1002 ± 0.0231; n for 95% lower bound > 0: **3**; n for 80% power (one-sided t, α=0.05): **3**.
- txc_pre_minus_pertoken @T4: observed 0.0600 ± 0.0493; n for 95% lower bound > 0: **4**; n for 80% power (one-sided t, α=0.05): **6**.
- txc_pre_minus_pertoken @T8: observed 0.1019 ± 0.0299; n for 95% lower bound > 0: **3**; n for 80% power (one-sided t, α=0.05): **3**.
- Criterion: one-sided 95% t lower bound > 0 on the paired TXC-pre - T-SAE diff at T = 8, AND exact sign-flip attainability (2^-n <= 0.05).
- **Recommendation:** total seeds needed **5** ⇒ **2 extra seeds**. Per extra seed (trained): txc_batchtopk_pre/T4, txc_batchtopk_pre/T8, tsae/T1 ⇒ 6 trained cells (+6 optional untrained counterparts). Headroom option: 3 extra seeds = 9 cells — one seed of slack against the plug-in sd estimate itself being an n=3 estimate; also reaches sign-flip p = 1/128 at the T8 cell.
- T4 is NOT cheaply boundable (n = 2 to bound, 3 for 80% power); the T-rise + trained−untrained margin carry that cell.

## Honesty notes

- n = 3 seeds: the exact one-sided sign-flip permutation test cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired direction is consistent in all 3 seeds', not as significance.
- The exact bootstrap distribution of a 3-value mean has 27 atoms (<= 10 distinct values); BCa endpoints are coarse and cannot extend past the extreme seed values.
- Pairing by seed was the right design a priori, but it bought no variance reduction here: at the T8 headline cell the across-seed correlation between the TXC-pre and T-SAE arms is r = 0.60, so the paired sd (0.0231) is not below the independent-arms value (0.0364). The cross-arch margin is therefore NOT bounded away from 0 at n = 3; the receipts that ARE significant at n = 3 are within-arch: the T = 2->8 rise and the trained-untrained margins (paired by seed WITHIN an arch, where the pairing does bind).
- The T = 2->8 trend test is exact with 216 relabelings (min p = 1/216), so it carries real resolution at n = 3.
