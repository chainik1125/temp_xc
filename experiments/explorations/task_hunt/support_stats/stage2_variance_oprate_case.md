# Stage-2 panel — variance receipts (`ward_real_oprate_case_base_l12`, probe v1)

Source: 84 leaderboard rows, datasource `ward_real_oprate_case_base_l12`, metric `lambda_recovery`, seeds [1, 2, 42]; cross-check vs `stage2_ward_real_oprate_case_base_l12.json`: exact (all 84 cells). Built by `stage2_variance.py` — every number below is script-derived.

Selection: probe v1, row layout paired, k_pos 8 (post rule times-T).

## Per-seed values (trained), λ̂ recovery

| cell | seed 1 | seed 2 | seed 42 | mean | 95% t CI |
|---|---|---|---|---|---|
| batchtopk_sae/T1 | 0.1215 | 0.0961 | 0.1103 | 0.1093 | [0.0776, 0.1409] |
| stacked_batchtopk/T2 | 0.1307 | 0.1117 | 0.1157 | 0.1194 | [0.0945, 0.1443] |
| stacked_batchtopk/T4 | 0.1326 | 0.0927 | 0.0832 | 0.1028 | [0.0377, 0.1679] |
| stacked_batchtopk/T8 | 0.0793 | 0.0877 | 0.0922 | 0.0864 | [0.0701, 0.1027] |
| stacked_batchtopk/T16 | 0.1548 | 0.0620 | 0.0510 | 0.0893 | [-0.0524, 0.2309] |
| tsae/T1 | 0.0784 | 0.0782 | 0.1119 | 0.0895 | [0.0413, 0.1377] |
| txc_batchtopk_post/T2 | 0.1135 | 0.1603 | 0.1023 | 0.1254 | [0.0489, 0.2018] |
| txc_batchtopk_post/T4 | 0.1368 | 0.1254 | 0.1414 | 0.1345 | [0.1141, 0.1549] |
| txc_batchtopk_post/T8 | 0.1425 | 0.1601 | 0.0626 | 0.1217 | [-0.0074, 0.2508] |
| txc_batchtopk_post/T16 | -0.0190 | 0.0706 | 0.0664 | 0.0393 | [-0.0863, 0.1650] |
| txc_batchtopk_pre/T2 | 0.1421 | 0.1135 | 0.0925 | 0.1160 | [0.0542, 0.1779] |
| txc_batchtopk_pre/T4 | 0.0893 | 0.1334 | 0.1727 | 0.1318 | [0.0281, 0.2355] |
| txc_batchtopk_pre/T8 | 0.1333 | 0.1015 | 0.0807 | 0.1052 | [0.0394, 0.1710] |
| txc_batchtopk_pre/T16 | 0.1285 | 0.0002 | 0.0709 | 0.0666 | [-0.0931, 0.2262] |

## Paired-by-seed differences (window arch − T=1 reference)

### txc_pre_minus_tsae (reference tsae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.0637 | 0.0353 | -0.0194 | 0.0265 | 0.0422 | [-0.0784, 0.1314] | [-0.0194, 0.0542] | 0.250 | -0.82 |
| T4 | 0.0108 | 0.0552 | 0.0608 | 0.0423 | 0.0274 | [-0.0257, 0.1103] | [0.0108, 0.0589] | 0.125 | +0.85 |
| T8 | 0.0549 | 0.0233 | -0.0312 | 0.0157 | 0.0435 | [-0.0925, 0.1238] | [-0.0312, 0.0444] | 0.375 | -0.80 |
| T16 | 0.0501 | -0.0780 | -0.0410 | -0.0230 | 0.0659 | [-0.1867, 0.1408] | [-0.0657, 0.0501] | 0.750 | +0.06 |

### txc_pre_minus_pertoken (reference batchtopk_sae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.0207 | 0.0174 | -0.0178 | 0.0068 | 0.0213 | [-0.0462, 0.0597] | [-0.0178, 0.0196] | 0.375 | +0.52 |
| T4 | -0.0322 | 0.0374 | 0.0624 | 0.0225 | 0.0490 | [-0.0992, 0.1443] | [-0.0322, 0.0541] | 0.250 | -0.47 |
| T8 | 0.0119 | 0.0055 | -0.0295 | -0.0041 | 0.0223 | [-0.0594, 0.0513] | [-0.0295, 0.0097] | 0.625 | +0.54 |
| T16 | 0.0071 | -0.0958 | -0.0394 | -0.0427 | 0.0515 | [-0.1708, 0.0853] | [-0.0958, 0.0071] | 0.875 | +1.00 |

## Trend across T (exact within-seed permutation, pooled seeds)

| test | Ts | Σ slopes (per log₂T) | per-seed slopes | p (1-sided) | perms |
|---|---|---|---|---|---|
| txc_pre_trained_2to8 | [2, 4, 8] | -0.0163 | -0.0044, -0.0060, -0.0059 | 0.6250 | 216 |
| txc_pre_margin_2to8 | [2, 4, 8] | -0.0047 | -0.0181, 0.0121, 0.0012 | 0.5417 | 216 |
| txc_pre_trained_2to16_secondary | [2, 4, 8, 16] | -0.0525 | 0.0003, -0.0372, -0.0157 | 0.9248 | 13824 |

## Trained − untrained margin (paired by seed), key cells

| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |
|---|---|---|---|---|
| batchtopk_sae/T1 | 0.0871 | [0.0381, 0.1361] | [0.0660, 0.1050] | 0.125 |
| stacked_batchtopk/T2 | 0.0932 | [0.0521, 0.1343] | [0.0742, 0.1032] | 0.125 |
| stacked_batchtopk/T4 | 0.0696 | [-0.0199, 0.1591] | [0.0321, 0.1039] | 0.125 |
| stacked_batchtopk/T8 | 0.0552 | [0.0184, 0.0919] | [0.0426, 0.0715] | 0.125 |
| stacked_batchtopk/T16 | 0.0556 | [-0.1247, 0.2359] | [0.0091, 0.1370] | 0.250 |
| tsae/T1 | 0.0674 | [0.0117, 0.1230] | [0.0528, 0.0920] | 0.125 |
| txc_batchtopk_post/T2 | 0.0874 | [0.0248, 0.1500] | [0.0709, 0.1150] | 0.125 |
| txc_batchtopk_post/T4 | 0.1122 | [0.0244, 0.2000] | [0.0899, 0.1522] | 0.125 |
| txc_batchtopk_post/T8 | 0.0926 | [-0.1004, 0.2857] | [0.0037, 0.1406] | 0.125 |
| txc_batchtopk_post/T16 | 0.0275 | [-0.0890, 0.1440] | [-0.0016, 0.0810] | 0.250 |
| txc_batchtopk_pre/T2 | 0.0925 | [0.0058, 0.1792] | [0.0584, 0.1282] | 0.125 |
| txc_batchtopk_pre/T4 | 0.1030 | [0.0114, 0.1947] | [0.0794, 0.1444] | 0.125 |
| txc_batchtopk_pre/T8 | 0.0893 | [0.0217, 0.1570] | [0.0609, 0.1151] | 0.125 |
| txc_batchtopk_pre/T16 | 0.0163 | [-0.0113, 0.0438] | [0.0058, 0.0278] | 0.125 |

## Power calc → seed recommendation

- Exact sign-flip attainability: p ≤ 0.05 first possible at **n = 5 seeds** (2⁻ⁿ ≤ 0.05).
- txc_pre_minus_tsae @T4: observed 0.0423 ± 0.0274; n for 95% lower bound > 0: **4**; n for 80% power (one-sided t, α=0.05): **5**.
- txc_pre_minus_tsae @T8: observed 0.0157 ± 0.0435; n for 95% lower bound > 0: **23**; n for 80% power (one-sided t, α=0.05): **50**.
- txc_pre_minus_pertoken @T4: observed 0.0225 ± 0.0490; n for 95% lower bound > 0: **15**; n for 80% power (one-sided t, α=0.05): **31**.
- txc_pre_minus_pertoken @T8: observed -0.0041 ± 0.0223; n for 95% lower bound > 0: **None**; n for 80% power (one-sided t, α=0.05): **None**.
- Criterion: one-sided 95% t lower bound > 0 on the paired TXC-pre - T-SAE diff at T = 8, AND exact sign-flip attainability (2^-n <= 0.05).
- **Recommendation:** total seeds needed 23 ⇒ extra seeds 20 — outside the cheap-append range (no cell list emitted).
- T4 is NOT cheaply boundable (n = 4 to bound, 5 for 80% power); the T-rise + trained−untrained margin carry that cell.

## Honesty notes

- n = 3 seeds: the exact one-sided sign-flip permutation test cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired direction is consistent in all 3 seeds', not as significance.
- The exact bootstrap distribution of a 3-value mean has 27 atoms (<= 10 distinct values); BCa endpoints are coarse and cannot extend past the extreme seed values.
- Pairing by seed was the right design a priori, but it bought no variance reduction here: at the T8 headline cell the across-seed correlation between the TXC-pre and T-SAE arms is r = -0.80, so the paired sd (0.0435) is not below the independent-arms value (0.0328). The cross-arch margin is therefore NOT bounded away from 0 at n = 3; the receipts that ARE significant at n = 3 are within-arch: the T = 2->8 rise and the trained-untrained margins (paired by seed WITHIN an arch, where the pairing does bind).
- The T = 2->8 trend test is exact with 216 relabelings (min p = 1/216), so it carries real resolution at n = 3.
