# Stage-2 panel — variance receipts (`fineweb_punctint_q_gpt2_l7`, probe v1)

Source: 24 leaderboard rows, datasource `fineweb_punctint_q_gpt2_l7`, metric `lambda_recovery`, seeds [1, 2, 42]; cross-check vs `stage2_fineweb_punctint_q_gpt2_l7.json`: exact (all 24 cells). Built by `stage2_variance.py` — every number below is script-derived.

Selection: probe v1, row layout paired, k_pos 8 (post rule times-T).

## Per-seed values (trained), λ̂ recovery

| cell | seed 1 | seed 2 | seed 42 | mean | 95% t CI |
|---|---|---|---|---|---|
| batchtopk_sae/T1 | 0.1131 | 0.1113 | 0.1172 | 0.1139 | [0.1065, 0.1213] |
| tsae/T1 | 0.1268 | 0.1107 | 0.1054 | 0.1143 | [0.0865, 0.1421] |
| txc_batchtopk_pre/T4 | 0.1470 | 0.1380 | 0.1427 | 0.1426 | [0.1315, 0.1536] |
| txc_batchtopk_pre/T8 | 0.1364 | 0.1430 | 0.1408 | 0.1401 | [0.1318, 0.1484] |

## Paired-by-seed differences (window arch − T=1 reference)

### txc_pre_minus_tsae (reference tsae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T4 | 0.0201 | 0.0274 | 0.0374 | 0.0283 | 0.0087 | [0.0068, 0.0498] | [0.0201, 0.0374] | 0.125 | +0.70 |
| T8 | 0.0096 | 0.0323 | 0.0354 | 0.0258 | 0.0141 | [-0.0092, 0.0608] | [0.0096, 0.0344] | 0.125 | -0.84 |

### txc_pre_minus_pertoken (reference batchtopk_sae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T4 | 0.0338 | 0.0267 | 0.0255 | 0.0287 | 0.0045 | [0.0176, 0.0398] | [0.0259, 0.0338] | 0.125 | +0.33 |
| T8 | 0.0233 | 0.0317 | 0.0236 | 0.0262 | 0.0047 | [0.0144, 0.0380] | [0.0234, 0.0317] | 0.125 | -0.12 |

## Trend across T (exact within-seed permutation, pooled seeds)

- **txc_pre_trend**: frozen 2->8 trend undefined: txc_batchtopk_pre present at T=[4, 8] only — a trend statistic over 2 T value(s) has no within-seed permutation resolution; the cells themselves are reported in the per-seed / paired sections

## Trained − untrained margin (paired by seed), key cells

| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |
|---|---|---|---|---|
| batchtopk_sae/T1 | 0.0704 | [0.0636, 0.0773] | [0.0676, 0.0731] | 0.125 |
| tsae/T1 | 0.0708 | [0.0459, 0.0957] | [0.0613, 0.0813] | 0.125 |
| txc_batchtopk_pre/T4 | 0.1034 | [0.0925, 0.1143] | [0.1005, 0.1081] | 0.125 |
| txc_batchtopk_pre/T8 | 0.0983 | [0.0774, 0.1192] | [0.0893, 0.1039] | 0.125 |

## Power calc → seed recommendation

- Exact sign-flip attainability: p ≤ 0.05 first possible at **n = 5 seeds** (2⁻ⁿ ≤ 0.05).
- txc_pre_minus_tsae @T4: observed 0.0283 ± 0.0087; n for 95% lower bound > 0: **3**; n for 80% power (one-sided t, α=0.05): **3**.
- txc_pre_minus_tsae @T8: observed 0.0258 ± 0.0141; n for 95% lower bound > 0: **3**; n for 80% power (one-sided t, α=0.05): **4**.
- txc_pre_minus_pertoken @T4: observed 0.0287 ± 0.0045; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **2**.
- txc_pre_minus_pertoken @T8: observed 0.0262 ± 0.0047; n for 95% lower bound > 0: **2**; n for 80% power (one-sided t, α=0.05): **3**.
- Criterion: one-sided 95% t lower bound > 0 on the paired TXC-pre - T-SAE diff at T = 8, AND exact sign-flip attainability (2^-n <= 0.05).
- **Recommendation:** total seeds needed **5** ⇒ **2 extra seeds**. Per extra seed (trained): txc_batchtopk_pre/T4, txc_batchtopk_pre/T8, tsae/T1 ⇒ 6 trained cells (+6 optional untrained counterparts). Headroom option: 3 extra seeds = 9 cells — one seed of slack against the plug-in sd estimate itself being an n=3 estimate; also reaches sign-flip p = 1/128 at the T8 cell.
- T4 is NOT cheaply boundable (n = 3 to bound, 3 for 80% power); the T-rise + trained−untrained margin carry that cell.

## Honesty notes

- n = 3 seeds: the exact one-sided sign-flip permutation test cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired direction is consistent in all 3 seeds', not as significance.
- The exact bootstrap distribution of a 3-value mean has 27 atoms (<= 10 distinct values); BCa endpoints are coarse and cannot extend past the extreme seed values.
- Pairing by seed was the right design a priori, but it bought no variance reduction here: at the T8 headline cell the across-seed correlation between the TXC-pre and T-SAE arms is r = -0.84, so the paired sd (0.0141) is not below the independent-arms value (0.0117). The cross-arch margin is therefore NOT bounded away from 0 at n = 3; the receipts that ARE significant at n = 3 are within-arch: the T = 2->8 rise and the trained-untrained margins (paired by seed WITHIN an arch, where the pairing does bind).
