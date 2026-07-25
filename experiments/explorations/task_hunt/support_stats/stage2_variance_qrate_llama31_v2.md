# Stage-2 panel — variance receipts (`fineweb_punctint_q_llama31_l14`, probe v2)

Source: 24 leaderboard rows, datasource `fineweb_punctint_q_llama31_l14`, metric `lambda_recovery_v2`, seeds [1, 2, 42]; cross-check vs `stage2_fineweb_punctint_q_llama31_l14.json`: exact (all 24 cells). Built by `stage2_variance.py` — every number below is script-derived.

Selection: probe v2, row layout flagged, k_pos 8 (post rule times-T).

## Per-seed values (trained), λ̂ recovery

| cell | seed 1 | seed 2 | seed 42 | mean | 95% t CI |
|---|---|---|---|---|---|
| batchtopk_sae/T1 | 0.2064 | 0.1884 | 0.2025 | 0.1991 | [0.1755, 0.2227] |
| tsae/T1 | 0.1908 | 0.2032 | 0.2017 | 0.1985 | [0.1817, 0.2153] |
| txc_batchtopk_pre/T4 | 0.2303 | 0.2237 | 0.2412 | 0.2317 | [0.2098, 0.2536] |
| txc_batchtopk_pre/T8 | 0.2435 | 0.3216 | 0.2562 | 0.2737 | [0.1696, 0.3778] |

## Paired-by-seed differences (window arch − T=1 reference)

### txc_pre_minus_tsae (reference tsae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T4 | 0.0395 | 0.0205 | 0.0395 | 0.0332 | 0.0110 | [0.0059, 0.0604] | [0.0205, 0.0395] | 0.125 | +0.03 |
| T8 | 0.0527 | 0.1184 | 0.0545 | 0.0752 | 0.0374 | [-0.0178, 0.1681] | [0.0533, 0.1184] | 0.125 | +0.71 |

### txc_pre_minus_pertoken (reference batchtopk_sae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T4 | 0.0238 | 0.0353 | 0.0387 | 0.0326 | 0.0078 | [0.0133, 0.0519] | [0.0238, 0.0375] | 0.125 | +0.64 |
| T8 | 0.0370 | 0.1332 | 0.0537 | 0.0746 | 0.0514 | [-0.0531, 0.2023] | [0.0426, 0.1332] | 0.125 | -1.00 |

## Trend across T (exact within-seed permutation, pooled seeds)

- **txc_pre_trend**: frozen 2->8 trend undefined: txc_batchtopk_pre present at T=[4, 8] only — a trend statistic over 2 T value(s) has no within-seed permutation resolution; the cells themselves are reported in the per-seed / paired sections

## Trained − untrained margin (paired by seed), key cells

| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |
|---|---|---|---|---|
| batchtopk_sae/T1 | 0.1301 | [0.0934, 0.1668] | [0.1148, 0.1443] | 0.125 |
| tsae/T1 | 0.1295 | [0.1275, 0.1316] | [0.1287, 0.1301] | 0.125 |
| txc_batchtopk_pre/T4 | 0.1491 | [0.1065, 0.1916] | [0.1391, 0.1689] | 0.125 |
| txc_batchtopk_pre/T8 | 0.1747 | [0.0601, 0.2893] | [0.1445, 0.2251] | 0.125 |

## Power calc → seed recommendation

- Exact sign-flip attainability: p ≤ 0.05 first possible at **n = 5 seeds** (2⁻ⁿ ≤ 0.05).
- txc_pre_minus_tsae @T4: observed 0.0332 ± 0.0110; n for 95% lower bound > 0: **3**; n for 80% power (one-sided t, α=0.05): **3**.
- txc_pre_minus_tsae @T8: observed 0.0752 ± 0.0374; n for 95% lower bound > 0: **3**; n for 80% power (one-sided t, α=0.05): **4**.
- txc_pre_minus_pertoken @T4: observed 0.0326 ± 0.0078; n for 95% lower bound > 0: **3**; n for 80% power (one-sided t, α=0.05): **3**.
- txc_pre_minus_pertoken @T8: observed 0.0746 ± 0.0514; n for 95% lower bound > 0: **4**; n for 80% power (one-sided t, α=0.05): **5**.
- Criterion: one-sided 95% t lower bound > 0 on the paired TXC-pre - T-SAE diff at T = 8, AND exact sign-flip attainability (2^-n <= 0.05).
- **Recommendation:** total seeds needed **5** ⇒ **2 extra seeds**. Per extra seed (trained): txc_batchtopk_pre/T4, txc_batchtopk_pre/T8, tsae/T1 ⇒ 6 trained cells (+6 optional untrained counterparts). Headroom option: 3 extra seeds = 9 cells — one seed of slack against the plug-in sd estimate itself being an n=3 estimate; also reaches sign-flip p = 1/128 at the T8 cell.
- T4 is NOT cheaply boundable (n = 3 to bound, 3 for 80% power); the T-rise + trained−untrained margin carry that cell.

## Honesty notes

- n = 3 seeds: the exact one-sided sign-flip permutation test cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired direction is consistent in all 3 seeds', not as significance.
- The exact bootstrap distribution of a 3-value mean has 27 atoms (<= 10 distinct values); BCa endpoints are coarse and cannot extend past the extreme seed values.
- Pairing by seed was the right design a priori, but it bought no variance reduction here: at the T8 headline cell the across-seed correlation between the TXC-pre and T-SAE arms is r = 0.71, so the paired sd (0.0374) is not below the independent-arms value (0.0425). The cross-arch margin is therefore NOT bounded away from 0 at n = 3; the receipts that ARE significant at n = 3 are within-arch: the T = 2->8 rise and the trained-untrained margins (paired by seed WITHIN an arch, where the pairing does bind).
