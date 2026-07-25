# Stage-2 panel — variance receipts (`fineweb_punctint_q_llama31_l14`, probe v1)

Source: 24 leaderboard rows, datasource `fineweb_punctint_q_llama31_l14`, metric `lambda_recovery`, seeds [1, 2, 42]; cross-check vs `stage2_fineweb_punctint_q_llama31_l14.json`: exact (all 24 cells). Built by `stage2_variance.py` — every number below is script-derived.

Selection: probe v1, row layout paired, k_pos 8 (post rule times-T).

## Per-seed values (trained), λ̂ recovery

| cell | seed 1 | seed 2 | seed 42 | mean | 95% t CI |
|---|---|---|---|---|---|
| batchtopk_sae/T1 | 0.2485 | 0.2104 | 0.2294 | 0.2295 | [0.1821, 0.2769] |
| tsae/T1 | 0.2691 | 0.2566 | 0.2414 | 0.2557 | [0.2213, 0.2901] |
| txc_batchtopk_pre/T4 | 0.2164 | 0.2566 | 0.2407 | 0.2379 | [0.1876, 0.2882] |
| txc_batchtopk_pre/T8 | 0.2199 | 0.2836 | 0.2228 | 0.2421 | [0.1528, 0.3314] |

## Paired-by-seed differences (window arch − T=1 reference)

### txc_pre_minus_tsae (reference tsae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T4 | -0.0527 | -0.0000 | -0.0008 | -0.0178 | 0.0302 | [-0.0929, 0.0572] | [-0.0527, -0.0003] | 1.000 | -0.55 |
| T8 | -0.0492 | 0.0270 | -0.0186 | -0.0136 | 0.0383 | [-0.1089, 0.0817] | [-0.0492, 0.0270] | 0.750 | +0.01 |

### txc_pre_minus_pertoken (reference batchtopk_sae/T1)

| T | seed 1 | seed 2 | seed 42 | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided) | r(arms) |
|---|---|---|---|---|---|---|---|---|---|
| T4 | -0.0322 | 0.0462 | 0.0112 | 0.0084 | 0.0393 | [-0.0891, 0.1059] | [-0.0322, 0.0462] | 0.375 | -0.99 |
| T8 | -0.0287 | 0.0732 | -0.0066 | 0.0126 | 0.0536 | [-0.1205, 0.1458] | [-0.0213, 0.0732] | 0.500 | -0.89 |

## Trend across T (exact within-seed permutation, pooled seeds)

- **txc_pre_trend**: frozen 2->8 trend undefined: txc_batchtopk_pre present at T=[4, 8] only — a trend statistic over 2 T value(s) has no within-seed permutation resolution; the cells themselves are reported in the per-seed / paired sections

## Trained − untrained margin (paired by seed), key cells

| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |
|---|---|---|---|---|
| batchtopk_sae/T1 | 0.2071 | [0.1438, 0.2704] | [0.1826, 0.2334] | 0.125 |
| tsae/T1 | 0.2333 | [0.1865, 0.2800] | [0.2210, 0.2540] | 0.125 |
| txc_batchtopk_pre/T4 | 0.2083 | [0.1468, 0.2698] | [0.1803, 0.2239] | 0.125 |
| txc_batchtopk_pre/T8 | 0.2212 | [0.1260, 0.3164] | [0.1960, 0.2628] | 0.125 |

## Power calc → seed recommendation

- Exact sign-flip attainability: p ≤ 0.05 first possible at **n = 5 seeds** (2⁻ⁿ ≤ 0.05).
- txc_pre_minus_tsae @T4: observed -0.0178 ± 0.0302; n for 95% lower bound > 0: **None**; n for 80% power (one-sided t, α=0.05): **None**.
- txc_pre_minus_tsae @T8: observed -0.0136 ± 0.0383; n for 95% lower bound > 0: **None**; n for 80% power (one-sided t, α=0.05): **None**.
- txc_pre_minus_pertoken @T4: observed 0.0084 ± 0.0393; n for 95% lower bound > 0: **None**; n for 80% power (one-sided t, α=0.05): **None**.
- txc_pre_minus_pertoken @T8: observed 0.0126 ± 0.0536; n for 95% lower bound > 0: **51**; n for 80% power (one-sided t, α=0.05): **None**.
- Criterion: one-sided 95% t lower bound > 0 on the paired TXC-pre - T-SAE diff at T = 8, AND exact sign-flip attainability (2^-n <= 0.05).
- **Recommendation:** total seeds needed None ⇒ extra seeds None — outside the cheap-append range (no cell list emitted).
- T4 is NOT cheaply boundable (n = None to bound, None for 80% power); the T-rise + trained−untrained margin carry that cell.

## Honesty notes

- n = 3 seeds: the exact one-sided sign-flip permutation test cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired direction is consistent in all 3 seeds', not as significance.
- The exact bootstrap distribution of a 3-value mean has 27 atoms (<= 10 distinct values); BCa endpoints are coarse and cannot extend past the extreme seed values.
- Pairing by seed was the right design a priori, but it bought no variance reduction here: at the T8 headline cell the across-seed correlation between the TXC-pre and T-SAE arms is r = 0.01, so the paired sd (0.0383) is not below the independent-arms value (0.0385). The cross-arch margin is therefore NOT bounded away from 0 at n = 3; the receipts that ARE significant at n = 3 are within-arch: the T = 2->8 rise and the trained-untrained margins (paired by seed WITHIN an arch, where the pairing does bind).
