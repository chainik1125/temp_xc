## Auto-updating dashboard — Phase 7 steering case study

All cells discovered with grades.jsonl + generations.jsonl ≥ 200 rows.

### Anchor: T-SAE k=20

unc=1.800, ≥1.5=1.167, ≥1.75=0.333, ≥2.0=0.283, ≥2.25=0.283, ≥2.5=0.283, AUC(1.5-3.0)=0.413


### Top 10 cells by Δ coh ≥ 1.5 (prereg)

| arch + protocol | n | Δ | base | seeds_data |
|---|---:|---:|---:|---|
| txc_h8_t2_kpos20_shifts2 per-position | 3 | +0.233 | 1.400 | n=3 |
| txc_galaxy4_t2_kw10_kp10 right-edge | 1 | +0.200 | 1.367 | n=1 |
| txc_h8_t2_kpos20_shifts2 right-edge | 3 | +0.072 | 1.239 | n=3 |
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 right-edge | 3 | +0.056 | 1.222 | n=3 |
| tsae_paper_k500 right-edge | 2 | +0.033 | 1.200 | n=2 |
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 per-position | 3 | +0.011 | 1.178 | n=3 |
| txc_h8_t3_kpos20_shifts3 per-position | 1 | +0.000 | 1.167 | n=1 |
| txc_bare_antidead_t5_kwin20 per-position | 1 | +0.000 | 1.167 | n=1 |
| txc_bare_antidead_t4_kpos20_grownChainFromT3 per-position | 1 | -0.033 | 1.133 | n=1 |
| txc_bare_antidead_t2_kpos20_ws_tsae_encoder per-position | 3 | -0.033 | 1.133 | n=3 |

### Top 10 cells by Δ coh ≥ 1.75

| arch + protocol | n | Δ | base | seeds_data |
|---|---:|---:|---:|---|
| txc_h8_t2_kpos20_shifts2 right-edge | 3 | +0.906 | 1.239 | n=3 |
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 right-edge | 3 | +0.889 | 1.222 | n=3 |
| txc_bare_antidead_t4_kpos20_grownChainFromT3 per-position | 1 | +0.800 | 1.133 | n=1 |
| txc_bare_antidead_t2_kpos20_ws_tsae_encoder per-position | 3 | +0.800 | 1.133 | n=3 |
| txc_h8_t5_kpos20_shifts5 per-position | 2 | +0.733 | 1.067 | n=2 |
| txc_bare_antidead_t2_kpos20_ws_tsae_encoder right-edge | 3 | +0.678 | 1.011 | n=3 |
| txc_bare_antidead_t2_kpos20 per-position | 3 | +0.644 | 0.978 | n=3 |
| txc_bare_antidead_t2_kpos20 right-edge | 3 | +0.622 | 0.956 | n=3 |
| txc_galaxy4_t2_kw10_kp10 right-edge | 1 | +0.600 | 0.933 | n=1 |
| txc_bare_antidead_t4_kpos20_grownChainFromT3 right-edge | 1 | +0.600 | 0.933 | n=1 |

### Top 10 cells by Δ coh ≥ 2.0

| arch + protocol | n | Δ | base | seeds_data |
|---|---:|---:|---:|---|
| txc_bare_antidead_t2_kpos20 per-position | 3 | +0.694 | 0.978 | n=3 |
| txc_bare_antidead_t2_kpos20 right-edge | 3 | +0.672 | 0.956 | n=3 |
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 per-position | 3 | +0.606 | 0.889 | n=3 |
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 right-edge | 3 | +0.472 | 0.756 | n=3 |
| txc_bare_antidead_t5_kpos20 right-edge | 2 | +0.400 | 0.683 | n=2 |
| txc_h8_t3_kpos20_shifts3 right-edge | 1 | +0.350 | 0.633 | n=1 |
| txc_h8_t3_kpos20_shifts3 per-position | 1 | +0.283 | 0.567 | n=1 |
| txc_bare_antidead_t5_kpos20_grownFromT2sd42 per-position | 1 | +0.283 | 0.567 | n=1 |
| txc_h8_t5_kpos20_shifts5 right-edge | 2 | +0.267 | 0.550 | n=2 |
| topk_sae right-edge | 2 | +0.267 | 0.550 | n=2 |

### Top 10 cells by Δ AUC(1.5-3.0)

| arch + protocol | n | Δ | base | seeds_data |
|---|---:|---:|---:|---|
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 right-edge | 3 | +0.340 | 0.753 | n=3 |
| txc_bare_antidead_t2_kpos20 right-edge | 3 | +0.331 | 0.745 | n=3 |
| txc_bare_antidead_t2_kpos20 per-position | 3 | +0.323 | 0.737 | n=3 |
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 per-position | 3 | +0.291 | 0.704 | n=3 |
| txc_h8_t3_kpos20_shifts3 right-edge | 1 | +0.278 | 0.691 | n=1 |
| txc_h8_t2_kpos20_shifts2 right-edge | 3 | +0.244 | 0.657 | n=3 |
| txc_galaxy4_t2_kw10_kp10 right-edge | 1 | +0.237 | 0.650 | n=1 |
| txc_bare_antidead_t2_kpos20_ws_tsae_encoder right-edge | 3 | +0.235 | 0.649 | n=3 |
| txc_bare_antidead_t3_kpos20 right-edge | 2 | +0.233 | 0.646 | n=2 |
| txc_h8_t3_kpos20_shifts3 per-position | 1 | +0.233 | 0.646 | n=1 |