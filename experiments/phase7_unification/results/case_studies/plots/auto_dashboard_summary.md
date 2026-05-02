## Auto-updating dashboard — Phase 7 steering case study

All cells discovered with grades.jsonl + generations.jsonl ≥ 200 rows.

### Anchor: T-SAE k=20

unc=1.678, ≥1.5=1.133, ≥1.75=0.411, ≥2.0=0.411, ≥2.25=0.411, ≥2.5=0.411, AUC(1.5-3.0)=0.574


### Top 10 cells by Δ coh ≥ 1.5 (prereg)

| arch + protocol | n | Δ | base | seeds_data |
|---|---:|---:|---:|---|
| txc_contrastive_h8_t2_kpos20_shifts2 right-edge | 3 | +0.444 | 1.578 | n=3 |
| txc_softmax_pool_h8_t2_kpos20_shifts2 right-edge | 3 | +0.333 | 1.467 | n=3 |
| txc_softmaxpool_t3_kpos20 tiled-broadcast | 3 | +0.311 | 1.444 | n=3 |
| txc_maxpool_t2_kpos20 per-position | 3 | +0.300 | 1.433 | n=3 |
| txc_galaxy4_t2_kw10_kp10 per-position | 3 | +0.300 | 1.433 | n=3 |
| txc_softmaxpool_t2_kpos20 per-position | 3 | +0.289 | 1.422 | n=3 |
| txc_h8_t2_kpos20_shifts2 per-position | 3 | +0.267 | 1.400 | n=3 |
| txc_softmaxpool_t3_kpos20 per-position | 3 | +0.222 | 1.356 | n=3 |
| txc_softmaxpool_t2_kpos20 tiled-broadcast | 3 | +0.200 | 1.333 | n=3 |
| txc_maxpool_t2_kpos20 right-edge | 3 | +0.189 | 1.322 | n=3 |

### Top 10 cells by Δ coh ≥ 1.75

| arch + protocol | n | Δ | base | seeds_data |
|---|---:|---:|---:|---|
| txc_softmaxpool_t3_kpos20 tiled-broadcast | 3 | +1.033 | 1.444 | n=3 |
| txc_softmaxpool_t2_kpos20 per-position | 3 | +1.011 | 1.422 | n=3 |
| txc_lsepool_t2_kpos20 per-position | 3 | +0.889 | 1.300 | n=3 |
| txc_h8_t2_kpos20_shifts2 right-edge | 3 | +0.828 | 1.239 | n=3 |
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 right-edge | 3 | +0.811 | 1.222 | n=3 |
| txc_softmaxpool_t3_kpos20 right-edge | 3 | +0.767 | 1.178 | n=3 |
| txc_maxpool_h8_t2_kpos20_shifts2 right-edge | 3 | +0.733 | 1.144 | n=3 |
| txc_maxpool_h8_t2_kpos20_shifts2 per-position | 3 | +0.733 | 1.144 | n=3 |
| txc_bare_antidead_t4_kpos20_grownChainFromT3 per-position | 1 | +0.722 | 1.133 | n=1 |
| txc_bare_antidead_t2_kpos20_ws_tsae_encoder per-position | 3 | +0.722 | 1.133 | n=3 |

### Top 10 cells by Δ coh ≥ 2.0

| arch + protocol | n | Δ | base | seeds_data |
|---|---:|---:|---:|---|
| txc_bare_antidead_t2_kpos20 per-position | 3 | +0.567 | 0.978 | n=3 |
| txc_bare_antidead_t2_kpos20 right-edge | 3 | +0.544 | 0.956 | n=3 |
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 per-position | 3 | +0.478 | 0.889 | n=3 |
| txc_maxpool_t2_kpos20 right-edge | 3 | +0.444 | 0.856 | n=3 |
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 right-edge | 3 | +0.344 | 0.756 | n=3 |
| txc_softmaxpool_t3_kpos20 tiled-broadcast | 3 | +0.322 | 0.733 | n=3 |
| txc_bare_antidead_t5_kpos20 right-edge | 2 | +0.272 | 0.683 | n=2 |
| txc_h8_t3_kpos20_shifts3 right-edge | 1 | +0.222 | 0.633 | n=1 |
| txc_softmaxpool_t5_kpos20 right-edge | 2 | +0.206 | 0.617 | n=2 |
| txc_h8_t3_kpos20_shifts3 per-position | 1 | +0.156 | 0.567 | n=1 |

### Top 10 cells by Δ AUC(1.5-3.0)

| arch + protocol | n | Δ | base | seeds_data |
|---|---:|---:|---:|---|
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 right-edge | 3 | +0.179 | 0.753 | n=3 |
| txc_bare_antidead_t2_kpos20 right-edge | 3 | +0.170 | 0.745 | n=3 |
| txc_bare_antidead_t2_kpos20 per-position | 3 | +0.163 | 0.737 | n=3 |
| txc_softmaxpool_t3_kpos20 tiled-broadcast | 3 | +0.161 | 0.735 | n=3 |
| txc_maxpool_h8_t2_kpos20_shifts2 per-position | 3 | +0.152 | 0.726 | n=3 |
| txc_softmaxpool_t2_kpos20 per-position | 3 | +0.152 | 0.726 | n=3 |
| txc_maxpool_h8_t2_kpos20_shifts2 right-edge | 3 | +0.140 | 0.715 | n=3 |
| txc_bare_antidead_t3_kpos20_grownFromT2sd42 per-position | 3 | +0.130 | 0.704 | n=3 |
| txc_maxpool_t2_kpos20 per-position | 3 | +0.122 | 0.696 | n=3 |
| txc_softmaxpool_t2_kpos20 tiled-broadcast | 3 | +0.117 | 0.691 | n=3 |