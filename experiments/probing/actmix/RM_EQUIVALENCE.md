# P1-RM ↔ btk-only weight-equivalence table (auto: rm_equivalence.py, protocol (a))

3/8 pairs IDENTICAL (torch.equal on every shared tensor).

| arch | seed | T | tensors | verdict | Δauc | extra keys |
|---|---|---|---|---|---|---|
| batchtopk_sae | 1 | None | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
| batchtopk_sae | 2 | None | 0 | **METRIC-IDENTICAL (weights remote)** | +0.00e+00 | — |
| batchtopk_sae | 42 | None | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
| txc_batchtopk_pre | 42 | 1 | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
| txc_batchtopk_pre | 42 | 2 | 0 | **METRIC-DIVERGES (weights remote)** | +4.56e-03 | — |
| txc_batchtopk_pre | 42 | 4 | 7 | **DIVERGES** | +1.54e-03 | threshold_set |
| txc_batchtopk_pre | 42 | 6 | 7 | **DIVERGES** | -1.63e-02 | threshold_set |
| txc_batchtopk_pre | 42 | 16 | 7 | **DIVERGES** | +2.46e-03 | threshold_set |
