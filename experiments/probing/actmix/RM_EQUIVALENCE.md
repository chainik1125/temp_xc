# P1-RM ↔ btk-only weight-equivalence table (auto: rm_equivalence.py, protocol (a))

3/4 pairs IDENTICAL (torch.equal on every shared tensor).

| arch | seed | T | tensors | verdict | Δauc | extra keys |
|---|---|---|---|---|---|---|
| batchtopk_sae | 1 | None | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
| batchtopk_sae | 2 | None | 0 | **METRIC-IDENTICAL (weights remote)** | +0.00e+00 | — |
| batchtopk_sae | 42 | None | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
| txc_batchtopk_pre | 42 | 1 | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
