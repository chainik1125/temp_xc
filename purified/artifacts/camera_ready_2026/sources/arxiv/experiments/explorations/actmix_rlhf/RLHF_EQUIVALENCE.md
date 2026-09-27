# RLHF relu-mix ↔ btk-only weight equivalence (CARD § 7 A3)

3/3 compared pairs IDENTICAL (torch.equal on every shared tensor).

| arch | seed | T | k_pos | tensors | verdict | Δauc_k20 | extra keys |
|---|---|---|---|---|---|---|---|
| batchtopk_sae | 42 | None | 500 | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
| txc_batchtopk_post | 42 | 16 | 1600 | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
| txc_batchtopk_post | 42 | 5 | 500 | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
