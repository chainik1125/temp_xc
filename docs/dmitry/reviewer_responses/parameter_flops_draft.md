---
author: Dmitry
date: 2026-07-28
tags:
  - in-progress
  - results
---

## Parameter count and inference-FLOPs draft

These tables use the parameters of the actual headline runs rather than a
single representative model size. Values are trainable parameters in millions
and dense-matmul inference cost in GFLOPs for one batch element.

One multiply-add is counted as two FLOPs. Costs cover one encoder-plus-decoder
forward and exclude TopK/threshold selection, bias additions, nonlinearities,
and training-only losses. The native forward unit is one token for TopK SAE,
SAE-Arditi, and T-SAE; one five-position window for TFA, Stacked SAE, and
TXC-base; one five-layer bundle for MLC; and one ten-position window for
TXC-pro. TXC-base `T=10` and `T=20` use ten- and twenty-position windows.

### Headline configurations

| Setting | Residual width | Dictionary widths and extents |
|---|---:|---|
| Sparse probing | 2,304 | TopK/TFA/MLC/TXC: 18,432; T-SAE: 16,384; TXC-base: `T={5,10,20}`; TXC-pro: `T_max=10` |
| Backtracking | 4,096 | All dictionaries: 32,768; base/MLC/Stacked: `T=L=5`; TXC-pro: `T_max=10` |
| Medical EM | 3,584 | SAE/TFA/TXC: 32,768; T-SAE paper-width: 16,384; T-SAE matched-width: 32,768; base/Stacked: `T=5`; TXC-pro: `T_max=10` |
| HH-RLHF | 2,304 | All displayed dictionaries: 18,432; TXC/Stacked: `T=5` |

### Trainable parameters (millions)

| Architecture | Sparse probing | Backtracking | Medical EM | HH-RLHF |
|---|---:|---:|---:|---:|
| TopK SAE / SAE-Arditi | 84.96 | 268.47 | 234.92 | 84.96 |
| T-SAE | 75.52 | 268.47 | 117.46 paper / 234.92 matched | 84.96 |
| TFA, `T=5`, bottleneck 64 | 732.60 | 2,315.33 | 2,298.55 | — |
| MLC, `L=5` | 424.70 | 1,342.23 | — | — |
| Stacked SAE, `T=5` | 424.78 | 1,342.36 | 1,174.59 | 424.78 |
| TXC-base, `T=5` | 424.70 | 1,342.23 | 1,174.46 | 424.70 |
| TXC-base, `T=10` | 849.39 | — | — | — |
| TXC-base, `T=20` | 1,698.76 | — | — | — |
| TXC-pro, `T_max=10` | 849.39 | 2,684.43 | 2,348.88 | — |

### Dense inference cost (GFLOPs per native forward)

| Architecture and native input | Sparse probing | Backtracking | Medical EM | HH-RLHF |
|---|---:|---:|---:|---:|
| TopK SAE / SAE-Arditi, one token | 0.170 | 0.537 | 0.470 | 0.170 |
| T-SAE, one token | 0.151 | 0.537 | 0.235 paper / 0.470 matched | 0.170 |
| TFA, five tokens | 8.601 | 27.181 | 26.510 | — |
| MLC, five-layer bundle | 0.849 | 2.684 | — | — |
| Stacked SAE, five-token window | 0.849 | 2.684 | 2.349 | 0.849 |
| TXC-base, five-token window | 0.849 | 2.684 | 2.349 | 0.849 |
| TXC-base, ten-token window | 1.699 | — | — | — |
| TXC-base, twenty-token window | 3.397 | — | — | — |
| TXC-pro, ten-token window | 1.699 | 5.369 | 4.698 | — |

For an equal-length five-token segment, multiply the per-token SAE/T-SAE
figures by five. Consequently, a matched-width SAE or T-SAE and a five-position
TXC have the same leading dense-matmul FLOPs for five reconstructed positions;
the TXC has approximately five times as many parameters because its weights are
position-specific. Sliding-window evaluation adds one native forward per
window.

### Formulas

Let `d` be residual width, `h` dictionary width, `T` the temporal or layer
extent, `s` the TFA sequence length, and `b=64` the TFA attention bottleneck.

| Architecture | Trainable parameters | Dense inference FLOPs |
|---|---:|---:|
| TopK SAE / T-SAE | `2dh+h+d` | `4dh` per token |
| Stacked SAE | `T(2dh+h+d)` | `4Tdh` per `T`-window |
| TXC / MLC | `2Tdh+h+Td` | `4Tdh` per `T`-window/bundle |
| TFA | `dh+d+2h²(1+1/b)+2h(1+1/b)` | `8sdh+4sh²(1+1/b)+2s²h(1+1/b)` |

The published T-SAE has no inference-time attention: Matryoshka, AuxK, and
contrastive terms affect training but add no trainable matrices or inference
FLOPs. TXC-pro's Matryoshka and contrastive terms likewise add no parameters;
its larger count comes from storing encoder and decoder slabs for all ten
positions and using the full ten-position model at probe time.

### Audit note

The sparse-probing headline T-SAE checkpoints use `d_SAE=16,384`, even though
the current appendix says every sparse-probing cell uses 18,432. That sentence
needs correction. The May 6 medical figure was built from 32,768-wide T-SAE
checkpoints; the 16,384-wide paper-setting result and the later 32,768-wide
matched rerun in the reviewer response should therefore remain explicitly
distinguished.
