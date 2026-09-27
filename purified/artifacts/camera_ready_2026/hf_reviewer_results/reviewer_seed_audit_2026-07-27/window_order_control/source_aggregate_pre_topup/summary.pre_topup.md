# Backtracking window-size sweep

This report is regenerated after every completed cell. Values are means over question-grouped outer folds at the largest registered sparse-probe feature budget. A row is not a three-seed result until all seeds are present.

| T | Seed | TXC ordered | Strongest TXC order control | TXC − positional SAE [95% question CI] | TXC − strongest learned control [95% question CI] | SAE invariant | Last-token SAE | Residual ordered |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 0.2273 | 0.2243 | +0.0203 [+0.0076, +0.0327] | -0.0087 [-0.0267, -0.0020] | 0.2312 | 0.2307 | 0.2385 |
| 2 | 2 | 0.2319 | 0.2305 | +0.0218 [+0.0065, +0.0361] | +0.0003 [-0.0106, +0.0030] | 0.2218 | 0.2108 | 0.2368 |
| 3 | 1 | 0.2321 | 0.2306 | +0.0296 [+0.0158, +0.0431] | -0.0047 [-0.0239, +0.0009] | 0.2241 | 0.2202 | 0.2493 |
| 3 | 2 | 0.2346 | 0.2325 | +0.0391 [+0.0258, +0.0526] | -0.0053 [-0.0142, +0.0015] | 0.2273 | 0.2185 | 0.2465 |
| 3 | 42 | 0.2365 | 0.2254 | +0.0434 [+0.0331, +0.0542] | +0.0035 [-0.0091, +0.0092] | 0.2213 | 0.2180 | 0.2445 |
| 4 | 1 | 0.2529 | 0.2388 | +0.0674 [+0.0490, +0.0880] | +0.0142 [+0.0017, +0.0196] | 0.2221 | 0.2152 | 0.2506 |
| 4 | 2 | 0.2569 | 0.2377 | +0.0644 [+0.0470, +0.0804] | +0.0183 [+0.0054, +0.0241] | 0.2147 | 0.2152 | 0.2462 |
| 4 | 42 | 0.2438 | 0.2297 | +0.0449 [+0.0279, +0.0599] | +0.0118 [-0.0031, +0.0167] | 0.2210 | 0.2031 | 0.2439 |
| 5 | 1 | 0.2650 | 0.2521 | +0.0843 [+0.0654, +0.1037] | +0.0100 [-0.0030, +0.0155] | 0.2349 | 0.2317 | 0.2512 |
| 5 | 2 | 0.2653 | 0.2537 | +0.0656 [+0.0488, +0.0810] | +0.0104 [-0.0002, +0.0171] | 0.2259 | 0.2128 | 0.2484 |
| 5 | 42 | 0.2607 | 0.2446 | +0.0793 [+0.0604, +0.0965] | +0.0154 [+0.0050, +0.0219] | 0.2197 | 0.2124 | 0.2426 |

## Seed aggregation

| T | Seeds complete | TXC ordered mean ± SD | SAE positional mean ± SD | Conservative order gap mean ± SD |
|---:|---:|---:|---:|---:|
| 2 | 2 | 0.2296 ± 0.0033 | 0.2086 ± 0.0023 | +0.0022 ± 0.0011 |
| 3 | 3 | 0.2344 ± 0.0022 | 0.1970 ± 0.0049 | +0.0049 ± 0.0054 |
| 4 | 3 | 0.2512 ± 0.0067 | 0.1923 ± 0.0067 | +0.0158 ± 0.0029 |
| 5 | 3 | 0.2637 ± 0.0025 | 0.1873 ± 0.0108 | +0.0135 ± 0.0023 |

![Backtracking detection versus window size](window_curve.png)

The fixed-probe order perturbations measure sensitivity to the learned ordered representation under covariate shift. The SAE positional stack and the train-fold-only residual control are the stronger tests of whether TXC adds value beyond multi-token support.
