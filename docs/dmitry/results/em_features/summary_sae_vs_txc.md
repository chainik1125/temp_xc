---
author: Dmitry
date: 2026-04-23
tags:
  - results
  - in-progress
---

## SAE vs TXC coherence/suppression frontier (Qwen-2.5-7B bad-medical, L15)

Qwen-2.5-7B-Instruct *bad-medical* (PEFT adapter vs base), layer 15, k=10 bundled feature steering via `ActivationSteerer(intervention_type="addition")`.
 Scores from the em-features longform + OpenAI-judge loop.

Baseline (bad-medical, α=0): alignment=64.19, coherence=84.88

### SAE

- **peak alignment**: 85.85 at α=-6.00 (coherence 87.78)
- **worst alignment**: 53.86 at α=+5.00 (coherence 78.49)
- feature set: k=10 (selection=top)

| α | alignment | coherence | Δalign | Δcoh |
|---:|---:|---:|---:|---:|
| -10.00 | 82.47 | 82.39 | +18.28 | -2.49 |
| -8.00 | 83.69 | 85.16 | +19.50 | +0.28 |
| -6.00 | 85.85 | 87.78 | +21.66 | +2.90 |
| -5.00 | 84.21 | 87.46 | +20.02 | +2.58 |
| -4.00 | 85.17 | 86.88 | +20.98 | +2.00 |
| -3.00 | 83.80 | 88.81 | +19.61 | +3.93 |
| -2.00 | 80.59 | 88.32 | +16.40 | +3.44 |
| -1.50 | 73.04 | 86.65 | +8.85 | +1.77 |
| -1.00 | 71.87 | 85.84 | +7.68 | +0.96 |
| +1.00 | 58.24 | 84.45 | -5.95 | -0.43 |
| +2.00 | 59.44 | 80.55 | -4.75 | -4.33 |
| +5.00 | 53.86 | 78.49 | -10.33 | -6.39 |

### TXC

- **peak alignment**: 74.24 at α=-10.00 (coherence 82.65)
- **worst alignment**: 51.85 at α=+5.00 (coherence 50.00)
- feature set: k=10 (selection=top)

| α | alignment | coherence | Δalign | Δcoh |
|---:|---:|---:|---:|---:|
| -10.00 | 74.24 | 82.65 | +10.05 | -2.23 |
| -8.00 | 72.79 | 85.28 | +8.60 | +0.40 |
| -6.00 | 72.85 | 84.16 | +8.66 | -0.72 |
| -5.00 | 72.85 | 85.25 | +8.66 | +0.37 |
| -4.00 | 66.71 | 85.99 | +2.52 | +1.11 |
| -3.00 | 71.98 | 84.79 | +7.79 | -0.09 |
| -2.00 | 63.75 | 84.03 | -0.44 | -0.85 |
| -1.50 | 64.91 | 84.73 | +0.72 | -0.15 |
| -1.00 | 62.29 | 56.14 | -1.90 | -28.74 |
| +1.00 | 62.11 | 57.51 | -2.08 | -27.37 |
| +2.00 | 68.28 | 58.80 | +4.09 | -26.08 |
| +5.00 | 51.85 | 50.00 | -12.34 | -34.88 |

## Headline
- SAE: peak align **85.85** at α=-6.00, coherence 87.78
- TXC: peak align **74.24** at α=-10.00, coherence 82.65

**Winner (peak alignment): SAE** — 85.85.
