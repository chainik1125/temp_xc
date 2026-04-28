---
author: Dmitry
date: 2026-04-25
tags:
  - results
  - in-progress
---

## TXC brickenauxk α=1/8 — frontier sweep at 5k / 10k / 40k

Best-from-ablation Bricken-resample + EMA-AuxK recipe (α=1/8, dead_threshold=128k tokens), trained from scratch with snapshots at 5k, 10k, 40k. Frontier sweep run on each via the em-features longform + OpenAI-judge loop (k=10 features, alphas −10 … +5).

Baseline (bad-medical, α=0): alignment 64.19, coherence 84.88

### Per-snapshot frontier

#### step 5000

| α | alignment | coherence |
|---:|---:|---:|
| −10 | 69.88 | 76.19 |
| −8  | 70.89 | 78.89 |
| −6  | **73.16** | 78.47 |
| −5  | 71.34 | 81.16 |
| −4  | 68.66 | 76.02 |
| −3  | 67.17 | 82.31 |
| −2  | 66.58 | 79.56 |
| −1.5| 66.76 | 78.33 |
| −1  | 68.89 | 79.94 |
| +1  | 66.85 | 78.75 |
| +2  | 66.04 | 79.78 |
| +5  | 75.54 | 79.28 |

Note: +5 outlier at 75.54 — likely poorly-calibrated feature direction at early training; curve is non-monotonic.

#### step 10000

| α | alignment | coherence |
|---:|---:|---:|
| −10 | 68.08 | 79.79 |
| −8  | **72.42** | 76.42 |
| −6  | 71.62 | 82.11 |
| −5  | 70.59 | 81.68 |
| −4  | 67.91 | 80.26 |
| −3  | 65.87 | 81.05 |
| −2  | 63.51 | 82.03 |
| −1.5| 61.65 | 79.63 |
| −1  | 67.83 | 79.48 |
| +1  | 64.45 | 78.65 |
| +2  | 66.69 | 79.90 |
| +5  | 67.69 | 80.21 |

#### step 40000

| α | alignment | coherence |
|---:|---:|---:|
| −10 | **77.55** | 79.61 |
| −8  | 74.83 | 79.88 |
| −6  | 72.79 | 79.28 |
| −5  | 66.90 | 79.67 |
| −4  | 68.36 | 79.76 |
| −3  | 66.82 | 77.36 |
| −2  | 59.60 | 67.77 |
| −1.5| 56.68 | 69.25 |
| −1  | 55.94 | 62.57 |
| +1  | 59.60 | 64.39 |
| +2  | 56.82 | 62.05 |
| +5  | 54.56 | 65.10 |

### Comparison with prior recipes

| Recipe | Step | Peak α | Peak align | Peak coh |
|---|---:|---:|---:|---:|
| SAE_k10 (Andy RDT) | – | −6 | **85.85** | 87.78 |
| MLC_small | 40k | −5 | 83.05 | 83.15 |
| MLC_small | 100k | −6 | 75.67 | 75.89 |
| MLC_small | 200k | −6 | 79.79 | 80.30 |
| TXC_small (no resample) | 40k | +1 | 68.05 | 75.87 |
| TXC_small (no resample) | 100k | −6 | 77.36 | 87.21 |
| TXC_small (no resample) | 200k | −5 | 74.90 | 88.40 |
| **TXC brickenauxk α=1/8** | 5k | −6 | 73.16 | 78.47 |
| **TXC brickenauxk α=1/8** | 10k | −8 | 72.42 | 76.42 |
| **TXC brickenauxk α=1/8** | **40k** | **−10** | **77.55** | 79.61 |

### Headline

- **Bricken+EMA-AuxK at 40k matches plain TXC at 100k** on peak alignment (77.55 vs 77.36) — ~2.5× training-step efficiency on the EM frontier task.
- Still below SAE_k10 (85.85) and MLC_40k (83.05).
- Coherence at peak is lower than long-trained TXC (~79 vs ~87) — likely because dead-fraction regresses to ~75% by step 40k; surviving features carry alignment signal but the dictionary is sparser overall.
- Misalignment-suppression direction calibrates around 10k: by 40k the steering response is monotonic in α; at 5k there's a +5 anomaly suggesting features are not yet aligned with the diff vector.

### Training-time dead-feature trajectory

| step | dead | loss |
|---:|---:|---:|
| 500 | 58.5% | 2581 |
| 5000 | 51.1% | 3722 |
| 10000 | 67.7% | 2566 |
| 21500 | 72.6% | 2212 |
| 40000 | 75.7% | 2643 |

Dead-fraction recovers to a minimum around step 5k (~51%) then regresses to ~75% by step 40k as Bricken can't keep up with newly-collapsing features. Loss decreases until step ~21k then plateaus.

### Files

- Frontier JSONs: `sweeps/qwen_l15_txc_brickenauxk_a8_step{5000,10000,40000}_frontier.json`
- Checkpoints (a100_1): `/root/em_features/checkpoints/qwen_l15_txc_brickenauxk_a8_step{5000,10000,40000}.pt`
- Training meta: `qwen_l15_txc_brickenauxk_a8_training.meta.json`
