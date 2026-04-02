---
author: Dmitry
date: 2026-03-26
tags:
  - results
  - in-progress
---

# BW factorial TopK results after 2k steps across rho

## Setup

These runs used the current `bw_factorial` model with a hard TopK bottleneck on the decoded latent coefficients.

Common settings:

- model: `bw_factorial`
- latent dimension: `n_features = 50`
- activation dimension: `d_model = 100`
- window length: `T = 5`
- TopK: `k = 5`
- training batch size: `64`
- learning rate: `3e-4`
- training steps: `2000`
- evaluation cadence: every `1000` steps
- evaluation set size: `200` sequences
- device: `cuda` on `a40_2`

The only thing varied across runs was the temporal persistence parameter `rho`.

## Timing

Training speed was very stable across all three runs.

- `rho = 0.0`: about `22.66-22.68 ms/step`
- `rho = 0.5`: about `21.90-22.40 ms/step`
- `rho = 1.0`: about `21.70-21.86 ms/step`

So the runtime cost of the model does not appear to depend much on `rho`.

## Main table

| rho | step | avg step ms | eval NMSE | eval AUC | mean max cos | R@0.9 | R@0.8 | eval L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0 | - | 0.9224 | 0.3212 | 0.3218 | 0.00 | 0.00 | 5.0 |
| 0.0 | 1000 | 22.68 | 0.2976 | 0.8380 | 0.8392 | 0.76 | 0.76 | 5.0 |
| 0.0 | 2000 | 22.66 | 0.2393 | 0.8628 | 0.8680 | 0.80 | 0.82 | 5.0 |
| 0.5 | 0 | - | 0.9205 | 0.2452 | 0.2444 | 0.00 | 0.00 | 5.0 |
| 0.5 | 1000 | 22.40 | 0.3041 | 0.8528 | 0.8518 | 0.68 | 0.76 | 5.0 |
| 0.5 | 2000 | 21.90 | 0.2028 | 0.9280 | 0.9328 | 0.86 | 0.88 | 5.0 |
| 1.0 | 0 | - | 0.8808 | 0.2504 | 0.2506 | 0.00 | 0.00 | 5.0 |
| 1.0 | 1000 | 21.70 | 0.6608 | 0.5128 | 0.5125 | 0.00 | 0.00 | 5.0 |
| 1.0 | 2000 | 21.86 | 0.6532 | 0.5524 | 0.5528 | 0.00 | 0.04 | 5.0 |

## Most important observations

### 1. The TopK constraint is behaving as intended

Across all runs and all checkpoints, the evaluation `L0` stayed pinned at `5.0`.

That means the hard TopK change is working cleanly: the model is not drifting to dense solutions as training proceeds.

### 2. `rho = 0.5` is the strongest regime at 2k steps

At `2000` steps:

- `rho = 0.5` reached `NMSE = 0.2028`
- `rho = 0.5` reached `AUC = 0.9280`
- `rho = 0.5` reached `mean max cosine = 0.9328`

This is better than both:

- `rho = 0.0`, which ended at `NMSE = 0.2393`, `AUC = 0.8628`
- `rho = 1.0`, which ended at `NMSE = 0.6532`, `AUC = 0.5524`

So in this early training regime, moderate persistence is the easiest regime for the model to exploit.

### 3. `rho = 0.0` still learns well

The iid case does not appear pathological for this model.

From step `0` to step `2000`, the model improved from:

- `NMSE = 0.9224` to `0.2393`
- `AUC = 0.3212` to `0.8628`

So even without temporal persistence, the learned latent dictionary plus BW bottleneck still finds a useful representation.

### 4. `rho = 1.0` is much harder

The fully persistent case improves only weakly by `2000` steps:

- `NMSE` goes from `0.8808` to `0.6532`
- `AUC` goes from `0.2504` to `0.5524`
- `R@0.9` stays at `0.00`

This is qualitatively different from the `rho = 0.0` and `rho = 0.5` runs.

So, at least with the current architecture and optimization, the model is **not yet handling the maximally persistent regime well**.

## Interpretation

The current picture seems to be:

- the BW-factorial TopK model is definitely trainable,
- it learns strong feature recovery by `2k` steps for `rho = 0.0` and especially `rho = 0.5`,
- but it struggles substantially for `rho = 1.0`.

One plausible interpretation is that the current model is good at using temporal information when persistence is present but not too extreme, while the fully persistent case may require either:

- longer training,
- a better observation model,
- a different initialization for the HMM parameters,
- or a cleaner coupling between the BW posterior and the continuous amplitude head.

## Short summary

After `2000` steps, the current `bw_factorial` TopK model works well for `rho = 0.0` and `rho = 0.5`, with the best performance at `rho = 0.5`, while `rho = 1.0` remains clearly underfit.

The good news is that:

- the model trains stably,
- the TopK sparsity constraint works,
- and the runtime per step is stable at about `22 ms`.

The main open issue is not stability, but performance in the strongest-persistence regime.
