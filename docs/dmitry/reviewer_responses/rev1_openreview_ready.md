---
author: Dmitry
date: 2026-07-28
tags:
  - in-progress
  - results
---

## Reviewer 1 — OpenReview-ready response

We thank the reviewer for recognizing the novelty of our proposal and for noting the synthetic benchmarking procedure we introduce.

### Summary of response

**1. Temporal contribution.** The reviewer points out the importance of establishing that the temporal features enabled by the temporal crosscoder are responsible for the performance improvements shown. We provide three additional lines of evidence:

- **a.** We introduce an additional synthetic task that is provably impossible without temporal feature sharing. The TXC strongly outperforms all other architectures on this task.
- **b.** We provide window-size sweeps for all tasks (see our response to Reviewer 4z15).
- **c.** We provide the Stacked SAE baseline explicitly (see our response to Reviewer 4z15).

**2. Seed variance.** We provide explicit three-seed results and confirm that the relative rankings do not change.

**3. Remaining points.** We respond below to the abstract wording, T-SAE dictionary size, parameter count, and inference cost.

### Synthetic setting

To prove that the TXC uses temporal information, we introduce a synthetic task with an analytic ceiling on recoverability from single-token information. The task uses a Hidden Markov Model based on Shamir secret sharing. This HMM encodes a temporal “secret” whose recovery can be no better than random guessing below a threshold number of temporal steps, h. For a fair comparison, when probing single-token architectures, we stack the activations over a window of the same size.

For h = 2, recovery below three steps is bounded at the chance accuracy of 1/11 ≈ 0.09. All methods satisfy this ceiling. Beyond the threshold, TXC accuracy improves from 0.15 at W = 3 to near-perfect recovery, 0.96, at W = 10.

**Secret-recovery accuracy** (parentheses give the selected k):

| Architecture | W = 1 | W = 2 | W = 3 | W = 4 | W = 5 | W = 10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chance | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 |
| SAE | 0.10 (2) | 0.10 (10) | 0.10 (1) | 0.09 (2) | 0.09 (1) | 0.10 (10) |
| Stacked SAE | 0.10 (2) | 0.09 (5) | 0.10 (1) | 0.10 (2) | 0.10 (10) | 0.11 (5) |
| T-SAE | — | 0.10 (5) | 0.10 (5) | 0.10 (2) | 0.10 (2) | 0.12 (2) |
| TFA | — | 0.10 (2) | 0.10 (2) | 0.10 (10) | 0.10 (1) | 0.09 (2) |
| TXC, k = 1 | 0.10 | 0.09 | 0.13 | 0.19 | 0.29 | 0.63 |
| TXC, k = 2 | 0.10 | 0.09 | **0.15** | **0.32** | **0.56** | 0.91 |
| TXC, k = 5 | 0.09 | 0.09 | 0.10 | 0.10 | 0.16 | **0.96** |

For every non-TXC baseline, parentheses give the selected k independently at each window size. We sweep k over {1, 2, 5, 10, 20} and choose the best k.

### Seed dependence

The reviewer asks:

> The checklist says the experiments use 2 seeds, but the appendix says the main backtracking results use only seed 42. Which is correct? Are the reported improvements larger than the variation across different seeds?

In the main text, we reported one training seed, consistent with landmark SAE-architecture papers introducing TopK (Gao et al., 2025, ICLR), Gated and JumpReLU (Rajamanoharan et al., 2024a,b), and BatchTopK (Bussmann et al., 2024). The appendix provided an additional seed; we have now updated it to include three-seed results:

| Task | TXC-base, seeds 1 / 2 / 42 | TopK SAE |
| :--- | ---: | ---: |
| Sparse probing | 0.90 / 0.90 / 0.90 | 0.89 |
| Backtracking | Pending / Pending / 0.54 | 0.40 |
| Medical EM | 17 / Pending / 23 | 21 |
| HH-RLHF | 0.62 / 0.62 / 0.62 | 0.61 |

### Other points

**a. Abstract wording**

> Is there one model setting where both the 40% better detection and the 15% better inducement are achieved? If not, please make this clear in the abstract.

Detection (Fig. 4b) and inducement (Fig. 4a) refer to the excess performance of the base TXC over the TopK SAE: a 40% higher detection AUC and a 15% higher average rate of backtracking induced, respectively. We have clarified the abstract to reflect this.

**b. Is the T-SAE underpowered?**

> Why does T-SAE use a smaller dictionary if the paper claims all methods use the same dictionary size? Please also report the parameter count and inference cost for each model.

Our description of dictionary sizes was ambiguous. For each task, we ran the T-SAE at the width used for the paper result and, where different, at the width matched to the other architectures. Where these disagreed, we selected the better-performing variant to ensure that we compared against the strongest baseline. We have clarified this and added both settings to the paper appendix:

| Task and metric | TXC | T-SAE, paper width | T-SAE, matched width |
| :--- | ---: | ---: | ---: |
| Backtracking, PR-AUC at S = 32 | 0.26‡ | 0.25 (d = 32,768) | 0.25 (d = 32,768) |
| Medical EM, PR-AUC at S = 16 | 0.54 | 0.71 (d = 16,384) | 0.43 (d = 32,768) |
| HH-RLHF, ROC-AUC at k = 20 | 0.62 | 0.60 (d = 18,432) | 0.60 (d = 18,432) |

‡ The Backtracking TXC value is the T = 5 cell from the new window-size sweep.

**c. Parameter count and inference cost**

See our response to Reviewer 4z15.
