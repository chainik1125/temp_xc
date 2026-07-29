---
author: Dmitry
date: 2026-07-28
tags:
  - in-progress
  - results
---

## 🟦 Reviewer 2 response

## OpenReview copy-and-paste version

We thank the reviewer for recognizing the novelty of applying crosscoders along the sequence axis and the value of the synthetic and real-world benchmarks. We provide responses to the specific question about temporal vs. generic capacity below.

> 🟦 Since the non-temporal MLC ties TXC on probing, can you isolate the temporal contribution from generic crosscoder capacity?

We provide several lines of evidence. First, the Stacked SAE matches the TXC's trainable parameter count and dense inference cost but cannot share information across sequence positions. We report both quantities for all architectures below, as requested by 🟩 Reviewer bbby.

**Trainable parameters (billions):**

| Architecture | Sparse | Backtracking | Medical EM | HH-RLHF |
| :--- | ---: | ---: | ---: | ---: |
| TopK SAE / SAE-Arditi | 0.085 | 0.27 | 0.23 | 0.085 |
| T-SAE | 0.076 | 0.27 | 0.12 / 0.23¹ | 0.085 |
| TFA (T = 5) | 0.73 | 2.3 | 2.3 | — |
| MLC (L = 5) | 0.42 | 1.3 | — | — |
| Stacked SAE (T = 5) | 0.42 | 1.3 | 1.2 | 0.42 |
| TXC-base (T = 5) | 0.42 | 1.3 | 1.2 | 0.42 |
| TXC-base (T = 10 / 20) | 0.85 / 1.7 | — | — | — |
| TXC-pro (max T = 10) | 0.85 | 2.7 | 2.3 | — |

**Dense inference cost (GFLOPs per forward pass):**

| Architecture and native input | Sparse | Backtracking | Medical EM | HH-RLHF |
| :--- | ---: | ---: | ---: | ---: |
| TopK SAE / SAE-Arditi, 1 token | 0.17 | 0.54 | 0.47 | 0.17 |
| T-SAE, 1 token | 0.15 | 0.54 | 0.24 / 0.47¹ | 0.17 |
| TFA, 5 tokens | 8.6 | 27 | 27 | — |
| MLC, 5 layers | 0.85 | 2.7 | — | — |
| Stacked SAE, 5 tokens | 0.85 | 2.7 | 2.3 | 0.85 |
| TXC-base, 5 tokens | 0.85 | 2.7 | 2.3 | 0.85 |
| TXC-base, 10 / 20 tokens | 1.7 / 3.4 | — | — | — |
| TXC-pro, 10 tokens | 1.7 | 5.4 | 4.7 | — |


Having established equal capacity, we compare the TXC with the Stacked SAE baseline directly. The TXC outperforms everywhere outside of EM:


**Stacked SAE:**

| Task and metric | Stacked SAE | TXC | Stacked / TXC | Floor |
| :--- | ---: | ---: | ---: | ---: |
| Sparse probing, mean AUC | 0.87 | 0.89 | 0.98 | 0.80 |
| Backtracking steering | 0.25 | 0.54 | 0.45 | — |
| Backtracking detection | 0.16 | 0.24 | 0.65 | — |
| Medical EM steering | Not run | 23 | — | — |
| Medical EM detection | 0.65† | 0.54 | 1.2† | 0.34 |
| HH-RLHF | 0.60 | 0.61 | 0.99 | **0.62** |


The matched-capacity Backtracking result isolates cross-position sharing from
generic parameter count: the Stacked SAE reaches only 0.45× the TXC's
steering effect and 0.65× its detection PR-AUC. This is the clearest
real-world evidence that the gain is not explained by capacity alone. We also
report the explicit dependence on window size:

**Window-size sweep** (percentage of the headline score):

| Task | Headline score | T = 1 | T = 2 | T = 4 | T = 5 | T = 6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse probing | 0.89 | 101% | 101% | 101% | 100% | 100% |
| Backtracking | 0.26 | 85% | 85% | 92% | 100% | 100% |
| Medical EM | 0.54 | 83% | 92% | 90% | 100% | 115% |
| HH-RLHF | 0.62 | 96% | 100% | 99% | 100% | 101% |

Longer context does not help static probing, but it materially helps
Backtracking and Medical EM. We do not expect temporal structure to help on
every task; rather, these results show both that temporal structure matters for
*some* important SAE applications *and* that TXCs can exploit it.

We also take the opportunity to answer 🟩 Reviewer bbby's request for an explicit accounting of capacity for the T-SAE:

**T-SAE dictionary-width control**

For each task, we ran the T-SAE at the width used for the paper result and,
where different, at the width matched to the other architectures. Where these
disagreed, we selected the better-performing variant to ensure that we
compared against the strongest baseline. We have clarified this and added both
settings to the paper appendix:

| Task | Metric | TXC | T-SAE, paper width | T-SAE, matched width |
| :--- | :--- | ---: | ---: | ---: |
| Backtracking | Detection PR-AUC at S = 32 | 0.26‡ | 0.25 (d = 32,768) | 0.25 (d = 32,768) |
| Medical EM | Detection PR-AUC at S = 16 | 0.54 | 0.71 (d = 16,384) | 0.43 (d = 32,768) |
| HH-RLHF | Preference ROC-AUC at k = 20 | 0.62 | 0.60 (d = 18,432) | 0.60 (d = 18,432) |
