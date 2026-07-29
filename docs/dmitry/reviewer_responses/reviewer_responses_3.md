---
author: Dmitry
date: 2026-07-28
tags:
  - in-progress
  - results
---

## Reviewer 3 response

We thank the reviewer for taking the time to review our manuscript (MS) and for appreciating the motivation behind our proposal. We appreciate the constructive suggestions on formatting, readability and proper citation.

### Performance improvements


> Experiments show a marginal improvement over existing works, such as T-SAE
> and MLC. This leaves the proposed TXC and TXC-pro primarily motivated by the
> backtracking results.

We emphasize three points.

#### 1. TXCs are the only architecture which reliably recovers ground truth temporal features

Prior work has emphasized that feature recovery cannot be established without ground truth (Venhoff et al., 2024; Makelov et al., 2024). In section 4, we find that the TXC is the only temporal architecture we tested that recovers ground-truth temporal features. In our response to Reviewer bbby we make this even more explicitly through a task with a formal no-go theorem. We summarize the ground truth recovery results in secton 4 below.

| Task and metric | SAE | T-SAE | TFA | TXC |
| :--- | ---: | ---: | ---: | ---: |
| Denoising, global \(R^2\) | 0.363 | 0.382 | 0.157 | **0.483** |
| Coupling, peak gAUC | 0.884 | 0.941 | 0.663 | **0.990** |
| Secret recovery, \(W=10\) | 0.10 | 0.12 | 0.09 | **0.96** |

<!--  -->
#### 2. Scale of improvement

The appropriate scale is performance relative to the regular SAE:

| Real-world task | T-SAE, % of SAE | TXC, % of SAE |
| :--- | ---: | ---: |
| Sparse probing | 101.5% | 101.5–101.8% |
| Backtracking | 40% | **135%** |
| Medical EM | **121%** | 92% |
| HH-RLHF, null result | 98% | 102% |

On the three informative tasks, TXC and T-SAE each improve over the SAE on two:
both improve sparse probing, TXC improves backtracking, and T-SAE improves
Medical EM. MLC reaches 102.3% of SAE performance on sparse probing but 62.5%
on backtracking. TXC also outperforms TFA on four of the five non-HH-RLHF
headline comparisons. These architectures therefore have comparable breadth
but different capability profiles.

#### 3. Matched benchmarking

The matched comparison is itself part of the contribution. Previous temporal
architectures were introduced and evaluated separately. To our knowledge, this
is the first work to evaluate T-SAE, TFA, MLC, and TXC on a common panel under
matched conditions. The finding that these architectures are complementary,
rather than universally ordered, is precisely what this comparison makes
visible.
