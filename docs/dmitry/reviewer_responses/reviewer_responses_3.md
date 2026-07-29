---
author: Dmitry
date: 2026-07-28
tags:
  - in-progress
  - results
---

## Reviewer 3 response

<!-- (1-3 sentences, synthetic) First, it has not previously been established in the
  literature that temporal architectures _actually_ recover temporal structure. As
  pointed out in (some reference work which makes this point, I think David
  Chanin's paper probably does but somebody else's would be better) this is not
  possible without a ground truth. We provide the appropriate generalization
  (maybe another word which has less connotations of a bolt on) and show that TXC
  outperform all other architectures where a temporal ground truth is available.
  In our response to reviewer xxx we make this even more explicitly through a
  formal no-go theorem.

  (Not in this version, but something to note for a potential rewrite is that we
  could here make the point that the synthetic benchmark is in and of itself
  valuable)

  (1-3 sentences - what is the scale?)

  Second, it is important to set the scale of comparison (This is a bad sentence
  but somethig like this). In the real world setting, TXCs achieve a capability
  uplift relative to SAEs roughly comparable to TSAEs (better: make this precise),
  and meaningfully outpeform TFA (neither of which have previously been
  benchmarked on a comparable panel of real world tasks). <insert relative to SAE
  table>. Moreover, as we emphasize in Sec. xx, we consider it significant that
  the capability profile of the TSAE is _different_ to the TXC, suggesting that
  both architectures can be used in parrallel, and tha the  TXC can accomodate the
  INFOCE penalty of the TSAE.

  (An alternative structure for this para is to say after the first sentence.
  Expressed as a percentage of SAE performance, the performance is: <table>

  after the table:
  Then TXCs achieve a capability uplift ....

  actually this is better because we lead with the data lets adopt that, but put
  the above version in comments)

  Third, something


  (Maybe we can close by using the forbidden jutsu of saying something like a more
  extensive temporal profiling of differet tasks, further developing the txc, and
  xxx are all valuable areas of investigation for the temporal feature analysis
  community, but they are out of scope for a single work. Basically flagging that
  what they're calling preliminary is basically setting a standard that compbines
  1 different papers) -->




We thank the reviewer for taking the time to review our manuscript (MS) and for appreciating the motivation behind our proposal. We appreciate the constructive suggestions on formatting, readability and proper citation.

### Performance improvements


> Experiments show a marginal improvement over existing works, such as T-SAE
> and MLC. This leaves the proposed TXC and TXC-pro primarily motivated by the
> backtracking results.

We emphasize three points.

#### 1. TXCs are the only architecture which reliably recovers ground truth temporal features

Prior work has emphasized that feature recovery cannot be established without ground truth (Venhoff et al., 2024; Makelov et al., 2024). In section 4, we find that the TXC is the only temporal architecture we tested that recovers ground-truth temporal features. In our response to Reviewer bbby we make this even more explicit through a task with a formal no-go theorem on per-token recoverability. We summarize these results below:


<!-- TODO: compactify as in others, express as ratio of SAE -->
| Task and metric | SAE | T-SAE | TFA | TXC |
| :--- | ---: | ---: | ---: | ---: |
| Denoising, global \(R^2\) | 0.363 | 0.382 | 0.157 | **0.483** |
| Coupling, peak gAUC | 0.884 | 0.941 | 0.663 | **0.990** |
| Secret recovery, \(W=10\) | 0.10 | 0.12 | 0.09 | **0.96** |

<!--  -->
#### 2. On sparse probing, TXC's improvement is comparable to T-SAE; across tasks, their capability profiles are complementary
<!-- TODO: We should agree that backtracking is the only panel where TXC clearly beats all other architectures. -->

Whilst this is a fair concern, this requires a scale of improvement. Previous work has used the natural comparison to conventional SAEs (Cite). That comparison for TXCs is:

<!-- TODO: Add TFA, 2 sf -->
| Task | T-SAE | TXC |
| :--- | ---: | ---: |
| Sparse probing | 101.5% | 101.5–101.8% |
| Backtracking | 40% | **135%** |
| Medical EM | **121%** | 92% |
| HH-RLHF, null result | 98% | 102% |

On the three informative tasks, TXC and T-SAE each improve over the SAE on two:
both improve sparse probing, TXC improves backtracking, and T-SAE improves
Medical EM. TXCs also outperforms TFA on everything but HH-RLHF. These architectures therefore have comparable breadth but different capability profiles.

#### 3. Matched benchmarking

We finally note that previous temporal architectures were introduced and evaluated separately. To our knowledge, this is the first work to evaluate T-SAE, TFA, MLC, and TXC on a common panel under matched conditions. The finding that these architectures are complementary, rather than substitutes is itself valuable.
