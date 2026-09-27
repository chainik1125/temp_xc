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

We thank the reviewer for taking the time to review our manuscript and for appreciating the motivation behind our proposal. We appreciate the constructive suggestions on formatting, readability and proper citation.

### Performance improvements

The core scientific concern raised by the reviewer is:

> Experiments show a marginal improvement over existing works, such as T-SAE
> and MLC. This leaves the proposed TXC and TXC-pro primarily motivated by the
> backtracking results.

We respond in three parts:

#### 1. This concern neglects the synthetic results

TXCs are the only architecture which reliably recovers ground truth temporal features. Prior work has emphasized that feature recovery cannot be established without ground truth (Venhoff et al., 2024; Makelov et al., 2024). In section 4, we find that the TXC is the only temporal architecture we tested that recovers ground-truth temporal features. In our response to Reviewer bbby we make this even more explicit through a task with a formal no-go theorem on per-token recoverability. We summarize these results below:

|Task|Metric|SAE|TFA|T-SAE|TXC|
|:---|:---|---:|---:|---:|---:|
|Denoising|Global \(R^2\)|0.363|0.157|0.382|**0.483**|
|Coupling|Peak gAUC|0.884|0.663|0.941|**0.990**|
|Secret recovery|\(W=10\), accuracy|0.10|0.09|0.12|**0.96**|

#### 2. The scale of improvement exceeds TFA and is comparable to TSAEs

In the real world setting, we agree that backtracking is the only evaluated task where the TXC clearly beats *all* other architectures. We emphasize, however, that assesing significance requires a scale. Previous work has used the natural comparison to conventional SAEs (Cite). The headline comparison relative to SAE performance is:

|Task|Metric|TFA|T-SAE|TXC|
|:---|:---|---:|---:|---:|
|Sparse probing|AUC-deficit reduction|-108%|**11%**|**11%**|
|Backtracking|Peak \(\Delta gc\)|86%|40%|**135%**|
|Medical EM|PR-AUC at \(S=16\)|100%|**121%**|92%|
|RLHF|Preference ROC-AUC at \(k=20\)|—|98%|102%|

The TXC improves over the SAE marginally in sparse probing, substantially in backtracking, and ties in the RLHF task. Emergent misalignment is the only context in which conventional SAEs outperform TXCs, and even this is window dependent (see above). Similarly, EM is the only task on which the TXC underperforms TFA. We agree that a broader panel of explicitly temporal behavioural evaluations would help to make this comparison clearer, and we regard this as a promising area for further work.

#### 3. Matched benchmarking has previously been missing

We finally note that previous temporal architectures were introduced and evaluated separately. To our knowledge, this is the first work to evaluate T-SAE, TFA, MLC, and TXC on a common panel under matched conditions and is hence useful to the community. The finding that these architectures are complementary, rather than substitutes, is itself valuable.

### Other points

1. Originality:

> Originality - Minor: The submission shares a similar title (Crosscoding Through Time) with Bayazit, Mueller, Bosselut. Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining. Arxiv 2025 / ACL 2026.

We agree it is confusing that both titles share the same first three words. They address fundamentally different questions, however. The reference work considers conventional crosscoders at different points in training, we consider an alternative SAE architecture. We do not consider this to bear on the originality of the present work. To avoid confusion, we will update our title to: "Temporal crosscoders: Sparse Feature Discovery Across Sequence Positions".

1. TXC definition:

> The main script does not include the TXC and TXC-Pro definitions or a reference point to page 12, Appendix A, making it hard to follow.

We promote the definitions in Appendix A to the main text.

1. Citations:

> TXC-pro uses Matryoshka [Learning > Multi-Level Features with Matryoshka Sparse Autoencoders Bhalla et al. ICLR 2026] The paper uses bad-medical-advice dataset (Line 243) but missing citation Model Organisms for Emergent Misalignment. Turner et al.

We have added the relevant citations, thank you.
