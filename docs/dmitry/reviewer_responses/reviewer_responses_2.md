---
author: Dmitry
date: 2026-07-28
tags:
  - in-progress
  - results
---

<!-- Thank the reviewer for appreciating the benchmarking. -->

## Reviewer 2 response

## OpenReview copy-and-paste version

We thank the reviewer for recognizing the novelty of applying crosscoders along the sequence axis and the value of the synthetic and real-world benchmarks.

> Since the non-temporal MLC ties TXC on probing, can you isolate the temporal contribution from generic crosscoder capacity?

We agree that sparse probing alone does not isolate a temporal advantage: MLC ties TXC, and TXC performance is nearly invariant to window size on this static task. We have narrowed our claim accordingly. Two new controls separate temporal context from generic crosscoder capacity on the explicitly temporal tasks.

**Window-size sweep** (entries are percentages of T=5):

$$
\\begin{array}{lrrrrrr}
&\\mathrm{Headline}&T1&T2&T4&T5&T6\\\\
\\mathrm{Sparse}&.89&101&101&101&100&100\\\\
\\mathrm{Backtrack}&.26&85&85&92&100&100\\\\
\\mathrm{Medical\ EM}&.54&83&92&90&100&115\\\\
\\mathrm{HH\!-\!RLHF}&.62&96&100&99&100&101
\\end{array}
$$

Longer context does not help static probing, but it materially helps Backtracking and Medical EM.

**Stacked SAE control at T=5:**

$$
\\begin{array}{lrrrr}
&\\mathrm{Stacked}&\\mathrm{TXC}&\\mathrm{Ratio}&\\mathrm{Floor}\\\\
\\mathrm{Sparse}&.869&.890&.98&.803\\\\
\\mathrm{BT\ steer}&.246&.541&.45&-\\\\
\\mathrm{BT\ detect}&.158&.242&.65&-\\\\
\\mathrm{Medical\ steer}&\\mathrm{not\ run}&22.88&-&-\\\\
\\mathrm{Medical\ detect}&.652^\\dagger&.540&1.21^\\dagger&.344\\\\
\\mathrm{HH\!-\!RLHF}&.602&.610&.99&\\mathbf{.617}
\\end{array}
$$

Backtracking steering values peak at m=-12; detection uses PR-AUC at S=8, Medical detection uses PR-AUC at S=16, and HH-RLHF uses preference AUC at k=20.

This control aggregates the same temporal window without sharing feature weights across positions. On causal Backtracking steering, it beats T-SAE (.164) and matches MLC (.246), showing that temporal aggregation helps, but reaches only .45x TXC's effect. Cross-position weight sharing supplies the remaining 2.2x gain. Conversely, Stacked SAE trails the per-token TopK SAE on static probing, so generic aggregation cannot explain that result. HH-RLHF remains a negative control: its random-dictionary floor (.617) exceeds both trained Stacked SAE (.602) and TXC (.610), indicating that this length-confounded metric is largely training-insensitive.

† Medical EM detection is not sparsity-calibrated: under train-to-rollout distribution shift, realized evaluation $L_0$ is 6--10x nominal for the reference architectures and approximately 32x nominal for Stacked SAE. The .652 result should therefore be treated as directional pending re-thresholding, although it exceeds the Stacked random-dictionary floor of .344. The Stacked Medical steering cell requires approximately 14.8K judged generations and was scoped to follow-up work.

### T-SAE dictionary-width control

Our description of dictionary sizes was ambiguous. We ran T-SAE variants whose widths were matched to the other architectures, as well as variants using the original paper width ($d_{\\mathrm{SAE}}=16{,}384$). Where these disagreed, we selected the better-performing variant to ensure that we compared against the strongest baseline. We have clarified this and added results for both settings to the paper appendix:

$$
\\begin{array}{l|l|c|c|c}
\\hline
\\text{Task} &
\\text{Headline metric} &
\\text{TXC} &
\\text{T-SAE at paper width} &
\\text{T-SAE at matched width} \\\\
\\hline
\\text{Backtracking} &
\\text{detection PR-AUC}@S{=}32 &
0.260^{\\dagger} &
0.245\\ (d{=}32{,}768) &
0.245\\ (d{=}32{,}768) \\\\
\\hline
\\text{Medical EM} &
\\text{detection PR-AUC}@S{=}16 &
0.540 &
0.710\\ (d{=}16{,}384) &
0.431\\ (d{=}32{,}768) \\\\
\\hline
\\text{HH-RLHF} &
\\text{preference ROC-AUC}@k{=}20 &
0.623 &
0.600\\ (d{=}18{,}432) &
0.599\\ (d{=}18{,}432) \\\\
\\hline
\\end{array}
$$

Here, *matched width* means matching the TXC's number of dictionary features, not its parameter count. The submitted Backtracking and HH-RLHF T-SAE checkpoints were already width-matched; the apparent mismatch came from presenting the global default without the task-specific overrides. $^{\\dagger}$ The interim Backtracking TXC value is Aniket's retained 20K-step seed-42 result, which we will replace with the completed 300K-step comparison. The completed matched-width Medical EM and HH-RLHF entries are single-seed (seed 42) reruns.
