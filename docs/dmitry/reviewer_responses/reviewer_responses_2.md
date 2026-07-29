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

$$
\\begin{array}{l|rrrr}
\\hline
\\text{Architecture} &
\\text{Sparse} &
\\text{Backtracking} &
\\text{Medical EM} &
\\text{HH-RLHF} \\\\
\\hline
\\text{TopK SAE / SAE-Arditi} &
0.085 & 0.27 & 0.23 & 0.085 \\\\
\\text{T-SAE} &
0.076 & 0.27 & 0.12/0.23^{*} & 0.085 \\\\
\\text{TFA }(T{=}5) &
0.73 & 2.3 & 2.3 & \\text{--} \\\\
\\text{MLC }(L{=}5) &
0.42 & 1.3 & \\text{--} & \\text{--} \\\\
\\text{Stacked SAE }(T{=}5) &
0.42 & 1.3 & 1.2 & 0.42 \\\\
\\text{TXC-base }(T{=}5) &
0.42 & 1.3 & 1.2 & 0.42 \\\\
\\text{TXC-base }(T{=}10/20) &
0.85/1.7 & \\text{--} & \\text{--} & \\text{--} \\\\
\\text{TXC-pro }(T_{\\max}{=}10) &
0.85 & 2.7 & 2.3 & \\text{--} \\\\
\\hline
\\end{array}
$$

**Dense inference cost (GFLOPs per native forward):**

$$
\\begin{array}{l|rrrr}
\\hline
\\text{Architecture and native input} &
\\text{Sparse} &
\\text{Backtracking} &
\\text{Medical EM} &
\\text{HH-RLHF} \\\\
\\hline
\\text{TopK SAE / SAE-Arditi, 1 token} &
0.17 & 0.54 & 0.47 & 0.17 \\\\
\\text{T-SAE, 1 token} &
0.15 & 0.54 & 0.24/0.47^{*} & 0.17 \\\\
\\text{TFA, 5 tokens} &
8.6 & 27 & 27 & \\text{--} \\\\
\\text{MLC, 5 layers} &
0.85 & 2.7 & \\text{--} & \\text{--} \\\\
\\text{Stacked SAE, 5 tokens} &
0.85 & 2.7 & 2.3 & 0.85 \\\\
\\text{TXC-base, 5 tokens} &
0.85 & 2.7 & 2.3 & 0.85 \\\\
\\text{TXC-base, 10/20 tokens} &
1.7/3.4 & \\text{--} & \\text{--} & \\text{--} \\\\
\\text{TXC-pro, 10 tokens} &
1.7 & 5.4 & 4.7 & \\text{--} \\\\
\\hline
\\end{array}
$$

$^{*}$ Medical T-SAE entries report paper-width / matched-width values. Costs
are per architecture's native forward. For an equal five-token segment, the
single-token SAE and T-SAE costs should therefore be multiplied by five.

We next compare the TXC with the Stacked SAE baseline directly. The TXC scores
higher on Sparse probing, both Backtracking metrics, and HH-RLHF; the Stacked
SAE scores higher on Medical EM detection, while its Medical EM steering
evaluation was not run.

**Stacked SAE control at T=5:**

$$
\\begin{array}{lrrrr}
&\\mathrm{Stacked}&\\mathrm{TXC}&\\mathrm{Ratio}&\\mathrm{Floor}\\\\
\\mathrm{Sparse}&.87&.89&.98&.80\\\\
\\mathrm{BT\ steer}&.25&.54&.45&-\\\\
\\mathrm{BT\ detect}&.16&.24&.65&-\\\\
\\mathrm{Medical\ steer}&\\mathrm{not\ run}&23&-&-\\\\
\\mathrm{Medical\ detect}&.65^\\dagger&.54&1.2^\\dagger&.34\\\\
\\mathrm{HH\!-\!RLHF}&.60&.61&.99&\\mathbf{.62}
\\end{array}
$$

$^{\dagger}$ The Medical EM detection comparison is not
sparsity-calibrated: realized evaluation $L_0$ exceeded its nominal target for
every architecture under train-to-rollout distribution shift. We therefore
treat this comparison as directional pending re-thresholding.

The matched-capacity Backtracking result isolates cross-position sharing from
generic parameter count: the Stacked SAE reaches only $0.45\times$ the TXC's
steering effect and $0.65\times$ its detection PR-AUC. This is the clearest
real-world evidence that the gain is not explained by capacity alone. We also
report the explicit dependence on window size:

**Window-size sweep** (entries after Headline are percentages of $T=5$):

$$
\\begin{array}{lrrrrrr}
&\\mathrm{Headline}&T1&T2&T4&T5&T6\\\\
\\mathrm{Sparse}&.89&101&101&101&100&100\\\\
\\mathrm{Backtrack}&.26&85&85&92&100&100\\\\
\\mathrm{Medical\ EM}&.54&83&92&90&100&115\\\\
\\mathrm{HH\!-\!RLHF}&.62&96&100&99&100&101
\\end{array}
$$

Longer context does not help static probing, but it materially helps
Backtracking and Medical EM. We do not expect temporal structure to help on
every task; rather, these results show both that temporal structure matters for
some important SAE applications *and* that TXCs can exploit it.

We also take the opportunity to answer 🟩 Reviewer bbby's request for an explicit accounting of capacity for the T-SAE:

### T-SAE dictionary-width control

Our description of dictionary sizes was ambiguous. For each task, we ran the
T-SAE at the width used for the paper result and, where different, at the width
matched to the other architectures. Where these disagreed, we selected the
better-performing variant to ensure that we compared against the strongest
baseline. We have clarified this and added both settings to the paper
appendix:

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
0.26^{\\ddagger} &
0.25\\ (d{=}32{,}768) &
0.25\\ (d{=}32{,}768) \\\\
\\hline
\\text{Medical EM} &
\\text{detection PR-AUC}@S{=}16 &
0.54 &
0.71\\ (d{=}16{,}384) &
0.43\\ (d{=}32{,}768) \\\\
\\hline
\\text{HH-RLHF} &
\\text{preference ROC-AUC}@k{=}20 &
0.62 &
0.60\\ (d{=}18{,}432) &
0.60\\ (d{=}18{,}432) \\\\
\\hline
\\end{array}
$$

$^{\ddagger}$ The Backtracking TXC value is the $T=5$ cell from the new
window-size sweep.
