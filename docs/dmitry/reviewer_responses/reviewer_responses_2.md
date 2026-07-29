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

We thank the reviewer for recognizing the novelty of applying crosscoders along the sequence axis and the value of the synthetic and real-world benchmarks. We provide responses to the specific question about temporal vs. generic capacity below.

> Since the non-temporal MLC ties TXC on probing, can you isolate the temporal contribution from generic crosscoder capacity?

We provide a number of lines of evidence. First, we provide the explicit window size dependence in each task for the base TXC:

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

Longer context does not help static probing, but it materially helps Backtracking and Medical EM. We do not in general expect temporal structure to be helpful on all tasks, but we do claim both that temporal structure is relevant for some important applications of SAEs _and_ that TXCs can exploit them.

As an additional comparison, we provide explicitly the stacked SAE baseline, which aggregates the same temporal window without sharing feature weights across positions. We see that the stacked SAE underperforms the TXC in all but the EM task.

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
0.26^{\\dagger} &
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

### Parameter count and inference cost

We also report the capacity and dense inference cost of every architecture in
the headline configurations. Parameter counts are in billions. Inference cost
is in GFLOPs per architecture's native forward: one token for TopK SAE and
T-SAE, five positions for TFA, Stacked SAE, and TXC-base, five layers for MLC,
and ten positions for TXC-pro. We count one multiply-add as two FLOPs and
exclude selection, bias additions, nonlinearities, and training-only losses.

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

$^{*}$ Medical T-SAE entries give paper-width / matched-width values. For an
equal five-token segment, the per-token SAE and T-SAE costs should be
multiplied by five. A matched-width per-token SAE or T-SAE and a five-position
TXC therefore have the same leading dense-matmul cost over five reconstructed
positions, while TXC stores approximately five times as many parameters
because its encoder and decoder weights are position-specific. Sliding-window
evaluation adds one native forward per window.

This also sharpens the Stacked SAE control: Stacked SAE and $T=5$ TXC are
effectively capacity- and inference-matched (1.3B parameters and 2.7 GFLOPs
each on Backtracking), but Stacked SAE reaches only
$0.45\\times$ TXC's causal steering effect. The Backtracking gain therefore
cannot be explained by generic parameter count or dense inference compute
alone.
