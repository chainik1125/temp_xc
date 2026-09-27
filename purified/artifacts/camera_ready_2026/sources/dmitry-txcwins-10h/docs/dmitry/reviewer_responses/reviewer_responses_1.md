---
author: Dmitry
date: 2026-07-28
tags:
  - in-progress
  - results
---

## Reviewer 1 response

<!-- TODO:  Extend a little bit to highlight the strengths of the paper -->


We thank the reviewer for recognizing the novelty of our proposal and for noting the synthetic benchmarking procedure we introduce.

We first provide a high level summary of our responses.

### Summary of response
<!-- TODO: maybe quote the reviewer -->
1. The reviewer points out the the importance of establishing that the temporal features allowed by the temporal crosscoder are responsible for the performance improvements shown. We provide five additional lines of evidence to show this.

<!-- TODO: this should be less what you did and more what the result is -->
<!-- TODO: maybe you can just  -->

<!-- Maybe can be a bit more concise here since this is not really crucial -->
    - a. First, we introduce an additional task in the synthetic setting which is provably impossible without temporal feature sharing. The TXC architecture strongly outperforms all other architectures on this task.

    - b. Second, we identify additional real world tasks which have more explicit temporal structure. The temporal crosscoder outperforms alternative architectures in these tasks.
    <!-- , and becomes uncompetitive when we introduce a shuffle control to eliminate temporal information. -->

    - c. Third, we provide the data for the scaling of TXC performance with window size for all real world tasks considered in the paper.

    - d. Fourth, we provide Stacked SAE results for all real world tasks. Stacked SAEs underperform TXCs for all tasks considered except Emergent Misalignment steering.

<!-- ============================================================
     ADDED ON THE `arxiv` BRANCH (mac-local, 2026-07-29): sycgen.
     Source branch `dmitry-txcwins-10h` is NOT modified.
     ============================================================ -->
    - e. Fifth, we add a real world task whose label no single token of text
      displays, and compare against pooled and stacked SAEs at matched
      sparsity over three seeds. The TXC improves with window size and is
      above both baselines at every window size.
<!-- ===================== END ADDED (sycgen) ===================== -->

<!-- Summary par -->


<!-- Would be good to have a less annoying way of saying 'other people don't do this' -->
2. We address the reviewers concerns about seeds by running three seeds for every headline experiment and confirming that the relative rankings do not change.
<!-- We also note that seed-averages are not typically reported in the SAE benchmarking literature. -->

3. We provide specific responses to the other points raised (abstract wording, T-SAE dictionary size, parameter count and inference cost )



### Synthetic setting

<!--
Our task adapts the finite-field polynomial construction underlying
Reed--Solomon codes (Reed and Solomon, 1960), in the threshold form closely
related to Shamir secret sharing (Shamir, 1979). This gives exactly the
guarantee we need: any $h$ observations contain no information about the
label, while $h+1$ observations suffice to recover it. Although Reed--Solomon
codes have been used as targets for neural decoders (Zhang et al., 2020), and
synthetic or formal-language tasks have been used to evaluate SAEs (Menon et
al., 2025; Chanin and Garriga-Alonso, 2026), to our knowledge this is the first
use of a polynomial threshold construction to benchmark temporal sparse
dictionary learning.
-->

In the synthetic setting, we prove that temporal information is responsible for performance. We do this by introducing a task with an analytic ceiling on recoverability from single token information. The task uses a Hidden Markov Model (HMM) based on Shamir secret sharing. This HMM has the property that secret recovery can be no better than random guessing below a threshold number of temporal steps $h$. The resulting performance for probing the secret on a given window size (for single-token architectures, we stack the activations in a same size window for a fair comparison).

For the $h=2$, recovery is bounded to chance accuracy is $1/11=0.091$ below *3* steps. All methods satisfy this ceiling; beyond the threshold, the TXC improves from $0.15$ at $W=3$ to near perfect recovery, $0.96$ at $W=10$.

**Secret-recovery accuracy:**

| Architecture | W = 1 | W = 2 | W = 3 | W = 4 | W = 5 | W = 10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chance | 0.091 | 0.091 | 0.091 | 0.091 | 0.091 | 0.091 |
| SAE, best k | 0.099 | 0.095 | 0.10 | 0.092 | 0.094 | 0.095 |
| Stacked SAE, best k | 0.10 | 0.092 | 0.10 | 0.097 | 0.098 | 0.11 |
| T-SAE, best k | — | 0.095 | 0.10 | 0.096 | 0.10 | 0.12 |
| TFA, best k | — | 0.096 | 0.10 | 0.10 | 0.098 | 0.094 |
| TXC, k = 1 | 0.10 | 0.091 | 0.13 | 0.19 | 0.29 | 0.63 |
| TXC, k = 2 | 0.098 | 0.087 | **0.15** | **0.32** | **0.56** | 0.91 |
| TXC, k = 5 | 0.092 | 0.088 | 0.097 | 0.097 | 0.16 | **0.96** |

For every non-TXC baseline, we select the best k independently at each window
size. We sweep k ∈ {1, 2, 5, 10} for SAE and Stacked SAE, and k ∈ {1, 2, 5,
10, 20} for T-SAE and TFA. Results use episode-disjoint
representation-training, probe-training, and validation sets, with one
evaluation window per episode.


### New tasks

We also show that the use of temporal information continues to hold in an additional, explicitly temporal, real world language task. The task we consider is system prompt following \cite{StruQ}. <short description of task>

<results on task>


### Window size experiments

We also report the temporal crosscoder's performance as a function of window
size. Sparse probing, Medical EM, and HH-RLHF use the seed-42 paper TXC
checkpoint (never TXC-pro). For Backtracking, we report Aniket's completed
20K-step seed-42 TXC-base detection sweep at $S=32$, so all five window sizes
come from one internally matched run rather than mixing 20K and 300K
checkpoints. Window-size entries are reported as a percentage of the
corresponding $T=5$ value and rounded to the nearest integer.

| Task | Headline score | T = 1 | T = 2 | T = 4 | T = 5 | T = 6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse probing | 0.89 | 101% | 101% | 101% | 100% | 100% |
| Backtracking | 0.26 | 85% | 85% | 92% | 100% | 100% |
| Medical EM | 0.54 | 83% | 92% | 90% | 100% | 115% |
| HH-RLHF | 0.62 | 96% | 100% | 99% | 100% | 101% |

<!-- Good way of udnerstanding the interest here is that it could just be that the txc allows you to allocate more weights to the mroe important parts of the window. -->
## Stacked SAE at T = 5 — filled (2026-07-27, training seed 42)
We address here the reviewer's request for the explicit stacked SAE comparison. Results are provided in the table below for a single seed. Except for the EM task, stacked SAE performance is below the worst performing seed for the TXC.

| Task | Metric | Stacked SAE, T = 5 | TXC | Stacked / TXC | Floor |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Sparse probing | Mean AUC (38 tasks) | 0.87 | 0.89 | 0.98× | 0.80 |
| Backtracking steering | Peak Δgc at magnitude 25 | 0.25 (m = −12) | 0.54 (m = −12) | 0.45× | — |
| Backtracking detection | PR-AUC at S = 8 | 0.16 | 0.24 | 0.65× | — |
| Medical EM steering | Δalignment at coherence ≥ 70 | Not run | 23 | — | — |
| Medical EM detection | PR-AUC at S = 16 | 0.65† | 0.54 | 1.2×† | 0.34 |
| HH-RLHF | Preference AUC at k = 20 | 0.60 | 0.61 | 0.99× | **0.62** |

Floor = the same architecture with an untrained (randomly initialized)
dictionary, evaluated identically. The TXC column uses the variant
holding each printed headline (TXC-base for steering, TXC-pro for
backtracking detection; EM steering value is the seed-42 headline from
the seed-dependence table below). The Medical EM steering cell for the
stacked baseline requires the full Wang stage-4 pipeline
($\sim$14.8K judged generations) and was scoped to follow-up work; the
detection row is the judge-free comparison available at matched
training scale. The stacked cells were trained at the published scale
for each task (Backtracking: $d_{\mathrm{SAE}}{=}32{,}768$,
$k_{\mathrm{pos}}{=}20$, $300$K steps, printed 25-magnitude grid,
Sonnet-4.6 judge; Sparse probing: locked 20K/bs1024 protocol; Medical
EM: 25K/bs1024, $k_{\mathrm{pos}}{=}25$; HH-RLHF:
$k_{\mathrm{win}}{=}500$ convention, realized $L_0$ $533$ vs $500$
nominal, $1/20$ length-spurious top features).

Reading. Temporal aggregation alone reaches mid-pack on the causal
benchmark — the stacked baseline peaks at the same magnitude as
TXC-base but at 0.45× its Δgc, while beating T-SAE
($0.16$) and matching MLC ($0.25$) — and does not explain the static
results: it trails the per-token TopK SAE on sparse probing. Weight
sharing across positions, not aggregation, carries the remaining
factor of $2.2$ on inducement. The HH-RLHF floor deserves emphasis: an
untrained stacked dictionary scores $0.62$, above both the trained
cell and the paper headline, and inside the three-seed spread of the
trained architectures (0.60–0.62, § Seed dependence) — the
preference-AUC metric is largely training-insensitive, consistent with
the manuscript's treatment of HH-RLHF as a length-confounded negative
control.

† EM caveat: realized evaluation L₀ runs far above nominal for every
architecture in this panel (JumpReLU thresholds under train-to-rollout
distribution shift; references 6–10× nominal, stacked approximately 32×), so
the EM detection comparison is
not sparsity-calibrated; treat the above-headline value as directional
until the panel is re-thresholded. The trained EM signal is real
relative to its floor ($0.65$ vs $0.34$, prevalence $0.32$).
Backtracking floors were not run (steering floors require judged
generations; scope was held to the trained seed-42 cells).

### Seed dependence

2. We address the reviewers concern about seed variance

<!-- Callout box here -->

"The checklist says the experiments use 2 seeds, but the appendix says the main backtracking results use only seed 42. Which is correct? Are the reported improvements larger than the variation across different seeds?"

We provide TXC-base results for random initialization seeds 1, 2, and 42 in
the table below. Outstanding evaluations are marked as pending.

| Task | Metric | TXC-base, seeds 1 / 2 / 42 | TopK SAE |
| :--- | :--- | ---: | ---: |
| Sparse probing | Mean AUC (38 tasks) | 0.90 / 0.90 / 0.90 | 0.89 |
| Backtracking | Peak Δgc at magnitude 25 | Pending / Pending / 0.54 | 0.40 |
| Medical EM | Δalignment at coherence ≥ 70 | 17 / 20 / 23 | 21 |
| HH-RLHF | Preference AUC at k = 20 | 0.62 / 0.62 / 0.62 | 0.61 |

The comparison column reports the conventional TopK SAE on the same metric.
The Medical steering endpoint uses the paper's full Wang protocol for all
three seeds.

### 3. Other

    - a. Abstract wording:

"Is there one model setting where both the 40% better detection and the 15% better inducement are achieved? If not, please make this clear in the abstract."

Detection (Fig 4b) and inducement (Fig 4a) refer to the excess performance of the base TXC over the TopK SAE: a 40% higher AUC for detection and a 15% higher average rate of genuine backtracking events induced, respectively. We have clarified the abstract to reflect this:

"Base temporal crosscoders can detect backtracking - a key reasoning behavior - at a 40% higher rate than conventional SAEs, and are 15% more effective in inducing it.




    - b. Is the T-SAE underpowered?

    "Why does T-SAE use a smaller dictionary if the paper claims all methods use the same dictionary size? Please also report the parameter count and inference cost for each model."

We agree that our description of dictionary sizes was ambiguous. For each
task, we ran the T-SAE at the width used for the paper result and, where
different, at the width matched to the other architectures. Where these
disagreed, we selected the better-performing variant to ensure that we
compared against the strongest baseline. We have clarified this and added both
settings to the paper appendix:

| Task | Metric | TXC | T-SAE, paper width | T-SAE, matched width |
| :--- | :--- | ---: | ---: | ---: |
| Backtracking | Detection PR-AUC at S = 32 | 0.26‡ | 0.25 (d = 32,768) | 0.25 (d = 32,768) |
| Medical EM | Detection PR-AUC at S = 16 | 0.54 | 0.71 (d = 16,384) | 0.43 (d = 32,768) |
| HH-RLHF | Preference ROC-AUC at k = 20 | 0.62 | 0.60 (d = 18,432) | 0.60 (d = 18,432) |

‡ The Backtracking TXC value is the T = 5 cell from the new window-size
sweep.




<!-- ============================================================
     ADDED ON THE `arxiv` BRANCH (mac-local, 2026-07-29): sycgen.
     Source branch `dmitry-txcwins-10h` is NOT modified.

     THIS is the readable copy: SINGLE-backslash LaTeX, matching the
     other working-section tables, which render correctly. The
     copy-and-paste section below carries the same table in that
     section's DOUBLE-backslash convention, for pasting only.
     ============================================================ -->

## A new real world task: how long since the user last challenged the model

We add a real world task whose label no single token of text displays. The data
is multi-turn dialogue where the user repeatedly questions the model's answers,
and the label at each position is the number of tokens since the user last
challenged it. A probe on one token's activation still recovers part of it, as
the residual stream has attended over the prefix, so the baselines here are not
blind: they read the same activations, and the question is whether the window
adds anything at matched sparsity. We compare the TXC against a pooled
SAE (the mean of the per-token codes across the window) and a stacked SAE (the
same codes concatenated), matched on sparsity: we sweep $k$ for both baselines
and read each one off at the sparsity the TXC actually uses. Three seeds;
Llama-3.1-8B, layer 14, $d_{\text{SAE}}=2048$. A per-token SAE reaches $0.482$.

$$
\begin{array}{l|cccc}
\hline
\text{Architecture} & T{=}2 & T{=}4 & T{=}8 & T{=}16 \\
\hline
\text{Pooled SAE} & 0.485 & 0.488 & 0.467 & 0.486^{*} \\
\text{Stacked SAE} & 0.468 & 0.412 & 0.149^{*} & 0.314^{*} \\
\hline
\text{TXC} & \mathbf{0.499} & \mathbf{0.523} & \mathbf{0.536} & \mathbf{0.577} \\
\hline
\text{TXC } L_0/\text{window} & 5.66 & 6.35 & 6.94 & 7.82 \\
\hline
\end{array}
$$

The TXC improves with window size, from $0.499$ at $T{=}2$ to $0.577$ at
$T{=}16$, and is above both baselines at every window size; at $T{=}2$ and
$T{=}4$ the margin over the pooled SAE is smaller than the variation across
seeds. Starred entries are baselines that cannot run as sparsely as the TXC,
so they are read off at a higher sparsity and the comparison favours them. The
stacked SAE's drop at $T\ge8$ comes from its input growing to $T\cdot d_\text{SAE}$, not from the architecture.

<!-- ===================== END ADDED (sycgen) ===================== -->

## OpenReview copy-and-paste version

We thank the reviewer for recognizing the novelty of our proposal and for noting the synthetic benchmarking procedure we introduce.

### Summary of response

1. The reviewer points out the importance of establishing that the temporal features enabled by the temporal crosscoder are responsible for the performance improvements shown. We provide three additional lines of evidence for this.

   **a.** First, we introduce an additional task in the synthetic setting that is provably impossible without temporal feature sharing. The TXC architecture strongly outperforms all other architectures on this task.

   **b.** Second, we provide window size sweeps for all tasks (in response to Reviewer 4z15).

   **c.** Third we provide the Stacked-SAE baseline explicitly (in response to Reviewer 4z15).

<!-- ============================================================
     ADDED ON THE `arxiv` BRANCH (mac-local, 2026-07-29): sycgen.
     Source branch `dmitry-txcwins-10h` is NOT modified.
     Everything between these markers is new.
     ============================================================ -->
   **d.** Fourth, we add a real world task whose label no single token of text displays, and compare against pooled and stacked SAEs at matched sparsity. The TXC improves with window size and is above both baselines.
<!-- ===================== END ADDED (sycgen) ===================== -->

2. We address the reviewer's concerns about seed variance by providing
explicit three-seed results. We confirm that the relative rankings do not
change.

3. We respond to the remaining points below, including the abstract wording,
T-SAE dictionary size, parameter count, and inference cost.

### Synthetic setting

To prove that the TXC uses temporal information, we introduce a synthetic task
with an analytic ceiling on recoverability from single-token information. The
task uses a Hidden Markov Model (HMM) based on Shamir secret sharing. This HMM
encodes a temporal “secret” whose recovery can be no better than random
guessing below a threshold number of temporal steps, h. For a fair comparison,
when probing single-token architectures, we stack the activations over a
window of the same size.

For h = 2, recovery below three steps is bounded at the chance accuracy of
1/11 ≈ 0.09. All methods satisfy this ceiling. Beyond the threshold, TXC
accuracy improves from 0.15 at W = 3 to near-perfect recovery, 0.96, at
W = 10.

**Secret-recovery accuracy:**

| Architecture | W = 1 | W = 2 | W = 3 | W = 4 | W = 5 | W = 10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chance | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 |
| SAE | 0.10 | 0.10 | 0.10 | 0.09 | 0.09 | 0.10 |
| Stacked SAE | 0.10 | 0.09 | 0.10 | 0.10 | 0.10 | 0.11 |
| T-SAE | — | 0.10 | 0.10 | 0.10 | 0.10 | 0.12 |
| TFA | — | 0.10 | 0.10 | 0.10 | 0.10 | 0.09 |
| TXC, k = 1 | 0.10 | 0.09 | 0.13 | 0.19 | 0.29 | 0.63 |
| TXC, k = 2 | 0.10 | 0.09 | **0.15** | **0.32** | **0.56** | 0.91 |
| TXC, k = 5 | 0.09 | 0.09 | 0.10 | 0.10 | 0.16 | **0.96** |

For every non-TXC baseline, we sweep k over {1, 2, 5, 10, 20} and report the
best accuracy independently at each window size.


<!-- ============================================================
     ADDED ON THE `arxiv` BRANCH (mac-local, 2026-07-29): sycgen.
     Source branch `dmitry-txcwins-10h` is NOT modified.
     Everything between these markers is new.

     LaTeX CONVENTION IN THIS SECTION: every backslash is DOUBLED --
     \\begin{array}, \\mathrm{...}, and rows end in \\\\ . The markdown
     renderer consumes one backslash before the math engine sees it, so a
     single \begin{array}{lcccc} loses its command and leaves stray braces,
     which reports as "Extra close brace or missing open brace" and the
     table does not render. Match the neighbouring tables, not raw LaTeX.
     Verified by rendering all 32 math blocks of this section with KaTeX
     after applying the markdown unescape: 0 failures.
     ============================================================ -->

### A new real world task: how long since the user last challenged the model

We add a real world task whose label no single token of text displays. The data
is multi-turn dialogue where the user repeatedly questions the model's answers,
and the label at each position is the number of tokens since the user last
challenged it. A probe on one token's activation still recovers part of it, as
the residual stream has attended over the prefix, so the baselines here are not
blind: they read the same activations, and the question is whether the window
adds anything at matched sparsity. We compare the TXC against a pooled
SAE (the mean of the per-token codes across the window) and a stacked SAE (the
same codes concatenated), matched on sparsity: we sweep $k$ for both baselines
and read each one off at the sparsity the TXC actually uses. Three seeds;
Llama-3.1-8B, layer 14, $d_{\\text{SAE}}=2048$. A per-token SAE reaches $0.482$.

$$
\\begin{array}{l|cccc}
\\hline
\\text{Architecture} & T{=}2 & T{=}4 & T{=}8 & T{=}16 \\\\
\\hline
\\text{Pooled SAE} & 0.485 & 0.488 & 0.467 & 0.486^{*} \\\\
\\text{Stacked SAE} & 0.468 & 0.412 & 0.149^{*} & 0.314^{*} \\\\
\\hline
\\text{TXC} & \\mathbf{0.499} & \\mathbf{0.523} & \\mathbf{0.536} & \\mathbf{0.577} \\\\
\\hline
\\text{TXC } L_0/\\text{window} & 5.66 & 6.35 & 6.94 & 7.82 \\\\
\\hline
\\end{array}
$$

The TXC improves with window size, from $0.499$ at $T{=}2$ to $0.577$ at
$T{=}16$, and is above both baselines at every window size; at $T{=}2$ and
$T{=}4$ the margin over the pooled SAE is smaller than the variation across
seeds. Starred entries are baselines that cannot run as sparsely as the TXC,
so they are read off at a higher sparsity and the comparison favours them. The
stacked SAE's drop at $T\\ge8$ comes from its input growing to $T\\cdot d_\\text{SAE}$, not from the architecture.

<!-- ===================== END ADDED (sycgen) ===================== -->

### Seed dependence

The reviewer asks:

> The checklist says the experiments use 2 seeds, but the appendix says the main backtracking results use only seed 42. Which is correct? Are the reported improvements larger than the variation across different seeds?

In the main text, we reported one training seed, consistent with landmark SAE-architecture papers introducing TopK (Gao et al., 2025, ICLR), Gated and JumpReLU (Rajamanoharan et al., 2024a,b), and BatchTopK (Bussmann et al., 2024). The appendix provided an additional seed; we have now updated it to include three-seed results, reproduced here:

| Task | TXC-base, seeds 1 / 2 / 42 | TopK SAE |
| :--- | ---: | ---: |
| Sparse probing | 0.90 / 0.90 / 0.90 | 0.89 |
| Backtracking | Pending / Pending / 0.54 | 0.40 |
| Medical EM | 17 / 20 / 23 | 21 |
| HH-RLHF | 0.62 / 0.62 / 0.62 | 0.61 |

### Other points

**a. Abstract wording**

> Is there one model setting where both the 40% better detection and the 15% better inducement are achieved? If not, please make this clear in the abstract.

Detection (Fig. 4b) and inducement (Fig. 4a) refer to the excess performance
of the base TXC over the TopK SAE: a 40% higher detection AUC and a 15% higher
average rate of backtracking induced, respectively. We have clarified the
abstract to reflect this.

**b. Is the T-SAE underpowered?**

> Why does T-SAE use a smaller dictionary if the paper claims all methods use the same dictionary size? Please also report the parameter count and inference cost for each model.

Our description of dictionary sizes was ambiguous. For each task, we ran the
T-SAE at the width used for the paper result and, where different, at the width
matched to the other architectures. Where these disagreed, we selected the
better-performing variant to ensure that we compared against the strongest
baseline. We have clarified this and added both settings to the paper
appendix:

| Task and metric | TXC | T-SAE, paper width | T-SAE, matched width |
| :--- | ---: | ---: | ---: |
| Backtracking, PR-AUC at S = 32 | 0.26‡ | 0.25 (d = 32,768) | 0.25 (d = 32,768) |
| Medical EM, PR-AUC at S = 16 | 0.54 | 0.71 (d = 16,384) | 0.43 (d = 32,768) |
| HH-RLHF, ROC-AUC at k = 20 | 0.62 | 0.60 (d = 18,432) | 0.60 (d = 18,432) |

‡ The Backtracking TXC value is the T = 5 cell from the new window-size
sweep.

**c**. Parameter count and inference cost

See response to Reviewer 4z15.
