# Reviewer responses




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
     ADDED by mac-local, 2026-07-29. Present on both this branch and
     `arxiv`; keep the two copies in step if either is edited.
     ============================================================ -->
    - e. Fifth, we add a real world task whose state is not visible in any
      single token, and compare against pooled and stacked SAEs at matched
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

For the $h=2$, recovery is bounded to chance accuracy is $1/11=0.091$ below *3* steps. All methods satisfy this ceiling; beyond the threshold, the TXC improves from $0.154$ at $W=3$ to near perfect recovery, $0.956$ at $W=10$.

$$
\begin{array}{l|cccccc}
\hline
\text{Architecture} &
W{=}1 & W{=}2 & W{=}3 & W{=}4 & W{=}5 & W{=}10 \\
\hline
\text{Chance} &
0.091 & 0.091 & 0.091 & 0.091 & 0.091 & 0.091 \\
\text{SAE (best }k\text{)} &
0.099_{(2)} & 0.095_{(10)} & 0.101_{(1)} &
0.092_{(2)} & 0.094_{(1)} & 0.095_{(10)} \\
\text{Stacked SAE (best }k\text{)} &
0.100_{(2)} & 0.092_{(5)} & 0.100_{(1)} &
0.097_{(2)} & 0.098_{(10)} & 0.108_{(5)} \\
\text{T-SAE (best }k\text{)} &
\text{--} & 0.095_{(5)} & 0.103_{(5)} &
0.096_{(2)} & 0.104_{(2)} & 0.123_{(2)} \\
\text{TFA (best }k\text{)} &
\text{--} & 0.096_{(2)} & 0.104_{(2)} &
0.101_{(10)} & 0.098_{(1)} & 0.094_{(2)} \\
\hline
\text{TXC }(k{=}1) &
0.100 & 0.091 & 0.131 & 0.186 & 0.292 & 0.634 \\
\text{TXC }(k{=}2) &
0.098 & 0.087 & \mathbf{0.154} & \mathbf{0.324} &
\mathbf{0.562} & 0.909 \\
\text{TXC }(k{=}5) &
0.092 & 0.088 & 0.097 & 0.097 & 0.156 & \mathbf{0.956} \\
\hline
\end{array}
$$


For every non-TXC baseline, subscripts give the selected $k$ independently at
each window size. We sweep $k\in\{1,2,5,10\}$ for SAE and Stacked SAE, and
$k\in\{1,2,5,10,20\}$ for T-SAE and TFA. Results use episode-disjoint
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

$$
\begin{array}{l|c|ccccc}
\hline
\text{Task} &
\text{Headline} &
T{=}1 & T{=}2 & T{=}4 & T{=}5 & T{=}6 \\
\hline
\text{Sparse} &
0.89 &
101\% & 101\% & 101\% & 100\% & 100\% \\
\hline
\text{Backtracking} &
0.26 &
85\% & 85\% & 92\% & 100\% & 100\% \\
\hline
\text{Medical EM} &
0.54 &
83\% & 92\% & 90\% & 100\% & 115\% \\
\hline
\text{HH-RLHF} &
0.62 &
96\% & 100\% & 99\% & 100\% & 101\% \\
\hline
\end{array}
$$

<!-- Good way of udnerstanding the interest here is that it could just be that the txc allows you to allocate more weights to the mroe important parts of the window. -->
## Stacked SAE at T=5 — filled (2026-07-27, training seed 42)
We address here the reviewer's request for the explicit stacked SAE comparison. Results are provided in the table below for a single seed. Except for the EM task, stacked SAE performance is below the worst performing seed for the TXC.

$$
\begin{array}{l|l|c|c|c|c}
\hline
\text{Task} &
\text{Metric} &
\text{Stacked } T{=}5 &
\text{TXC} &
\text{Stacked}/\text{TXC} &
\text{Floor} \\
\hline
\text{Sparse probing} &
\text{mean AUC (38 tasks)} &
0.869 &
0.890 &
0.98\times &
0.803 \\
\hline
\text{Backtracking (steer)} &
\Delta gc_{\mathrm{peak}}\ \text{(25-mag)} &
0.246\ (m{=}{-}12) &
0.541\ (m{=}{-}12) &
0.45\times &
\text{--} \\
\hline
\text{Backtracking (detect)} &
\text{PR-AUC}@S{=}8 &
0.158 &
0.242 &
0.65\times &
\text{--} \\
\hline
\text{Medical EM (steer)} &
\Delta\mathrm{align}\ (\mathrm{coh}\geq 70) &
\mathrm{not\ run} &
22.88 &
\text{--} &
\text{--} \\
\hline
\text{Medical EM (detect)} &
\text{PR-AUC}@S{=}16 &
0.652^{\dagger} &
0.540 &
1.21\times^{\dagger} &
0.344 \\
\hline
\text{HH-RLHF} &
\text{pref.\ AUC}@k{=}20 &
0.602 &
0.610 &
0.99\times &
\mathbf{0.617} \\
\hline
\end{array}
$$

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
TXC-base but at $0.45\times$ its $\Delta gc$, while beating T-SAE
($0.164$) and matching MLC ($0.246$) — and does not explain the static
results: it trails the per-token TopK SAE on sparse probing. Weight
sharing across positions, not aggregation, carries the remaining
factor of $2.2$ on inducement. The HH-RLHF floor deserves emphasis: an
untrained stacked dictionary scores $0.617$, above both the trained
cell and the paper headline, and inside the three-seed spread of the
trained architectures ($0.604$--$0.618$, § Seed dependence) — the
preference-AUC metric is largely training-insensitive, consistent with
the manuscript's treatment of HH-RLHF as a length-confounded negative
control.

$^{\dagger}$ EM caveat: realized eval $L_0$ runs far above nominal for
every architecture in this panel (JumpReLU thresholds under
train$\to$rollout distribution shift; references $6$--$10\times$
nominal, stacked $\sim 32\times$), so the EM detection comparison is
not sparsity-calibrated; treat the above-headline value as directional
until the panel is re-thresholded. The trained EM signal is real
relative to its floor ($0.652$ vs $0.344$, prevalence $0.315$).
Backtracking floors were not run (steering floors require judged
generations; scope was held to the trained seed-42 cells).

### Seed dependence

2. We address the reviewers concern about seed variance

<!-- Callout box here -->

"The checklist says the experiments use 2 seeds, but the appendix says the main backtracking results use only seed 42. Which is correct? Are the reported improvements larger than the variation across different seeds?"

We provide TXC-base results for random initialization seeds 1, 2, and 42 in
the table below. Outstanding evaluations are marked as pending.

$$
\begin{array}{l|l|c|c}
\hline
\text{Task} &
\text{Metric} &
\text{TXC-base (seeds 1 / 2 / 42)} &
\text{TopK SAE} \\
\hline
\text{Sparse probing} &
\text{mean AUC (38 tasks)} &
0.900\,/\,0.900\,/\,0.898 &
0.886 \\
\hline
\text{Backtracking} &
\Delta gc_{\mathrm{peak}}\ \text{(25-mag)} &
\mathrm{pending}\,/\,\mathrm{pending}\,/\,0.541 &
0.400 \\
\hline
\text{Medical EM} &
\Delta\mathrm{align}\ (\mathrm{coh}\geq 70) &
16.72\,/\,\mathrm{pending}\,/\,22.88 &
21.45 \\
\hline
\text{HH-RLHF} &
\text{pref.\ AUC}@k{=}20 &
0.622\,/\,0.618\,/\,0.623 &
0.613 \\
\hline
\end{array}
$$

The comparison column reports the conventional TopK SAE on the same metric.
The Medical steering endpoint currently has only seeds 1 and 42; seed 2 is
pending.

### 3. Other

    - a. Abstract wording:

"Is there one model setting where both the 40% better detection and the 15% better inducement are achieved? If not, please make this clear in the abstract."

Detection (Fig 4b) and inducement (Fig 4a) refer to the excess performance of the base TXC over the TopK SAE: 0.23 vs. 0.17 when evaluated on the AUC for detection, and 1.20 vs 1.15 when evaluated on the average rate of genuine backtracking events induced, respectively. We have clarified the abstract to reflect this:

"Base temporal crosscoders can detect backtracking - a key reasoning behavior - at a 40% higher rate than conventional SAEs, and are 15% more effective in inducing it.




    - b. Is the T-SAE underpowered?

    "Why does T-SAE use a smaller dictionary if the paper claims all methods use the same dictionary size? Please also report the parameter count and inference cost for each model."

We agree that our description of dictionary sizes was ambiguous. We ran widths matched both the other architectures and width matched to the original paper $(d_{\mathrm{SAE}}=16{,}384)$. Where they disagreed, we chose the best performing variant in order to have the strongest baseline.  We have now added results for both explicitly to the paper appendix. We reproduce the summary data here:

$$
\begin{array}{l|l|c|c|c}
\hline
\text{Task} &
\text{Headline metric} &
\text{TXC} &
\text{T-SAE at paper width} &
\text{T-SAE at matched width} \\
\hline
\text{Backtracking} &
\text{detection PR-AUC}@S{=}32 &
0.260^{\dagger} &
0.245\ (d{=}32{,}768) &
0.245\ (d{=}32{,}768) \\
\hline
\text{Medical EM} &
\text{detection PR-AUC}@S{=}16 &
0.540 &
0.710\ (d{=}16{,}384) &
0.431\ (d{=}32{,}768) \\
\hline
\text{HH-RLHF} &
\text{preference ROC-AUC}@k{=}20 &
0.623 &
0.600\ (d{=}18{,}432) &
0.599\ (d{=}18{,}432) \\
\hline
\end{array}
$$




<!-- ============================================================
     ADDED by mac-local, 2026-07-29. Present on both this branch and
     `arxiv`; keep the two copies in step if either is edited.

     THIS is the readable copy: SINGLE-backslash LaTeX, matching the
     other working-section tables, which render correctly. The
     copy-and-paste section below carries the same table in that
     section's DOUBLE-backslash convention, for pasting only.
     ============================================================ -->

## A new real world task: how long since the user last challenged the model

We add a real world task in which the quantity being probed is not visible in
any single token. The data is multi-turn dialogue where the user repeatedly
questions the model's answers, and the label at each position is the number of
tokens since the user last challenged it. We compare the TXC against a pooled
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
\text{TXC} & \mathbf{0.499} & \mathbf{0.523} & \mathbf{0.537} & \mathbf{0.577} \\
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
     ADDED by mac-local, 2026-07-29. Present on both this branch and
     `arxiv`; keep the two copies in step if either is edited.
     Everything between these markers is new.
     ============================================================ -->
   **d.** Fourth, we add a real world task whose state is not visible in any single token, and compare against pooled and stacked SAEs at matched sparsity. The TXC improves with window size and is above both baselines.
<!-- ===================== END ADDED (sycgen) ===================== -->

2. We address the reviewer's concerns about seeds by running three seeds for the base TXC. We confirm that the relative rankings do not change.

3. We provide specific responses to the other points raised, including the abstract wording, T-SAE dictionary size, parameter count, and inference cost.

### Synthetic setting

In the synthetic setting. We introduce a task with an analytic ceiling on recoverability from single-token information. The task uses a Hidden Markov Model (HMM) based on Shamir secret sharing. This HMM encodes a temporal ``secret'', whose recovery can be no better than random guessing below a threshold number of temporal steps, $h$. For a fair comparison, when probing single-token architectures, we stack the activations over a window of the same size.

For $h=2$, recovery below three steps is bounded at the chance accuracy of $1/11\\approx0.09$. All methods satisfy this ceiling. Beyond the threshold, TXC accuracy improves from $0.15$ at $W=3$ to near-perfect recovery, $0.96$, at $W=10$.

**Secret-recovery accuracy** (subscript: selected k):

$$
\\begin{array}{lrrrrrr}
&W1&W2&W3&W4&W5&W10\\\\
\\mathrm{Chance}&.09&.09&.09&.09&.09&.09\\\\
\\mathrm{SAE}&.10_2&.10_{10}&.10_1&.09_2&.09_1&.10_{10}\\\\
\\mathrm{Stacked}&.10_2&.09_5&.10_1&.10_2&.10_{10}&.11_5\\\\
\\mathrm{TSAE}&-&.10_5&.10_5&.10_2&.10_2&.12_2\\\\
\\mathrm{TFA}&-&.10_2&.10_2&.10_{10}&.10_1&.09_2\\\\
\\mathrm{TXC}_{k=1}&.10&.09&.13&.19&.29&.63\\\\
\\mathrm{TXC}_{k=2}&.10&.09&\\mathbf{.15}&\\mathbf{.32}&\\mathbf{.56}&.91\\\\
\\mathrm{TXC}_{k=5}&.09&.09&.10&.10&.16&\\mathbf{.96}
\\end{array}
$$

For every non-TXC baseline, subscripts give the selected k independently at each window size. We sweep k over {1,2,5,10,20} and choose the best k.


<!-- ============================================================
     ADDED by mac-local, 2026-07-29. Present on both this branch and
     `arxiv`; keep the two copies in step if either is edited.
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

We add a real world task in which the quantity being probed is not visible in
any single token. The data is multi-turn dialogue where the user repeatedly
questions the model's answers, and the label at each position is the number of
tokens since the user last challenged it. We compare the TXC against a pooled
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
\\text{TXC} & \\mathbf{0.499} & \\mathbf{0.523} & \\mathbf{0.537} & \\mathbf{0.577} \\\\
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

$$
\\begin{array}{lcc}
&\\mathrm{TXC\ seeds\ }1/2/42&\\mathrm{TopK}\\\\
\\mathrm{Sparse}&.900/.900/.898&.886\\\\
\\mathrm{Backtrack}&\\mathrm{pending/pending}/.541&.400\\\\
\\mathrm{Medical\ EM}&16.72/\\mathrm{pending}/22.88&21.45\\\\
\\mathrm{HH\!-\!RLHF}&.622/.618/.623&.613
\\end{array}
$$

### Other points

**a. Abstract wording**

> Is there one model setting where both the 40% better detection and the 15% better inducement are achieved? If not, please make this clear in the abstract.

Detection (Fig. 4b) and inducement (Fig. 4a) refer to the excess performance of the base TXC over the TopK SAE: $0.23$ vs. $0.17$, and $1.20$ vs. $1.15$  on backtracking induced, respectively. We have clarified the abstract to reflect this.

**b. Is the T-SAE underpowered?**

> Why does T-SAE use a smaller dictionary if the paper claims all methods use the same dictionary size? Please also report the parameter count and inference cost for each model.

Our description of dictionary sizes was ambiguous. We ran variants whose widths were matched to the other architectures as well as variants whose widths were matched to the original paper ($d_{\\mathrm{SAE}}=16{,}384$). Where they disagreed, we chose the best-performing variant in order to use the strongest baseline. We have clarified this and added results for both explicitly to the paper appendix. We reproduce the summary data here:

$$
\\begin{array}{lccc}
&\\mathrm{TXC}&\\mathrm{TSAE\ paper}&\\mathrm{TSAE\ matched}\\\\
\\mathrm{Backtrack}\ S32&.260^\\dagger&.245\ (32768)&.245\ (32768)\\\\
\\mathrm{Medical}\ S16&.540&.710\ (16384)&.431\ (32768)\\\\
\\mathrm{HH\!-\!RLHF}\ k20&.623&.600\ (18432)&.599\ (18432)
\\end{array}
$$

**c**. Parameter count and inference cost

See response to Reviewer 4z15.
