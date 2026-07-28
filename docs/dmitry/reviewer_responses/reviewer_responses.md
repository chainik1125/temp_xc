# Reviewer responses




## Reviewer 1 response

<!-- TODO:  Extend a little bit to highlight the strengths of the paper -->


We thank the reviewer for recognizing the novelty of our proposal and for noting the synthetic benchmarking procedure we introduce. 

We first provide a high level summary of our responses.

### Summary of response
<!-- TODO: maybe quote the reviewer -->
1. The reviewer points out the the importance of establishing that the temporal features allowed by the temporal crosscoder are responsible for the performance improvements shown. We provide four additional lines of evidence to show this.

<!-- TODO: this should be less what you did and more what the result is -->
<!-- TODO: maybe you can just  -->

<!-- Maybe can be a bit more concise here since this is not really crucial -->
    - a. First, we introduce an additional task in the synthetic setting which is provably impossible without temporal feature sharing. The TXC architecture strongly outperforms all other architectures on this task.

    - b. Second, we identify additional real world tasks which have more explicit temporal structure. The temporal crosscoder outperforms alternative architectures in these tasks.
    <!-- , and becomes uncompetitive when we introduce a shuffle control to eliminate temporal information. -->

    - c. Third, we provide the data for the scaling of TXC performance with window size for all real world tasks considered in the paper.

    - d. Fourth, we provide Stacked SAE results for all real world tasks. Stacked SAEs underperform TXCs for all tasks considered except Emergent Misalignment steering.

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
size. Every numerical headline and $T=5$ entry uses the seed-42 paper TXC
checkpoint (never TXC-pro), so each numerical $T=5$ entry is exactly $100\%$
by construction. Cells marked ``run'' are exact-paper-recipe reruns in
progress; we exclude earlier auxiliary and TXC-pro checkpoints. Window-size
entries are reported as a percentage of the corresponding seed-42 paper
headline and rounded to the nearest integer.

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
0.25 &
84\% & \mathrm{run} & \mathrm{run} & 100\% & \mathrm{run} \\
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
\begin{array}{l|c|c}
\hline
\text{Task} &
\text{Chance } B &
\text{Stacked } T{=}5 \\
\hline
\text{Sparse} &
0.50 &
95\% \\
\hline
\text{Backtracking} &
0.13 &
26\% \\
\hline
\text{Medical EM} &
0.31 &
150 \\
\hline
\text{HH-RLHF} &
0.50 &
93\% \\
\hline
\end{array}
$$



### Seed dependence

2. We address the reviewers concern about seed variance

<!-- Callout box here -->

"The checklist says the experiments use 2 seeds, but the appendix says the main backtracking results use only seed 42. Which is correct? Are the reported improvements larger than the variation across different seeds?"

We provide TXC results for three random initialization seeds in the table below.


### 3. Other

    - a. Abstract wording:

"Is there one model setting where both the 40% better detection and the 15% better inducement are achieved? If not, please make this clear in the abstract."

Detection (Fig 4b) and inducement (Fig 4a) refer to the excess performance of the base TXC over the TopK SAE: 0.23 vs. 0.17 when evaluated on the AUC for detection, and 1.20 vs 1.15 when evaluated on the average rate of genuine backtracking events induced, respectively. We have clarified the abstract to reflect this:

"Base temporal crosscoders can detect backtracking - a key reasoning behavior - at a 40% higher rate than conventional SAEs, and are 15% more effective in inducing it.




    - b. Is the T-SAE underpowered?

    "Why does T-SAE use a smaller dictionary if the paper claims all methods use the same dictionary size? Please also report the parameter count and inference cost for each model."

    <!--  -->
