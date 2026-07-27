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

    - b. Second, we identify additional real world tasks which have more explicit temporal structure. The temporal crosscoder outperforms alternative architectures in these tasks, and becomes uncompetitive when we introduce a shuffle control to eliminate temporal information.

    - c. Third, we introduce a position-shuffled control and document the improvement of the TXC with window size for all real world tasks considered in the paper.

    - d. Fourth, we provide Stacked SAE results for all real world tasks. Stacked SAEs underperform TXCs for all tasks considered.

<!-- Summary par -->


<!-- Would be good to have a less annoying way of saying 'other people don't do this' -->
2. We address the reviewers concerns about seeds by running three seeds for every headline experiment and confirming that the relative rankings do not change. We also note that seed-averages are not typically reported in the SAE benchmarking literature.

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

In the synthetic setting, we are able to prove that temporal information is responsible for performance. We do this by introducing a task with an analytic ceiling on recoverability from single token information. The task uses a HMM based on Samir secret sharing. In this task, ground truth feature recovery is upper bounded by random guessing below a threshold number of temporal steps $h$. 

**Theorem (known context threshold).** On the degree-$h$ polynomial-clock task,
any method given $h$ or fewer time steps can do no better than random
guessing. With $h+1$ noiseless time steps, the label is uniquely recoverable.
Thus, $h+1$ is the first window length at which performance can even in
principle rise above chance. Intuitively, one point cannot identify the slope
of a line but two can; likewise, two points cannot identify the curvature of a
parabola but three can.

<!-- CLEAN_POLYNOMIAL_CLOCK_RESULTS: insert the verified h=1,2,3 table here. -->

### Position shuffled 
