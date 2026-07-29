## Reviewer responses


## Meta:

The paper proposes to extend the crosscoder from cross-layer to cross-position temporally, it is acknowledged that this is a novel extension. However, the reviewers all would like to see more evidence/comparisons that clearly approves the usefulness of temporal aggregation. 🟧 Reviewer EAxU also provided some suggestions for further improving the presentation.


## 🟩 Reviewer 1
Summary:
The paper introduces temporal crosscoders, which extend the crosscoder idea from the layer axis to the sequence axis: a window of token activations is compressed into one shared sparse latent space and then decoded back per position, so a feature can capture multi-token patterns. To evaluate this, the authors build TempBench, pairing two synthetic benchmarks with four real-model tasks. The main finding is that temporal aggregation helps when the signal is spread over nearby tokens but hurts on per-token or length-correlated tasks.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths The idea of changing the logic of the crosscoder from the layer axis to the sequence axis is very interesting, and the synthetic benchmark is clear and well built (Sec. 4). The honesty about negative results (Secs. 5.3, 5.4, 6) is also appreciated.

Weaknesses I have spotted two main weaknesses:

the main results are based on only one random seed. The checklist says the case studies use two seeds (Q7), but the appendix states that all the main results in Fig. 4 and Table 2 were obtained using only training seed 42 (App. F.12). In addition, the confidence intervals in Fig. 11 are large and overlap, making it unclear whether the improvements are reliable. As a result, the paper never shows both state-of-the-art performance and strong robustness at the same time.

The biggest weakness, however, is that the paper never proves that its main claimed contribution, so the fact that crosscoder-style cross-position weight sharing, and its temporal version is actually responsible for the gains on the main benchmark. The paper even describes the right baseline, Stacked SAE, which isolates temporal aggregation from cross-position weight sharing and was evaluated on C7 (App. A), but this comparison is missing from Fig. 4 and Table 2. Moreover, the claim that temporal aggregation is the key idea is weakened by the sparse probing results. The best-performing model is MLC, which uses cross-layer rather than temporal aggregation, while changing the temporal window length has almost no effect on performance. Overall, the evidence suggests that aggregating information across different dimensions is helpful, but it does not show that temporal aggregation is the reason for the improvements.

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:
Is there one model setting where both the 40% better detection and the 15% better inducement are achieved? If not, please make this clear in the abstract.
The checklist says the experiments use 2 seeds, but the appendix says the main backtracking results use only seed 42. Which is correct? Are the reported improvements larger than the variation across different seeds?
Why are Stacked SAE detection and inducement results not included in Fig. 4 or Table 2? Without this comparison, it is unclear whether the gains come from temporal aggregation or from cross-position weight sharing.
Since MLC performs best in sparse probing, and changing the temporal window has little effect, what evidence shows that the improvement is specifically due to temporal aggregation?
Why does T-SAE use a smaller dictionary if the paper claims all methods use the same dictionary size? Please also report the parameter count and inference cost for each model.
Limitations:
YES

Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.




## 🟦 Reviewer 2

Summary:
This paper proposes a novel method towards the disentanglement of features across sequence positions. It expands the application of crosscoders to crosscodes temporally, where each crosscoder encoder-decoder pair concerns about one position's activation. It also introduces TempBench, a matched-condition comparison of temporal dictionary architectures across two synthetic benches. It also compares the crosscoder-based methods with previous temporal methods including T-SAE and TopK SAEs, etc in real world scenario including sparse coding, backtracking and emergent misalignment.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths
Applying crosscoders to the sequence axis is a natural extension, given the current application of crosscoders in cross-layer and cross-model.
The synthetic benchmark is well-designed, with rigorous formalization with the HMM framework.
The selection of real world evaluation, especially sparse probing and emergent misalignment, are issues worthy concerning.
Weaknesses
I cannot identify specific weaknesses in this paper.

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:
Since the non-temporal MLC ties TXC on probing, can you isolate the temporal contribution from generic crosscoder capacity?
Limitations:
yes

Rating: 5: Accept: Technically solid paper, with high potential value on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.


## 🟧 Reviewer 3
Summary:
The paper presents a new method, Temporal Crosscoders, an adaptation of the crosscoder architecture proposed for temporal feature discovery in LLMs, and introduces TempBench, which includes both synthetic and real-world tasks for evaluating temporal structures. The paper's main claims:

Introduce a new temporal crosscoders architecture for capturing temporal structure in LLMs.
Generalize existing synthetic benchmarks for ground-truth feature recovery to the temporal setting.
Combine four real-world evaluations with synthetic evaluations into TempBench for comparing dictionary-learning architectures.
Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
The main strength of the paper is in its idea, and the motivation behind Temporal Crosscoders is a well-motivated direction.

The main weaknesses of the paper are that the results seem preliminary, and major parts of the manuscript are hard to read and need to be revisited.

Experiments show a marginal improvement over existing works, such as T-SAE and MLC. This leaves the proposed TXC and TXC-pro primarily motivated by the backtracking results.

The main script does not include the TXC and TXC-Pro definitions or a reference point to page 12, Appendix A, making it hard to follow.

Some core citations used in the method seem to be missing: TXC-pro uses Matryoshka [Learning Multi-Level Features with Matryoshka Sparse Autoencoders Bhalla et al. ICLR 2026] The paper uses bad-medical-advice dataset (Line 243) but missing citation Model Organisms for Emergent Misalignment. Turner et al.

Originality - Minor: The submission shares a similar title (Crosscoding Through Time) with Bayazit, Mueller, Bosselut. Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining. Arxiv 2025 / ACL 2026. The reviewer acknowledges that the two papers use "temporal" in different ways, but citing and explicitly stating the distinction would help avoid confusion in the literature.

Quality: 1: poor
Clarity: 1: poor
Significance: 2: not good
Originality: 2: not good
Questions:
I would suggest revisiting the manuscript, specifically, connecting the parts of the Appendix and the parts of the main paper.

In line 459, a method named SAE-arditi is presented, but is missing from the reference or results.

Section F.1 refers to section F.13, it would help to unify them into one table.

Limitations:
yes

Rating: 1: Strong Reject: For instance, a paper with well-known results or unaddressed ethical considerations.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
