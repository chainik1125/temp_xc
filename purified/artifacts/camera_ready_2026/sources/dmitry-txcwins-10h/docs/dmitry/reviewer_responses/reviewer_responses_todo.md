## 🟩 Reviewer 1


## TODO

1. Random seeds
    - Do we have more than one?
    - I think we only report one and that is standard in the SAE literature? Have not seen seed averaged SAEs reported before, but can check.
2. Stacked SAE baseline
3. Shuffle control
4. Synthetic controls
5. 

## Full text
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
