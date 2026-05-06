## 1 Introduction

Recent advancements in large language model (LLM) post-training has led to a new class of *reasoning models*, which can leverage test-time compute to achieve enhanced performance on reasoning-intensive benchmarks (Wei et al., 2022; Guo et al., 2025; Yeo et al., 2025; Yang et al., 2025; Ye et al., 2025). These models tend to exhibit an emergent behavior often referred to as *backtracking*, where after progressing down a reasoning path or coming up with a candidate answer, a model will explore alternative strategies (Venhoff et al., 2025). Empirically, the presence of backtracking accounts for a substantial fraction of the accuracy gap between base models and their reasoning-fine-tuned counterparts (Niklas Muennighoff, 2025).

Prior work has shown that this behavior can be reliably induced using *steering vectors* derived from activation differences at sentences classified as backracking (Venhoff et al., 2025). While Venhoff et al. have shown that steering vectors can be used to control backtracking behavior, the fundamental mechanism underlying this behavior remains poorly understood.

In this work, we perform a deeper investigation into backtracking steering vectors and investigate how and where they emerge in model activations. Concretely, we find that a backtracking steering vector can be computed using (1) activations at an offset token position preceding the backtracking event, allowing capture of upstream, causally relevant concepts, and (2) activations sampled only from the base model, suggesting that the underlying mechanism of backtracking partially emerges from a concept already represented in base model activations (Fig. [1](https://arxiv.org/html/2507.12638v1#S1.F1)). Crucially, while this representation is shared by both base and reasoning models, the representation only induces backtracking in the reasoning model, suggesting that it has been repurposed as an input to the backtracking mechanism during reasoning finetuning.

Figure: Figure 1: Steering vectors derived from base model activations induce backtracking when used to steer the reasoning-finetuned model. Green highlights represent tokens from which our backtracking steering vectors are computed, red highlights indicate the start of backtracking.
Refer to caption: https://arxiv.org/html/2507.12638/x1.png

In Sec. [3](https://arxiv.org/html/2507.12638v1#S3), we show that backtracking steering vectors derived from base model activations reliably invoke backtracking when used to steer the reasoning model (Fig. [1](https://arxiv.org/html/2507.12638v1#S1.F1)). In Sec. [4](https://arxiv.org/html/2507.12638v1#S4), we use logit lens and probing to show that the backtracking-inducing direction is non-trivial: it does not directly boost the logits of backtracking-related keywords (e.g. “Wait”). We additionally find this direction is densely present in model activations across contexts, suggesting that it is not the sole factor mediating backtracking behavior.

Taken together, our findings sharpen our understanding of how reasoning capabilities emerge in finetuned models. Rather than learning reasoning behavior entirely from scratch, these models respond to and repurpose signals already present in the base models. This suggests that base models may possess latent reasoning capabilities which are unexpressed until they are extracted by the finetuning process. Broadly, we hope this work inspires focused, fine-grained investigations into various aspects of reasoning processes through the lens of interpretability.

## 2 Preliminaries

### 2.1 Deriving steering vectors

We derive steering vectors using the Difference-of-Means (DoM) method (Rimsky et al., 2024; Venhoff et al., 2025). Full derivation and details are provided in Appendix [A](https://arxiv.org/html/2507.12638v1#A1) for completeness.

### 2.2 Detection of backtracking

Following (Venhoff et al., 2025), we use a GPT-4o judge to identify backtracking events in 300 reasoning model output traces. Separately, we operationalize backtracking as the fraction of output tokens matching a predefined keyword set (e.g. “Wait” or “But”), which has been shown to coincide with backtracking (Galichin et al., 2025). We define this proxy metric as:

$$ $b:=\frac{1}{N}\sum_{\mathrm{word}\in\begin{subarray}{c}\mathrm{reasoning}\\ \mathrm{trace}\end{subarray}}\mathbb{I}[{\mathrm{word}\in\mathcal{B}}]$ (1) $$

where $N$ counts the number of words in the reasoning trace, and $\mathcal{B}=\{\texttt{wait},\texttt{hmm}\}$ is the set of backtracking keywords. Further justifications of the metric are provided in Appendix [C](https://arxiv.org/html/2507.12638v1#A3), where we explore the consistency between LLM (GPT-4o) judges, keyword detection, and human judges in identifying whether a reasoning output component is backtracking or not. We demonstrate that our keyword metric is an acceptable indicator of backtracking events for our purposes.

## 3 Steering vector analysis

Extending previous work on steering vectors and reasoning models (Venhoff et al., 2025), we study steering vectors with two key properties: (1) Negative token offset: In addition to deriving backtracking steering vectors from token positions where backtracking occurs, we use token positions with a negative offset from the actual backtracking event. The fact that these vectors are derived from positions before backtracking occurs suggests that these directions are causally relevant to the model’s decision to backtrack. (2) Using base model activations: Beyond sampling activations from the finetuned reasoning model only, we derive steering vectors separately on residual stream activations from both the base and reasoning models on the same reasoning traces. We refer to these as “base-derived” and “reasoning-derived” steering vectors where appropriate.

### 3.1 Deriving steering vectors with a negative offset

Fig. [2](https://arxiv.org/html/2507.12638v1#S3.F2) measures the effectiveness of backtracking steering vectors across different steering offsets and magnitudes. We find that the optimal offset for $10^{\mathrm{th}}$-layer residual stream of DeepSeek-R1-Distill-Llama-8B is $\sim-13$ to $-8$. This window usually covers the beginning of the sentence prior to backtracking. We also show that these optimally derived steering vectors outperform the no-offset baseline. We fix this offset for the remainder of our analysis.

Figure: Figure 2: The effect of steering as a function of token window offset and steering vector magnitude. Steering vectors are derived from the layer 10 residual stream of the reasoning model. The colorbar reflects the metric value from Eq. ([1](https://arxiv.org/html/2507.12638v1#S2.E1)) averaged over multiple generated steered reasoning traces.
Refer to caption: https://arxiv.org/html/2507.12638/x2.png

### 3.2 Using base model activations

We compute steering vectors separately from base and reasoning model residual stream activations at layer 10, and examine the effect these vectors have when added to each models’ activations during generation. In Fig. [3](https://arxiv.org/html/2507.12638v1#S3.F3) we report the striking result that base-derived steering vectors reliably induce backtracking when used to steer the *reasoning* model, and have comparable performance to their reasoning-derived counterparts. Additionally, we find that neither base-derived nor reasoning-derived steering vectors invoke backtracking behavior in the base model.

Figure: Figure 3: Proportion of backtracking-related tokens generated by both base and reasoning models when steered with base-derived or reasoning-derived steering vectors. Gray lines represent error bars of one standard deviation. Note that the base model never exhibits backtracking behavior, even when steered with the reasoning model-derived backtracking-inducing vector.
Refer to caption: https://arxiv.org/html/2507.12638/x3.png

We additionally examine backtracking steering vectors computed at each of the 32 layers in the studied models, and find both base-derived and reasoning-derived steering vectors are most effective around layer 10. Readers are referred to Appendix [B.1](https://arxiv.org/html/2507.12638v1#A2.SS1) and Fig. [B.1](https://arxiv.org/html/2507.12638v1#A2.F1) for more detailed presentations.

We find that base- and reasoning-derived steering vectors have high cosine similarity of $\sim 0.74$, suggesting that they are capturing the same representation. While the existence of effective, mostly parallel steering vectors strongly indicates a shared representation between base and reasoning models, we find that only the reasoning model uses this representation to initiate backtracking (Fig. [3](https://arxiv.org/html/2507.12638v1#S3.F3)). In light of this, we conjecture that the extracted backtracking steering vectors may actually represent some more abstract concept, and that base and reasoning models use this concept differently for downstream generation.

### 3.3 Validation against baselines

To ensure that the increased backtracking behavior we observe is not merely a consequence of any perturbation in activations, we compare the effect of our steering vector to that of various baselines: (1) overall mean - adding the mean activation to activations; (2) noise - adding random Gaussian noise; (3) self-amplification - increasing activation magnitudes by adding each activation to itself scaled by a coefficient; (4) deduction - steering vectors from tokens labeled as deductive reasoning steps; and (5) initializing - steering vectors from problem setup tokens.

As shown in Fig. [4](https://arxiv.org/html/2507.12638v1#S3.F4), the backtracking steering vector significantly outperforms all tested baselines, validating that we have identified a meaningful direction rather than an artifact of arbitrary perturbations.

Figure: Figure 4: Comparison of various baselines used to steer the reasoning model, measured by the “Wait” metric. The base model-derived, negative-offset backtracking steering vector (ours) clearly has a significant effect.
Refer to caption: https://arxiv.org/html/2507.12638/x4.png

Notably, adding Gaussian noise to activations has a nontrivial effect on the fraction of output words which are “Wait”. We observe anecdotally that this type of intervention results in coherent outputs with increased propensity for backtracking. We leave investigation of this phenomenon as a direction for future work.

## 4 Steering vector directions represent nontrivial concepts

In this section, we investigate what our identified backtracking-inducing vectors encode. Naively, one might expect these vectors to simply boost the output logit probability for trigger tokens like “wait” (Niklas Muennighoff, 2025). We use logit lens to show that token-level attributes cannot explain backtracking behavior in our steering vectors. Given the effectiveness of our base-derived vector, optimists might anticipate (in view of refusal directions (Arditi et al., 2024)) that backtracking is mediated by this single direction. However, probing experiments reveal that the backtracking-inducing direction we find is not alone sufficient to fully explain backtracking behavior.

### 4.1 Logit lens

One possible trivial explanation for the backtracking-inducing behavior of our steering vectors is that they merely boost the probability of backtracking-related tokens via a direct projection onto the unembedding directions for these tokens. We refute this explanation by showing that the base-derived vector *does not* have significant positive projections onto the relevant unembed directions. We compute a “backtracking score” $s$ by computing the projection of our vectors onto the unembed matrix, with directions for irrelevant tokens (non-backtracking-related) masked out:

$$ $s(\mathbf{v}):=(W_{U}\mathbf{v})\cdot\frac{\mathbf{a}}{\|\mathbf{a}\|_{1}},a_{ i}=\left\{\begin{array}[]{cc}1&\text{if $\mathrm{Decode}(i)\in\mathcal{B}$}\\ 0&\text{otherwise}\end{array}\right.$ (2) $$

where $\mathbf{v}$ is the steering vector, $W_{U}$ is the unembedding matrix of either base or fine-tuned model and $\mathbf{a}$ is a mask which selects for backtracking keywords. In this section, we use $\mathcal{B}=\{\texttt{wait},\texttt{but}\}$ (^1^11This covers all tokens in the vocabulary which contain “wait” or “but”, case insensitive. Examples: _Wait, _wait, Wait, etc.).

Fig. [5](https://arxiv.org/html/2507.12638v1#S4.F5) reports the backtracking scores of base- and reasoning-derived steering vectors computed from hidden activations at different layers. Combined with findings in the previous section, we observe that (1) the base-derived steering vectors do not decode to backtracking keywords, yet are successful in eliciting backtracking in the fine-tuned model; (2) although later-layer steering vectors can be attributed to token-level logit boosts, they are less effective when used for steering. From this, we claim that our layer-10 base-derived steering vector is capturing a more abstract concept, causally relevant for backtracking.

Figure: Figure 5: Backtracking scores of steering vectors trained on base model (blue) or reasoning-finetuned model (orange) activations at different layers, when projected onto base (light) or reasoning (dark) model unembedding matrices.
Refer to caption: https://arxiv.org/html/2507.12638/x5.png

### 4.2 Probing

To better understand our identified backtracking-inducing direction, we conduct case studies in which we use this direction for probing. We examine the magnitude of the projection of these vectors onto centered model activations and attempt to identify an interpretable semantic meaning.

Figure: Figure 6: A sample output generated by the reasoning model without steering. Tokens are highlighted in green when the projection of the base model-derived backtracking steering vector onto layer 10 activations is positive, with darker green representing higher magnitudes. “Wait” tokens outlined in red for clarity.
Refer to caption: https://arxiv.org/html/2507.12638/x6.png

We find that, unexpectedly, the base-derived steering direction is densely present in model activations (Fig. [6](https://arxiv.org/html/2507.12638v1#S4.F6)), and that it does not cleanly correlate with backtracking when used as a probe. These results lead us to hypothesize that our identified direction may be one of several heuristics the reasoning model uses to trigger backtracking. Our observation that our identified direction is effective at moderate (but not small) steering strengths indicates that the actual backtracking mechanism may involve a linear combination of such heuristics. We leave further investigation of this hypothesis to future work.

## 5 Conclusion and Outlook

In this work, we have provided evidence that the emergent backtracking behavior expressed by the DeepSeek-R1-Distill-Llama-8B reasoning model arises, at least in part, through the repurposing of pre-existing representations in the base model from which it was finetuned. More specifically, we have identified a direction present in Llama-3.1-8B residual stream activations which, when added to reasoning model activations, systematically induces backtracking. The fact that this representation exists in the base model without inducing backtracking there offers a key insight: that the reasoning finetuning process may elicit backtracking behavior by repurposing latent representations already present in the base model, rather than learning the entire mechanism from scratch.

Our analysis has several limitations. First, we examined a single reasoning model; further investigation to validate the robustness of our findings across different model families and scales is required in order to claim our findings are general. Second, our identified steering direction is densely present in model activations and we find instances both where backtracking occurs while the direction is not present, and instances where the direction is present but backtracking does not occur. This indicates that we have only identified one component of the backtracking mechanism, and the remainder of the mechanism remains unknown. Our results should be interpreted primarily as an existence proof for latent reasoning-related representations in base models, rather than a comprehensive explanation of reasoning behavior.

Our work highlights the value of interpretability tools in uncovering nuanced insights regarding the mechanisms underlying emergent capabilities in LLMs. We hope that these insights will ultimately lead to more transparent and controllable artificial intelligence.

## Acknowledgements

We acknowledge the support of the MATS 8.0 program during which foundational experiments for this work were completed. C.L. thanks Arthur Conmy and Shivam Raval for useful comments on related research topics, and Shivaji Sondhi and The Leverhulme Trust for kind support provoiding GPU compute, and introduction to the field of interpretability more generally.

## References

- Arditi et al. (2024)
Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., and Nanda, N.
Refusal in language models is mediated by a single direction.
*arXiv preprint arXiv:2406.11717*, 2024.
- Galichin et al. (2025)
Galichin, A., Dontsov, A., Druzhinina, P., Razzhigaev, A., Rogov, O. Y., Tutubalina, E., and Oseledets, I.
I have covered all the bases here: Interpreting reasoning features in large language models via sparse autoencoders.
*arXiv preprint arXiv:2503.18878*, 2025.
- Guo et al. (2025)
Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.
Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.
*arXiv preprint arXiv:2501.12948*, 2025.
- Niklas Muennighoff (2025)
Niklas Muennighoff, Zitong Yang, W. S. X. L. L. L. F.-F. H. H. L. Z. P. L. E. C. T. H.
s1: Simple test-time scaling.
*arXiv preprint arXiv:2501.19393*, 2025.
- Rimsky et al. (2024)
Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., and Turner, A.
Steering llama 2 via contrastive activation addition.
In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp.  15504–15522, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
doi: 10.18653/v1/2024.acl-long.828.
URL [https://aclanthology.org/2024.acl-long.828/](https://aclanthology.org/2024.acl-long.828/).
- Venhoff et al. (2025)
Venhoff, C., Arcuschin, I., Torr, P., Conmy, A., and Nanda, N.
Understanding reasoning in thinking language models via steering vectors.
In *Workshop on Reasoning and Planning for Large Language Models*, 2025.
URL [https://openreview.net/forum?id=OwhVWNOBcz](https://openreview.net/forum?id=OwhVWNOBcz).
- Wei et al. (2022)
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al.
Chain-of-thought prompting elicits reasoning in large language models.
*Advances in neural information processing systems*, 35:24824–24837, 2022.
- Yang et al. (2025)
Yang, X.-W., Zhu, X.-Y., Wei, W.-D., Zhang, D.-C., Shao, J.-J., Zhou, Z., Guo, L.-Z., and Li, Y.-F.
Step back to leap forward: Self-backtracking for boosting reasoning of language models.
*arXiv preprint arXiv:2502.04404*, 2025.
URL [https://arxiv.org/abs/2502.04404](https://arxiv.org/abs/2502.04404).
- Ye et al. (2025)
Ye, G., Pham, K. D., Zhang, X., Gopi, S., Peng, B., Li, B., Kulkarni, J., and Inan, H. A.
On the emergence of thinking in llms i: Searching for the right intuition.
*arXiv preprint arXiv:2502.06773*, 2025.
URL [https://arxiv.org/abs/2502.06773](https://arxiv.org/abs/2502.06773).
- Yeo et al. (2025)
Yeo, E.-H., Lin, Z., Han, T., Lin, M., Wang, J., Wang, J., Zhang, J., and Zhuo, D.
Demystifying long chain-of-thought reasoning in llms.
*arXiv preprint arXiv:2502.03373*, 2025.
URL [https://arxiv.org/abs/2502.03373](https://arxiv.org/abs/2502.03373).

## Appendix A Deriving steering vectors

In this work, we derive steering vectors using the Difference-of-Means (DoM) method (Rimsky et al., 2024; Venhoff et al., 2025). We use DeepSeek-R1-Distill-Llama-8B to generate a set of tokenized reasoning traces $\mathcal{R}$. We then extract a dataset $\mathcal{D}=\{(p_{i},S_{i})|i\in[|\mathcal{R}|]\}$, where $p_{i}$ is the tokenized prompt of the $i^{\mathrm{th}}$ reasoning trace in $\mathcal{R}$, and $S_{i}$ is a subset of sequence positions where we would like to train the steering vector (e.g. fixed offset preceding sentence terminations). We further extract $\mathcal{D}_{+}\subset\mathcal{D}$ where the backtracking-related token is present, as annotated by GPT-4o. The steering vector $\mathbf{v}$ is computed as

$$ $\displaystyle\mathrm{MeanAct}(\mathcal{D}):=\frac{1}{|\mathcal{D}|}\sum_{i=1}^ {|\mathcal{D}|}\frac{1}{|S_{i}|}\sum_{s\in S_{i}}A(p_{i})[s]\$ $\displaystyle\mathbf{v}=\mathrm{MeanAct}(\mathcal{D}_{+})-\mathrm{MeanAct}( \mathcal{D})$ (3) $$

where $A(p_{i})$ is the target hidden state (at some target layer) when forward-passing the tokenized prompt $p_{i}$, and $[s]$ accesses the subset sequence positions.

Following the methodology of (Venhoff et al., 2025), we generate a dataset for computing steering vectors:

- 1.
We generate 300 prompts within 10 different categories using Claude Sonnet 3.7.
- 2.
We use DeepSeek-R1-Distill-Llama-8B to generate reasoning traces for these prompts.
- 3.
We use GPT-4o to classify sentences in the generated reasoning traces as “backtracking” or otherwise.

## Appendix B More on training steering vectors

### B.1 Base-derived and reasoning-derived steering vectors from all hidden layers

Figure: (a) Downstream backtracking behavior response to base-derived steering vectors, as measured by proportion of output tokens in the steered reasoning trace which are backtracking-related.
Refer to caption: https://arxiv.org/html/2507.12638/x7.png

## Appendix C Consistency between human, LLM and keyword judges for backtracking detection

Throughout our analyses, we employ a keyword counting metric to measure the presence of backtracking in various steering regimes. We refer to this metric here as the “keyword judge”. This metric classifies a reasoning trace component as “backtracking” by the presence of keyword tokens. We believe this is a reasonable proxy for detecting backtracking behavior.

#### LLM judge

When generating a labeled dataset for deriving steering vectors, we prompt GPT-4o to annotate reasoning traces based on a prescribed sentence taxonomy c.f. (Venhoff et al., 2025).

#### Keyword judge

As explained above, to judge whether a reasoning trace component is backtracking or not, we look for the pattern wait in the decoded text.

In Table [C.1](https://arxiv.org/html/2507.12638v1#A3.T1), we analyze the consistency between LLM and keyword judges when classifying reasoning sentences generated by the fine-tuned model with and without backtracking steering, and at various steering strengths. While backtracking occurs in fewer than two percent of sentences on average, we see F1 scores above 60% at intermediate steering strengths, showing that we have reasonable agreement between detections made by either LLM or keyword judge.

On the other hand, the numbers still signal a nontrivial gap between the classification decisions made by LLM judge and the keyword judge - To further resolve such discrepancies, we introduce ourselves as human judges for backtracking ground truth. Extensive case study further unravels inadequacy of LLM and keywords as judges. Quantitatively, we find that $\sim 83\%$ of sentences identified with the keyword metric are true examples of backtracking when we use our own judgement as the ground truth. Qualitatively, we find that some of the discrepancy between keyword and LLM judges stems from the LLM classifiying backtracking sentences as something other than backtracking, like “uncertainty-estimation”.

In summary, we believe the LLM judge and keyword metrics are both reasonable proxies for “true” backtracking for our purposes. However, applications where high precision and accuracy is critical will require more sophisticated metrics.

**Table C.1: Metrics of fit for keyword judge against LLM judge (treated as ground truth). The dataset we used consists of 300 questions featuring basic logic, geometry and probability.**
| Steering Strength | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| 0 | 63.36% | 64.91% | 64.12% |
| 4 | 62.50% | 66.67% | 64.52% |
| 8 | 48.11% | 68.00% | 56.35% |
| 12 | 47.66% | 62.93% | 54.24% |