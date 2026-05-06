## Contents
- 1 Introduction
- 2 Preliminaries
  - 2.1 Attribution Patching
  - 2.2 Computing Steering Vectors
  - 2.3 DeepSeek Thinking Models
- 3 Identifying Reasoning Behaviors for Steering
- 4 Extracting and Evaluating Steering Vectors
  - 4.1 Experimental Setup
  - 4.2 Locating Causally Relevant Activations
  - 4.3 Selecting Final Steering Vectors
  - 4.4 Evaluation of Steering Vectors
  - 4.5 Results of Steering
- 5 Related Work
- 6 Conclusion and Future Work
- Acknowledgements
- References
- Appendix A Details on the Annotation Process
- Appendix B Annotated Example
- Appendix C Cosine Similarity Between Feature Vectors
- Appendix D Cosine Similarity with Embedding and Unembedding Vectors
  - D.1 Methodology
  - D.2 Results and Analysis
- Appendix E Steered Example (Adding Knowledge)

## Abstract

Abstract Recent advances in large language models (LLMs) have led to the development of thinking language models that generate extensive internal reasoning chains before producing responses. While these models achieve improved performance, controlling their reasoning processes remains challenging. This work presents a steering approach for thinking LLMs by analyzing and manipulating specific reasoning behaviors in DeepSeek-R1-Distill models.
Through a systematic experiment on 500 tasks across 10 diverse categories, we identify several reasoning behaviors exhibited by thinking models, including expressing uncertainty, generating examples for hypothesis validation, and backtracking in reasoning chains. We demonstrate that these behaviors are mediated by linear directions in the model’s activation space and can be controlled using steering vectors.
By extracting and applying these vectors, we provide a method to modulate specific aspects of the model’s reasoning process, such as its tendency to backtrack or express uncertainty. Our approach offers practical tools for steering reasoning processes in thinking models in a controlled and interpretable manner. We validate our steering method using three DeepSeek-R1-Distill models, demonstrating consistent control across different model architectures.

## 1 Introduction

A recent trend in the development of large language models (LLMs) has been *thinking* LLMs, which generate extensive internal reasoning chains before producing responses (Reynolds & McDonell, 2021; Nye et al., 2021; Wei et al., 2022). Examples include OpenAI’s o1 (OpenAI, 2024) and DeepSeek’s R1 (DeepSeek-AI, 2025). These models have achieved remarkable improvements in performance (Chollet, 2024), yet controlling and understanding their internal reasoning processes remains challenging.

To address this challenge, we develop a steering approach for thinking LLMs by analyzing reasoning behaviors in DeepSeek-R1-Distill models. We focus on several reasoning behaviors exhibited by these models, including their tendency to express uncertainty, backtrack in reasoning chains, and generate examples for hypothesis testing. While these behaviors may not constitute a complete taxonomy of reasoning mechanisms, they provide a practical foundation for developing steering methods.

We investigate whether these reasoning behaviors of thinking LLMs can be directly controlled using *steering vectors*, a method that has been shown to allow precise behavioral control in LLMs (Subramani et al., 2022; Turner et al., 2023; Zou et al., 2023; Panickssery et al., 2023; Templeton et al., 2024; Arditi et al., 2024a). By extracting and applying steering vectors, we provide a means to modulate the internal reasoning dynamics of thinking LLMs in a controlled manner.

In summary, our work presents the following key contributions:

- 1.
We develop a method to extract steering vectors for specific reasoning behaviors in DeepSeek-R1-Distill models, enabling precise control over aspects of their thinking process, such as *backtracking* and *uncertainty estimation*.
- 2.
We demonstrate the effectiveness of our steering approach through an empirical evaluation on over $500$ tasks across $10$ diverse categories, showing consistent control across multiple DeepSeek-R1-Distill model architectures and sizes.

Figure: Figure 1: *Steering* on DeepSeek-R1’s backtracking feature vector changes the model’s behavior. Depending on whether we add or subtract this vector to the activations at inference time, the model increases or decreases its tendency to abandon its current approach and explore alternative strategies for the task at hand. Highlighted sections indicate instances of this behavior.
Refer to caption: x1.png

Our steering approach provides a practical method for controlling reasoning processes in thinking models in an interpretable manner, opening new possibilities for fine-grained control of model behavior. To facilitate reproducibility and enable further research, we make our complete experimental codebase and datasets publicly available.(^1^11[https://github.com/cvenhoff/steering-thinking-llms](https://github.com/cvenhoff/steering-thinking-llms))

## 2 Preliminaries

### 2.1 Attribution Patching

A fundamental challenge in analyzing the behavior of large language models (LLMs) is identifying the specific components and layers responsible for a given behavior. A widely used technique for addressing this challenge is activation patching (Meng et al., 2022). Activation patching works by replacing the activations of a specific model component with those from a counterfactual example, which differs only in a specific aspect of the behavior being analyzed. If this intervention significantly alters the model’s output with respect to the observed behavior, the modified component can be marked as playing a key role in implementing this behavior. The patching effect is quantified as the change in a relevant output metric:

$$ $\Delta L=L(\mathbf{x}_{\text{clean}}\mid\text{do}(\mathbf{a}=\mathbf{a}_{\text{patch}}))-L(\mathbf{x}_{\text{clean}}),$ $$

where $L$ is a metric measuring the difference in model outputs (e.g., KL-divergence), $\mathbf{a}$ is the original activation, and $\mathbf{a}_{\text{patch}}$ is the counterfactual activation.

Since activation patching is computationally expensive, a more efficient linear approximation known as attribution patching (Nanda, 2023; Syed et al., 2023) is often used, which utilizes the gradients of the model’s activations with respect to the metric:

$$ $\Delta L\approx(\mathbf{a}_{\text{patch}}-\mathbf{a}_{\text{clean}})^{T}\cdot\frac{\partial}{\partial\mathbf{a}_{\text{clean}}}L(\mathbf{x}_{\text{clean}}\mid\text{do}(\mathbf{a}=\mathbf{a}_{\text{clean}})).$ $$

### 2.2 Computing Steering Vectors

The Difference of Means method is a widely used technique for extracting steering vectors in LLMs (Turner et al., 2024; Arditi et al., 2024b). This technique is based on constructing contrastive datasets that differ in a specific concept and computing the difference in their mean activations of a model.
Formally, let $D_{+}$ and $D_{-}$ be two datasets where samples in $D_{+}$ exhibit a given concept, while samples in $D_{-}$ do not. Given a model component, we compute the Difference of Means vector as:

$$ $\mathbf{u}=\frac{1}{|D_{+}|}\sum_{p_{i}\in D_{+}}\mathbf{a}(p_{i})-\frac{1}{|D_{-}|}\sum_{p_{j}\in D_{-}}\mathbf{a}(p_{j})$ $$

where $a(p_{i})$ and $a(p_{j})$ represent the activations of the model components over the prompts from the respective datasets.
This vector $u$ captures the primary direction in activation space that differentiates the two datasets with respect to the target concept.
In cases where explicitly matched counterfactuals are unavailable, a common heuristic is to define $D_{+}$ as the set of all samples exhibiting the target behavior, while $D_{-}$ consists of the full dataset. In this scenario, the Difference of Means vector is computed by subtracting the overall mean activation from the mean activation of the behavior-associated examples.
This isolates the direction in the activation space most associated with the target behavior while reducing the influence of general model biases.

### 2.3 DeepSeek Thinking Models

As mentioned in the introduction, *thinking* models are a type of language model designed to generate long chains of internal reasoning before arriving at a final answer.
Examples of this type of models include QwQ (Qwen Team, 2024), Gemini 2.0 Flash Thinking (GDM, 2024), o1 (OpenAI, 2024), and DeepSeek-R1 (DeepSeek-AI, 2025).

In this work, we focus on characterizing the thinking mechanisms of DeepSeek-R1, a recent *thinking* model that has achieved a similar performance to o1-preview on the ARC-AGI-Pub dataset (Knoop, 2025; Chollet, 2024).
DeepSeek-R1 is a language model trained through a multi-stage process that combines large-scale reinforcement learning (RL) with the strategic use of supervised fine-tuning (SFT).
The model’s architecture uses a Mixture-of-Experts (MoE) approach with 37B activated parameters and 671B total parameters.

The DeepSeek team has distilled R1’s reasoning capabilities into smaller, dense models ranging from 1.5B to 70B parameters, based on both Qwen and Llama architectures.
These distilled models achieve similar or better performance than frontier production models like GPT-4o and Claude 3.5 Sonnet at several math and coding benchmarks (DeepSeek-AI, 2025).
We use the Qwen-14B and Llama-8B distilled models of DeepSeek R1 for our analysis.

## 3 Identifying Reasoning Behaviors for Steering

Figure: Figure 2: Comparison of behavioral patterns across five DeepSeek-R1-Distill models and five baseline models on $100$ randomly selected tasks from our dataset (cf. [Section 4.1](https://arxiv.org/html/2506.18167v4#S4.SS1)). The plot on the left shows the fraction of sentences annotated with each behavioral category. The plot on the right shows the average number of sentences per response. Thinking models generate substantially longer responses ($27.6$ vs $14.4$ sentences on average) and exhibit a higher fractions of backtracking, uncertainty estimation and example testing behaviors, but lower fractions of knowledge augmentation.
Refer to caption: x2.png

To identify reasoning behaviors suitable for steering, we examined $100$ reasoning chains generated by DeepSeek-R1 and $100$ answers produced by GPT-4o, using tasks across diverse categories (see [Section 4.1](https://arxiv.org/html/2506.18167v4#S4.SS1)).
Consistent with prior findings (DeepSeek-AI, 2025), we observe that one of the key differences is that the DeepSeek-R1 distills tend to explore many different approaches when solving a task, whereas the base models follow a more linear reasoning trajectory. Notably, the thinking models can express uncertainty about its current approach, generate examples or scenarios to test hypotheses, and backtrack to intermediate steps when revising its reasoning.

Based on these observations, we identify the following behavioral patterns in the DeepSeek-R1 distill reasoning processes as targets for our steering approach:

- •
Initialization: The model rephrases the task and articulates initial thoughts, typically at the beginning of the reasoning chain.
- •
Deduction: The model derives conclusions based on its current approach and assumptions.
- •
Knowledge Augmentation: The model incorporates external knowledge to refine its reasoning.
- •
Example Testing: The model generates examples or scenarios to validate its working hypothesis.
- •
Uncertainty Estimation: The model explicitly states its confidence or uncertainty regarding its reasoning.
- •
Backtracking: The model abandons its current approach and explores an alternative strategy.

An annotated example of these behaviors is provided in [Appendix B](https://arxiv.org/html/2506.18167v4#A2). To systematically annotate reasoning chain in our experiments, we employ GPT-4o (see [Appendix A](https://arxiv.org/html/2506.18167v4#A1) for details). To quantify the prevalence of these reasoning behaviors, we generate reasoning chains using five DeepSeek-R1-Distill models (Qwen-14B, Qwen-32B, Llama-70B, Qwen-1.5B, and Llama-8B) and compare them against five baseline models (Qwen2.5-Math-1.5B, Qwen2.5-32B-Instruct, Qwen2.5-14B-Instruct, Llama-3.3-70B-Instruct, and Llama-3.1-8B-Instruct) across $100$ diverse tasks (see [Section 4.1](https://arxiv.org/html/2506.18167v4#S4.SS1)). We then automatically annotate each reasoning chain and compute the fraction of sentences annotated with each behavioral category.
Results are presented in [Figure 2](https://arxiv.org/html/2506.18167v4#S3.F2).

The results reveal several key distinctions between thinking and baseline models. Most notably, thinking models generate substantially longer responses, averaging $27.6$ sentences vs $14.4$ for baseline models. Both model types devote similar fractions of their responses to initialization and deduction, suggesting that these processes are comparable across model types. In contrast, baseline models show lower fractions of backtracking, uncertainty estimation, and, to a lesser extent, example testing, instead allocating more of their responses to knowledge augmentation.

These patterns suggest that uncertainty estimation and backtracking are prominent behaviors that distinguish thinking models, followed by example testing and knowledge addition.
Given that initialization and deduction do not seem to be behaviors specific to thinking models, steering on these seems conceptually ill-defined; initialization occurs once at the start and deduction is fundamental to any LLM, therefore we omit them from the empirical study on steering vectors.

## 4 Extracting and Evaluating Steering Vectors

In this section, we demonstrate that the reasoning behaviors of the DeepSeek-R1-Distill models characterized in [Section 3](https://arxiv.org/html/2506.18167v4#S3) are mediated by linear directions, the steering vectors. We assess the causal effect of these vectors by comparing the model’s original reasoning chains to those generated under positive and negative steering (adding and subtracting the steering vectors). Our findings indicate that the DeepSeek-R1-Distill models have distinct mechanisms to achieve their reasoning process. Additionally, our steering vectors provide an efficient way to influence these models’ reasoning behavior, for example, increasing their tendency to backtrack or modulate their inherent uncertainty in their own reasoning.

### 4.1 Experimental Setup

For our experiments, we generate a dataset of $500$ tasks across 10 categories using Claude 3.5 Sonnet (see [Table 1](https://arxiv.org/html/2506.18167v4#S4.T1)).
We conduct our experiments on three DeepSeek-R1-Distill models: Qwen-14B, Qwen-1.5B, and Llama-8B. When generating a reasoning chain, we use greedy decoding and $1000$ max tokens per response.

**Table 1: Task categories used to analyze reasoning behaviors.**
| Category | Description |
| --- | --- |
| Mathematical Logic | Problems requiring formal logical operations, mathematical proofs, and numerical reasoning |
| Spatial Reasoning | Tasks involving visualization, geometric manipulation, and understanding spatial relationships |
| Verbal Logic | Problems focused on language-based reasoning, syllogisms, and verbal analogies |
| Pattern Recognition | Questions requiring identification and continuation of sequences or abstract patterns |
| Lateral Thinking | Problems that require creative, non-linear approaches to reach unconventional solutions |
| Causal Reasoning | Tasks involving understanding cause-and-effect relationships and making causal inferences |
| Probabilistic Thinking | Problems requiring reasoning about uncertainty, probability, and statistical concepts |
| Systems Thinking | Questions about complex systems, interconnected components, and emergent behaviors |
| Creative Problem Solving | Open-ended problems requiring novel approaches and innovative solutions |
| Scientific Reasoning | Tasks involving hypothesis formation, experimental design, and evidence evaluation |

### 4.2 Locating Causally Relevant Activations

To extract robust steering vectors, we first identify the activations where these vectors are linearly represented within the model. We focus on the residual stream activations, i.e., the outputs of each transformer layer. Given a reasoning chain generated by a DeepSeek-R1-Distill model, we identify both the token positions and layers where the steering vector is active. This process consists of two key steps:

- 1.
Identifying relevant token positions: Determine which tokens in the reasoning chain correspond to a specific behavioral category.
- 2.
Determining causally relevant layers: Use attribution patching ([Section 2](https://arxiv.org/html/2506.18167v4#S2)) to evaluate which layers contribute causally to the behavior in question.

To obtain token positions associated with each behavioral category, we generate $500$ reasoning chains with the tasks introduced in [Section 4.1](https://arxiv.org/html/2506.18167v4#S4.SS1), using both DeepSeek-R1-Distill models and then annotate them automatically with GPT-4o. Since the DeepSeek-R1-Distill models are autoregressive, we consider for each category both the token position preceding the start of a token-sequence annotated with the current category and the annotated sequence itself as the causally relevant token positions (up to $10$ tokens). This ensures that we capture both the decision point where the model transitions into the behavior and the behavior’s execution phase.

To identify the causally relevant layers for each behavioral category, we first extract a steering vector candidate from every layer using the Difference of Means method ([Section 2.2](https://arxiv.org/html/2506.18167v4#S2.SS2)):

$$ $\mathbf{u}_{\ell}^{c}=\frac{1}{|D_{+}|}\sum_{p_{i}\in D_{+}}\mathbf{\bar{a}}_{\ell}^{c}(p_{i})-\frac{1}{|D_{-}|}\sum_{p_{j}\in D_{-}}\mathbf{a}_{\ell}^{c}(p_{j}),\quad\text{with}\quad\mathbf{\bar{a}}_{\ell}^{c}(p_{i})=\frac{1}{|\text{seq}_{c}(p_{i})|}\sum_{t\in\text{seq}_{c}(p_{i})}\mathbf{a}_{\ell}(t).$ $$

where:

- •
$\mathbf{a}_{\ell}(t)$ represents the residual stream activation at layer $\ell$ for token position $t$.
- •
$\text{seq}_{c}(p)$ is the set of all token sequences within prompt $p$ that are annotated with category $c$, including the preceding token position.
- •
$\mathbf{\bar{a}}_{\ell}^{c}(p_{i})$ denotes the mean activation across all token positions within the annotated sequences of category $c$ at layer $\ell$.
- •
$D_{+}$ consists of prompts containing at least one sequence labeled with category $c$, while $D_{-}$ represents the full dataset.

The resulting vector $\mathbf{u}_{\ell}^{c}$ serves as a candidate steering vector for each layer. To ensure consistent scaling across different sequence lengths and behaviors, we normalize each steering vector to have the same magnitude as the mean overall activation:

$$ $\mathbf{u}_{\ell}^{c,\text{norm}}=\mathbf{u}_{\ell}^{c}\cdot\frac{\|\mathbf{\bar{a}}_{\ell}^{\text{overall}}\|}{\|\mathbf{u}_{\ell}^{c}\|}$ $$

where $\mathbf{\bar{a}}_{\ell}^{\text{overall}}$ is the mean activation across all tokens in the dataset at layer $\ell$. This normalization helps account for varying sequence lengths and ensures steering vectors have comparable magnitudes across different behavioral categories.

### 4.3 Selecting Final Steering Vectors

Figure: Figure 3: Causal impact of candidate steering vectors across model layers. The y-axis represents the absolute mean KL-divergence for the next-token logit distribution when removing the steering vector at each layer. The steering vectors for all reasoning mechanisms have similar peaks in the middle layers of the respective models.
Refer to caption: x3.png

To determine the final steering vectors, we apply attribution patching ([Section 2.1](https://arxiv.org/html/2506.18167v4#S2.SS1)) to quantify the causal relevance of each vector in its respective layer. Specifically, we consider the following patching experiment: Given a candidate steering vector $\mathbf{u}_{\ell}^{c}$ for a specific behavioral category, we add it to the residual stream activation preceding a token-sequence annotated with one of the relevant behaviors. Therefore, we define the patched activation as:

$$ $\mathbf{a}_{\ell}^{\text{patched}}=\mathbf{a}_{\ell}+\mathbf{u}_{\ell}^{c}.$ $$

If this intervention leads to a significant change in the KL divergence of the next-token prediction, then the steering vector in layer $\ell$ is causally relevant for the given behavior. We approximate the patching effect for this experiment with:

$$ $\Delta L\approx(\mathbf{u}_{\ell}^{c})^{T}\cdot\frac{\partial}{\partial\mathbf{a}_{\ell}}L(\mathbf{x}_{\text{clean}}\mid\text{do}(\mathbf{a}_{\ell}=\mathbf{a}_{\text{clean}})),$ $$

where $\mathbf{u}_{\ell}^{c}=(\mathbf{a}_{\ell}^{\text{patched}}-\mathbf{a}_{\ell})$.
We average the absolute patching effect for each category over all category-sequences in all $500$ reasoning chains. The results are shown in [Figure 3](https://arxiv.org/html/2506.18167v4#S4.F3). Based on these results, we can select the causally most relevant steering vectors from the layers where the patching scores are highest. However, we avoid selecting early layers that show excessive correlation with embedding tokens, as these layers primarily capture token-specific representations rather than behavioral patterns ([Appendix D](https://arxiv.org/html/2506.18167v4#A4)).

### 4.4 Evaluation of Steering Vectors

To evaluate the effectiveness of our extracted steering vectors, we apply them at the selected layers (see [Table 2](https://arxiv.org/html/2506.18167v4#S4.T2)) and observe their influence on the model’s reasoning process. Steering is implemented by adding or subtracting the extracted steering vectors $\mathbf{u}_{\ell}^{c}$ to the residual stream activations at inference time.
By applying this intervention, we can increase or decrease behaviors such as backtracking, uncertainty estimation, and example testing, providing a direct mechanism for manipulating the model’s reasoning process.

### 4.5 Results of Steering

Figure: Figure 4: Effect of applying the steering vector for each reasoning behavior across different distill models. The y-axis shows the change in the fraction of tokens exhibiting each behavior when applying positive or negative steering. Positive steering increases behaviors such as backtracking and uncertainty estimation, while negative steering suppresses or significantly reduces them, confirming the causal influence of our extracted vectors.
Refer to caption: x6.png

**Table 2: Selected layers for each behavioral category based on attribution patching results. For each model, we select the layer with the maximum score from the attribution patching experiments, ignoring early layers that are highly correlated with embedding tokens.**
| Behavioral Category | DeepSeek-R1-Distill<br>Llama-8B | DeepSeek-R1-Distill<br>Qwen-1.5B | DeepSeek-R1-Distill<br>Qwen-14B |
| --- | --- | --- | --- |
| Uncertainty Estimation | 12 | 18 | 29 |
| Example Testing | 12 | 15 | 29 |
| Backtracking | 12 | 17 | 29 |
| Adding Knowledge | 12 | 18 | 24 |

We apply each steering vector to $50$ unseen reasoning tasks and analyze how the model’s reasoning behavior changes. The results, presented in [Figure 4](https://arxiv.org/html/2506.18167v4#S4.F4), demonstrate that our extracted vectors effectively control the model’s reasoning patterns.
[Appendix E](https://arxiv.org/html/2506.18167v4#A5) includes a full example of positive and negative steering on the “Adding Knowledge” vector for the DeepSeek-R1-Distill-Qwen-14B model.

As shown in [Figure 4](https://arxiv.org/html/2506.18167v4#S4.F4), positive steering increases behaviors such as backtracking, uncertainty estimation, and example testing, while negative steering reduces them. These effects are consistent across both DeepSeek-R1-Distill models, reinforcing the hypothesis that Thinking LLMs encode these reasoning mechanisms as linear directions in their activation space. Our findings confirm that steering vectors provide a reliable and efficient method for interpreting and controlling the internal reasoning dynamics of thinking large language models. Further analysis of the cosine similarity between different steering vectors reveals that most behavioral categories correspond to distinct directions in the model’s activation space (see [Appendix C](https://arxiv.org/html/2506.18167v4#A3)).

## 5 Related Work

Recent work has explored methods for steering and interpreting language models by identifying meaningful directions or features within their internal representation spaces. Subramani et al. (2022) show that extracting latent steering vectors from pretrained language models can systematically alter the generated text. Similarly, Turner et al. (2023) propose *activation engineering*, modifying model activations at inference time to control outputs, in contrast to prompt engineering or fine-tuning. Extending this line of research, Panickssery et al. (2023) introduce Contrastive Activation Addition (CAA), which derives a “contrastive” vector by averaging differences in residual stream activations between positive and negative examples of a target behavior. Adding this vector to a model’s activations elicits more desirable outputs without retraining. Beyond these methods, Zou et al. (2023) propose *Representation Engineering*, offering a top-down approach to refining and analyzing internal representations for greater transparency.

Applying these methods for fine-grained control of language models through systematic manipulation of their internal representations Li et al. (2023) propose an *inference-time intervention* to encourage more truthful responses without additional training. Additionally, Arditi et al. (2024a) has shown that refusal behavior can be localized to a single direction in latent space, that minimally affect other capabilities, enabling targeted interventions to encourage refusal or jailbreak the model.

A related line of work focuses on leveraging these internal representations for reasoning and chain-of-thought. Zhao et al. (2025) found a steering vector to efficiently elicit long chain-of-thought in language models. Dutta et al. (2024) analyzed the neural sub-structures within Llama-2 7B that facilitate multistep reasoning over fictional ontologies and found that the model deploys multiple parallel pathways for step-by-step reasoning. In concurrent work, Hazra et al. (2025) train sparse autoencoders on DeepSeek’s 671B-parameter R1 reasoning model, uncovering internal features associated with reasoning behaviors, such as backtracking, which can be used for steering interventions.

## 6 Conclusion and Future Work

This work presents a steering approach for controlling reasoning behaviors in thinking LLMs, with a specific focus on DeepSeek-R1-Distill models. We demonstrate that several reasoning behaviors exhibited by these models, including expressing uncertainty, backtracking, and example testing, can be effectively controlled using steering vectors. While our analysis does not claim to provide a complete taxonomy of reasoning mechanisms, it establishes a practical framework for steering specific aspects of model behavior.
Our key findings indicate that:

- •
Several reasoning behaviors in thinking models can be isolated to specific directions in the model’s activation space, enabling precise control through steering vectors.
- •
Our steering approach is effective across a diverse set of 500 tasks, demonstrating robust control over targeted reasoning behaviors.
- •
The method generalizes across different model architectures within the DeepSeek-R1-Distill family, showing consistent steering effects.

These results provide practical tools for modulating reasoning capabilities in thinking models. The ability to adjust specific aspects of the reasoning process through steering vectors opens new possibilities for adapting these models to different tasks and requirements.

Despite these promising results, our work has several limitations that suggest directions for future research. The current automated annotation process using GPT-4o, while efficient, has occasionally produced false positives and false negatives in identifying reasoning patterns. Future work should focus on developing more robust annotation methods, potentially incorporating multiple models or human validation to improve accuracy. Additionally, while our analysis centers on DeepSeek-R1-Distill models, the generalization of these findings to other models that have undergone reward-based RL training instead of being fine-tuned on thinking models (e.g., Qwen’s QwQ) remains an open question. Extending this research to a broader range of models would provide deeper insights into the universality of thinking mechanisms and their practical applications.
By addressing these limitations, future research can further advance the understanding and control of thinking models, paving the way for more reliable and adaptable AI systems.

## Acknowledgements

We would like to thank the ML Alignment & Theory Scholars (MATS) program for supporting this research, and in particular John Teichman and Cameron Holmes for being great research managers. We would also like to thank reviewers from the Workshop on Reasoning and Planning for Large Language Models at ICLR 2025 for extremely helpful feedback on early drafts of this paper.

## References

- Arditi et al. (2024a)
Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda.
Refusal in Language Models Is Mediated by a Single Direction.
*arXiv*, June 2024a.
doi: 10.48550/arXiv.2406.11717.
- Arditi et al. (2024b)
Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda.
Refusal in language models is mediated by a single direction.
*arXiv*, abs/2406.11717, 2024b.
URL [https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717).
- Chollet (2024)
François Chollet.
OpenAI o3 Breakthrough High Score on ARC-AGI-Pub, December 2024.
URL [https://arcprize.org/blog/oai-o3-pub-breakthrough](https://arcprize.org/blog/oai-o3-pub-breakthrough).
- DeepSeek-AI (2025)
DeepSeek-AI.
Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement ‘learning, 2025.
- Dutta et al. (2024)
Subhabrata Dutta, Joykirat Singh, Soumen Chakrabarti, and Tanmoy Chakraborty.
How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning, 2024.
URL [https://arxiv.org/abs/2402.18312](https://arxiv.org/abs/2402.18312).
- GDM (2024)
GDM.
Gemini flash thinking: Gemini 2.0 Flash Thinking Experimental, 2024.
URL [https://deepmind.google/technologies/gemini/flash-thinking/](https://deepmind.google/technologies/gemini/flash-thinking/).
- Hazra et al. (2025)
Dron Hazra, Max Loeffler, Murat Cubuktepe, Levon Avagyan, Liv Gorton, Mark Bissell, Owen Lewis, Thomas McGrath, and Daniel Balsam.
Under the hood of a reasoning model, 2025.
URL [https://www.goodfire.ai/blog/under-the-hood-of-a-reasoning-model](https://www.goodfire.ai/blog/under-the-hood-of-a-reasoning-model).
Accessed: 2025-06-22.
- Knoop (2025)
Mike Knoop.
R1-Zero and R1 Results and Analysis, January 2025.
URL [https://arcprize.org/blog/r1-zero-r1-results-analysis](https://arcprize.org/blog/r1-zero-r1-results-analysis).
- Li et al. (2023)
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg.
Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.
*arXiv*, June 2023.
doi: 10.48550/arXiv.2306.03341.
- Meng et al. (2022)
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov.
Locating and editing factual associations in gpt.
In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 17359–17372. Curran Associates, Inc., 2022.
URL [https://proceedings.neurips.cc/paper_files/paper/2022/file/6f1d43d5a82a37e89b0665b33bf3a182-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/6f1d43d5a82a37e89b0665b33bf3a182-Paper-Conference.pdf).
- Nanda (2023)
Neel Nanda.
Attribution patching: Activation patching at industrial scale.
2023.
https://www.neelnanda.io/mechanistic-interpretability/attribution-patching.
- Nye et al. (2021)
Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena.
Show your work: Scratchpads for intermediate computation with language models, 2021.
- OpenAI (2024)
OpenAI.
Learning to reason with LLMs, 9 2024.
URL [https://openai.com/index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/).
- Panickssery et al. (2023)
Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner.
Steering Llama 2 via Contrastive Activation Addition.
*arXiv*, December 2023.
doi: 10.48550/arXiv.2312.06681.
- Qwen Team (2024)
Qwen Team.
Qwq: Reflect deeply on the boundaries of the unknown, 11 2024.
URL [https://qwenlm.github.io/blog/qwq-32b-preview/](https://qwenlm.github.io/blog/qwq-32b-preview/).
- Reynolds & McDonell (2021)
Laria Reynolds and Kyle McDonell.
Prompt programming for large language models: Beyond the few-shot paradigm, 2021.
- Subramani et al. (2022)
Nishant Subramani, Nivedita Suresh, and Matthew E. Peters.
Extracting Latent Steering Vectors from Pretrained Language Models.
*arXiv*, May 2022.
doi: 10.48550/arXiv.2205.05124.
- Syed et al. (2023)
Aaquib Syed, Can Rager, and Arthur Conmy.
Attribution patching outperforms automated circuit discovery.
In *NeurIPS Workshop on Attributing Model Behavior at Scale*, 2023.
URL [https://openreview.net/forum?id=tiLbFR4bJW](https://openreview.net/forum?id=tiLbFR4bJW).
- Templeton et al. (2024)
Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan.
Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet.
*Transformer Circuits Thread*, 2024.
URL [https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).
- Turner et al. (2023)
Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid.
Steering Language Models With Activation Engineering.
*arXiv*, August 2023.
doi: 10.48550/arXiv.2308.10248.
- Turner et al. (2024)
Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid.
Activation addition: Steering language models without optimization.
*arXiv*, abs/2308.10248, 2024.
URL [https://arxiv.org/abs/2308.10248](https://arxiv.org/abs/2308.10248).
- Wei et al. (2022)
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou.
Chain-of-thought prompting elicits reasoning in large language models.
In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.
ISBN 9781713871088.
- Zhao et al. (2025)
Zekai Zhao, Qi Liu, Kun Zhou, Zihan Liu, Yifei Shao, Zhiting Hu, and Biwei Huang.
Activation control for efficiently eliciting long chain-of-thought ability of language models, 2025.
URL [https://arxiv.org/abs/2505.17697](https://arxiv.org/abs/2505.17697).
- Zou et al. (2023)
Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks.
Representation Engineering: A Top-Down Approach to AI Transparency.
*arXiv*, October 2023.
doi: 10.48550/arXiv.2310.01405.

## Appendix A Details on the Annotation Process

We use the following prompt to automatically annotate LLM responses:

## Appendix B Annotated Example

Riddle: What has cities, but no houses; forests, but no trees; and rivers, but no water?

DeepSeek R1 Response:

Automatically annotated response: (colored by assigned label)

## Appendix C Cosine Similarity Between Feature Vectors

To better understand the relationships between different reasoning behaviors in thinking models, we analyze the cosine similarity between the extracted steering vectors for different behavioral categories. This analysis provides insights into how distinct these reasoning mechanisms are in the model’s representational space and whether certain behaviors share similar underlying features.

We compute pairwise cosine similarities between the steering vectors for five key behavioral categories: initializing, backtracking, uncertainty-estimation, adding-knowledge, and deduction. For each pair of behavioral categories, we calculate the cosine similarity between their corresponding steering vectors at the layers identified as causally relevant through attribution patching ([Section 4.3](https://arxiv.org/html/2506.18167v4#S4.SS3)).

The cosine similarity between two feature vectors $\mathbf{u}^{c_{1}}$ and $\mathbf{u}^{c_{2}}$ is computed as:

$$ $\text{sim}(\mathbf{u}^{c_{1}},\mathbf{u}^{c_{2}})=\frac{\mathbf{u}^{c_{1}}\cdot\mathbf{u}^{c_{2}}}{|\mathbf{u}^{c_{1}}||\mathbf{u}^{c_{2}}|}$ $$

[Figure 5](https://arxiv.org/html/2506.18167v4#A3.F5) presents the cosine similarity heatmaps for both DeepSeek-R1-Distill models. Several key observations emerge from this analysis:

- 1.
Distinct reasoning mechanisms: Most behavioral categories show relatively low cosine similarities with each other, indicating that they correspond to distinct directions in the model’s activation space. This supports our hypothesis that different reasoning behaviors are mediated by separate linear directions.
- 2.
Uncertainty and backtracking correlation: Both models show moderate positive correlation between uncertainty-estimation and backtracking behaviors, which aligns with our intuition that models often express uncertainty before deciding to change their reasoning approach. Despite this cosine similarity overlap, our steering experiments (see [Section 4.5](https://arxiv.org/html/2506.18167v4#S4.SS5)) demonstrate that these represent fundamentally different mechanisms in the model’s reasoning process.
- 3.
Model-specific patterns: While the overall structure is similar across models, there are notable differences in the specific similarity values, suggesting that different model architectures may encode these reasoning behaviors with varying degrees of separation.

These results demonstrate that the extracted steering vectors capture meaningful and largely orthogonal directions in the model’s representational space, validating our approach for isolating specific reasoning mechanisms in thinking language models.

Figure: (a) DeepSeek-R1-Distill-Llama-8B
Refer to caption: x7.png

## Appendix D Cosine Similarity with Embedding and Unembedding Vectors

To better understand the relationship between our extracted steering vectors and the model’s embedding space, we analyze the cosine similarity between steering vectors and the model’s embedding and unembedding matrices across different layers. This analysis helps explain why certain layers are more effective for steering than others.

### D.1 Methodology

We compute the cosine similarity between each steering vector at each layer and the model’s embedding matrix (model.embed_tokens.weight) and unembedding matrix (model.lm_head.weight). For each steering vector $\mathbf{u}_{\ell}^{c}$ at layer $\ell$ and behavioral category $c$, we calculate:

$$ $\displaystyle\text{sim}_{\text{embed}}(\mathbf{u}_{\ell}^{c})$ $\displaystyle=\max_{i}\cos(\mathbf{u}_{\ell}^{c},\mathbf{E}_{i})$ (1) $\displaystyle\text{sim}_{\text{unembed}}(\mathbf{u}_{\ell}^{c})$ $\displaystyle=\max_{j}\cos(\mathbf{u}_{\ell}^{c},\mathbf{U}_{j})$ (2) $$

where $\mathbf{E}_{i}$ represents the $i$-th row of the embedding matrix and $\mathbf{U}_{j}$ represents the $j$-th row of the unembedding matrix.

### D.2 Results and Analysis

[Figure 6](https://arxiv.org/html/2506.18167v4#A4.F6) shows the layer-wise similarity patterns for both models. The analysis reveals distinct architectural differences:

DeepSeek-R1-Distill-Llama-8B: Shows high similarity with embedding vectors in the first layers, explaining the high KL divergence observed at the beginning of the attribution plots ([Figure 3](https://arxiv.org/html/2506.18167v4#S4.F3)). This suggests that Llama retains token representation information in early layers, making steering vectors in these layers too correlated with specific tokens to be effective for behavioral control.

DeepSeek-R1-Distill-Qwen-14B: Does not exhibit this pattern, with more uniform similarities across layers. This architectural difference allows for more flexibility in layer selection for steering.

These findings inform our layer selection strategy: we prioritize layers with high KL divergence in attribution plots while avoiding early layers that show excessive correlation with embedding tokens. This approach ensures that our steering vectors capture behavioral patterns rather than token-specific representations.

Figure: (a) Embedding similarities - Llama-8B
Refer to caption: x9.png

## Appendix E Steered Example (Adding Knowledge)

Task:

Original Response:

Positively Steered Response

Negatively Steered Response