## Contents
- 1 Introduction
- 2 Methodology
  - 2.1 Background
    - Transformers.
    - Chat models.
  - 2.2 Datasets and models
    - Datasets.
    - Models.
  - 2.3 Extracting a refusal direction
    - Difference-in-means.
    - Selecting a single vector.
  - 2.4 Model interventions
    - Activation addition.
    - Directional ablation.
  - 2.5 Evaluation of refusal and harmfulness
    - Refusal score.
    - Safety score.
- 3 Refusal is mediated by a single direction
  - 3.1 Bypassing refusal via directional ablation
  - 3.2 Inducing refusal via activation addition
- 4 A white-box jailbreak via weight orthogonalization
  - 4.1 Weight orthogonalization
  - 4.2 Comparison to other jailbreaks
  - 4.3 Measuring model coherence
- 5 Mechanistic analysis of adversarial suffixes
  - 5.1 Adversarial suffixes suppress the refusal-mediating direction
  - 5.2 Adversarial suffixes hijack the attention of important heads
- 6 Related work
  - Understanding refusal in language models.
  - Features as directions.
  - Undoing safety fine-tuning.
  - Jailbreaks.
- 7 Discussion
  - Limitations.
  - Ethical considerations.
- Acknowledgments and Disclosure of Funding
  - Author contributions.
  - Acknowledgements.
  - Tooling.
  - Disclosure of funding.
- References
- Appendix A Dataset details
  - A.1 Harmful instructions
  - A.2 Harmless instructions
- Appendix B Refusal metric: an efficient proxy for measuring refusal
- Appendix C Direction selection
  - C.1 Direction selection algorithm
  - C.2 Direction selection for each model
- Appendix D Refusal evaluation
  - D.1 Refusal score
  - D.2 Safety score
  - D.3 Challenges of evaluating refusal
  - D.4 Reporting of confidence intervals
- Appendix E Weight orthogonalization is equivalent to directional ablation
- Appendix F Jailbreak evaluation
  - F.1 Comparing to other jailbreaks
  - F.2 System prompts
- Appendix G Model coherence evaluation
  - G.1 Language model evaluation
  - G.2 TruthfulQA accuracy
  - G.3 CE loss evaluation
- Appendix H Adversarial suffix analysis
  - H.1 Adversarial suffix generation
  - H.2 Reporting of confidence intervals
- Appendix I Comparison to other methodologies
  - I.1 Comparison to activation addition
  - I.2 Comparison to fine-tuning
- Appendix J The “refusal direction” is also present in base models
- Appendix K Extended results
  - K.1 Bypassing refusal - examples
  - K.2 Inducing refusal - examples
- Appendix L Further experiments with orthogonalized models
  - L.1 Does orthogonalization just prevent the model from parroting standard refusal strings?
  - L.2 The orthogonalized model behaves similarly on harmless instructions
  - L.3 The orthogonalized model may have trouble understanding its new refusal behavior
- Appendix M Use of existing assets
  - M.1 Models
  - M.2 Datasets
- Appendix N Compute statement

## Abstract

Abstract Conversational large language models are fine-tuned for both instruction-following and safety, resulting in models that obey benign requests but refuse harmful ones.
While this refusal behavior is widespread across chat models, its underlying mechanisms remain poorly understood.
In this work, we show that refusal is mediated by a one-dimensional subspace, across 13 popular open-source chat models up to 72B parameters in size.
Specifically, for each model, we find a single direction such that erasing this direction from the model’s residual stream activations prevents it from refusing harmful instructions, while adding this direction elicits refusal on even harmless instructions.
Leveraging this insight, we propose a novel white-box jailbreak method that surgically disables refusal with minimal effect on other capabilities.
Finally, we mechanistically analyze how adversarial suffixes suppress propagation of the refusal-mediating direction.
Our findings underscore the brittleness of current safety fine-tuning methods.
More broadly, our work showcases how an understanding of model internals can be leveraged to develop practical methods for controlling model behavior. 2 2 2 Code available at https://github.com/andyrdt/refusal_direction .

## 1 Introduction

Deployed large language models (LLMs) undergo multiple rounds of fine-tuning to become both helpful and harmless: to provide helpful responses to innocuous user requests, but to refuse harmful or inappropriate ones (Bai et al., 2022).
Naturally, large numbers of users and researchers alike have attempted to circumvent these defenses using a wide array of jailbreak attacks (Wei et al., 2023; Xu et al., 2024; Chu et al., 2024) to
uncensor model outputs,
including fine-tuning techniques (Yang et al., 2023; Lermen et al., 2023; Zhan et al., 2023).
While the consequences of a successful attack on current chat assistants are modest, the scale and severity of harm from misuse could increase dramatically if frontier models are endowed with increased agency and autonomy (Anthropic, 2024).
That is, as models are deployed in higher-stakes settings and
are able to take actions in the real world, the ability to robustly refuse a request to cause harm is an essential requirement of a safe AI system. Inspired by the rapid progress of mechanistic interpretability (Nanda et al., 2023; Bricken et al., 2023; Marks et al., 2024; Templeton et al., 2024) and activation steering (Zou et al., 2023a; Turner et al., 2023; Rimsky et al., 2023), this work leverages the internal representations of chat models to better understand refusal.

It is widely hypothesized that LLMs represent features, or concepts, as linear directions in activation space (Mikolov et al., 2013; Bolukbasi et al., 2016; Elhage et al., 2022; Park et al., 2023b). Recent works have studied the linear representation of particular features such as
harmlessness (Zou et al., 2023a; Wolf et al., 2024; Zheng et al., 2024),
truth (Marks and Tegmark, 2023; Li et al., 2024),
humor (von Rütte et al., 2024),
sentiment (Tigges et al., 2023),
language (Bricken et al., 2023),
topic (Turner et al., 2023),
and many others.
Moreover, these feature directions have been shown to be effective causal mediators of behavior, enabling fine-grained steering of model outputs (Rimsky et al., 2023; Turner et al., 2023; Zou et al., 2023a; Templeton et al., 2024).

Figure: Figure 1: Ablating the “refusal direction” reduces refusal rates and elicits unsafe completions. We evaluate each model over 100 harmful instructions from JailbreakBench (Chao et al., 2024).
Refer to caption: https://ar5iv.labs.arxiv.org/html/2406.11717/assets/x1.png

In this work, we show that refusal is mediated by a one-dimensional subspace across 13 popular open-source chat models up to 72B parameters in size. Specifically, we use a small set of contrastive pairs (Burns et al., 2022; Zou et al., 2023a; Rimsky et al., 2023) of harmful and harmless instructions to identify a single difference-in-means direction (Rimsky et al., 2023; Marks and Tegmark, 2023; Belrose, 2023) that can be intervened upon to circumvent refusal on harmful prompts, or induce refusal on harmless prompts (§[3](#S3)).
We then use this insight to design a simple white-box jailbreak via an interpretable rank-one weight edit that
effectively disables refusal with minimal impact on other capabilities (§[4](#S4)). We conclude with a preliminary mechanistic investigation into how adversarial suffixes (Zou et al., 2023b), a popular prompt-based jailbreak technique, interfere with the propagation of the refusal direction across token positions (§[5](#S5)).

Our work is a concrete demonstration that insights derived from interpreting model internals can be practically useful, both for better understanding existing model vulnerabilities and identifying new ones.
Our findings make clear how defenseless current open-source chat models are, as even a simple rank-one weight modification can nearly eliminate refusal behavior.
We hope that our findings serve as a valuable contribution to the conversation around responsible release of open-source models.

## 2 Methodology

### 2.1 Background

#### Transformers.

Decoder-only transformers (Liu et al., 2018) map input tokens $\mathbf{t}=(t_{1},t_{2},\ldots,t_{n})\in\mathcal{V}^{n}$ to output probability distributions $\mathbf{y}=(\mathbf{y}_{1},\mathbf{y}_{2},\ldots,\mathbf{y}_{n})\in\mathbb{R}^{n\times|\mathcal{V}|}$. Let $\mathbf{x}_{i}^{(l)}(\mathbf{t})\in\mathbb{R}^{d_{\text{model}}}$ denote the residual stream activation of the token at position $i$ at the start of layer $l$.(^1^11We shorten $\mathbf{x}_{i}^{(l)}(\mathbf{t})$ to $\mathbf{x}_{i}^{(l)}$ when the input $\mathbf{t}$ is clear from context or unimportant.) Each token’s residual stream is initialized to its embedding $\mathbf{x}_{i}^{(1)}=\mathtt{Embed}(t_{i})$, and then undergoes a series of transformations across $L$ layers. Each layer’s transformation includes contributions from attention and MLP components:

$$ $\displaystyle\tilde{\mathbf{x}}_{i}^{(l)}=\mathbf{x}_{i}^{(l)}+\mathtt{Attn}^{(l)}(\mathbf{x}_{1:i}^{(l)}),\quad\mathbf{x}_{i}^{(l+1)}=\tilde{\mathbf{x}}_{i}^{(l)}+\mathtt{MLP}^{(l)}(\tilde{\mathbf{x}}_{i}^{(l)}).$ (1) $$

The final logits $\mathtt{logits}_{i}=\mathtt{Unembed}(\mathbf{x}_{i}^{(L+1)})\in\mathbb{R}^{|\mathcal{V}|}$ are then transformed into probabilities over output tokens $\mathbf{y}_{i}=\mathtt{softmax}(\mathtt{logits}_{i})\in\mathbb{R}^{|\mathcal{V}|}$.(^2^22This high-level description omits details such as positional embeddings and layer normalization.)

#### Chat models.

Chat models are fine-tuned for instruction-following and dialogue (Ouyang et al., 2022; Touvron et al., 2023).
These models use *chat templates* to structure user queries.
Typically, a chat template takes the form <user>{instruction}<end_user><assistant>.
We use *post-instruction tokens* to refer to all template tokens after the instruction, and denote the set of positional indices corresponding to these post-instruction tokens as $I$.
Our analysis focuses on activations in this region to understand how the model formulates its response.

### 2.2 Datasets and models

#### Datasets.

We construct two datasets: $\mathcal{D}_{\text{harmful}}$, a dataset of harmful instructions drawn from AdvBench (Zou et al., 2023b), MaliciousInstruct (Huang et al., 2023), TDC2023 (Mazeika et al., 2024, 2023), and HarmBench (Mazeika et al., 2024); and $\mathcal{D}_{\text{harmless}}$, a dataset of harmless instructions sampled from Alpaca (Taori et al., 2023). Each dataset consists of train and validation splits of 128 and 32 samples, respectively.
We apply filtering to ensure that the train and validation splits do not overlap with the evaluation datasets used in §[3](#S3) and §[4](#S4). See §[A](#A1) for further details about the datasets, including representative examples.

#### Models.

To assess the generality of our findings, we study a diverse set of safety fine-tuned models, spanning 1.8 to 72 billion parameters in size. We consider both models aligned by preference optimization (APO) and aligned by fine-tuning (AFT) (Meade et al., 2024).
All models included in the study are specified in [Table 1](#S2.T1).(^3^33Unless explicitly stated otherwise, all models examined in this study are chat models. As a result, we often omit the terms Chat or Instruct when referring to these models (e.g. we often abbreviate “Qwen 1.8B Chat” as “Qwen 1.8B”).)

**Table 1: Model families, sizes, alignment training type, and references.**
| Model family | Sizes | Alignment type | Reference |
| --- | --- | --- | --- |
| Qwen Chat | 1.8B, 7B, 14B, 72B | AFT | Bai et al. (2023) |
| Yi Chat | 6B, 34B | AFT | Young et al. (2024) |
| Gemma It | 2B, 7B | APO | Team et al. (2024) |
| Llama-2 Chat | 7B, 13B, 70B | APO | Touvron et al. (2023) |
| Llama-3 Instruct | 8B, 70B | APO | AI@Meta (2024) |

### 2.3 Extracting a refusal direction

#### Difference-in-means.

To identify the “refusal direction” in the model’s residual stream activations, we compute the difference between the model’s mean activations when run on harmful and harmless instructions.
This technique, known as *difference-in-means* (Belrose, 2023), effectively isolates key feature directions, as demonstrated in prior work (Rimsky et al., 2023; Marks and Tegmark, 2023; Tigges et al., 2023). For each layer $l\in[L]$ and post-instruction token position $i\in I$, we calculate the mean activation $\boldsymbol{\upmu}_{i}^{(l)}$ for harmful prompts from $\mathcal{D}_{\text{harmful}}^{\text{(train)}}$ and $\boldsymbol{\upnu}_{i}^{(l)}$ for harmless prompts from $\mathcal{D}_{\text{harmless}}^{\text{(train)}}$:

$$ $\displaystyle\boldsymbol{\upmu}_{i}^{(l)}=\frac{1}{|\mathcal{D}_{\text{harmful}}^{\text{(train)}}|}\sum_{\mathbf{t}\in\mathcal{D}_{\text{harmful}}^{\text{(train)}}}\mathbf{x}_{i}^{(l)}(\mathbf{t}),\quad\boldsymbol{\upnu}_{i}^{(l)}=\frac{1}{\lvert\mathcal{D}_{\text{harmless}}^{\text{(train)}}\rvert}\sum_{\mathbf{t}\in\mathcal{D}_{\text{harmless}}^{\text{(train)}}}\mathbf{x}_{i}^{(l)}(\mathbf{t}).$ (2) $$

We then compute the difference-in-means vector $\mathbf{r}_{i}^{(l)}=\boldsymbol{\upmu}_{i}^{(l)}-\boldsymbol{\upnu}_{i}^{(l)}$.
Note that each such vector is meaningful in both (1) its direction, which describes the direction that mean harmful and harmless activations differ along, and (2) its magnitude, which quantifies the distance between mean harmful and harmless activations.

#### Selecting a single vector.

Computing the difference-in-means vector $\mathbf{r}_{i}^{(l)}$ for each post-instruction token position $i\in I$ and layer $l\in[L]$ yields a set of $|I|\times L$ candidate vectors.
We then select the single most effective vector $\mathbf{r}_{i^{*}}^{(l^{*})}$ from this set by evaluating
each candidate vector over validation sets $\mathcal{D}_{\text{harmful}}^{\text{(val)}}$ and $\mathcal{D}_{\text{harmless}}^{\text{(val)}}$.
This evaluation measures each candidate vector’s ability to bypass refusal when ablated and to induce refusal when added, while otherwise maintaining minimal change in model behavior.
A more detailed description of our selection algorithm is provided in §[C](#A3).
We notate the selected vector as $\mathbf{r}$, and its corresponding unit-norm vector as $\hat{\mathbf{r}}$.

### 2.4 Model interventions

#### Activation addition.

Given a difference-in-means vector $\mathbf{r}^{(l)}\in\mathbb{R}^{d_{\text{model}}}$ extracted from layer $l$, we can modulate the strength of the corresponding feature via simple linear interventions. Specifically, we can *add* the difference-in-means vector to the activations of a harmless input to shift them closer to the mean harmful activation, thereby inducing refusal:

$$ $\displaystyle\mathbf{x}^{(l)^{\prime}}\leftarrow\mathbf{x}^{(l)}+\mathbf{r}^{(l)}.$ (3) $$

Note that for activation addition, we intervene only at layer $l$, and across all token positions.

#### Directional ablation.

To investigate the role of a direction $\hat{\mathbf{r}}\in\mathbb{R}^{d_{\text{model}}}$ in the model’s computation, we can erase it from the model’s representations using *directional ablation*.
Directional ablation “zeroes out” the component along $\hat{\mathbf{r}}$ for every residual stream activation $\mathbf{x}\in\mathbb{R}^{d_{\text{model}}}$:

$$ $\displaystyle\mathbf{x}^{\prime}\leftarrow\mathbf{x}-\hat{\mathbf{r}}\hat{\mathbf{r}}^{\intercal}\mathbf{x}.$ (4) $$

We perform this operation at every activation $\mathbf{x}_{i}^{(l)}$ and $\tilde{\mathbf{x}}_{i}^{(l)}$, across all layers $l$ and all token positions $i$.
This effectively prevents the model from ever representing this direction in its residual stream.

### 2.5 Evaluation of refusal and harmfulness

When generating model completions for evaluation, we always use greedy decoding and a maximum generation length of 512 tokens, as suggested in Mazeika et al. (2024).
We then evaluate each model completion based on whether it constitutes a refusal, and whether it contains harmful content.
We separate these evaluations into two scores: a *refusal score* and a *safety score*.

#### Refusal score.

Refusals often contain characteristic phrases, such as "I’m sorry" or "As an AI". Following prior work (Liu et al., 2023; Zou et al., 2023b; Xu et al., 2023; Robey et al., 2023; Shah et al., 2023a; Lermen et al., 2023), we compile a set of these common “refusal substrings”.
If a model completion includes at least one such substring, it is classified as a refusal (refusal_score=1); otherwise, it is classified as a non-refusal (refusal_score=0).
The full set of refusal substrings is provided in §[D.1](#A4.SS1).

As has been previously noted (Qi et al., 2023; Huang et al., 2023; Meade et al., 2024; Shah et al., 2023a), this string-matching approach has limitations.
While effective at detecting memorized refusals, it does not assess whether the completion is coherent or contains harmful content.
To address these limitations, we use a complementary metric that evaluates the harmfulness of a completion.

#### Safety score.

In order to measure the harmfulness of a model completion, we use Meta Llama Guard 2 (Team, 2024), a widely-used open-source model fine-tuned to accurately detect harmful content. We prompt this model to classify each model completion as safe (safety_score=1) or unsafe (safety_score=0).
More details are provided in §[D.2](#A4.SS2).

## 3 Refusal is mediated by a single direction

For each model, we extract a single difference-in-means vector $\mathbf{r}$ via the methodology described in §[2.3](#S2.SS3).
We then show that this single direction is both necessary and sufficient for refusal.
In §[3.1](#S3.SS1), we show that ablating this direction $\hat{\mathbf{r}}$ effectively disables the model’s ability to refuse harmful requests.
In §[3.2](#S3.SS2), we show that adding $\mathbf{r}$ to the model’s activations induces refusal on harmless instructions.

### 3.1 Bypassing refusal via directional ablation

To bypass refusal, we perform directional ablation on the “refusal direction” $\hat{\mathbf{r}}$, ablating it from activations at all layers and all token positions.
With this intervention in place, we generate model completions over JailbreakBench (Chao et al., 2024), a dataset of 100 harmful instructions.

Results are shown in [Figure 1](#S1.F1).
Under no intervention, chat models refuse nearly all harmful requests, yielding high refusal and safety scores.
Ablating $\hat{\mathbf{r}}$ from the model’s residual stream activations, labeled as *directional ablation*, reduces refusal rates and elicits unsafe completions.

Figure: Figure 2: Ablation of the “refusal direction” can effectively bypass refusal on harmful instructions. This example is taken from Llama-3 70B Instruct. For more examples, see §[K.1](#A11.SS1).

### 3.2 Inducing refusal via activation addition

To induce refusal, we add the difference-in-means vector $\mathbf{r}$ to activations in layer $l^{*}$, the layer that the $\mathbf{r}$ was originally extracted from. We perform this intervention at all token positions. With this intervention in place, we generate model completions over 100 randomly sampled harmless instructions from Alpaca.

Results are shown in [Figure 3](#S3.F3).
Under no intervention, chat models typically do not refuse harmless instructions.
Adding $\mathbf{r}$ to the model’s residual stream activations, labeled as *activation addition*, results in the model refusing even harmless requests.

Figure: Figure 3: Adding the “refusal direction" induces refusal on 100 harmless instructions from Alpaca.
Refer to caption: https://ar5iv.labs.arxiv.org/html/2406.11717/assets/x2.png

Figure: Figure 4: Adding the “refusal direction” to residual stream activations can induce refusal on harmless instructions. This example is taken from Gemma 7B It. For more examples, see §[K.2](#A11.SS2).

## 4 A white-box jailbreak via weight orthogonalization

In this section, we propose a novel white-box jailbreak method through *weight orthogonalization*.
This technique directly modifies model weights to eliminate the representation of the refusal direction, resulting in a model that retains its original capabilities but no longer refuses harmful instructions.
This new approach offers a simpler way to jailbreak open-source models compared to prior methodologies involving fine-tuning (Lermen et al., 2023; Yang et al., 2023; Zhan et al., 2023), as it does not require gradient-based optimization nor any examples of harmful completions.

### 4.1 Weight orthogonalization

In §[2.4](#S2.SS4), we described *directional ablation* as an inference-time intervention that prevents the model from representing a direction $\hat{\mathbf{r}}$: during a forward pass, we zero out the $\hat{\mathbf{r}}$ component from every intermediate residual stream activation ([Equation 4](#S2.E4)).
We can equivalently implement this operation by directly modifying component weights to never write to the $\hat{\mathbf{r}}$ direction in the first place.
Specifically, we can take each matrix $W_{\text{out}}\in\mathbb{R}^{d_{\text{model}}\times d_{\text{input}}}$ that writes to the residual stream, and orthogonalize its column vectors with respect to $\hat{\mathbf{r}}$:

$$ $\displaystyle W_{\text{out}}^{\prime}\leftarrow W_{\text{out}}-\hat{\mathbf{r}}\hat{\mathbf{r}}^{\intercal}W_{\text{out}}.$ (5) $$

In a transformer architecture, the matrices that write to the residual stream are: the embedding matrix, the positional embedding matrix, attention out matrices, and MLP out matrices. Orthogonalizing all of these matrices, as well as any output biases, with respect to the direction $\hat{\mathbf{r}}$ effectively prevents the model from ever writing $\hat{\mathbf{r}}$ to its residual stream.

Note that this weight modification is equivalent to the previously described inference-time directional ablation, as shown explicitly in §[E](#A5).
Therefore, the performance of the inference-time intervention in bypassing refusal, presented in §[3.1](#S3.SS1), also exactly characterizes that of the direct weight modification.

### 4.2 Comparison to other jailbreaks

In this section, we compare our methodology to other existing jailbreak techniques using the standardized evaluation setup from HarmBench (Mazeika et al., 2024). Specifically, we generate completions over the HarmBench test set of 159 “standard behaviors”, and then use their provided classifier model to determine the attack success rate (ASR), which is the proportion of completions classified as successfully bypassing refusal.
We evaluate our weight orthogonalization method on models included in the HarmBench study, and report its ASR alongside those of alternative jailbreaks.
For brief descriptions of each alternative jailbreak, see §[F.1](#A6.SS1).

**Table 2: HarmBench attack success rate (ASR) across various jailbreaking methods. Our method is labeled as Ortho. The baseline “direct response” rate with no jailbreak applied is labeled as DR. We differentiate *general* jailbreaks, which are applied across all prompts generically, from *prompt-specific* jailbreaks, which are optimized for each prompt individually. All evaluations use the model’s default system prompt. We also report ASR without system prompt in blue.**
|  | General | Prompt-specific |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chat model | Ortho | GCG-M | GCG-T | Human | DR | GCG | AP | PAIR |
| Llama-2 7B | 22.6 (79.9) | 20.0 | 16.8 | 00.1 | 0.0 | 34.5 | 17.0 | 07.5 |
| Llama-2 13B | 06.9 (61.0) | 08.7 | 13.0 | 00.6 | 0.5 | 28.0 | 14.5 | 15.0 |
| Llama-2 70B | 04.4 (62.9) | 05.5 | 15.2 | 00.0 | 0.0 | 36.0 | 15.5 | 07.5 |
| Qwen 7B | 79.2 (74.8) | 73.3 | 48.4 | 28.4 | 7.0 | 79.5 | 67.0 | 58.0 |
| Qwen 14B | 84.3 (74.8) | 75.5 | 46.0 | 31.5 | 9.5 | 83.5 | 56.0 | 51.5 |
| Qwen 72B | 78.0 (79.2) | - | 36.6 | 42.2 | 8.5 | - | - | 54.5 |

[Table 2](#S4.T2) shows that our weight orthogonalization method, labeled as Ortho, fares well compared to other general jailbreak techniques.
Across the Qwen model family, our general method is even on par with prompt-specific jailbreak techniques like GCG (Zou et al., 2023b), which optimize jailbreaks for each prompt individually.

Note that HarmBench’s evaluation methodology specifies that each model’s default system prompt should be used during evaluation.
While this approach is sensible for assessing the robustness of black-box systems, it is less applicable for white-box scenarios where attackers have full access to the model and can easily exclude the system prompt.
Thus, we report ASR both with and without the system prompt.

We observe a significant discrepancy in ASR for the Llama-2 models when comparing results with and without the system prompt.
We speculate that our intervention, while disabling the model’s natural refusal mechanism, does not hinder its ability to follow instructions.
Hence, when the Llama-2 system prompt, which explicitly instructs the model to avoid harmful content, is included, the orthogonalized model is less likely to generate harmful completions.
In contrast, the Qwen system prompt is much shorter and lacks explicit guidance to avoid harmful content, which may contribute to the smaller discrepancy in ASR for these models.
For a more detailed discussion of our method’s sensitivity to system prompts, see §[F.2](#A6.SS2).

### 4.3 Measuring model coherence

A reasonable concern with any new jailbreak technique is that, in addition to circumventing refusal, it may also degrade the model’s overall quality (Souly et al., 2024).
However, qualitatively, we observe that models maintain their coherence after undergoing weight orthogonalization.
While §[3.1](#S3.SS1) and §[4.2](#S4.SS2) show that our method effectively bypasses refusal, in this subsection we quantitatively evaluate how the modification alters a model’s general capabilities.

**Table 3: Model evaluations. For each evaluation, we report the orthogonalized model’s performance, followed by the baseline model’s performance, followed by the absolute increase or decrease. We display the largest model from each model family. Full results are reported in §[G.1](#A7.SS1).**
| Chat model | MMLU | ARC | GSM8K | TruthfulQA |
| --- | --- | --- | --- | --- |
| Gemma 7B | 51.8 / 51.7 (+0.1) | 51.7 / 51.5 (+0.2) | 31.3 / 32.0 (-0.7) | 44.7 / 47.1 (-2.4) |
| Yi 34B | 73.5 / 74.9 (-1.4) | 65.6 / 64.9 (+0.7) | 65.5 / 65.0 (+0.5) | 51.9 / 55.4 (-3.5) |
| Llama-2 70B | 63.1 / 63.0 (+0.1) | 65.2 / 65.4 (-0.2) | 54.5 / 53.0 (+1.5) | 51.8 / 52.8 (-1.0) |
| Llama-3 70B | 79.8 / 79.9 (-0.1) | 71.5 / 71.8 (-0.3) | 90.8 / 91.2 (-0.4) | 59.5 / 61.8 (-2.3) |
| Qwen 72B | 76.5 / 77.2 (-0.7) | 67.2 / 67.6 (-0.4) | 76.3 / 75.5 (+0.8) | 55.0 / 56.4 (-1.4) |

For each model and its orthogonalized version, we run four common language model evaluations:
MMLU (Hendrycks et al., 2020),
ARC (Clark et al., 2018),
GSM8K (Cobbe et al., 2021),
and TruthfulQA (Lin et al., 2021).
All evaluations are run using LM Evaluation Harness (Gao et al., 2023), with settings consistent with Open LLM Leaderboard (Beeching et al., 2023).(^4^44 As of June 2024, Open LLM Leaderboard does not use chat templates in evaluation prompts, and we follow the same practice to remain consistent. Note that we are interested in detecting *relative changes in performance*, not in measuring absolute performance.)

[Table 3](#S4.T3) displays that, for MMLU, ARC, and GSM8K, orthogonalized models perform similarly to baseline models.
In §[G.1](#A7.SS1), we show that this holds across other models in our suite, with additional evaluations of WinoGrande (Sakaguchi et al., 2021) and TinyHellaSwag (Polo et al., 2024).
Except for Qwen 7B and Yi 34B, all evaluation metrics for orthogonalized models lie within 99% confidence intervals of original performance.

Interestingly, accuracy on TruthfulQA consistently drops for orthogonalized models.
This phenomenon is consistent with Yang et al. (2023), where it was observed that fine-tuning away safety guardrails results in decreased accuracy on TruthfulQA.
Examining specific questions in TruthfulQA reveals that the dataset veers close to the territory of refusal, with categories including “misinformation”, “stereotypes”, and “conspiracies”, and thus it may intuitively make sense that model behavior differs meaningfully on this evaluation dataset.
See §[G.2](#A7.SS2) for further discussion of TruthfulQA performance.

In addition to standard language model evaluations, we also evaluate differences in CE loss, both on standard text corpora and model-specific generations (§[G.3](#A7.SS3)).
These loss metrics suggest that directional ablation is more surgical than activation addition based methods (§[I.1](#A9.SS1)).

## 5 Mechanistic analysis of adversarial suffixes

Safety fine-tuned chat models are vulnerable to *adversarial suffix* attacks (Zou et al., 2023b): there exist carefully constructed strings such that appending these strings to the end of a harmful instruction bypasses refusal and elicits harmful content.
Effective adversarial suffixes are usually not human interpretable, and the mechanisms by which they work are not well understood.
In this section, we mechanistically analyze the effect of an adversarial suffix on Qwen 1.8B Chat.

### 5.1 Adversarial suffixes suppress the refusal-mediating direction

Figure: Figure 5: Cosine similarity between last token residual stream activations and refusal direction.
Refer to caption: https://ar5iv.labs.arxiv.org/html/2406.11717/assets/x3.png

We first identify a single adversarial suffix that effectively bypasses refusal in Qwen 1.8B Chat.
The suffix is displayed in §[H](#A8), along with details of its generation.
To study the effect of this adversarial suffix, we sample 128 refusal-eliciting harmful instructions from JailbreakBench and the HarmBench test set. For each instruction, we run the model three times: first with the unedited instruction, second with the adversarial suffix appended, and third with a freshly-sampled random suffix of the same length appended.
By comparing the adversarial suffix to random suffixes, we aim to control for the effect of appending any suffix at all.
For each run, we cache the last token activations and visualize their cosine similarity with the refusal-mediating direction.
We also compare to a baseline of 128 harmless instructions from Alpaca that do not elicit refusal. [Figure 5](#S5.F5) shows that the expression of the refusal direction is very high for harmful instructions, and remains high when a random suffix is appended. The expression of the refusal direction after appending the adversarial suffix is heavily suppressed, and closely resembles that of harmless instructions.

### 5.2 Adversarial suffixes hijack the attention of important heads

Figure: (a) Attention head outputs at last token position, projected onto refusal direction.
Refer to caption: https://ar5iv.labs.arxiv.org/html/2406.11717/assets/x4.png

To further investigate how the refusal direction is suppressed, we examine the contributions of individual attention head and MLP components to the refusal direction.
We quantify each component’s contribution to this direction using *direct feature attribution* (DFA) (Makelov et al., 2024; Kissane et al., 2024): each component’s direct contribution can be measured by projecting its output onto the refusal direction.
We select the top eight attention heads with the highest DFA on harmful instructions, and then investigate how their behavior changes when suffixes are appended.
[Figure 6(a)](#S5.F6.sf1) shows that the direct contributions of these heads to the refusal direction are significantly suppressed when the adversarial suffix is appended, as compared with no suffix and random suffixes.

To understand how the outputs of these attention heads are altered, we examine their attention patterns.
[Figure 6(b)](#S5.F6.sf2) illustrates that the adversarial suffix effectively “hijacks” the attention of these heads. Normally, these heads focus on the instruction region of the prompt, which contains harmful content. With the adversarial suffix appended, these heads shift their attention to the suffix region, and away from the harmful instruction.

## 6 Related work

#### Understanding refusal in language models.

Wei et al. (2024) demonstrate that removing a set of safety-critical neurons and ranks can degrade safety mechanisms while preserving utility.
Zheng et al. (2024) and Zou et al. (2023a) both use contrastive pairs of harmful and harmless inputs to identify the model’s representation of harmfulness, asserting that this direction is distinct from the model’s representation of refusal.
Zheng et al. (2024) argue this by showing that safety prompts shift activations in a distinct direction, while Zou et al. (2023a) show that the representation is not significantly altered by adversarial suffixes.
Note that this is *in contrast* to our findings in §[5.1](#S5.SS1) that the “refusal direction” is significantly *suppressed* in the presence of an adversarial suffix.
Rimsky et al. (2023) use contrastive multiple-choice completions, finding that steering with the resulting vector is effective at modulating refusal in multiple-choice settings but not in long-form generation.

#### Features as directions.

Extracting feature directions from contrastive pairs of inputs is an established technique (Burns et al., 2022; Rimsky et al., 2023; Zou et al., 2023a).
It is well understood that adding feature vectors to the residual stream can modify behavior (Turner et al., 2023; Zou et al., 2023a; Tigges et al., 2023; Rimsky et al., 2023; Marks and Tegmark, 2023; Li et al., 2024), although details on how and where to intervene vary (Jorgensen et al., 2023; von Rütte et al., 2024).

Various works show that
directions in activation space have more “feature-like” properties than neurons do (Elhage et al., 2022; Mikolov et al., 2013; Park et al., 2023b; Li et al., 2021; Geiger et al., 2024; Hernandez and Andreas, 2021; Nanda et al., 2023; Bolukbasi et al., 2016).
Recent works use sparse autoencoders to discover feature directions in an unsupervised manner
(Bricken et al., 2023; Cunningham et al., 2023; Templeton et al., 2024).
The assumption that features are represented linearly has been effective for erasing concepts from language models (Belrose et al., 2024; Belrose, 2023; Ravfogel et al., 2020; Guerner et al., 2023; Haghighatkhah et al., 2022; Shao et al., 2022).

#### Undoing safety fine-tuning.

It is well known that
fine-tuning on malicious examples is sufficient to undo safety guardrails (Lermen et al., 2023), even with minimal degradation of overall capabilities (Yang et al., 2023; Zhan et al., 2023).
Undoing refusal via fine-tuning requires examples of harmful instructions and completions, while our method requires only harmful instructions.
Note however that fine-tuning can weaken safety guardrails even when data is benign (Qi et al., 2023; Pelrine et al., 2023).
Mechanistic interpretability works have provided initial evidence suggesting that fine-tuning does not significantly alter relevant internal circuitry (Lee et al., 2024; Prakash et al., 2024; Jain et al., 2023).
For example, Lee et al. (2024) fine-tune a model to make it less toxic, and find this behavioral modification can be undone simply by scaling a small number of MLP weights.

#### Jailbreaks.

The research area of circumventing restrictions on LLM behavior by *modifying the input* has seen many different directions of work.
Many models are vulnerable to *social engineering attacks*
(Wei et al., 2023; Shah et al., 2023b; Perez et al., 2022).
One hypothesis for why this works is that such prompts modify the LLM assistant’s "persona"
(Andreas, 2022; Shanahan et al., 2023; Park et al., 2023a).
Preliminary experiments in §[L](#A12) suggest that our method does not change the model’s chat personality or behavior outside of refusal.

Optimized *adversarial suffixes* (Zou et al., 2023b; Andriushchenko et al., 2024; Liao and Sun, 2024) can be appended to prompts to bypass refusal.
In contrast, our method does not require any modifications to the input prompt, but has the obvious limitation that we require access to the model’s weights.
However, note that transferability of jailbreak prompts optimized on open-weight models on black-box models is unclear (Meade et al., 2024).
Jailbreak prompts may have significant impact on model performance (Souly et al., 2024), whereas our method does not (§[4.3](#S4.SS3)).

## 7 Discussion

In this work, we demonstrate that refusal behavior is consistently mediated by a single direction across a diverse set of open-source chat models.
Based on this understanding, we propose a simple yet effective white-box jailbreak method that directly modifies model weights to disable the refusal mechanism while retaining model coherence. Our work demonstrates the practical utility of model-internals based interpretability:
by studying refusal through the lens of model internals, we were able to create a simple yet effective jailbreak method.
The simplicity of the model’s refusal mechanism, and the ease of circumventing it in the white-box setting, raise concerns about the robustness of current alignment techniques.

#### Limitations.

Our study has several limitations. First, while we evaluate a broad range of open-source models, our findings may not generalize to untested models, especially those at greater scale, including current state-of-the-art proprietary models and those developed in the future.
Second, the methodology we used to extract the “refusal direction” is likely not optimal and relies on several heuristics.
We see this paper as more of an existence proof that such a direction exists, rather than a careful study of how best to extract it, and we leave methodological improvements to future work.
Third, our analysis of adversarial suffixes does not provide a comprehensive mechanistic understanding of the phenomenon, and is restricted to a single model and a single adversarial example.
Fourth, it is difficult to measure the coherence of a chat model, and we consider each metric used flawed in various ways. We use multiple varied metrics to give a broad view of coherence, and welcome more rigorous analysis in future work.

#### Ethical considerations.

Any work on jailbreaking LLMs must ask the question of whether it enables novel harms.
It is already widely known that open-source model weights can be jailbroken via fine-tuning.
Our method, which can yield a jailbroken version of a 70B parameter model using less than $5 of compute, is simpler than previous fine-tuning methods, requiring neither gradient-based optimization nor a dataset of harmful completions.
While we acknowledge that our methodology marginally lowers the bar for jailbreaking open-source model weights, we believe that it does not substantially alter the risk profile of open sourcing models.

Although the risk of misuse posed by today’s language models may be relatively low (Anthropic, 2024; Mouton et al., 2024), the rapid advancement of state-of-the-art model capabilities suggests that this risk could become significant in the near future.
Our work contributes to the growing body of literature that highlights the fragility of current safety mechanisms, demonstrating that they can easily be circumvented and are insufficient to prevent the misuse of open-source LLMs.
Building a scientific consensus around the limitations of current safety techniques is crucial for informing future policy decisions and research efforts.

## Acknowledgments and Disclosure of Funding

#### Author contributions.

AA led the research project, and led the writing of the paper.
AA discovered and validated that ablating a single direction bypasses refusal, and came up with the weight orthogonalization trick.
OO ran initial experiments identifying that it is possible to jailbreak models via activation addition.
AA and OO implemented and ran all experiments presented in the paper, with DP helping to run model coherence evaluations.
DP investigated behavior of the orthogonalized models, suggested more thorough evaluations, and assisted with the writing of the paper.
AS ran initial experiments testing the causal efficacy of various directional interventions, and identified the suffix used in §[5](#S5) as universal for Qwen 1.8B Chat.
NR first proposed the idea of trying to extract a linear refusal direction (Rimsky, 2023), and advised the initial project to mechanistically understand refusal in Llama-2 7B Chat (Arditi and Obeso, 2023).
WG advised on methodology and experiments, and assisted with the writing and framing of the paper.
NN acted as primary supervisor for the project, providing guidance and feedback throughout.

#### Acknowledgements.

AA and OO began working on the project as part of the Supervised Program for Alignment Research (SPAR) program, mentored by NR.
AA and AS continued working on the project as part of the ML Alignment & Theory Scholars (MATS) program, mentored by NN.

For support throughout the research process, we thank McKenna Fitzgerald, Rocket Drew, Matthew Wearden, Henry Sleight, and the rest of the MATS team, and also Arthur Conmy.
We also thank the staff at Lighthaven and London Initiative for Safe AI (LISA) for cultivating great environments in which to conduct research.
We thank Florian Tramèr for generous help with compute resources, and for his comments on an earlier draft.

#### Tooling.

For our exploratory research, we used TransformerLens (Nanda and Bloom, 2022). For our experimental pipeline, we use HuggingFace Transformers (Wolf et al., 2020), PyTorch (Paszke et al., 2019), and vLLM (Kwon et al., 2023).
We used Together AI remote inference to compute the safety_score metric quickly.

#### Disclosure of funding.

AA and OO are funded by Long-Term Future Fund (LTFF).
AA and AS are funded by AI Safety Support (AISS).

## References

- AI@Meta (2024)
AI@Meta.
Llama 3 model card.
2024.
URL [https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md).
- Andreas (2022)
Jacob Andreas.
Language models as agent models.
*arXiv preprint arXiv:2212.01681*, 2022.
- Andriushchenko et al. (2024)
Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion.
Jailbreaking leading safety-aligned LLMs with simple adaptive attacks.
*arXiv preprint arXiv:2404.02151*, 2024.
- Anthropic (2024)
Anthropic.
Anthropic’s responsible scaling policy, 2024.
[https://www.anthropic.com/news/anthropics-responsible-scaling-policy](https://www.anthropic.com/news/anthropics-responsible-scaling-policy). Accessed on: May 20, 2024.
- Arditi and Obeso (2023)
Andy Arditi and Oscar Obeso.
Refusal mechanisms: initial experiments with Llama-2-7b-chat.
Alignment Forum, 2023.
URL [https://www.alignmentforum.org/posts/pYcEhoAoPfHhgJ8YC/refusal-mechanisms-initial-experiments-with-llama-2-7b-chat](https://www.alignmentforum.org/posts/pYcEhoAoPfHhgJ8YC/refusal-mechanisms-initial-experiments-with-llama-2-7b-chat).
- Bai et al. (2023)
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al.
Qwen technical report.
*arXiv preprint arXiv:2309.16609*, 2023.
- Bai et al. (2022)
Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al.
Training a helpful and harmless assistant with reinforcement learning from human feedback.
*arXiv preprint arXiv:2204.05862*, 2022.
- Beeching et al. (2023)
Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf.
Open LLM leaderboard.
[https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), 2023.
- Belrose (2023)
Nora Belrose.
Diff-in-means concept editing is worst-case optimal: Explaining a result by Sam Marks and Max Tegmark, 2023.
[https://blog.eleuther.ai/diff-in-means/](https://blog.eleuther.ai/diff-in-means/). Accessed on: May 20, 2024.
- Belrose et al. (2024)
Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman.
LEACE: Perfect linear concept erasure in closed form.
*Advances in Neural Information Processing Systems*, 36, 2024.
- Biderman et al. (2024)
Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, Leo Gao, Jonathan Tow, Baber Abbasi, Alham Fikri Aji, Pawan Sasanka Ammanamanchi, Sidney Black, Jordan Clive, et al.
Lessons from the trenches on reproducible evaluation of language models.
*arXiv preprint arXiv:2405.14782*, 2024.
- Bolukbasi et al. (2016)
Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai.
Man is to computer programmer as woman is to homemaker? Debiasing word embeddings.
*Advances in neural information processing systems*, 29, 2016.
- Bricken et al. (2023)
Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah.
Towards monosemanticity: Decomposing language models with dictionary learning.
*Transformer Circuits Thread*, 2023.
[https://transformer-circuits.pub/2023/monosemantic-features/index.html](https://transformer-circuits.pub/2023/monosemantic-features/index.html).
- Burns et al. (2022)
Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt.
Discovering latent knowledge in language models without supervision.
*arXiv preprint arXiv:2212.03827*, 2022.
- Chao et al. (2023)
Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong.
Jailbreaking black box large language models in twenty queries.
*arXiv preprint arXiv:2310.08419*, 2023.
- Chao et al. (2024)
Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J Pappas, Florian Tramer, et al.
JailbreakBench: An open robustness benchmark for jailbreaking large language models.
*arXiv preprint arXiv:2404.01318*, 2024.
- Chu et al. (2024)
Junjie Chu, Yugeng Liu, Ziqing Yang, Xinyue Shen, Michael Backes, and Yang Zhang.
Comprehensive assessment of jailbreak attacks against LLMs.
*arXiv preprint arXiv:2402.05668*, 2024.
- Clark et al. (2018)
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.
Think you have solved question answering? Try ARC, the AI2 reasoning challenge.
*arXiv preprint arXiv:1803.05457*, 2018.
- Cobbe et al. (2021)
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al.
Training verifiers to solve math word problems.
*arXiv preprint arXiv:2110.14168*, 2021.
- Cunningham et al. (2023)
Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey.
Sparse autoencoders find highly interpretable features in language models.
*arXiv preprint arXiv:2309.08600*, 2023.
- Elhage et al. (2022)
Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah.
Toy models of superposition.
*Transformer Circuits Thread*, 2022.
[https://transformer-circuits.pub/2022/toy_model/index.html](https://transformer-circuits.pub/2022/toy_model/index.html).
- Gao et al. (2020)
Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al.
The Pile: An 800GB dataset of diverse text for language modeling.
*arXiv preprint arXiv:2101.00027*, 2020.
- Gao et al. (2023)
Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou.
A framework for few-shot language model evaluation, 12 2023.
URL [https://zenodo.org/records/10256836](https://zenodo.org/records/10256836).
- Geiger et al. (2024)
Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, and Noah Goodman.
Finding alignments between interpretable causal variables and distributed neural representations.
In *Causal Learning and Reasoning*, pages 160–187. PMLR, 2024.
- Guerner et al. (2023)
Clément Guerner, Anej Svete, Tianyu Liu, Alexander Warstadt, and Ryan Cotterell.
A geometric notion of causal probing.
*arXiv preprint arXiv:2307.15054*, 2023.
- Haghighatkhah et al. (2022)
Pantea Haghighatkhah, Antske Fokkens, Pia Sommerauer, Bettina Speckmann, and Kevin Verbeek.
Better hit the nail on the head than beat around the bush: Removing protected attributes with a single projection.
*arXiv preprint arXiv:2212.04273*, 2022.
- Hendrycks et al. (2020)
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
Measuring massive multitask language understanding.
*arXiv preprint arXiv:2009.03300*, 2020.
- Hernandez and Andreas (2021)
Evan Hernandez and Jacob Andreas.
The low-dimensional linear geometry of contextualized word representations.
*arXiv preprint arXiv:2105.07109*, 2021.
- Hu et al. (2021)
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
LoRA: Low-rank adaptation of large language models.
*arXiv preprint arXiv:2106.09685*, 2021.
- Huang et al. (2023)
Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai Li, and Danqi Chen.
Catastrophic jailbreak of open-source LLMs via exploiting generation.
*arXiv preprint arXiv:2310.06987*, 2023.
- Jain et al. (2023)
Samyak Jain, Robert Kirk, Ekdeep Singh Lubana, Robert P Dick, Hidenori Tanaka, Edward Grefenstette, Tim Rocktäschel, and David Scott Krueger.
Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks.
*arXiv preprint arXiv:2311.12786*, 2023.
- Jiang et al. (2023)
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al.
Mistral 7B.
*arXiv preprint arXiv:2310.06825*, 2023.
- Jorgensen et al. (2023)
Ole Jorgensen, Dylan Cope, Nandi Schoots, and Murray Shanahan.
Improving activation steering in language models with mean-centring.
*arXiv preprint arXiv:2312.03813*, 2023.
- Kissane et al. (2024)
Connor Kissane, Robert Krzyzanowski, Arthur Conmy, and Neel Nanda.
Sparse autoencoders work on attention layer outputs.
Alignment Forum, 2024.
URL [https://www.alignmentforum.org/posts/DtdzGwFh9dCfsekZZ](https://www.alignmentforum.org/posts/DtdzGwFh9dCfsekZZ).
- Kwon et al. (2023)
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica.
Efficient memory management for large language model serving with PagedAttention.
In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023.
- Lee et al. (2024)
Andrew Lee, Xiaoyan Bai, Itamar Pres, Martin Wattenberg, Jonathan K Kummerfeld, and Rada Mihalcea.
A mechanistic understanding of alignment algorithms: A case study on DPO and toxicity.
*arXiv preprint arXiv:2401.01967*, 2024.
- Lermen et al. (2023)
Simon Lermen, Charlie Rogers-Smith, and Jeffrey Ladish.
LoRA fine-tuning efficiently undoes safety training in Llama 2-Chat 70B.
*arXiv preprint arXiv:2310.20624*, 2023.
- Li et al. (2021)
Belinda Z Li, Maxwell Nye, and Jacob Andreas.
Implicit representations of meaning in neural language models.
*arXiv preprint arXiv:2106.00737*, 2021.
- Li et al. (2024)
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg.
Inference-time intervention: Eliciting truthful answers from a language model.
*Advances in Neural Information Processing Systems*, 36, 2024.
- Liao and Sun (2024)
Zeyi Liao and Huan Sun.
AmpleGCG: Learning a universal and transferable generative model of adversarial suffixes for jailbreaking both open and closed LLMs.
*arXiv preprint arXiv:2404.07921*, 2024.
- Lin et al. (2021)
Stephanie Lin, Jacob Hilton, and Owain Evans.
TruthfulQA: Measuring how models mimic human falsehoods.
*arXiv preprint arXiv:2109.07958*, 2021.
- Liu et al. (2018)
Peter J Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and Noam Shazeer.
Generating Wikipedia by summarizing long sequences.
*arXiv preprint arXiv:1801.10198*, 2018.
- Liu et al. (2023)
Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao.
AutoDAN: Generating stealthy jailbreak prompts on aligned large language models.
*arXiv preprint arXiv:2310.04451*, 2023.
- Makelov et al. (2024)
Aleksandar Makelov, George Lange, and Neel Nanda.
Towards principled evaluations of sparse autoencoders for interpretability and control.
*arXiv preprint arXiv:2405.08366*, 2024.
- Marks and Tegmark (2023)
Samuel Marks and Max Tegmark.
The geometry of truth: Emergent linear structure in large language model representations of true/false datasets.
*arXiv preprint arXiv:2310.06824*, 2023.
- Marks et al. (2024)
Samuel Marks, Can Rager, Eric J Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller.
Sparse feature circuits: Discovering and editing interpretable causal graphs in language models.
*arXiv preprint arXiv:2403.19647*, 2024.
- Mazeika et al. (2023)
Mantas Mazeika, Andy Zou, Norman Mu, Long Phan, Zifan Wang, Chunru Yu, Adam Khoja, Fengqing Jiang, Aidan O’Gara, Ellie Sakhaee, Zhen Xiang, Arezoo Rajabi, Dan Hendrycks, Radha Poovendran, Bo Li, and David Forsyth.
TDC 2023 (LLM edition): the Trojan Detection Challenge.
In *NeurIPS Competition Track*, 2023.
- Mazeika et al. (2024)
Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al.
HarmBench: A standardized evaluation framework for automated red teaming and robust refusal.
*arXiv preprint arXiv:2402.04249*, 2024.
- Meade et al. (2024)
Nicholas Meade, Arkil Patel, and Siva Reddy.
Universal adversarial triggers are not universal.
*arXiv preprint arXiv:2404.16020*, 2024.
- Mikolov et al. (2013)
Tomáš Mikolov, Wen-tau Yih, and Geoffrey Zweig.
Linguistic regularities in continuous space word representations.
In *Proceedings of the 2013 conference of the north american chapter of the association for computational linguistics: Human language technologies*, pages 746–751, 2013.
- Min et al. (2023)
Bonan Min, Hayley Ross, Elior Sulem, Amir Pouran Ben Veyseh, Thien Huu Nguyen, Oscar Sainz, Eneko Agirre, Ilana Heintz, and Dan Roth.
Recent advances in natural language processing via large pre-trained language models: A survey.
*ACM Computing Surveys*, 56(2):1–40, 2023.
- Mouton et al. (2024)
Christopher A. Mouton, Caleb Lucas, and Ella Guest.
*The Operational Risks of AI in Large-Scale Biological Attacks: Results of a Red-Team Study*.
RAND Corporation, Santa Monica, CA, 2024.
doi: 10.7249/RRA2977-2.
- Nanda and Bloom (2022)
Neel Nanda and Joseph Bloom.
TransformerLens.
[https://github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), 2022.
- Nanda et al. (2023)
Neel Nanda, Andrew Lee, and Martin Wattenberg.
Emergent linear representations in world models of self-supervised sequence models.
*arXiv preprint arXiv:2309.00941*, 2023.
- Ouyang et al. (2022)
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
Training language models to follow instructions with human feedback.
*Advances in neural information processing systems*, 35:27730–27744, 2022.
- Park et al. (2023a)
Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein.
Generative agents: Interactive simulacra of human behavior.
In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*, pages 1–22, 2023a.
- Park et al. (2023b)
Kiho Park, Yo Joong Choe, and Victor Veitch.
The linear representation hypothesis and the geometry of large language models.
*arXiv preprint arXiv:2311.03658*, 2023b.
- Paszke et al. (2019)
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al.
Pytorch: An imperative style, high-performance deep learning library.
*Advances in neural information processing systems*, 32, 2019.
- Pelrine et al. (2023)
Kellin Pelrine, Mohammad Taufeeque, Michał Zając, Euan McLean, and Adam Gleave.
Exploiting novel GPT-4 APIs.
*arXiv preprint arXiv:2312.14302*, 2023.
- Perez et al. (2022)
Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving.
Red teaming language models with language models.
*arXiv preprint arXiv:2202.03286*, 2022.
- Polo et al. (2024)
Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin.
tinyBenchmarks: evaluating LLMs with fewer examples.
*arXiv preprint arXiv:2402.14992*, 2024.
- Prakash et al. (2024)
Nikhil Prakash, Tamar Rott Shaham, Tal Haklay, Yonatan Belinkov, and David Bau.
Fine-tuning enhances existing mechanisms: A case study on entity tracking.
*arXiv preprint arXiv:2402.14811*, 2024.
- Qi et al. (2023)
Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson.
Fine-tuning aligned language models compromises safety, even when users do not intend to!
*arXiv preprint arXiv:2310.03693*, 2023.
- Ravfogel et al. (2020)
Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg.
Null it out: Guarding protected attributes by iterative nullspace projection.
*arXiv preprint arXiv:2004.07667*, 2020.
- Rimsky (2023)
Nina Rimsky.
Red-teaming language models via activation engineering.
Alignment Forum, 2023.
URL [https://www.alignmentforum.org/posts/iHmsJdxgMEWmAfNne/red-teaming-language-models-via-activation-engineering](https://www.alignmentforum.org/posts/iHmsJdxgMEWmAfNne/red-teaming-language-models-via-activation-engineering).
- Rimsky et al. (2023)
Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner.
Steering Llama 2 via contrastive activation addition.
*arXiv preprint arXiv:2312.06681*, 2023.
- Robey et al. (2023)
Alexander Robey, Eric Wong, Hamed Hassani, and George J Pappas.
SmoothLLM: Defending large language models against jailbreaking attacks.
*arXiv preprint arXiv:2310.03684*, 2023.
- Sakaguchi et al. (2021)
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi.
WinoGrande: An adversarial Winograd schema challenge at scale.
*Communications of the ACM*, 64(9):99–106, 2021.
- Shah et al. (2023a)
Muhammad Ahmed Shah, Roshan Sharma, Hira Dhamyal, Raphael Olivier, Ankit Shah, Dareen Alharthi, Hazim T Bukhari, Massa Baali, Soham Deshmukh, Michael Kuhlmann, et al.
LoFT: Local proxy fine-tuning for improving transferability of adversarial attacks against large language model.
*arXiv preprint arXiv:2310.04445*, 2023a.
- Shah et al. (2023b)
Rusheb Shah, Quentin Feuillade-Montixi, Soroush Pour, Arush Tagade, Stephen Casper, and Javier Rando.
Scalable and transferable black-box jailbreaks for language models via persona modulation, 2023b.
- Shanahan et al. (2023)
Murray Shanahan, Kyle McDonell, and Laria Reynolds.
Role play with large language models.
*Nature*, 623(7987):493–498, 2023.
- Shao et al. (2022)
Shun Shao, Yftah Ziser, and Shay B Cohen.
Gold doesn’t always glitter: Spectral removal of linear and nonlinear guarded attribute information.
*arXiv preprint arXiv:2203.07893*, 2022.
- Shen et al. (2023)
Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang.
"Do anything now": Characterizing and evaluating in-the-wild jailbreak prompts on large language models.
*arXiv preprint arXiv:2308.03825*, 2023.
- Shin et al. (2020)
Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh.
AutoPrompt: Eliciting knowledge from language models with automatically generated prompts.
In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Association for Computational Linguistics, 2020.
- Souly et al. (2024)
Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, et al.
A StrongREJECT for empty jailbreaks.
*arXiv preprint arXiv:2402.10260*, 2024.
- Taori et al. (2023)
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto.
Stanford Alpaca: An instruction-following LLaMA model.
[https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), 2023.
- Team et al. (2024)
Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al.
Gemma: Open models based on Gemini research and technology.
*arXiv preprint arXiv:2403.08295*, 2024.
- Team (2024)
Llama Team.
Meta Llama Guard 2.
[https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard2/MODEL_CARD.md](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard2/MODEL_CARD.md), 2024.
- Templeton et al. (2024)
Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan.
Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet.
*Transformer Circuits Thread*, 2024.
URL [https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).
- Tigges et al. (2023)
Curt Tigges, Oskar John Hollinsworth, Atticus Geiger, and Neel Nanda.
Linear representations of sentiment in large language models.
*arXiv preprint arXiv:2310.15154*, 2023.
- Touvron et al. (2023)
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.
Llama 2: Open foundation and fine-tuned chat models.
*arXiv preprint arXiv:2307.09288*, 2023.
- Turner et al. (2023)
Alex Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDiarmid.
Activation addition: Steering language models without optimization.
*arXiv preprint arXiv:2308.10248*, 2023.
- von Rütte et al. (2024)
Dimitri von Rütte, Sotiris Anagnostidis, Gregor Bachmann, and Thomas Hofmann.
A language model’s guide through latent space.
*arXiv preprint arXiv:2402.14433*, 2024.
- Wang et al. (2023)
Tony T Wang, Miles Wang, Kaivu Hariharan, and Nir Shavit.
Forbidden facts: An investigation of competing objectives in Llama-2.
*arXiv preprint arXiv:2312.08793*, 2023.
- Wei et al. (2023)
Alexander Wei, Nika Haghtalab, and Jacob Steinhardt.
Jailbroken: How does LLM safety training fail?
*arXiv preprint arXiv:2307.02483*, 2023.
- Wei et al. (2024)
Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, and Peter Henderson.
Assessing the brittleness of safety alignment via pruning and low-rank modifications.
*arXiv preprint arXiv:2402.05162*, 2024.
- Wolf et al. (2020)
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush.
Transformers: State-of-the-art natural language processing.
In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45, Online, October 2020. Association for Computational Linguistics.
URL [https://www.aclweb.org/anthology/2020.emnlp-demos.6](https://www.aclweb.org/anthology/2020.emnlp-demos.6).
- Wolf et al. (2024)
Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, and Amnon Shashua.
Tradeoffs between alignment and helpfulness in language models.
*arXiv preprint arXiv:2401.16332*, 2024.
- Xu et al. (2023)
Nan Xu, Fei Wang, Ben Zhou, Bang Zheng Li, Chaowei Xiao, and Muhao Chen.
Cognitive overload: Jailbreaking large language models with overloaded logical thinking.
*arXiv preprint arXiv:2311.09827*, 2023.
- Xu et al. (2024)
Zihao Xu, Yi Liu, Gelei Deng, Yuekang Li, and Stjepan Picek.
LLM jailbreak attack versus defense techniques – a comprehensive study.
*arXiv preprint arXiv:2402.13457*, 2024.
- Yang et al. (2023)
Xianjun Yang, Xiao Wang, Qi Zhang, Linda Petzold, William Yang Wang, Xun Zhao, and Dahua Lin.
Shadow alignment: The ease of subverting safely-aligned language models.
*arXiv preprint arXiv:2310.02949*, 2023.
- Young et al. (2024)
Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al.
Yi: Open foundation models by 01.AI.
*arXiv preprint arXiv:2403.04652*, 2024.
- Zellers et al. (2019)
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.
HellaSwag: Can a machine really finish your sentence?
*arXiv preprint arXiv:1905.07830*, 2019.
- Zhan et al. (2023)
Qiusi Zhan, Richard Fang, Rohan Bindu, Akul Gupta, Tatsunori Hashimoto, and Daniel Kang.
Removing RLHF protections in GPT-4 via fine-tuning.
*arXiv preprint arXiv:2311.05553*, 2023.
- Zheng et al. (2024)
Chujie Zheng, Fan Yin, Hao Zhou, Fandong Meng, Jie Zhou, Kai-Wei Chang, Minlie Huang, and Nanyun Peng.
Prompt-driven LLM safeguarding via directed representation optimization.
*arXiv preprint arXiv:2401.18018*, 2024.
- Zou et al. (2023a)
Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al.
Representation engineering: A top-down approach to AI transparency.
*arXiv preprint arXiv:2310.01405*, 2023a.
- Zou et al. (2023b)
Andy Zou, Zifan Wang, J Zico Kolter, and Matt Fredrikson.
Universal and transferable adversarial attacks on aligned language models.
*arXiv preprint arXiv:2307.15043*, 2023b.

## Appendix A Dataset details

### A.1 Harmful instructions

To construct $\mathcal{D}_{\text{harmful}}^{\text{(train)}}$, we randomly sample a total of 128 harmful instructions from AdvBench (Zou et al., 2023b), MaliciousInstruct (Huang et al., 2023), and TDC2023 (Mazeika et al., 2024, 2023).

To construct $\mathcal{D}_{\text{harmful}}^{\text{(val)}}$, we sample 32 instructions from the HarmBench validation set (Mazeika et al., 2024). We use only the “standard behaviors”, and exclude instructions that require context or concern copyright violations.

In §[3](#S3), we evaluate over JailbreakBench (Chao et al., 2024), a dataset containing 100 harmful instructions, spanning 10 categories: harassment/discrimination, malware/hacking, physical harm, economic harm, fraud/deception, disinformation, sexual/adult content, privacy, expert advice, government decision-making.

In §[4](#S4), we evaluate over the HarmBench test set (Mazeika et al., 2024). We consider only the 159 “standard behaviors”, and exclude instructions that require context or concern copyright violations.
These harmful instructions span 6 categories: cybercrime & unauthorized intrusion, chemical & biological weapons/drugs, misinformation & disinformation, harassment & bullying, illegal activities, general harm.

Note that we perform filtering to ensure that $\mathcal{D}_{\text{harmful}}^{\text{(train)}}$, $\mathcal{D}_{\text{harmful}}^{\text{(val)}}$, and the two evaluation datasets are all pairwise disjoint, containing no overlapping instructions.

Figure: Figure 7: A random sample of instructions from $\mathcal{D}_{\text{harmful}}^{\text{(train)}}$.

### A.2 Harmless instructions

To construct the harmless datasets, we sample instructions from Alpaca (Taori et al., 2023). $\mathcal{D}_{\text{harmless}}^{\text{(train)}}$ contains 128 instructions, and $\mathcal{D}_{\text{harmless}}^{\text{(val)}}$ contains 32 instructions.

In §[3](#S3), we evaluate over 100 instructions from Alpaca.

Note that $\mathcal{D}_{\text{harmless}}^{\text{(train)}}$, $\mathcal{D}_{\text{harmless}}^{\text{(val)}}$, and the evaluation dataset are all pairwise disjoint, containing no overlapping instructions.

Figure: Figure 8: A random sample of instructions from $\mathcal{D}_{\text{harmless}}^{\text{(train)}}$.

## Appendix B Refusal metric: an efficient proxy for measuring refusal

Evaluating whether a model refuses a particular instruction is most accurately done by generating a full completion using greedy decoding, and then assessing whether the generated text constitutes a refusal.
However, this process can be computationally expensive, especially when working with large models and a large number of instructions.
To address this, we define a more efficient proxy for estimating the likelihood of a model refusing a given instruction without requiring generation ([Figure 10](#A2.F10)).

We observe that each model tends to have a small set of characteristic phrases that it typically uses to begin its refusals ([Figure 9](#A2.F9)). This allows us to approximate the probability of refusal by examining the model’s next token probability distribution at the last token position, which corresponds to the start of its completion.

Formally, for each model, we define a set of refusal tokens $\mathcal{R}\subseteq\mathcal{V}$, which contains the tokens most likely to begin the model’s refusals ([Table 4](#A2.T4)).
We can then estimate the probability of refusal $P_{\text{refusal}}$ as the sum of the probabilities assigned to the tokens in $\mathcal{R}$.
Given a vector of next token probabilities $\mathbf{p}=(p_{1},p_{2},\ldots,p_{|\mathcal{V}|})\in\mathbb{R}^{|\mathcal{V}|}$, we define

$$ $\displaystyle P_{\text{refusal}}(\mathbf{p}):=\sum_{t\in\mathcal{R}}p_{t}.$ (6) $$

To create a more informative “refusal metric”, we take the log-odds of $P_{\text{refusal}}$. This transformation helps to better distinguish between extreme probabilities that are close to 0 or 1 (Wang et al., 2023).

$$ $\displaystyle\texttt{refusal_metric}(\mathbf{p})$ $\displaystyle:=\mathrm{logit}\left(P_{\text{refusal}}(\mathbf{p})\right)$ (7) $\displaystyle\ =\log\left(\frac{P_{\text{refusal}}(\mathbf{p})}{1-P_{\text{refusal}}(\mathbf{p})}\right)$ (8) $\displaystyle\ =\log\left(\sum_{t\in\mathcal{R}}p_{t}\right)-\log\left(\sum_{t\in\mathcal{V}\setminus\mathcal{R}}p_{t}\right).$ (9) $$

We use this metric to filter out instructions in our test and validation datasets: for harmful instructions, we filter out prompts yielding $\texttt{refusal_metric}<0$, and for harmless instruction, we filter out prompts yielding $\texttt{refusal_metric}>0$. We also use this metric to quickly evaluate the efficacy of interventions over the validation set (§[C](#A3)).

Figure: (a) Top-10 token probabilities for Gemma 2B It, averaged over 128 harmful instructions.
Refer to caption: https://ar5iv.labs.arxiv.org/html/2406.11717/assets/x6.png

Figure: Figure 10: The refusal metric separates harmful and harmless instructions for Gemma 2B It. Refusals generally begin with token $234285$ (corresponding to ‘I’). Setting $\mathcal{R}_{\textsc{Gemma}}=\{234285\}$ yields a refusal metric that is an efficient proxy for assessing whether the model will refuse.
Refer to caption: https://ar5iv.labs.arxiv.org/html/2406.11717/assets/x8.png

**Table 4: The refusal token set $\mathcal{R}$ that we use for each model family, along with the refusal phrases corresponding to each token.**
| Model family | Refusal token set $\mathcal{R}$ | Corresponding refusal phrases |
| --- | --- | --- |
| Qwen Chat | { $40,2121$ } | { "I’m sorry", "As an AI" } |
| Gemma It | { $235285$ } | { "I cannot" } |
| Yi Chat | { $59597$ } | { "I’m sorry" } |
| Llama-2 Chat | { $306$ } | { "I cannot" } |
| Llama-3 Instruct | { $40$ } | { "I cannot" } |

## Appendix C Direction selection

### C.1 Direction selection algorithm

Given a set of difference-in-means vectors $\{\mathbf{r}_{i}^{(l)}|i\in I,l\in\lceil L\rceil\}$, we want to select the best vector $\mathbf{r}_{i^{*}}^{(l^{*})}$. For each vector $\mathbf{r}_{i}^{(l)}$, we compute the following:

- •
bypass_score: under directional ablation of $\mathbf{r}_{i}^{(l)}$, compute the average refusal metric across $\mathcal{D}_{\text{harmful}}^{\text{(val)}}$.
- •
induce_score: under activation addition of $\mathbf{r}_{i}^{(l)}$, compute the average refusal metric across $\mathcal{D}_{\text{harmless}}^{\text{(val)}}$.
- •
kl_score: run the model on $\mathcal{D}_{\text{harmless}}^{\text{(val)}}$ with and without directional ablation of $\mathbf{r}_{i}^{(l)}$, and compute the average KL divergence between the probability distributions at the last token position.

We then select $\mathbf{r}_{i^{*}}^{(l^{*})}$ to be the direction with minimum bypass_score, subject to the following conditions:

- •
$\texttt{induce_score}>0$
–
This condition filters out directions that are not *sufficient* to induce refusal.
- •
$\texttt{kl_score}<0.1$
–
This condition filters out directions that significantly change model behavior on harmless prompts when ablated.
- •
$l<0.8L$
–
This condition ensures that the direction is not too close to the unembedding directions. Intuitively, one could disable refusal by preventing the model from writing to refusal unembed directions, e.g. directions corresponding to the ‘I’ or ‘As’ unembedding directions, and this would directly prevent the model from outputting these refusal tokens. However, we restrict our search to higher level features, and do not prevent the model from outputting specific tokens (see §[L.1](#A12.SS1)).

Using the compute setup described in §[N](#A14),
this direction selection procedure takes about an hour to run for the largest models (72 billion parameters), and faster for smaller models.

### C.2 Direction selection for each model

We report details of direction selection for each model in [Table 5](#A3.T5), including the token position $i^{*}$ and layer $l^{*}$ from which the direction was sourced, along with the direction’s corresponding metrics.

**Table 5: Direction selection details for each model. Note that $i^{*}=-1$ indicates that the direction is selected from the last token position, $i^{*}=-2$ the second-to-last token position, and so on. Also note that the layer index $l^{*}$ starts from index 0, while $L$ indicates the total number of layers.**
| Chat model | $i^{*}$ | $l^{*}/L$ | bypass_score | induce_score | kl_score |
| --- | --- | --- | --- | --- | --- |
| Qwen 1.8B | $-1$ | $15/24$ | $-4.415$ | $1.641$ | $0.077$ |
| Qwen 7B | $-1$ | $17/32$ | $-5.355$ | $1.107$ | $0.069$ |
| Qwen 14B | $-1$ | $23/40$ | $-5.085$ | $1.606$ | $0.014$ |
| Qwen 72B | $-1$ | $62/80$ | $-4.246$ | $1.885$ | $0.034$ |
| Yi 6B | $-5$ | $20/32$ | $-6.693$ | $1.968$ | $0.046$ |
| Yi 34B | $-1$ | $37/60$ | $-11.14$ | $1.865$ | $0.069$ |
| Gemma 2B | $-2$ | $10/18$ | $-14.435$ | $6.709$ | $0.067$ |
| Gemma 7B | $-1$ | $14/28$ | $-12.239$ | $6.851$ | $0.091$ |
| Llama-2 7B | $-1$ | $14/32$ | $-5.295$ | $5.941$ | $0.073$ |
| Llama-2 13B | $-1$ | $26/40$ | $-4.377$ | $2.794$ | $0.092$ |
| Llama-2 70B | $-1$ | $21/80$ | $-4.565$ | $5.191$ | $0.036$ |
| Llama-3 8B | $-5$ | $12/32$ | $-9.715$ | $7.681$ | $0.064$ |
| Llama-3 70B | $-5$ | $25/80$ | $-7.839$ | $0.126$ | $0.021$ |

## Appendix D Refusal evaluation

### D.1 Refusal score

For a model completion $\mathbf{c}$, $\texttt{refusal_score}(\mathbf{c})=1$ if $\mathbf{c}$ contains at least one “refusal substring”; otherwise $\texttt{refusal_score}(\mathbf{c})=0$.

[Figure 11](#A4.F11) displays the “refusal substrings” that are used in the refusal_score evaluation.
Note that the substring comparison is not sensitive to capitalization, and checks that the phrases are contained *anywhere* in the completion, not just at the start.

Figure: Figure 11: The set of “refusal substrings”, used to evaluate the refusal_score of a given completion.

### D.2 Safety score

For a model completion $\mathbf{c}$, $\texttt{safety_score}(\mathbf{c})=1$ if Meta Llama Guard 2 (Team, 2024) outputs that the completion is safe; otherwise $\texttt{safety_score}(\mathbf{c})=0$.

[Figure 12](#A4.F12) displays the template used to prompt Meta Llama Guard 2.
Note that we use the default prompt specified by the model providers, without any custom alteration.

Figure: Figure 12: The default prompt template for Meta Llama Guard 2, used to evaluate the safety_score of a given completion.

### D.3 Challenges of evaluating refusal

Assessing whether a completion constitutes a successful jailbreak is complex.
In this subsection, we highlight scenarios that are ambiguous, motivating our use of two metrics.

[Figure 13](#A4.F13) and [Figure 14](#A4.F14) display cases in which the model does not explicitly refuse, but also does not provide a harmful response.
In these scenarios, refusal_score=0 while safety_score=1.

[Figure 15](#A4.F15) displays a case in which the model initially refuses, but then goes on to give a harmful response.
In these scenarios, refusal_score=1 while safety_score=0.

### D.4 Reporting of confidence intervals

In [Figure 1](#S1.F1) and [Figure 3](#S3.F3), we display error bars corresponding to standard error (SE), computed as $\text{SE}=\sqrt{\frac{p(1-p)}{n}}$.
In both cases, $n=100$.

Figure: Figure 13: *Challenges of evaluating refusal*. The model completion does not explicitly use a refusal phrase (refusal_score=0), but it does not contain harmful content (safety_score=1). This completion is taken from the orthogonalized Llama-3 70B Chat model.

Figure: Figure 14: *Challenges of evaluating refusal*. The model completion does not use a refusal phrase (refusal_score=0), but it does not contain harmful content (safety_score=1). This completion is taken from the orthogonalized Gemma 7B It model.

Figure: Figure 15: *Challenges of evaluating refusal*. The model completion does use a refusal phrase (refusal_score=1), but it also contains harmful content (safety_score=0). This completion is taken from the orthogonalized Qwen 72B Chat model.

## Appendix E Weight orthogonalization is equivalent to directional ablation

To show the equivalence of directional ablation and weight orthogonalization, we consider all matrices that directly write contributions to the residual stream.

Let $W_{\text{out}}\in\mathbb{R}^{d_{\text{model}}\times d_{\text{input}}}$ be a matrix that writes to the residual stream, mapping vectors from $\mathbb{R}^{d_{\text{input}}}$ to $\mathbb{R}^{d_{\text{model}}}$.(^5^55Note that $d_{\text{input}}$ varies depending on which matrix is being considered. For example it would be the vocabulary size $|\mathcal{V}|$ if considering the embedding matrix, or the hidden MLP dimension $d_{\text{hidden}}$ if considering the MLP down-projection matrix, etc.)
Let the unit norm vector $\hat{\mathbf{r}}\in\mathbb{R}^{d_{\text{model}}}$ denote the direction to be ablated.

Now let $\mathbf{x}_{\text{pre}}\in\mathbb{R}^{d_{\text{model}}}$ denote the residual stream activation before $W_{\text{out}}$ adds a contribution to the residual stream, let $\mathbf{x}_{\text{post}}$ denote the residual stream activation after, and let $\mathbf{t}\in\mathbb{R}^{d_{\text{input}}}$ denote the input to $W_{\text{out}}$:

$$ $\displaystyle\mathbf{x}_{\text{post}}=\mathbf{x}_{\text{pre}}+W_{\text{out}}\mathbf{t}.$ (10) $$

With directional ablation, we take $\mathbf{x}_{\text{post}}$ and zero out its projection onto $\hat{\mathbf{r}}$:

$$ $\displaystyle\mathbf{x}_{\text{post}}^{\prime}$ $\displaystyle=\mathbf{x}_{\text{post}}-\hat{\mathbf{r}}\hat{\mathbf{r}}^{\intercal}\mathbf{x}_{\text{post}}$ (11) $\displaystyle=(\mathbf{x}_{\text{pre}}+W_{\text{out}}\mathbf{t})-\hat{\mathbf{r}}\hat{\mathbf{r}}^{\intercal}(\mathbf{x}_{\text{pre}}+W_{\text{out}}\mathbf{t})$ (12) $\displaystyle=\mathbf{x}_{\text{pre}}+W_{\text{out}}\mathbf{t}-\hat{\mathbf{r}}\hat{\mathbf{r}}^{\intercal}\mathbf{x}_{\text{pre}}-\hat{\mathbf{r}}\hat{\mathbf{r}}^{\intercal}W_{\text{out}}\mathbf{t}$ (13) $\displaystyle=\mathbf{x}_{\text{pre}}-\hat{\mathbf{r}}\hat{\mathbf{r}}^{\intercal}\mathbf{x}_{\text{pre}}+(W_{\text{out}}-\hat{\mathbf{r}}\hat{\mathbf{r}}^{\intercal}W_{\text{out}})\mathbf{t}.$ (14) $$

Supposing that directional ablation was similarly applied after all previous contributions to the residual stream, we have that $\hat{\mathbf{r}}^{\intercal}\mathbf{x}_{\text{pre}}=0$:

$$ $\displaystyle\mathbf{x}_{\text{post}}^{\prime}$ $\displaystyle=\mathbf{x}_{\text{pre}}+(W_{\text{out}}-\hat{\mathbf{r}}\hat{\mathbf{r}}^{\intercal}W_{\text{out}})\mathbf{t}$ (15) $\displaystyle=\mathbf{x}_{\text{pre}}+W_{\text{out}}^{\prime}\mathbf{t}$ (16) $$

where $W_{\text{out}}^{\prime}=W_{\text{out}}-\hat{\mathbf{r}}\hat{\mathbf{r}}^{\intercal}W_{\text{out}}$, as specified by weight orthogonalization in [Equation 5](#S4.E5).

## Appendix F Jailbreak evaluation

### F.1 Comparing to other jailbreaks

To compare the effectiveness of our jailbreak method to other methods in the literature, we report the 5 top performing jailbreak attacks, as ranked by HarmBench attack success rate (ASR) on the Llama-2 model family: GCG is the algorithm from Zou et al. (2023b) which optimizes an adversarial suffix for each prompt and each model;
GCG-M is the multi-prompt version of GCG, trained over multiple prompts for each model;
GCG-T is the transferable version of GCG-M, trained over multiple prompts and across multiple models;
AP is the AutoPrompt approach from Shin et al. (2020);
PAIR is a black-box method from Chao et al. (2023).

We also include comparisons to the set of human-written
“Do Anything Now” adversarial templates from Shen et al. (2023) (Human), and the “direct response” baseline without any jailbreaks in place (DR).

### F.2 System prompts

[Table 2](#S4.T2) shows that the attack success rate (ASR) of our weight orthogonalization methodology is sensitive to system prompts.
As shown in [Figure 16](#A6.F16), the Llama-2 Chat default system prompt includes explicit guidelines to avoid harmful or inappropriate content.
These guidelines function as in-context instructions that an instruction-following model will comply with, even if its natural propensity to refuse a request is removed.
In contrast, the Qwen Chat default system prompt, shown in [Figure 17](#A6.F17), is minimal and lacks specific directives about safety or ethics.
As a result, the orthogonalized Qwen Chat models demonstrate a high ASR even when the system prompt applied, as its system prompt does not have specify guidance to curb harmful outputs.

Figure: Figure 16: The default system prompt for Llama-2 Chat.

Figure: Figure 17: The default system prompt for Qwen Chat.

## Appendix G Model coherence evaluation

### G.1 Language model evaluation

**Table 6: Model evaluations. For each evaluation, we report the orthogonalized model’s performance, followed by the baseline model’s performance, followed by the absolute increase or decrease.**
| Chat model | MMLU | TinyHellaSwag | ARC | WinoGrande | GSM8K | TruthfulQA |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen 1.8B | 43.0 / 43.1 (-0.1) | 48.2 / 49.3 (-1.1) | 37.6 / 38.7 (-1.1) | 59.6 / 59.0 (+0.6) | 29.7 / 30.0 (-0.3) | 37.1 / 41.7 (-4.6) |
| Qwen 7B | 54.8 / 56.8 (-2.0) | 76.3 / 73.1 (+3.2) | 52.0 / 51.7 (+0.3) | 72.0 / 72.5 (-0.5) | 41.8 / 48.1 (-6.3) | 47.9 / 51.6 (-3.7) |
| Qwen 14B | 66.1 / 65.9 (+0.2) | 77.3 / 79.5 (-2.2) | 60.3 / 61.3 (-1.0) | 74.8 / 74.7 (+0.1) | 59.3 / 60.3 (-1.0) | 50.4 / 52.9 (-2.5) |
| Qwen 72B | 76.5 / 77.2 (-0.7) | 86.5 / 85.3 (+1.2) | 67.2 / 67.6 (-0.4) | 80.7 / 80.8 (-0.1) | 76.3 / 75.5 (+0.8) | 55.0 / 56.4 (-1.4) |
| Yi 6B | 62.6 / 63.2 (-0.6) | 78.1 / 76.8 (+1.3) | 56.6 / 57.4 (-0.8) | 72.9 / 72.2 (+0.7) | 39.0 / 40.6 (-1.6) | 44.2 / 50.1 (-5.9) |
| Yi 34B | 73.5 / 74.9 (-1.4) | 83.6 / 84.6 (-1.0) | 65.6 / 64.9 (+0.7) | 78.9 / 80.2 (-1.3) | 65.5 / 65.0 (+0.5) | 51.9 / 55.4 (-3.5) |
| Gemma 2B | 36.8 / 36.9 (-0.1) | 57.1 / 55.2 (+1.9) | 43.0 / 43.3 (-0.3) | 60.5 / 61.5 (-1.0) | 10.8 / 11.1 (-0.3) | 40.4 / 45.8 (-5.4) |
| Gemma 7B | 51.8 / 51.7 (+0.1) | 46.5 / 44.9 (+1.6) | 51.7 / 51.5 (+0.2) | 66.6 / 66.5 (+0.1) | 31.3 / 32.0 (-0.7) | 44.7 / 47.1 (-2.4) |
| Llama-2 7B | 46.8 / 47.5 (-0.7) | 76.8 / 77.6 (-0.8) | 53.0 / 53.7 (-0.7) | 71.7 / 72.6 (-0.9) | 22.7 / 23.1 (-0.4) | 41.6 / 45.3 (-3.7) |
| Llama-2 13B | 53.6 / 53.6 (+0.0) | 82.3 / 83.2 (-0.9) | 60.4 / 60.3 (+0.1) | 73.4 / 74.3 (-0.9) | 35.3 / 35.6 (-0.3) | 42.6 / 44.0 (-1.4) |
| Llama-2 70B | 63.1 / 63.0 (+0.1) | 84.8 / 84.8 (+0.0) | 65.2 / 65.4 (-0.2) | 79.7 / 80.2 (-0.5) | 54.5 / 53.0 (+1.5) | 51.8 / 52.8 (-1.0) |
| Llama-3 8B | 65.0 / 65.8 (-0.8) | 79.6 / 82.1 (-2.5) | 62.3 / 62.4 (-0.1) | 75.9 / 75.5 (+0.4) | 74.3 / 75.9 (-1.6) | 48.3 / 51.7 (-3.4) |
| Llama-3 70B | 79.8 / 79.9 (-0.1) | 85.4 / 86.1 (-0.7) | 71.5 / 71.8 (-0.3) | 83.4 / 83.6 (-0.2) | 90.8 / 91.2 (-0.4) | 59.5 / 61.8 (-2.3) |

Except on TruthfulQA, orthogonalization has a very small effect on general performance benchmarks.
We observe less than 1% performance drop on average, with the difference to the baseline performance being indistinguishable from noise in most cases.
The main exceptions are Qwen 7B, which has statistically significant drops on MMLU and GSM8K, and Yi 34B with drops on MMLU and Winogrande.

For MMLU, we use the default settings from LM Evaluation Harness (Gao et al., 2023; Biderman et al., 2024) as of May 2024. For the other benchmarks, we use the default settings, with the exception that models are run using vLLM (Kwon et al., 2023).

TinyHellaSwag from TinyBenchmarks (Polo et al., 2024) is a statistically informative 400-sample subset of the larger
HellaSwag (Zellers et al., 2019) test set. Polo et al. (2024) claim a 2% average error compared to the full-sized counterparts.

### G.2 TruthfulQA accuracy

TruthfulQA measures the performance of language models in generating truthful and accurate responses, particularly in areas prone to human misconceptions and falsehoods.
[Table 6](#A7.T6) displays clearly that TruthfulQA performance is consistently worse for orthogonalized models as compared with unmodified models.
TruthfulQA contains questions that touch on sensitive topics such as misinformation, stereotypes, and conspiracies.
For such questions, models with and without safety guardrails may understandably generate different responses.
[Figure 18](#A7.F18) displays an example of a conspiracy-flavored question from TruthfulQA, and the contrasting responses from Llama-3 8B Instruct and its orthogonalized version.

Figure: Figure 18: A question from TruthfulQA, and corresponding completions from Llama-3 8B Instruct and its orthogonalized version.

### G.3 CE loss evaluation

In addition to standard language model evaluations, we also check changes in cross-entropy (CE) loss over various datasets.
For each chat model and its orthogonalized version we compute CE loss over a sample of The Pile (Min et al., 2023).
The Pile consists of scraped webtext, and so we do not append any chat template when evaluating CE loss.

We note that some chat models are especially sensitive to chat templates, and behave poorly without them.
Thus, we also evaluate over Alpaca (Taori et al., 2023), which is a chat dataset consisting of instructions and completions.
We format each instruction according to each model’s chat template, and compute CE loss only over the completion tokens.

We further note that some chat models, seemingly Gemma 7B It in particular, have high CE loss on text that is out of distribution, e.g. completions from Alpaca.
To account for this, we take each baseline model, and generate completions on 100 instructions from Alpaca.
We then compute CE loss over these “on-distribution” completion tokens.

All CE loss values are reported in [Table 7](#A7.T7).
We denote the orthogonalized model as “Ablation”.
We also compare to activation addition methodology, labeled “Act Add”, where rather than ablating the refusal direction, we *subtract* the difference-in-means vector.
See §[I.1](#A9.SS1) for a more detailed discussion of bypassing refusal via activation addition.

**Table 7: Model performance as measured by CE loss across different datasets.**
|  | CE Loss (The Pile) | CE Loss (Alpaca) | CE Loss (On-distribution) |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chat model | Baseline | Ablation | Act Add | Baseline | Ablation | Act Add | Baseline | Ablation | Act Add |
| Qwen 1.8B | 2.921 | 2.938 | 3.259 | 1.779 | 1.784 | 2.038 | 0.284 | 0.293 | 0.586 |
| Qwen 7B | 2.259 | 2.277 | 2.388 | 1.615 | 1.631 | 1.697 | 0.242 | 0.278 | 0.479 |
| Qwen 14B | 2.070 | 2.078 | 2.230 | 1.602 | 1.606 | 1.713 | 0.212 | 0.218 | 0.443 |
| Qwen 72B | 1.944 | 1.971 | 2.097 | 1.740 | 1.768 | 2.124 | 0.147 | 0.162 | 0.380 |
| Yi 6B | 2.019 | 2.017 | 2.205 | 1.889 | 1.882 | 2.078 | 0.277 | 0.311 | 0.731 |
| Yi 34B | 1.862 | 1.872 | 2.002 | 1.971 | 2.008 | 2.066 | 0.191 | 0.259 | 0.680 |
| Gemma 2B | 3.506 | 3.489 | 3.739 | 2.090 | 2.101 | 2.179 | 0.254 | 0.311 | 0.853 |
| Gemma 7B | 5.975 | 5.963 | 6.051 | 2.336 | 2.335 | 2.356 | 0.201 | 0.228 | 0.656 |
| Llama-2 7B | 2.220 | 2.214 | 2.333 | 1.609 | 1.586 | 1.584 | 0.118 | 0.126 | 0.460 |
| Llama-2 13B | 2.082 | 2.087 | 2.325 | 1.563 | 1.591 | 1.642 | 0.102 | 0.116 | 0.336 |
| Llama-2 70B | 1.970 | 1.969 | 2.010 | 1.657 | 1.659 | 1.630 | 0.067 | 0.070 | 0.169 |
| Llama-3 8B | 2.348 | 2.362 | 2.469 | 1.912 | 1.944 | 1.912 | 0.195 | 0.213 | 0.441 |
| Llama-3 70B | 2.121 | 2.117 | 2.274 | 1.980 | 1.978 | 1.928 | 0.116 | 0.126 | 0.265 |

## Appendix H Adversarial suffix analysis

### H.1 Adversarial suffix generation

Using a custom implementation of Greedy Coordinate Gradient (\textsscGCG) (Zou et al., 2023b), we generated 100 adversarial suffixes of token-length 20, each of which was optimized for a particular behavior from AdvBench.

Of these 100 suffixes, we found one suffix in particular that performs well across a wide range of harmful prompts. The suffix is shown in [Figure 19](#A8.F19).

Figure: Figure 19: The adversarial suffix studied in §[5](#S5). This suffix is generally effectively in bypassing refusal in Qwen 1.8B Chat.

While we would ideally perform analysis over a larger number of suffixes and models, we found it difficult to find suffixes that are universal across prompts and transferable across models (Meade et al., 2024).
We therefore restrict our analysis to a single model, Qwen 1.8B Chat, and a single suffix.

### H.2 Reporting of confidence intervals

In [Figure 5](#S5.F5), for each layer and scenario, we display the standard deviation (SD) of cosine similarities across 128 prompts, computed as $\text{SD}=\sqrt{\frac{\sum{(x_{i}-\overline{x})^{2}}}{n}}$.
In this case, $n=128$.

## Appendix I Comparison to other methodologies

In §[3.1](#S3.SS1), we use *directional ablation* to bypass refusal.
In §[4](#S4), we show how this can be implemented as a direct weight modification, and then analyze the modification’s effect on refusal and coherence.

In this section, we compare directional ablation to two other weight modification methodologies: activation addition and fine-tuning.

### I.1 Comparison to activation addition

Figure: Figure 20: A visualization of activation addition (abbreviated as “act add”) in the negative refusal direction. The intervention pulls harmful activations towards harmless activations, effectively bypassing refusal. However, note that the intervention pushes harmless activations far out of distribution. This figure displays activations from Gemma 7B It, computed over 128 harmful and harmless prompts.
Refer to caption: https://ar5iv.labs.arxiv.org/html/2406.11717/assets/x9.png

In §[2.4](#S2.SS4), we described how to induce refusal using activation addition.
Given a difference-in-means vector $\mathbf{r}^{(l)}\in\mathbb{R}^{d_{\text{model}}}$ extracted from layer $l$, we can *add* this vector at layer $l$ in order to shift activations towards refusal ([Equation 3](#S2.E3)).
Similarly, we can *subtract* this vector at layer $l$ in order to shift activations away from refusal:

$$ $\displaystyle\mathbf{x}^{(l)^{\prime}}\leftarrow\mathbf{x}^{(l)}-\mathbf{r}^{(l)}.$ (17) $$

We perform this intervention at all token positions.
Note that this intervention can be implemented as a direct weight modification by subtracting $\mathbf{r}^{(l)}$ from the bias term of $\texttt{MLP}^{(l-1)}$.

As shown in [Figure 21](#A9.F21), this activation addition intervention is effective in bypassing refusal.
The decreases in refusal score and safety score are comparable to those achieved by directional ablation ([Figure 1](#S1.F1)).
However, [Table 7](#A7.T7) displays that the activation addition intervention, labeled as *act add*, causes increased loss over harmless data, in particular compared to directional ablation.

[Figure 20](#A9.F20) displays a visualization of activation addition in the negative refusal direction, and suggests an intuitive explanation of the intervention’s behavior on harmful and harmless prompts.
On harmful inputs, adding the negative refusal direction shifts the harmful activations towards harmless activations, with respect to projection onto the refusal direction.
With low projection onto the refusal direction, this intervention leads to low rates of refusal.
However, on harmless inputs, adding the negative refusal direction shifts the harmless activations off distribution, resulting in increased perplexity.

Note that, in comparison to activation addition, directional ablation shifts harmful activations towards harmless activations, while also not shifting harmless activations too far off distribution.

Figure: Figure 21: Performing activation addition in the negative “refusal direction”, displayed in dots, reduces refusal rates and elicits unsafe completions. It is approximately as effective as directional ablation at bypassing refusal ([Figure 1](#S1.F1)).
Refer to caption: https://ar5iv.labs.arxiv.org/html/2406.11717/assets/x10.png

### I.2 Comparison to fine-tuning

**Table 8: Refusal and CE loss evaluation metrics for Llama-3 8B Instruct, comparing the interventions of directional ablation, activation addition, and fine-tuning.**
|  | Refusal | CE Loss |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Intervention | Refusal score | Safety score | The Pile | Alpaca | On-distribution |
| No intervention | 0.95 | 0.97 | 2.348 | 1.912 | 0.195 |
| Directional ablation | 0.01 | 0.15 | 2.362 | 1.944 | 0.213 |
| Activation addition | 0.01 | 0.16 | 2.469 | 1.912 | 0.441 |
| Fine-tuning | 0.00 | 0.08 | 2.382 | 1.626 | 0.273 |

Prior work has established that fine-tuning is effective in removing safety guardrails of chat models (Yang et al., 2023; Lermen et al., 2023; Zhan et al., 2023).

We replicate this result by fine-tuning Llama-3 8B Instruct.
First, we construct a dataset of harmful instruction-completion pairs.
For the harmful instructions, we sample instructions from AdvBench, MaliciousInstruct, TDC2023, and HarmBench.
To generate corresponding harmful completions, we use Mistral 7B Instruct (Jiang et al., 2023), a competent chat model with low refusal rates.
For each harmful instruction, we generate 5 completions, and then select a single completion satisfying both refusal_score=0 and safety_score=0.
If no completions satisfy this condition, then the instruction is discarded.
After this filtering, we were left with a dataset of 243 harmful instruction-completion pairs.

We then fine-tuned Llama-3 8B Instruct on the constructed dataset, applying LoRA (Hu et al., 2021) with rank=16 and alpha=32 for 4 epochs.
The LoRA fine-tuning was performed on an A100 GPU with 80GB of VRAM, and took approximately 10 minutes.

Evaluations of refusal and CE loss are displayed in [Table 8](#A9.T8).
In accordance with prior work, we confirm fine-tuning to be very effective in disabling refusal.
We speculate that the decrease in CE loss over Alpaca could be due to the similarity between the distributions of Mistral Instruct completions and Alpaca completions, and as a result, fine-tuning over Mistral Instruct completions leads to a decreased CE loss over Alpaca completions.

We note that, although the LoRA fine-tuning process itself is straightforward and efficient, creating a high-quality dataset of harmful instruction-completion pairs requires non-trivial effort.
In comparison, directional ablation (and its equivalent implementation via weight orthogonalization) requires just a dataset of harmful instructions, without the need for any harmful completions.

## Appendix J The “refusal direction” is also present in base models

Throughout this work, we consider only *chat models*, models that have undergone fine-tuning to follow benign instructions and refuse harmful ones.
Prior to this fine-tuning process, models are essentially just next-token predictors, referred to as *base models*.
By default, base models are not instruction-following.
For instance, if prompted with a question, a base model is likely to output another question, rather than an answer to the original question.
In particular, base models do not refuse harmful requests.

In §[3](#S3), we argue that refusal in chat models is mediated by a single direction in activation space.
One natural question is whether this direction, or feature, is learned from scratch during safety fine-tuning specifically to mediate refusal, or whether this direction is already present in the base model and gets repurposed, or “hooked into”, during safety fine-tuning.

To investigate this question, we check the expression of the refusal direction in chat and base models.
For each model, we sample 128 harmful and harmless instructions, run inference on both the chat model (e.g. Llama-3 8B Instruct) and its corresponding base model (e.g. Llama-3 8B), and cache all intermediate activations at the last token position.(^6^66Note that we append a newline character to the end of each instruction, and consider activations only at this token position. For chat models, we prepend the portion of the chat template that comes before the instruction. For base models, we do not prepend anything before the instruction.)
We then take the refusal direction extracted from the corresponding chat model (§[2.3](#S2.SS3)), and examine the cosine similarity of each activation with this direction.

[Figure 22](#A10.F22) displays the results for four distinct models.
We find that, similarly to the chat models, corresponding base models have high cosine similarity with the refusal direction when run on harmful prompts, and low cosine similarity when run on harmless prompts.
This suggests that, rather than developing the “refusal direction” from scratch during fine-tuning, this representation exists already in the base model, and is repurposed for refusal during safety fine-tuning.

Figure: Figure 22: Cosine similarity of activations with the refusal direction, for base (dotted lines) and chat (solid lines) models. The refusal direction is expressed similarly in base and chat models.
Refer to caption: https://ar5iv.labs.arxiv.org/html/2406.11717/assets/x11.png

## Appendix K Extended results

As throughout the paper, all generations are generated deterministically using greedy decoding.

### K.1 Bypassing refusal - examples

Figure: Figure 23: Examples of bypassing refusal by ablating the “refusal direction”. These completions are taken from Llama-3 70B Instruct.

### K.2 Inducing refusal - examples

Figure: Figure 24: Examples of inducing refusal by adding the “refusal direction”. These completions are taken from Gemma 7B It.

## Appendix L Further experiments with orthogonalized models

As throughout the paper, all generations are generated deterministically using greedy decoding.
All experiments in this section are performed on Llama-3 8B Instruct.

### L.1 Does orthogonalization just prevent the model from parroting standard refusal strings?

One possible way to prevent a model from refusing is to directly block it from outputting any of the standard refusal strings, such as "Sorry, I cannot", or "As a language model".
Experiments shown in [Figure 25](#A12.F25) show that this is not the case for our weight orthogonalization methodology, as the orthogonalized model is able to generate the same strings that the unmodified model uses to refuse a harmful request.
This suggests that the modification works at a higher conceptual level, rather than at the level of suppressing output tokens.

Figure: Figure 25: The unmodified model (no intervention) refuses the first request with the string "I cannot provide instructions on how to make a bomb". When explicitly requested to output this string, the orthogonalized model (intervention) is able to do so.

### L.2 The orthogonalized model behaves similarly on harmless instructions

In general, we notice that the orthogonalized model behaves very similarly to the non-modified model on harmless instructions.
[Figure 26](#A12.F26) displays completions on a random sample of harmless prompts from Alpaca.
Generations from the unmodified model and the orthogonalized model appear to be indistinguishable, and often the generations are exactly the same.

Figure: Figure 26: Generations over a random sample of harmless instructions from Alpaca. Generations from the unmodified model (no intervention) are essentially indistinguishable from the orthogonalized model (intervention).

### L.3 The orthogonalized model may have trouble understanding its new refusal behavior

As our intervention targets refusal directly, it is natural to ask how the resulting model answers meta-questions about its new refusal behavior: does it understand its behavioral modification, or does it default to explaining its original refusal behavior?
As shown in [Figure 27](#A12.F27), the new model seems likely to answer meta-refusal questions consistently with its previous refusal behavior.
However, its explanations seem not to be coherent.

Figure: Figure 27: The orthogonalized model gives the same answer to a meta-reasoning question about its own refusal behavior as the original model. However, its reasoning seems incoherent: it claims the reason for not explaining how to make a Molotov cocktail is “this is a factual question, but it’s not a request for a creative response”.

## Appendix M Use of existing assets

### M.1 Models

**Table 9: The list of models used in this work.**
| Model | Source | Accessed via | License |
| --- | --- | --- | --- |
| Qwen Chat | Bai et al. (2023) | [Link](https://huggingface.co/Qwen/Qwen-1_8B-Chat) | Tongyi Qianwen Research License |
| Yi Chat | Young et al. (2024) | [Link](https://huggingface.co/01-ai/Yi-6B-Chat) | Apache License 2.0 |
| Gemma It | Team et al. (2024) | [Link](https://huggingface.co/google/gemma-2b-it) | Gemma Terms of Use |
| Llama-2 Chat | Touvron et al. (2023) | [Link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | Llama 2 Community License |
| Llama-3 Instruct | AI@Meta (2024) | [Link](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Meta Llama 3 Community License |
| Llama Guard 2 | Team (2024) | [Link](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B) | Meta Llama 3 Community License |
| HarmBench Classifier | Mazeika et al. (2024) | [Link](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls) | MIT License |
| Mistral Instruct | Jiang et al. (2023) | [Link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | Apache License 2.0 |

### M.2 Datasets

**Table 10: The list of datasets used in this work.**
| Dataset | Source | Accessed via | License |
| --- | --- | --- | --- |
| AdvBench | Zou et al. (2023b) | [Link](https://github.com/llm-attacks/llm-attacks) | MIT License |
| TDC2023 | Mazeika et al. (2024, 2023) | [Link](https://github.com/centerforaisafety/tdc2023-starter-kit) | MIT License |
| HarmBench | Mazeika et al. (2024) | [Link](https://github.com/centerforaisafety/HarmBench/tree/main) | MIT License |
| JailbreakBench | Chao et al. (2024) | [Link](https://github.com/JailbreakBench/jailbreakbench/tree/main) | MIT License |
| MaliciousInstruct | Huang et al. (2023) | [Link](https://github.com/princeton-sysml/jailbreak_llm) | MIT License |
| Alpaca | Taori et al. (2023) | [Link](https://huggingface.co/datasets/tatsu-lab/alpaca) | Apache License 2.0 |
| The Pile | Gao et al. (2020) | [Link](https://huggingface.co/datasets/monology/pile-uncopyrighted) | MIT License |
| MMLU | Hendrycks et al. (2020) | [Link](https://huggingface.co/datasets/cais/mmlu) | MIT License |
| ARC | Clark et al. (2018) | [Link](https://huggingface.co/datasets/allenai/ai2_arc) | CC-BY-SA-4.0 |
| GSM8K | Cobbe et al. (2021) | [Link](https://huggingface.co/datasets/openai/gsm8k) | MIT License |
| WindoGrande | Sakaguchi et al. (2021) | [Link](https://huggingface.co/datasets/allenai/winogrande) | Apache License 2.0 |
| TruthfulQA | Lin et al. (2021) | [Link](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Apache License 2.0 |
| TinyHellaSwag | Polo et al. (2024) | [Link](https://huggingface.co/datasets/tinyBenchmarks/tinyHellaswag) | MIT License |

## Appendix N Compute statement

Most experiments presented in this paper were run on a cluster of eight NVIDIA RTX A6000 GPUs with 48GB of memory.
All experiments on models with $\leq$ 14B parameters are run using a single 48GB memory GPU. For larger models, we use four 48BG memory GPUs in parallel.

Generating and selecting the directions, as described in Subsection [2.3](#S2.SS3), takes approximately 5 minutes for smaller models of size $\leq$ 14B, and approximately 1 hour for the larger models.