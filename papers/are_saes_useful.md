## 1 Introduction

Dictionary learning is a popular method for interpreting LLM activations ; most notably, demonstrated the promise of sparse autoencoders (SAEs) and sparked considerable follow-up work. This includes papers focused on improving the downstream cross entropy loss of SAE reconstructions , improving SAE training efficiency , finding weaknesses in SAEs and proposing solutions , using SAEs to understand model representation structure , training large suites of SAEs , and developing benchmarks for SAEs .

Unfortunately, we lack a “ground truth” to know whether SAEs truly extract the interpretable concepts used by language models. Prior work has largely avoided this limitation by evaluating SAEs with proxy metrics like reconstruction loss . These metrics are tractable to optimize for, but do not necessarily align with mechanistic interpretability’s (MI) goal of better understanding neural networks (see and for surveys of MI). We argue that if SAEs truly advance MI’s goal, they should improve performance on a real, hard-to-fake model control or explainability task.

However, despite the extensive body of work on SAEs, there are relatively few cases where SAEs have been shown to help on such a task: show that SAE feature circuits can help identify and remove bias from classifiers; show that SAEs can identify misaligned features learned by a preference model; and show that SAE feature ablations are better at preventing regex output than baselines in a setting with scarce supervised data. While these results are promising, they all study a single example in detail, and consider comparative baselines with varying levels of rigor.

At the same time, there are also surprisingly few negative results finding that SAEs do not help on downstream tasks; in fact, we are only aware of , which finds that SAE latents are worse than neurons at disentangling geographic data, and , which finds that pinning related SAE latents is less effective than baselines for unlearning bioweapon knowledge. Thus, it is not clear whether SAEs are just one result away from being differentially useful, or if MI should seek fundamentally different methods. We defer a thorough examination of related work to Appendix A.

In this work, we attempt to fairly evaluate the utility of SAEs by examining their competitive advantage on a concrete task: training probes from language model activations to targets . Probing has two important qualities:

- 1.
Probing is practically useful: Probing has been used to investigate LLM representations , detect safety relevant quantities , remove knowledge from models , and catch synthetic sleeper agents .
- 2.
SAEs might reasonably improve probing: Theoretically, SAE latents are a more interpretable basis of model activations, and we hypothesize that this inductive bias will help train probes in difficult regimes. Recent work has also found some positive results for the utility of SAE probes .

Thus, we curate 113 linear probing datasets from a variety of settings and train linear probes on corresponding SAE latent activations (see Figure 2). We compare to a suite of baseline methods across 4 difficult probing regimes: 1) data scarcity, 2) class imbalance, 3) label noise, and 4) co-variate shift.
Unfortunately, we find that SAE probes fail to offer a consistent overall advantage when added to a simulated practitioner’s toolkit.

Further, we explore areas that suggest SAEs may be valuable for, including detecting attributes distributed over multiple tokens and identifying dataset issues. Although SAEs initially seemed promising in these settings, we were able to achieve the same results with improved baselines. Our results underscore the necessity for MI works to rigorously design baselines when evaluating the utility of interpretability techniques.

Figure: Figure 2: Left: An illustration of our SAE probing method. We pass in training activation vectors from each class and train an $L_{1}$ regularized logistic regression probe on the latents that differ the most between classes. Right: We ensure robustness of our results with the ”quiver of arrows” approach (see Section 2.4): we add SAE regression into a set of methods, and see if the test accuracy of the best method (chosen by validation accuracy) increases.
Refer to caption: https://arxiv.org/html/2502.16681/x2.png

**Table 1: Example probing tasks. Only tasks with short prompts shown for conciseness.**
| Dataset Name | Prompt | Target |
| --- | --- | --- |
| 26_headline_isfrontpage | Sebelius’s Slow-Motion Resignation From the Cabinet. | 1 |
|  | MI5 References Emerge in Phone Hacking Lawsuit. | 0 |
| 36_sciq_tf | Q: Binary fission is an example of which type of production? A: asexual | 1 |
|  | Q: What occurs when light interacts with our atmosphere? A: prism effect | 0 |
| 114_nyc_borough_Manhattan | INWOOD BASKETBALL COURTS | 1 |
|  | PS 18 JOHN G WHITTIER | 0 |
| 149_twt_emotion_happiness | All moved in to our new apartment. So exciting | 1 |
|  | is noow calmmm eating polvoron .. yuumm | 0 |
| 155_athlete_sport_basketball | Jimmy Butler | 1 |
|  | Steve Balboni | 0 |

## 2 Methodology

We apply probes to the hidden states of two language models, Gemma-2-9B and Llama-3.1-8B . Our main paper results use Gemma-2-9B. We replicate core results on Llama-3.1-8B in Appendix F. We use JumpReLU  sparse autoencoders (SAEs) from Gemma Scope for Gemma-2-9B and TopK  SAEs from Llama Scope for Llama-3.1-8B. See Appendix A for further background on SAEs.

### 2.1 Classification Datasets

We collect a diverse set of 113 binary classification datasets listed in LABEL:tab:app_all_tasks (Appendix B), including datasets that we expect to be challenging for probes.
For example, 26_headline_isfrontpage requires probes to identify front-page headlines and 136_glue_mnli_entailment requires probes to identify logically entailment. For other example datasets, refer to Table 1. All datasets are titled in the form $\mathrm{ID}_\mathrm{description}$. We originally collected datasets for other purposes and discarded some of them; thus, while we have 113 datasets, $\mathrm{ID}$ ranges from $[5,163]$.

All datasets contain $\mathrm{prompts}$ and $\mathrm{targets}$. Probes are tasked with predicting the $\mathrm{target}$ from the model’s hidden activations when run on a $\mathrm{prompt}$. We focus on binary classification, since SAEs are mostly thought of as representing binarized latents (see ). Thus, $\mathrm{target}$ is either 0 0 or $1$. The prompts in our datasets range in length from 5 tokens to a (left-truncated) maximum of 1024 tokens.

### 2.2 Probing Strategy

To train a probe on a given dataset, we first run the model on all $\mathrm{prompts}$ from that dataset to generate model activations at each layer $l$ on the last token, $X_{-1}^{l}$.$X_{-1}^{l}$ is of shape $\mathrm{(len(prompts),model_dim)}$. We probe the last token to ensure the target information was present in the preceding context. For activation probes, we then train a probe $p$ to map from $X_{-1}^{l}\mapsto t$, where $t$ are the $\mathrm{targets}$ in our dataset.

Our SAE probe training technique is summarized in Figure 2. We first pass the training dataset $X_{-1}^{l}$ through the SAE encoder, resulting in a batch of vectors in the SAE latent space $Z=\mathrm{SAE}(X_{-1}^{l})$ of shape $(\mathrm{len(prompts)},W)$, where $W$ is the width of the SAE. We do not train probes directly on the SAE latent space because we hypothesize that a small number of SAE latents encodes the desired concept. Instead, we create a basis of latents with the highest average absolute difference between the set of training prompts $T_{1}$ with $\mathrm{target}=1$ and the set of training prompts $T_{0}$ with $\mathrm{target}=0$. More formally, if $\mathrm{SAE}(X_{-1}^{l})\in\mathbb{R}^{W}$, where $W\gg d_{\textrm{model}}$, we choose $k\ll W$ and find indices $\mathcal{I}$ as follows:

$$ $\displaystyle\mathcal{I}=\operatorname*{arg\,top\,k}_{i\in\{W\}}\left|\frac{1} {|T_{1}|}\sum_{j\in T_{1}}Z_{j,i}-\frac{1}{|T_{0}|}\sum_{j\in T_{0}}Z_{j,i}\right|$ (1) $$

We then train a probe $p_{\textrm{SAE}}$ to map from $Z[:,\mathcal{I}]\mapsto t$.

Note that previous work in SAE probing has aggregated SAE latents across tokens to provide a richer input space for SAE probes . However, most probing studies operate on the final token, which we choose to emulate. In Section 5, we present results considering probes on multiple tokens.

Throughout this study, we use the area under the ROC curve (AUC) to evaluate the quality of a probe. AUC is the likelihood that the classifier assigns a higher probability of $\mathrm{target}=1$ to a $\mathrm{target}=1$ example than to a $\mathrm{target}=0$ example. Using AUC allows us to comprehensively assess probe performance agnostic to classification thresholds. For additional details on AUC, see Section C.1.

Often, a probe $p$ has hyperparameters $h_{p}$ we would like to optimize.
We select $h_{p}$ that has the maximal validation AUC using the cross-validation strategy described in Table 4. We then test $p$ with optimal $h_{p}$ on a held out test set to calculate $\mathrm{AUC}_{p}^{\mathrm{test}}$. All datasets have at least $100$ testing examples, with most having more (the average test set size is $1945$).

### 2.3 Probing Methods

We use the following 5 probing methods, with hyperparameters in parenthesis:

- 1.
Logistic Regression ($\mathrm{L}_{1}$ regularization for SAE probes, $\mathrm{L}_{2}$ otherwise).
- 2.
PCA Regression (# of PCA components to reduce $X_{-1}^{l}$ to before using unregularized logistic regression).
- 3.
K-Nearest Neighbors (KNN) (# of nearest neighbors).
- 4.
XGBoost ($\mathrm{n_estimators}$, $\mathrm{max_depth}$, $\mathrm{learning_rate}$, $\mathrm{subsample}$, $\mathrm{colsample_bytree}$, $\mathrm{reg_alpha}$, $\mathrm{reg_lambda}$, and $\mathrm{min_child_weight}$).
- 5.
Multilayer Perceptron (MLP) (network depth, hidden state width, learning rate, and weight decay).

We evaluate 10 hyperparameter values $h_{p}$ for each probing method $p$. For the first three probing methods, we use grid search, while for MLPs and XGBoost, we randomly sample from a hyperparameter grid to manage the larger search space. For additional details on hyperparameter ranges, see Section C.3. We use all five probing methods as baselines, but for SAE probes we only use logistic regression.

### 2.4 Experimental Setup: Quiver of Arrows

To evaluate whether SAE probes provide an advantage over baselines, we use an evaluation metric we call the “Quiver of Arrows”: we ask whether adding SAE probes (a new “arrow”) to the set of existing probing methods available to a practitioner (the “quiver”) increases performance compared to a practitioner without access to SAE probes. In other words, over a collection of methods we choose the best method by validation AUC and then report the test AUC of that method. We can then compare the marginal improvement of adding SAE probes to a practitioner’s toolkit by comparing the Quiver of Arrows AUC of baeline methods with and without SAE probes. We describe the Quiver of Arrows more formally in Section C.4.

The quiver of arrows approach is designed as a robustness check to fairly evaluate the benefit of adding SAE probes to a practioner’s toolkit. If we instead constructed probes from a large set of SAEs and chose the best one based on test AUC, we could be tricked into thinking SAEs outperform baselines because we accessed the held-out test set. With the quiver of arrows approach, we emulate the information a real practioner has access to, allowing us to make a robust recommendation on the usefulness of SAE probes.

Figure: Figure 3: For a given width, using higher L0 and constructing probes with a larger basis of latents ($k$) is more performant.
Refer to caption: https://arxiv.org/html/2502.16681/x3.png

## 3 Comparing Probing Techniques in Different Regimes

### 3.1 Standard Conditions

We initially assess probes under standard conditions, characterized by sufficient data for probe convergence and balanced classes. We find that baseline methods perform the best on layer 20 (see Figure 13(a), Section D.1), so we run experiments with this layer. We train baselines on 1024 data points (or the maximum number of points in the dataset).

We first conduct a preliminary investigation to cut down the large space of Gemma Scope SAEs. We train probes for all SAEs using $k$ latents (for logspaced values of $k$) on all datasets using 1024 training examples. We calculate the average test AUC for each SAE probe across all datasets. We find that SAE $\mathrm{width}$ is relatively unimportant to probe success, while larger $\mathrm{L0}$s lead to more performant probes (Section D.2). Additionally, as we might expect, using a larger value of $k$ leads to better probe performance, as shown in Figure 3.

Figure: Figure 4: In standard conditions, when SAE probes are added to the quiver, we find a slight decrease in performance.
Refer to caption: https://arxiv.org/html/2502.16681/x4.png

Thus, throughout the paper we train probes using the largest $\mathrm{L0}$ for SAEs $\mathrm{width}=16\mathrm{k},\mathrm{width}=131\mathrm{k}$, and $\mathrm{width}=1\mathrm{M}$. We use $k=16$ to construct easily interpretable probes that potentially overfit less and use $k=128$ for performance.

Using this set of probes, we consider our quiver of arrows approach in standard conditions. SAEs are chosen as the “arrow” for 14/113 tasks, however, we see a slight decrease in performance when they are added to the quiver in Figure 4.

It is unsurprising that SAE probes underperform baselines in standard conditions, as their inductive bias is marginal given a large, balanced dataset. Thus, we now investigate more difficult settings to test if the inductive bias of SAEs translates into a competitive advantage for probing.

Figure: Figure 5: For three of the datasets in Table 1, we visualize the performance when SAE probes are in the quiver (dashed) versus when they are not (solid) for the regimes of data scarcity (left), class imbalance (middle), and label noise (right). In all three regimes, we see that on average (bottom row), SAEs do not help.
Refer to caption: https://arxiv.org/html/2502.16681/x5.png

Figure: Figure 6: For the regimes of data scarcity, class imbalance, and label noise, we see no average improvement across datasets when adding SAEs to the quiver. Shading represents 95% confidence intervals.
Refer to caption: https://arxiv.org/html/2502.16681/x6.png

### 3.2 Data Scarcity, Class Imbalance, and Label Noise

In this section, we consider more difficult probing regimes: limiting the training data (Data Scarcity), changing the relative frequency of $\mathrm{target}=1$ examples (Class Imbalance), and randomly flipping a fraction of the $\mathrm{targets}$ (Label Noise). All of these settings are realistic workflows - for example, a researcher observing a rare model phenomena would like a probe that generalizes with few total examples or few examples of the desired class, while a researcher working with collected user data wants a generalizable probe even if some fraction of users respond randomly.

Each regime is characterized by a parameter which we vary to compare SAEs and baselines.

- •
Data Scarcity - We use 20 values of $n$ logspaced in $[2,1024]$, where $n$ is the number of training examples. We use a standard test set.
- •
Class Imbalance - We use 19 values of $\mathrm{ratio}$ linearly spaced between $[0.05,0.95]$, where $\mathrm{ratio}=\frac{n_{1}}{n}$ and $n_{1}$ is the number of $\mathrm{target}=1$ training examples. The test set uses the same $\mathrm{ratio}$.
- •
Label Noise - We use 11 values of $\mathrm{fraction}$ linearly spaced between $[0,0.5]$, where $\mathrm{fraction}=\frac{n_{\mathrm{corrupted}}}{n}$. We use a standard test set (uncorrupted).

We evaluate the first two regimes with the quiver of arrows method. However, the quiver of arrows method relies on the validation data being representative of the test data. This assumption fails for the label noise setting since the validation data is corrupted. Thus, a hypothetical practitioner would likely deploy their most performant method from other settings instead of allowing corrupted validation AUC to arbitrarily choose a method. To emulate this, we compare logistic regression and the $\mathrm{width}=16\mathrm{k}$, $k=128$ SAE probe head-to-head in the setting of label noise.

In Figure 5, we visualize the SAE versus non-SAE quivers for a sample of datasets listed in Table 1 across the three regimes. Additionally, we visualize the average performance difference between the SAE and non-SAE quivers for each regime across all datasets in Figure 6. At all parameter values in each regime, SAEs show no meaningful improvement over baselines. This is not because SAEs are not chosen from the quiver; in Figure 16 we see that SAEs are chosen for up to 40 datasets in each regime. SAEs simply underperform baselines in these settings. See Figure 16 for results for individual datasets.

### 3.3 Covariate Shift

We now investigate if SAE probes are more resilient to distribution shifts in prompts. To model this covariate shift, we create or use 8 out-of-distribution (OOD) datasets: two pre-existing GLUE-X datasets which are designed as “extreme” versions of tasks 87 and 90 to test grammaticality and logical entailment respectively; three datasets (tasks 66, 67, and 73) which we alter the language of; and three datasets (tasks 5, 6, and 7) where we use syntactical alterations to names or use cartoon characters instead of historical figures. We train probes on these datasets in standard settings and evaluate on 300 covariate shifted test examples.

Like the label noise setting, our validation data is unfaithful to our test data. Thus, we compare logistic regression to a single SAE probe. We construct SAE probes from the $\mathrm{width}=131\mathrm{k},\mathrm{L0}=114$ SAE, as latent descriptions are available for this SAE on Neuronpedia . Our results (Figure 7) show that baselines outperform SAE probes when generalizing to covariate shifted data.

## 4 Interpretability

While we find that SAE probes are not helpful in traditional probing settings, one intrinsic advantage of SAE probes is that their input basis is interpretable. To leverage this, we use the technique of automatic interpretation, or autointerp, to create natural language descriptions of top latents for each dataset (see ).
We investigate three applications of labeled latents:

- 1.
Probe Interpretability: In Section 4.1, we investigate why SAE probes fail to perform well by pruning latents that o1 ranks as spurious.
- 2.
Latent Interpretability: In Section 4.2, we generate autointerp explanations of each dataset’s top latent to find spurious latents and latents that fit well to our probes in a way not explained by the autointerp label.
- 3.
Detecting Dataset Quality Issues: In Section 4.3, we invesigate spurious dataset features and label errors using insights from the top latent descriptions and firing patterns.

Figure: Figure 7: SAE probes often generalize worse than baseline logistic regression with covariate shift.
Refer to caption: https://arxiv.org/html/2502.16681/x7.png

### 4.1 Probe Intepretability: Pruning and Latent Generalization

We first use autointerp to investigate why SAE probes fail to generalize to covariate shift. We have two initial hypotheses: 1) the latents SAE probes rely on leverage spurious correlations in the training data and 2) the latents are not spurious, but they are not robust to the distribution shift.

First, we investigate hypothesis 1. Specifically, we attempt to improve SAE probes OOD performance by pruning latents deemed spurious by their autointerp description. We focus on three OOD tasks that SAE probes perform poorly on: 7_hist_fit_ispolitician, 66_living-room, and 90_glue_qnli (Figure 7). We use the $\mathrm{width}=131\mathrm{k},\mathrm{L0}=114$ SAE.

For each task, we take the $k=8$ top latents by mean difference and use Claude-3.5-Sonnet to generate latent descriptions with autointerp. We then use OpenAI’s o1 model to rank the relevance of each latent’s description to the task. We also prompt o1 to downrank spurious latents given the OOD transformation. An example of this procedure for the task 66_living-room, which identifies if the phrase “living room” is in an English sentence, is shown in Table 5, Section G.1. For this task, o1 ranked latent 12274, which identifies “mentions of living rooms” first, while ranking latent 51330, which identifies “objects and materials related to scientific experiments,” last.

We then construct a probe with the top $k$ latents using o1’s relevance rank for $k\in[1,8]$. If there are spurious latents in the $k=8$ probe, we expect a probe with $k<8$ to have better OOD generalization. We visualize each task’s performance by $k$ in Figure 20, Section G.1. For two of the datasets, 66_living-room and 90_glue_qnli, we see that pruning works, with a 0.024 and 0.052 increase in AUC between the $k=8$ and $k=1$ probe for each dataset. However, for 7_hist_fig_ispolitician, OOD test AUC increases by 0.077 after pruning. This indicates that the task 7_hist_fig_ispolitician requires $k=8$ latents to represent.

Our preliminary results show that pruning helps SAE probes generalize. However, the drop between in-distribution (ID) and OOD performance is much larger than the modest improvement from pruning. This indicates that spurious correlations are not the primary reason SAE probes fail to generalize. Thus, we consider the second hypothesis - that the underlying SAE latents do not generalize well OOD.

On the task 66_living-room, latent 122774 has an ID test AUC of 0.99 while only having an OOD test AUC of 0.64. Since the OOD transformation involves translating the prompts to French, we hypothesize that this latent is active on the phrase “living room” in English but not other languages. We use GPT-4o to translate “living room” to 15 languages, and in Figure 8 we indeed observe that latent 122774 is more active on the English translation than all other languages, and it is not active on the French translation at all. Thus, latent 122774 does not generalize OOD, which explains why the SAE probe also failed to generalize.

Figure: Figure 8: Latent 122774 is most active on the English translation of “living room,” and does not fire on French.
Refer to caption: https://arxiv.org/html/2502.16681/x8.png

### 4.2 Latent Interpretability

We next generate autointerp descriptions for the most determinative latent for each dataset, allowing us to identify interesting categories of latents. Specifically, we consider each of the top 128 latents for each dataset for the $\mathrm{width}=131\mathrm{k},\mathrm{L0}=114$ SAE, and evaluate each latent’s $k=1$ SAE probe in standard settings. We then generate an autointerp explanation with GPT-4o for the best latent.

Table 5 (Section G.2) contains a breakdown of a selection of interesting top latents that we find. We see that some latents’ descriptions fit their tasks, like latent 81210 for 5_hist_fig_ismale, which activates on “references to female individuals.” Another interesting category is incorrect autointerp labels (e.g. for 125_wold_country_Italy, latent 50817 has 0.989 AUC, implying (correctly) that it fires on Italian concepts; however, the description reads “names of researchers and scientific authors”). However, some latents appear to be spurious and unrelated to the semantics of their dataset. For example, 22_headline_isobama, which targets headlines published during the Obama administration, is classified with $0.782$ AUC by latent 10555, which activates on “strings of numbers, often with mathematical notations.”

Note that these findings may be possible using baseline classifiers. For example, we could take a baseline classifier and apply it to model hidden states on tokens from the Pile and then examine maximally activating examples. However, a practical advantage for SAEs is that the infrastructure to perform autointerp is pre-existing through platforms like Neuronpedia, and a theoretical advantage is that the baseline classifier can only identify the single most relevant coarse-grained feature, while the decomposability of SAE probes into latents allows for identifying many independent features of various importance.

### 4.3 Detecting Dataset Quality Issues

We now investigate the top latents of two datasets, 87_glue_cola and 110_aimade_humangpt3. We find that although the latents identify errors in each dataset, baseline methods are also able to identify these dataset errors.

#### 4.3.1 GLUE CoLA

We first examine 87_glue_cola, an established linguistic acceptability dataset. CoLA prompts are standard English sentences with $\mathrm{target}=1$ if the sentence is grammatically correct and 0 0 otherwise. Latent 369585 from the $\mathrm{width}=1\mathrm{M}$ SAE has a test AUC of 0.76 and appears to fire on ungrammatical text. Surprisingly, when we look at prompts that latent 369585 fires on, those that it “disagrees” with the labels on (ones that are labeled grammatical), often appear to be ungrammatical. Although we first found this result with SAE probes, we later found that logistic regression was also capable of the same identification. In Table 7, we show the sentences that a baseline logistic regression classifier and latent 369585 mark as ungrammatical while the label is grammatical; most such sentences are truly ungrammatical. Thus, both SAE and baseline methods lead us to hypothesize that the CoLA dataset is partially mislabeled.

To test our hypothesis, we choose 3 LLMs - GPT-4o , Claude-3.5-Sonnet-New, and Llama-3.1-405B Instruct - to judge the linguistic acceptability of 1000 random CoLA prompts. We take the LLM majority vote as the ensembled, clean label. This is analogous to the experiment performed in that validated the CoLA dataset by ensembling the predictions of native English speakers, but with LLM judges instead of English speakers. We find that 2̃5% of CoLA labels are mislabeled by this metric (higher than the 13% disagreement rate from ). Notably, the Table 7 sentences are identified as ungrammatical by the ensemble while being labeled as grammatical in CoLA.

Figure: Figure 9: Latent 369585 outperforms a dense SAE probe and logistic regression when testing on mislabeled CoLA examples.
Refer to caption: https://arxiv.org/html/2502.16681/x9.png

Next, we train a baseline logistic regression, a $k=128$ dense SAE probe, and the latent 369585 classifier on the original CoLA labels. We then test on a held-out set with the clean, ensembled predictions. In Figure 9, we find that the original labels, the baseline classifier, the dense SAE probe, and latent 369585 all have about the same accuracy on the clean labels. Remarkably, when we test all classifiers on examples which were misclassified in CoLA (where the ensembled labels disagree with the original labels), we find that the latent 369585 classifier outperforms the dense SAE probe, which itself outperforms the baseline considerably.

#### 4.3.2 AI vs Human

Figure: Figure 10: Left: Top 3 latents (w/ descriptions) by mean activation difference between AI generated and human SAE latent activations. Right: Histograms of the identity of the 4 most common last tokens in AI generated and human text.
Refer to caption: https://arxiv.org/html/2502.16681/x10.png

We also investigate 110_aimade_humangpt3, which tests if probes can distinguish between human and ChatGPT-3.5 generated text. Latent 105150 has an AUC of 0.82 on this task, yet appears to fire primarily on periods and punctuation.(^2^22See [Neuronpedia](https://www.neuronpedia.org/gemma-2-9b/20-gemmascope-res-131k/105150) for examples.) Since this latent is syntactical and unrelated to the content of the dataset, we hypothesize it represents a spurious correlation in the dataset.

We examine the final token of each prompt in the dataset and indeed find a spurious feature: AI text is more likely to end in a period while human text is more likely to end in a space (see Figure 10). Although we discovered this by investigating SAE latents, we could reach the same conclusion with baselines: in Table 8 (Section G.4), we apply the logistic regression classifier to 2.8 million tokens from the Pile, and find that it is most active on punctuation tokens.

## 5 Why Didn’t This Work: Illusions of SAE Probes

report that SAE probes outperform baselines slightly. A natural question is why our experiments fail to replicate this result. One possible explanation is that ’s findings were based on a single dataset, whereas we evaluate across multiple datasets. We find that SAE probes outperform baselines on only a small subset of these datasets (2.2%, see Figure 11). It is possible that the dataset used by falls within this subset; however, without access to it, we cannot confirm this.

A separate explanation involves a potential illusion due to insufficiently strong baselines. Unlike our work, uses multi-token SAE probing: they aggregate the maximum value for each latent across all prompt tokens, while we only use last token latents. Crucially, they similarly max-pool each model dimension across prompt tokens for their comparative baseline, even though activations are unlikely to have privileged dimensions (in Table 9 we show that pooling activations in this way works poorly). We implement multi-token SAE probing using max-aggregation on $60$ random datasets from our list and $k=128$; see Table 9 for full results on these datasets. We also implement a better baseline: we train attention-pooled probes of the form

$$ $\displaystyle\left[\textrm{softmax}_{t\in[1,CtxLen]}\{X^{l}_{t}\cdot q\}\right ]\cdot\left[X^{l}\cdot v\right]$ (2) $$

for $q,v\in\mathbb{R}^{d}$. We find that when compared to the last-token baseline, max-pooled SAE probes win $19.6\%$ of the time, a considerable improvement over win rate of last-token SAE probes (2.2%, see Figure 11). However, when implementing attention-pooled baselines and using the quiver of arrows approach to select between pooled and last-token strategies for SAE probes and baselines, the SAE probe win rate drops more than $50\%$ to $8.7\%$ (see Figure 11).

Figure: Figure 11: Illustration of a potentially misleading result we find: when comparing activation probes to pooled SAE probes, SAE probe win rate increases considerably, but adding in a “pooled” attention-inspired probe brings down the SAE win-rate.
Refer to caption: https://arxiv.org/html/2502.16681/x11.png

## 6 Assessing Improvements in SAE Architectures

While SAE probes do not robustly outperform baselines, a separate interesting question is whether recent SAE architectural developments have improved SAE probing performance. We investigate this question by examining the performance of eight different SAE architectures released in the last two years, as outlined in Table 2.

**Table 2: Timeline of SAE Architecture Improvements.**
| SAE Architecture | Publication Date |
| --- | --- |
| ReLU (original) | October 4, 2023 |
| ReLU (updated) | April 26, 2024 |
| Gated | May 1, 2024 |
| TopK | June 6, 2024 |
| JumpReLU | July 19, 2024 |
| BatchTopK | July 19, 2024 |
| p-annealing | July 31, 2024 |
| Matryoshka | December 19, 2024 |

We use $\mathrm{width}=16\mathrm{k}$ Gemma-2-2B layer $12$ SAEs with a variety of $\mathrm{L0}$ values trained by . We create $k=16,128$ SAE probes on all regimes and datasets for each architecture, and additionally $k=1$ for standard conditions. In Figure 12, we plot each SAE architecture’s $k=16$ probe performance in standard conditions. We find the average test AUC of each SAE and then take the max across $\mathrm{L0}$ for each SAE architecture (using a single $\mathrm{L0}$ value is somewhat noisy, see Appendix I). This metric tests how expressive individual SAE latents are for different SAE architectures. While we see a slight positive trend for probing performance with more recently released SAE architectures, the effect is not statistically significant. For plots in all regimes, see Appendix I; in other regimes there seems to be a possible slight improvement with newer SAE architectures.

Figure: Figure 12: We see that there is a slight uptick in probing performance compared to baselines in standard conditions with more recent architectures. However, the spread of the data is considerably larger than this improvement.
Refer to caption: https://arxiv.org/html/2502.16681/x12.png

## 7 Conclusion

Our evaluation of sparse autoencoders (SAEs) reveals fundamental limitations in current SAE methodologies. Despite expectations that interpretable SAE latents would provide a competitive advantage in probing, we found no improvement over traditional methods across multiple regimes and over 100 datasets. Some failures expose critical weaknesses – for instance, SAE latents struggle to be robust to distributional shifts and fail to represent complex concepts – while others are more mundane – lower L0 SAEs create worse sparse probes. More broadly, our findings highlight the need for the field of mechanistic interpretability to evaluate techniques with rigorous baselines. While prior work reported advantages for SAE probes, our robust quiver of arrows methodology, use of stronger baselines, and evaluation on a large set of datasets demonstrate these advantages to be illusory. Even in our own analysis, initial conclusions favoring SAE interpretability were later overturned when proper baselines were considered. We do not consider this work to be a wholesale critique of the SAE paradigm. Instead, we view it as a single rigorously evaluated datapoint to contextualize SAE utility and to motivate a more thorough examination of methods in mechanistic interpretability.

Limitations We study the performance of SAE probes as a proxy for SAE utility. While we believe that probing performance is a more effective measurement than typical SAE evaluations like downstream cross-entropy loss or reconstruction error, it is ultimately still a proxy metric. We chose to study difficult probing regimes to try to find cases where the inductive bias of SAE latents outweighed imperfect SAE reconstruction, but it is possible that even a basis of “true” model representations would offer only a mild inductive bias advantage over traditional linear probing. We are thus excited for future work studying probing on toy models (e.g. models trained on board games ) where the “true” model features are known.

Finally, it is possible that further optimization of the SAE probe baseline might increase performance such that it beats baseline methods. For example, find that SAE probing with a number of modifications (multi-token probes, binarization of latents, and probing with $k=$ full SAE dimension) beats baseline methods on safety-relevant datasets (albeit only comparing to single-token baseline probes). However, we believe a main takeaway of our paper is that equivalent effort should be put into optimizing baselines. Indeed, we examine the effect of binarizing latents in Appendix J and find that it does not significantly help.
