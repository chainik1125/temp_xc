## Contents
- 1 Introduction
  - This work.
- 2 Preliminaries
  - Notations.
  - Sparse Coding.
  - Sparse Autoencoders (SAEs).
- 3 Temporal Structure in Language Model Activations
  - Increasing intrinsic dimensionality.
  - Non-stationarity: Context explains bulk of signal variance.
- 4 Temporal Priors of Sparse Autoencoders
  - Proposition 4.1 (Independence prior over time) .
  - Corollary 4.1.1 (Assumptions of time-invariant sparsity) .
  - Proposition 4.2 (Restrictive Sparsity Budget Leads to Support Switching in SAEs) .
- 5 Temporal Feature Analysis: Towards Predictive Approaches for Interpreting Neural Representations
  - Temporal Feature Analysis.
  - Sanity Checking Temporal Feature Analysis.
- 6 Capturing dynamic structure with Temporal Feature Analysis
  - Experimental Setup.
  - 6.1 The Geometry of Stories: A Narrative-Driven Domain
    - UMAP of Latent Codes Suggests Models Temporally Straighten Activations.
    - Predictive Component Captures Local Event Boundaries.
  - 6.2 Garden Path Sentences: Ambiguities Resolved via Temporal Structure
  - 6.3 In-Context Representations defined by entirely temporal relations
- 7 Further Investigation of Temporal Feature Analysis
  - 7.1 Geometry of latent codes in an OOD dialogue domain
  - 7.2 Automatic Interpretability of Novel Codes
- 8 Discussion
  - 8.1 Future Work
    - Features as Low-Dimensional Manifolds.
    - Temporal Straightening in Language Model Representations.
    - Priors in Temporal Feature Analysis.
    - Training Dynamics and Possible Variants of Temporal Feature Analysis
- Acknowledgments
- Appendix
  - Appendix A Experimental details
    - A.1 Autocorrelation Between Activations
    - A.2 Projection Analysis
    - A.3 U-statistic
      - A.3.1 Why is U-stat an intrinsic dimensionality measure
    - A.4 Surrogate
    - A.5 Analysis of Stories: UMAP Projections, Dendrograms, Similarity Heatmaps, and Event Boundary Detection
      - Data.
      - Analysis Details.
    - A.6 Analyzing Garden Path Sentences: Dendrograms and Phrase Similarity
      - Data.
      - Analysis Details.
    - A.7 Analyzing In-Context Representations: Visualization and Counterfactual Experiments
      - Data.
      - Analysis Details.
  - Appendix B Details on Training Temporal Feature Analyzers and SAEs
    - B.1 Training, Harvesting, and Inference for Temporal Feature Analysis and SAEs
      - Compute.
      - SAEs’ training.
      - Training procedure.
      - Activations normalization.
  - Appendix C Further Results: Language Model Representations (U-Statistic and Autocorrelation)
    - C.1 U-Statistic across domains for Gemma-2-2B
    - C.2 Autocorrelation across domains for Gemma-2-2B
  - Appendix D More Gemma Results
    - D.1 Hierarchical clustering of codes from story tokens
      - D.1.1 Story 1
      - D.1.2 Story 2
      - D.1.3 Story 3
    - D.2 Code: Analyzing Another Domain
    - D.3 Further Results: Similarity Maps on Stories
    - D.4 Further Results: Dendrograms and Evaluations on Garden Path Sentences
      - D.4.1 Dendrograms
    - D.5 Further Results: Counterfactual Generation with In-Context Representations
  - Appendix E Further Results: Temporal Feature Analysis on LLama-3.1-8B
    - E.1 Geometry, Dendrograms, and Spectra
      - E.1.1 Story 1
      - E.1.2 Story 2
      - E.1.3 Story 3
      - E.1.4 Comparing Eigenspectrum with Slow vs. Fast Features
    - E.2 Event Boundaries and Noise Stability
      - E.2.1 Event Boundaries
    - E.3 In-Context Representations
  - Appendix F Further Results: Temporal Feature Analysis with Only Predictive Component
  - Appendix G Further Results: Emergent Separation of Predictive and Novel Code in Temporal Feature Analysis
  - Appendix H Proofs
    - H.1 Priors on the Sparse Code for various SAEs
      - Proposition H.1 (Independence priors over time) .
      - Proof.
      - Bayesian Interpretation.
    - H.2 Sparsity over time and implications
      - Corollary H.1.1 (Assumptions of time-invariant sparsity) .
      - Proof.
      - Proposition H.2 (Restrictive Sparsity Budget Leads to Support Switching in SAEs) .
      - Proof.
  - Appendix I Further Theoretical Results: SAE-wise Characterization of Priors over concepts and time
    - Proposition I.1 (SAE Priors on Sparse Code) .
    - I.0.1 ReLU SAE
    - I.0.2 TopK SAE
    - I.0.3 BatchTopK SAE
    - I.0.4 JumpReLU SAE

## Abstract

Abstract Recovering meaningful concepts from language model activations is a central aim of interpretability. While existing feature extraction methods aim to identify concepts that are independent directions, it is unclear if this assumption can capture the rich temporal structure of language. Specifically, via a Bayesian lens, we demonstrate that Sparse Autoencoders (SAEs) impose priors that assume independence of concepts across time, implying stationarity. Meanwhile, language model representations exhibit rich temporal dynamics, including systematic growth in conceptual dimensionality, context-dependent correlations, and pronounced non-stationarity, in direct conflict with the priors of SAEs. Taking inspiration from computational neuroscience, we introduce a new interpretability objective—Temporal Feature Analysis—which possesses a temporal inductive bias to decompose representations at a given time into two parts: a predictable component, which can be inferred from the context, and a residual component, which captures novel information unexplained by the context. Temporal Feature Analyzers correctly parse garden path sentences, identify event boundaries, and more broadly delineate abstract, slow-moving information from novel, fast-moving information, while existing SAEs show significant pitfalls in all the above tasks. Overall, our results underscore the need for inductive biases that match the data in designing robust interpretability tools. Code available at https://github.com/eslubana/TemporalFeatureAnalysis .

### 1 Introduction

Given the success of Language Models (LMs) (bubeck2023sparks; geminigold), there is growing interest in understanding how such models incrementally update over sequences of tokens to exhibit complex behaviors (murthy2025inside; lindsey2025emergent; lindsey2025biology; lepori2025just; bigelow2025belief; tuckute2024driving; klindt2025superposition).
Interpretability research aims to make such analyses tractable, offering tools for hypothesis design, testing, and intervention based on evaluation of intermediate activations (geiger2025abstraction; sharkey2025open; bereska2024mechanistic).
Often, such work builds on hypothesized computational models of how concepts are encoded in a neural network’s representations, e.g., the linear representation hypothesis (LRH) (elhage2022superposition; arora2018linear), correspondingly motivating tools such as sparse autoencoders (SAEs) (gao2024scaling; cunningham2023sparse) for unsupervised extraction of a dictionary of vectors that (ideally) mediate human-interpretable concepts (mueller2025questrightmediatorsurveying).

A central challenge in “bottom-up” approaches to interpretability, like SAEs, is the mismatch between the assumptions of their underlying implementational account and the precise behavior or computation they intend to explain (jonas2017could; geiger2025abstraction; costa2025flat) (see Fig. [1](https://arxiv.org/html/2511.01836v3#S1.F1)).
For instance, since LRH posits that different concepts correspond to directions in activation space that can be independently manipulated, it implicitly claims the data distribution can be factorized into independently varying latent variables (allen2024unpicking).
This mismatch between the structure of the data distribution and strong priors codified in LRH can lead to misleading or pathological explanations when using SAEs to understand neural networks (chanin2025absorptionstudyingfeaturesplitting; bricken2023monosemanticity; hindupur2025projecting).
This raises a set of critical questions for using SAEs to interpret models trained on sequential data like language.
Specifically, since language exhibits rich temporal structure at multiple scales (marsenwilson-1980; thompson1999temporal)—e.g., sentences contain dependencies that link words across time (gibson2000dependency; mcelree2003memory), upcoming words can be anticipated from context (hale2001probabilistic; levy2008expectation), and discourse imposes structure over longer timescales through phenomena like event boundaries (zacks2007event; baldassano2017discovering)—one can ask what assumptions about temporal structures do SAEs make? How do these assumptions align with the actual temporal structure present in a LM’s activations?

Figure: Figure 1: The mismatch between SAE assumptions and temporal structure of language. An illustrative sentence describing attributes of Harry Potter is shown. When passed into a language model (LM), it leads to activations $\bm{x}_{t}$ that include concepts within them (possibly entangled): note the presence of large numbers of shared attributes over time, which manifest as correlations across time of activations. Sparse Autoencoders (SAEs) implicitly have an independence (i.i.d.) prior across time $t$ over their latents and thereby over concepts, which clashes with the true structure of language.
Refer to caption: x1.png

###### This work.

Building on the Bayesian interpretation of sparse coding (olshausen1996emergence; olshausen1997sparse)—the framework that motivates SAEs—we rephrase the optimization objective of SAEs as a MAP (maximum a posteriori) estimation problem.
This allows us to make explicit prior assumptions about temporal structure embedded in SAEs, showing they implicitly assume concepts are uncorrelated across time and the number of concepts necessary to explain an activation is time-invariant—that is, the information present at each token position is independent of information at other positions and uniformly distributed.
As we empirically show, these assumptions stand in stark contrast to the actual temporal structure present in language and language model representations, and can result in empirically observed pathologies in SAEs, such as feature splitting (bricken2023monosemanticity; chanin2025absorptionstudyingfeaturesplitting; bussmann2025learning).

These results then motivate us to draw a broader parallel between SAEs and computational neuroscience approaches for understanding neural data.
Specifically, population-level analyses of neural recordings have revealed that representations often lie on structured manifolds (khona2022attractor; nogueira2023geometry; sohn2019bayesian), challenging the reductionist assumption in sparse coding that computations occur via independently firing, monosemantic features (eichenbaum2018barlow; saxena2019towards; barack2021two).
This motivated a paradigm shift towards more structured analysis protocols—methods designed around the generative process of the behavior one aims to explain (schneider2023learnable; chen2018sparse).
Motivated by this and similar findings of intricate geometrical structure in neural network representations (fel2025into; gurnee2025when; modell2025origins), we propose Temporal Feature Analysis, a new protocol for interpreting language model activations that incorporates explicit inductive biases about temporal structure.
Our approach decomposes activations at each timestep into two orthogonal components: a predictable component, obtained by projecting current representations onto past context using a learned attention mechanism, and a novel component, representing residual information orthogonal to the predictable component.
That is, we assume the *novel* component—not the total representation—is uncorrelated over time.
This allows correlations between total codes and hence enables our method to capture the temporal structure of LM activations.
Overall, we make the following contributions in this paper.

- •
Highlighting temporal structure in language model (LM) representations (Sec. [3](https://arxiv.org/html/2511.01836v3#S3)). We demonstrate rich temporal dynamics in LM representations, which show nonstationarity through systematic growth in conceptual dimensions with time, time-varying activation correlations, and strong correlations with recent context, that are at odds with the implicit priors of SAEs.
- •
Precise characterization of prior assumptions of existing SAEs (Sec. [4](https://arxiv.org/html/2511.01836v3#S4)). Adopting a Bayesian perspective on the SAE objective, we show that SAEs implicitly assume an independent and identically distributed (i.i.d.) prior across time on their latents, and thereby on concepts. As a consequence, SAEs fail to capture local correlations in LM activations.
- •
Incorporating data-informed temporal inductive biases into a novel SAE architecture (Sec. [5](https://arxiv.org/html/2511.01836v3#S5)). Using empirically observed temporal structure in LM activations, we design a novel interpretability protocol—called Temporal Feature Analysis—which decomposes the model activation into two components: a predictive component, which captures slow-moving contextual information, and a novel component, which captures fast-moving stimulus-driven information.
- •
Using Temporal Feature Analysis to reveal temporal structures in LM representations (Sec. [6](https://arxiv.org/html/2511.01836v3#S6)). We show Temporal Feature Analysis’s ability to extract temporal structures by studying linguistic input with temporal structure at different scales: garden path sentences, stories, and in-context dialogue. Specifically, we show the extracted predictive codes: (1) correctly parse garden path sentences, (2) decompose stories into events, (3) capture precise structure of in-context representations, and (4) delineate user-assistant interactions through smooth trajectories. Meanwhile, the novel codes allow automatic interpretability pipelines, akin to SAEs.

### 2 Preliminaries

###### Notations.

Let bold, lowercase letters represent vectors (e.g., $\bm{z}$). Subscripts on vectors denote different samples (e.g., $\bm{z}_{i}$), while superscripts denote the index within the vector, leading to a scalar (e.g., $z^{k}$). We denote model activations by $\bm{x}\in\mathbb{R}^{n}$, SAE latents (sparse code) by $\bm{z}\in\mathbb{R}^{M}$, and the dictionary by $\bm{D}\in\mathbb{R}^{n\times M}$ ($M$ is the dictionary size).

###### Sparse Coding.

Sparse dictionary learning (olshausen1996emergence; olshausen1997sparse) expresses data as a sparse linear combination of dictionary elements, where both the weights and dictionary are learned from data. Intuitively, the dictionary behaves as a data-adaptive overcomplete basis; i.e., it typically has more elements that the dimension of ambient space. The optimization problem involved in this framework is $\operatorname*{arg\,min}_{\bm{D},\bm{z}}\frac{1}{N}\sum_{i=1}^{N}\|\bm{x}_{i}-\bm{D}\bm{z}_{i}\|_{2}^{2}+\lambda\mathcal{R}(\bm{z}_{i})$, where $\mathcal{R}(\cdot)$, typically chosen to be the $\ell_{1}$-norm, is a sparsity-inducing regularizer. Sparsity assists in picking the fewest most relevant dictionary atoms to explain a given data point.

###### Sparse Autoencoders (SAEs).

SAEs aim to disentangle (bengio2013representation; higgins2018towards; olahreps) neural network activations into human-interpretable concepts (cunningham2023sparse; bricken2023monosemanticity). Specifically, SAEs transform their inputs (i.e., neural network activations) into a latent representation which is encouraged to be sparse. As shown by hindupur2025projecting, this is achieved by solving the sparse coding problem using a specific parametric form for the sparse codes:

$$ $\begin{split}\operatorname*{arg\,min}_{\bm{D},\bm{z}}\frac{1}{T}\sum_{i=1}^{T}\|\bm{x}_{i}-\bm{D}\bm{z}_{i}\|_{2}^{2}+\lambda\mathcal{R}(\bm{z}_{i}),\quad\text{s.t. }\quad\bm{z}_{k}=f_{\mathtt{SAE}}(\bm{x}_{k})\;\forall k,\;\tilde{g}(\bm{z}_{1},\dots,\bm{z}_{T})=0,\end{split}$ (1) $$

where $\mathcal{R}(\cdot)$ is the regularizer (typically the $L_{1}$ norm), $f_{\mathtt{SAE}}$ is the SAE encoder architecture, and $\tilde{g}(\cdot)$ captures SAE-specific sparsity constraints on $\bm{z}$.
$f_{\mathtt{SAE}}$ is typically a single hidden layer as in the ReLU SAE (bricken2023monosemanticity; cunningham2023sparse), TopK SAE (gao2024scaling; makhzani2013k), JumpReLU SAE (rajamanoharan2024jumpingaheadimprovingreconstruction) and BatchTopK SAE (bussmann2024batchtopksparseautoencoders),
though recent work has also explored alternative architectures inspired by sparse coding algorithms to capture specific structures, e.g., hierarchies (muchane2025incorporatinghierarchicalsemanticssparse; costa2025flat).

### 3 Temporal Structure in Language Model Activations

Figure: Figure 2: Temporal structure of LLM activations reveals nonstationarity. We use Pile samples (monology2021pile-uncopyrighted) to analyze temporal structure from activations of two pretrained LMs, comparing it to a surrogate signal that is stationary in nature (see App. [A.4](https://arxiv.org/html/2511.01836v3#A1.SS4)). (a, e) Intrinsic dimension of model activations and stationary surrogate. (b, f) Autocorrelations $A(\bm{x}_{t},\bm{x}_{t-\tau})$ as a function of sequence position ($t$) and lag ($\tau$). (c, g) Autocorrelation of the stationary surrogate. (d, h) Variance explained by projecting current representation $\bm{x}_{t}$ onto past context window $\{\bm{x}_{t-1},\dots,\bm{x}_{t-w}\}$ with different sizes $w$, along with a baseline. Results consistently show representations getting ‘denser’ over time and being significantly more structured than a stationary surrogate.
Refer to caption: x2.png

To contextualize the prior assumptions made by SAEs about temporal structure in LMs’ activations, we first perform an empirical characterization of such temporal structure in pretrained LMs.
Specifically, since LMs are trained to generate coherent text by learning the distribution of natural language, one can expect their representations capture the rich phenomenology of its sequential structure (elman1990finding); indeed, recent work has in fact found LM representations to be predictive of human neural recordings during language comprehension (hosseini2024universality; schrimpf2021neural; tuckute2024driving; hong2024scale; georgiou2023using).
Motivated by this, we perform two experiments relevant to our discussion (see Fig. [2](https://arxiv.org/html/2511.01836v3#S3.F2)): (i) measuring intrinsic dimensionality—an approximation of the number of concepts necessary to explain the data, which can be expected to increase in a monotonic manner with time (zhong2024random; can2025statistical; barak2014working; meister2021revisiting)—and (ii) signal nonstationarity—which assesses whether model activations reflect the contextual relations between phrases of a passage  (zacks2007event).

###### Increasing intrinsic dimensionality.

Fig. [2](https://arxiv.org/html/2511.01836v3#S3.F2) (a,e) show the dimensionality of the underlying manifold structure (intrinsic dimension) in model activations. We estimate the intrinsic dimensionality at a fixed position across a set of sequences with the U-statistic (App. Sec. [A.3](https://arxiv.org/html/2511.01836v3#A1.SS3)). For language model activations, this metric increases steadily with sequence position. On the other hand, a stationary surrogate of the data (see App. Sec. [A.4](https://arxiv.org/html/2511.01836v3#A1.SS4)) shows nearly constant intrinsic dimension over time. This indicates that model activations get ‘denser’, i.e., they possess more information over time. Correspondingly, the number of concepts needed to explain them varies with context.

###### Non-stationarity: Context explains bulk of signal variance.

Subplots (b), (f) show the autocorrelation of model activations (App. Sec. [A.1](https://arxiv.org/html/2511.01836v3#A1.SS1)), which is noticeably different at different sequence positions (x-axis), while the stationary surrogate, as expected, shows nearly position-invariant autocorrelation values (subplots (c), (g)). This finding is a clear signature of time-dependent correlation structure, and therefore of non-stationarity. We quantify the similarity of a representation with its context in subplots (d), (h). Specifically, we project representations of token $\bm{x}_{t}$ at a given time $t$ onto the subspace spanned by preceeding representations in the context $\{\bm{x}_{<t}\}$. These subplots show that up to $80\%$ variance of $\bm{x}_{t}$ is explained by a context of 500 tokens, further highlighting strong cross-temporal correlations. Significant variance in the representation at time $t$, $\bm{x}_{t}$, can be predicted (expressed) using representations from the past context.

### 4 Temporal Priors of Sparse Autoencoders

We now state the prior assumptions made by existing SAEs regards sequential structure in an input, contrasting these assumptions with the empirical results shown in Sec. [3](https://arxiv.org/html/2511.01836v3#S3).
Specifically, building on the arguments used by olshausen1997sparse to formalize the problem of sparse coding, we note that the SAE training objective (Eq. [1](https://arxiv.org/html/2511.01836v3#S2.E1)) can be interpreted from a Bayesian lens: minimize the negative log posterior $\operatorname*{arg\,min}_{\{\bm{z}_{t}\}}-\log P(\bm{z}_{1},\dots,\bm{z}_{T}\mid\bm{x}_{1},\dots,\bm{x}_{T})$ of the data, which, by Bayes’ rule, can be written as the sum of log likelihood (MSE) and log prior (the regularizer $\mathcal{R}$).
From this lens, SAEs’ prior assumptions on sequential structure in LM activations can be described as follows.

###### Proposition 4.1 (Independence prior over time) .

Consider the SAE maximum aposteriori (MAP) objective from Eq. [1](https://arxiv.org/html/2511.01836v3#S2.E1). Since the sparsity constraints are additive over time, this objective has an independent and identically distributed (i.i.d.) prior over time:

$$ $P(\bm{z}_{1},\dots,\bm{z}_{T})\propto\prod_{t=1}^{T}\exp\left(-\lambda\mathcal{R}(\bm{z}_{i})-\tilde{\lambda}\tilde{g}(\bm{z}_{i})\right)=\prod_{i}P(\bm{z}_{i}).$ (2) $$

A more precise version of the claim for specific SAE architectures is provided in Appendix [H.1](https://arxiv.org/html/2511.01836v3#A8.SS1).
Intuitively, the claim above says that SAEs assume an independence of latents, and hence the concepts underlying the generative process of language, over time.
This directly conflicts with the rich contextual structure of LM activations we empirically observed in Fig. [2](https://arxiv.org/html/2511.01836v3#S3.F2)b–d, f–h.
Crucially, this also implies that SAEs assume the sparsity of latent codes necessary to explain model activations to be time-invariant, as stated formally in the corollary below.

###### Corollary 4.1.1 (Assumptions of time-invariant sparsity) .

As a consequence of the i.i.d. priors over time from Prop. [2](https://arxiv.org/html/2511.01836v3#S4.E2), standard SAEs assume that sparsity of representations emerges from a fixed distribution independently over time (i.i.d.), i.e., $P(\|\bm{z}_{1}\|_{0},\dots,\|\bm{z}_{T}\|_{0})=\prod_{t}P(\|\bm{z}_{t}\|_{0}).$

SAEs thus assume that sparsity—which, in their underlying generative model of activations corresponds to the number of concepts necessary for explaining the data (bricken2023monosemanticity; elhage2022superposition)—remains approximately constant over time.
This again does not align with the increasing dimensionality of representations observed in LM activations (see Fig. [2](https://arxiv.org/html/2511.01836v3#S3.F2)a,e).
Correspondingly, if enough concepts aggregate over context such that a model’s activations become ‘denser’ than the assumed sparsity budget, the assumption of time-invariant sparsity implies SAEs can fail to capture the temporal structure inherent in language, as formally stated below.

###### Proposition 4.2 (Restrictive Sparsity Budget Leads to Support Switching in SAEs) .

Suppose data $\bm{x}$ lies on a $C^{1}$ manifold $\mathcal{M}\subset\mathbb{R}^{d}$. For $\bm{x}\in\mathcal{M}$, let $T_{\bm{x}}\mathcal{M}$ denote the tangent space at $\bm{x}$, and $m(\bm{x})=\dim T_{\bm{x}}\mathcal{M}$ its dimension. Suppose an SAE $S$ with latent code $S(x)$ has a sparsity budget $|S(\bm{x})|=K$. If $K\leq m(\bm{x})$, under the assumption of low error (Eq. [1](https://arxiv.org/html/2511.01836v3#S2.E1)), in some neighborhood $\mathcal{N}_{\bm{x}}$ of $\bm{x}$, $\exists\;\bm{x}_{1},\bm{x}_{2}\in\mathcal{N}_{\bm{x}}$ s.t. $S(\bm{x}_{1})\neq S(\bm{x}_{2})$, i.e., support switching occurs in the SAE latents.

Figure: Figure 3: Sparsity splits local structure. (left) Temporally correlated inputs can yield geometrically structured activations. (right) If the sparsity budget is lower than the intrinsic dimensionality of the activation geometry, an SAE is incentivized to partition the manifold into local regions such that even nearby points map to disjoint codes and local structure is lost.
Refer to caption: final_figures/theory.png

Intuitively, the claim above says that even if an SAE achieves low reconstruction error, its specific geometric assumptions—use of independently firing directions and restricted intrinsic dimensionality of data—will lead to a “splitting” of the input data into several regions such that different sets of latents will activate for even locally nearby points (see Fig. [3](https://arxiv.org/html/2511.01836v3#S4.F3)).
Consequently, if the sparsity budget is too little or if model activations are more dense than the assumed budget, then local geometry present in the activations will be lost in the SAE latent codes—as is arguably already observed empirically with phenomena like feature splitting (chanin2025absorptionstudyingfeaturesplitting; bussmann2025learning; bricken2023monosemanticity).
For the context of our paper, as per our results in Fig. [2](https://arxiv.org/html/2511.01836v3#S3.F2) (a,e), temporal correlations can increase activations’ intrinsic dimensionality and, when viewed from the lens of the claim above, we arrive at the conclusion that SAEs can fail to capture temporal structure present in activations.
We empirically validate this claim in Sec. [6](https://arxiv.org/html/2511.01836v3#S6).
More broadly, we note there is growing evidence that neural network representations possess rich geometric structures across a broad set of modalities (gurnee2025when; fel2025into; park2025iclrincontextlearningrepresentations; engels2024not; costa2025flat; kantamneni2025language; pearce2025tree).
As per Prop. [4.2](https://arxiv.org/html/2511.01836v3#S4.Thmproposition2), the use of SAEs for interpreting behaviors that invoke such structures can be misleading and should be approached carefully (cf. (sengupta2018manifold)).

###### Proposition 4.1 (Independence prior over time) .

###### Corollary 4.1.1 (Assumptions of time-invariant sparsity) .

###### Proposition 4.2 (Restrictive Sparsity Budget Leads to Support Switching in SAEs) .

### 5 Temporal Feature Analysis: Towards Predictive Approaches for Interpreting Neural Representations

Figure: Figure 4: Schematic of Temporal Feature Analysis. Temporal Feature Analyzers decompose activations $\bm{x}_{t}$ into two components: a predictable component, obtained by projecting $\bm{x}_{t}$ onto a context direction (derived from the past $\bm{x}_{<t}$ using attention), and a sparse, novel component orthogonal to the predictable component that captures new information seen at time $t$.
Refer to caption: final_figures/architecture_latest.png

As stated in Sec. [2](https://arxiv.org/html/2511.01836v3#S2), sparse coding, a framework designed in computational neuroscience to understand neural representations in biological brains (olshausen1996emergence; olshausen1997sparse), inspired SAEs as a framework for interpreting artificial neural networks (bricken2023monosemanticity; cunningham2023sparse).
In fact, the parallels between these communities can be made deeper: motivated by observations of intricate geometry of neural representations derived out of multi-dimensional population analyses (khona2022attractor; nogueira2023geometry; sohn2019bayesian), there were calls in computational neuroscience to discard the limiting reduction assumed in sparse coding that computations occur via a set of independently firing, monosemantic features (eichenbaum2018barlow; saxena2019towards; barack2021two; seung1996brain; chung2021neural)—similar to our arguments in Sec. [3](https://arxiv.org/html/2511.01836v3#S3), [4](https://arxiv.org/html/2511.01836v3#S4) (and results that follow in Sec. [6](https://arxiv.org/html/2511.01836v3#S6)).
Correspondingly, a need for more structured protocols was suggested (eichenbaum2018barlow; barack2021two; chen2019sparse; sengupta2018manifold), leading to methods that were motivated by the generative process of the behavior one is trying to explain (schneider2023learnable; chen2018sparse; berkes2005slow; wiskott2002slow; linderman2017bayesian; yu2008gaussian).
We argue a similar paradigm shift is needed in language model interpretability: given that we train models to learn the distribution of highly structured data, we ought to embrace the fact that neural network activations can exhibit intricate geometrical organization.
These geometrical objects may in fact be the units of computation that are necessary for describing model computation (perich2025neural; eichenbaum2018barlow; modell2025origins), and hence we must design interpretability approaches to identify them.
In what follows, as an attempt to qualify our arguments, we propose one such approach, titled Temporal Feature Analysis, that focuses on the temporal structure of LM activations.
We also point the reader to relevant related work from identifiability literature (joshi2025identifiable; klindt2020towards).

###### Temporal Feature Analysis.

In computational neuroscience, when analyzing data from dynamical domains (e.g., audio, language, or video), a commonly made assumption is that there is contextual information present in the recent history that informs the next state—this part of the signal is deemed predictable (chen2019sparse; millidge2024predictive), slow-changing (berkes2005slow; klindt2020towards), invariant (olshausen2007learning; hyvarinen2003bubbles), or dense (tasissa2022improvingdiscriminativereconstructionsimultaneous).
Meanwhile, the remaining signal corresponds to new bits of information added by the observed state at the next timestep—this part can be deemed novel, fast-changing, variant, or sparse with respect to the context.
We argue LM activations are amenable to a similar generative model.
Specifically, our observations in Sec. [3](https://arxiv.org/html/2511.01836v3#S3) show that activations $\bm{x}_{t}$ at time $t$ are correlated with the context and can be decomposed into two such parts as well.
Motivated by this, we propose the following model of LM activations:

$$ $\displaystyle\bm{x}_{t}=\bm{x}_{p,t}+\bm{x}_{n,t},\quad\text{s.t. }\quad\bm{x}_{p,t}=\bm{D}\bm{z}_{p,t}\,\,\text{and}\,\,\bm{x}_{n,t}=\bm{D}\bm{z}_{n,t},$ (3) $$

where $\bm{x}_{p,t}$ denotes a predictable component of the signal that captures the correlations of $\bm{x}_{t}$ with past data $\left\{\bm{x}_{<t}\right\}$, while $\bm{x}_{n,t}$ denotes a novel component that represents new, i.e., uncorrelated with the past, information added by the current token $\bm{x}_{t}$.
To obtain $\bm{z}_{p,t}$, we project $\bm{x}_{t}$ onto $\{\bm{x}_{<t}\}$ to explain the predictable variance in $\bm{x}_{t}$ as a convex combination of past data.
Specifically, we apply an encoder (ReLU plus a linear map) to take the inputs to a latent space $\bm{v}$ and perform an attention operation $f$ in that space, yielding $\bm{z}_{p,t}=f(\ \{\phi(D^{T}\bm{x}_{1}),\dots,\phi(D^{T}\bm{x}_{t-1})\},\phi(D^{T}\bm{x}_{t}))$; here $\bm{x}_{t}$ defines the query and the remaining context serves as keys.
Meanwhile, $\bm{z}_{n,t}$ is defined using a standard SAE: $\bm{z}_{n,t}=\tilde{f}_{\mathtt{SAE}}(\bm{x}_{t},\bm{z}_{p,t})=\sigma(\bm{D}^{T}(\bm{x}_{t}-\bm{D}\bm{z}_{p,t}))$; we use a standard SAE nonlinearity (either TopK or BatchTopK) to instantiate $\sigma$, applying it to $\bm{x}_{t}-\bm{D}\bm{z}_{p,t}$ to derive $\bm{z}_{n,t}$.
See Fig. [4](https://arxiv.org/html/2511.01836v3#S5.F4) for an overall schematic of the encoding process.
The learning objective in Temporal Feature Analysis follows.

$$ $\begin{split}\operatorname*{arg\,min}_{\bm{D},\bm{z}}\frac{1}{T}&\sum_{i=1}^{T}\|\bm{x}_{i}-\bm{D}(\bm{z}_{p,i}+\bm{z}_{n,i})\|_{2}^{2}+\lambda\mathcal{R}(\bm{z}_{n,i}),\\ \text{s.t. }\,\,\,\bm{z}_{p,k}&=f_{\mathtt{SAE}}(\{\bm{x}_{<k}\}\bm{x}_{k})\,\,\,\text{and}\,\,\bm{z}_{n,k}=\tilde{f}_{\mathtt{SAE}}(\bm{x}_{k},\bm{z}_{p,k})\,\,\forall k.\end{split}$ (4) $$

Relating to Sec. [4](https://arxiv.org/html/2511.01836v3#S4), we note the prior assumption in Temporal Feature Analysis is that the residual $\bm{z}_{n,t}=\bm{z}_{t}-\bm{z}_{p,t}$, which captures the novel information in $\bm{x}_{t}$ remaining after removing the projections onto the past context, is i.i.d. over time.
This prior allows correlations between the total codes, and thereby between concepts, across time, instead of assuming that all concepts are time-independent.
However, we do note this prior is still an assumption: one can reasonably argue the residuals need not be i.i.d..
We nevertheless state this point to be explicit about assumptions involved in our work.

Figure: Table 1: Temporal Feature Analysis and SAEs achieves similar NMSE across domains (Simple Stories, Webtext, Code).

###### Sanity Checking Temporal Feature Analysis.

Before analyzing how different approaches represent the temporal structure of language, we demonstrate that Temporal Feature Analysis performs on par with SAEs on standard metrics such as reconstruction error.
Specifically, we train a Temporal Feature Analyzer and standard SAEs (ReLU, TopK, BatchTopK) on $1$B token activations extracted from Gemma-2-2B (team2024gemma) from the Pile-Uncopyrighted dataset (monology2021pile-uncopyrighted).
We also analyze a baseline of the prediction only module from Temporal Feature Analysis, reported as ‘Pred. only’, which can be expected to underperform since predicting the next-token representation is likely to be more difficult than reconstructing it.
Results are provided in Tab. [1](https://arxiv.org/html/2511.01836v3#S5.T1), [2](https://arxiv.org/html/2511.01836v3#S5.T2) and show competitive performance between all protocols, except Pred. only.
One can also assess which part of a fully trained Temporal Feature Analyzer is more salient in defining its performance, i.e., does the estimated predictive part $\hat{\bm{x}}_{p,t}$ contribute more to the reconstruction $\hat{\bm{x}}$ or does the estimated novel part $\hat{\bm{x}}_{n,t}$.
Results are reported in Tab. [3](https://arxiv.org/html/2511.01836v3#S5.T3).
We see that the error vectors, i.e., $\bm{x}-\hat{\bm{x}}_{p}$ and $\bm{x}-\hat{\bm{x}}_{n}$, are approximately orthogonal, suggesting the modules computing predictive and novel codes capture separate bits of information from the input signal.
Furthermore, we find that a bulk of the reconstructed signal $\hat{\bm{x}}_{t}$ (in the sense of norm) is captured by the predictive code—in fact, the percentage contribution of the predictive code is $\sim$80%, in line with numbers observed in Fig. [2](https://arxiv.org/html/2511.01836v3#S3.F2)d.
However, analyzing the reconstruction performance, we see the predictive component primarily contributes to achieving a good NMSE, while the novel component is more responsible for explaining the input signal variance.
These results align with the generative model assumed in Temporal Feature Analysis (Eq. [3](https://arxiv.org/html/2511.01836v3#S5.E3)).
Specifically, NMSE captures the average reconstruction, and hence a slower moving, contextual signal can expect to dominate its calculation; in fact, we see the attention layer used for defining the prediction module produces block structured attention maps related to sub-event chunks (see Fig. [13](https://arxiv.org/html/2511.01836v3#S8.F13)), suggesting more granular temporal dynamics are being captured by predictive part (we elaborate on this last point in the next section).
Meanwhile, variance assesses changes per dimension and timestamp in the signal, which better matches the inductive bias imposed on the novel part.

### 6 Capturing dynamic structure with Temporal Feature Analysis

Figure: Figure 5: Temporal Feature Analyzers unroll stories, decomposing into events. We consider model activations from a story and compute pairwise similarity of codes extracted from different interpretability protocols. (a) We see predictive codes from Temporal Feature Analysis organizes in hierarchical block structures that seem to align with (sub)event boundaries in the analyzed story, while the novel code primarily emphasizes sudden changes in the narrative; meanwhile, standard SAEs show a mixture of the two structures, with a stronger similarity to the structure exhibited by the novel codes. (b) We confirm the alignment of predictive codes with event boundaries by running an off-the-shelf hierarchical clustering algorithm, finding the token clusters indeed correspond to (sub)events occurring in the story as the narrative proceeds. Running this process on SAEs, we find this process yields temporally incoherent clusters that are primarily defined by lexical information.
Refer to caption: x3.png

We now assess the ability of different SAEs towards capturing temporal structure in language model representations, offering empirical evidence for our arguments in Sec. [4](https://arxiv.org/html/2511.01836v3#S4).
We contextualize these results with respect to our proposed protocol of Temporal Feature Analysis, hence assessing what new information about model activations a temporally-informed interpretability protocol buys us.

###### Experimental Setup.

We analyze standard SAEs (ReLU, TopK, and BatchTopK) and Temporal Feature Analyzers trained on 1B model activations from Layer 15 of a Gemma-2-2B model extracted from the Pile Uncopyrighted dataset (monology2021pile-uncopyrighted).
For evaluation domains, we again seek inspiration from cognitive and computational neuroscience literature on assessing humans’ abilities to process temporal structure in language.
These settings capture a spectrum of interaction between a system’s prior knowledge versus temporally offered information about a concept, i.e., how the concept gets used within the given context.
Specifically, we first use stories as a testbed for evaluation.
Motivated by prior work on understanding how narrative structures are parsed by humans (rumelhart1975notes; thorndyke1977cognitive; baldassano2017discovering; nastase2021narratives), we first analyze whether local event structures from stories are reflected in latent codes derived out of SAEs and Temporal Feature Analysis.
This help us evaluate how different protocols represent local vs. global (over time) semantic information.
To investigate a more syntactic variant of this experiment, we analyze latent codes extracted from garden path sentences, i.e., grammatically valid but ambiguous phrases in which the constituents, when parsed according to their typical syntactic role, lead to an incorrect overall sentence parse (e.g., “The old man the road”).
Thus, this setting captures a domain wherein the context requires interpreting constituent words in a manner that goes against the most likely interpretation from a model’s pretraining prior, i.e., the temporal information dominates the interpretation of a constituent.
While humans can resolve these sentences by backtracking (hale2001probabilistic; hanna2024incremental; levy2008expectation), LMs operate in an autoregressive manner and hence must capture the correct sentence parse in the forward direction.
Given that LMs possess the ability to do so (li2024incremental; hanna2024incremental), we compare whether latent codes extracted using SAEs and Temporal Feature Analysis relate sentence constituents according to the valid parse.
Finally, motivated by literature on cognitive maps (behrens2018cognitive), wherein a subject has to infer a latent in-context structure that relates two observations (tokens)—a capability LMs have been show to possess (park2025iclrincontextlearningrepresentations)—we assess whether different interpretability protocols’ latent codes capture these in-context representations.
In this last setting, a model cannot rely on its prior knowledge at all, since the behavior is entirely guided by the in-context, temporal structure.

#### 6.1 The Geometry of Stories: A Narrative-Driven Domain

###### UMAP of Latent Codes Suggests Models Temporally Straighten Activations.

We consider the TinyStories datasets (eldan2023tinystories) for its relatively straightforward narrative structures, and qualitatively analyze the geometry of latent codes extracted from model activations when processing these stories.
Visualizing the latent codes in a low-dimensional basis via a 3D UMAP projection (mcinnes2018umap), we see SAEs yield a highly irregular and unstructured geometry (see Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)a).
Calculating Tortuosity (bullitt2003measuring), a measure of how aligned local arcs are with respect to the global structure of a curve, we see very high values emerge for SAEs’ latent codes geometry, suggesting sudden changes in the local similarity as a story unravels. To further understand the results above, we highlight a specific token (‘and’) from the story, the UMAP analysis shows that standard SAEs generally just cluster tokens by lexical identity.
This is further corroborated by running a hierarchical clustering algorithm on the latent codes (dendrogram), finding temporally incoherent, but lexically related clusters (see Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b).

Figure: Table 4: Kernel similarity (CKA) between latent codes and model activations. Predictive codes from Temporal Feature Analyzers show strong similarity to slow-changing part of activations, while novel codes and SAEs primarily capture the fast-changing part.
Refer to caption: x4.png

Performing the experiments above on latent codes extracted from Temporal Feature Analyzers, we see that while the novel component exhibits behavior that is similar to standard SAEs’, the predictive component shows a smooth, regular curve that clusters together several tokens from the story around a similar point in space.
Again running hierarchical clustering, the extracted dendrograms suggest these clusters qualitatively correspond to events occurring in the story.
We emphasize this finding is very similar to the phenomenon of temporal straightening observed in neural recordings of human subjects when processing stories (xutemporal) and sequential visual data (henaff2019perceptual)!
Straightening simplifies future state prediction and recent findings show that such straightening is exhibited by language models (hosseini2023large); however, Temporal Feature Analysis recovers this phenomenon in an unsupervised manner.

Quantifying Similarity to Slow vs. Fast Moving Signals. To further quantify the straightening claim, we compute the temporal Fourier transform of the model activations and divide the frequency spectrum into two halves at a critical frequency $f_{c}$ such that the energy (i.e., sum of squared value of phase information) in the frequencies below $f_{c}$ equals that of the remaining ones.
We call the first split “slow part” of a sequence, and latter the “fast part”.
We then compute the correlation matrix defined by the slow and fast parts, compute their spectrum, and analyze how similar these spectrums are to the ones defined using different SAEs’ codes.
Results are shown in Fig. [6](https://arxiv.org/html/2511.01836v3#S6.F6).
We clearly see the spectrum extracted from the predictive component of Temporal Feature Analysis approximates that of the slow part of the representation, while the novel component’s spectrum is similar to that of the fast moving part.
Meanwhile, spectra of SAEs is primarily similar to the fast part, suggesting lack of ability to capture longer range dependencies necessary for interpreting narrative-driven texts like most language domains.
To quantify this further, we also measure kernel similarities (CKA) between the kernels respectively defined by the slow moving and fast moving signal to the different SAE codes (see Tab. [4](https://arxiv.org/html/2511.01836v3#S6.T4)).
Results clearly show Standard SAEs and novel codes are primarily similar to fast moving part of the representation, while predictive codes are similar to the slow moving part.

Figure: Figure 7: Predictive codes decompose stories into events. (a) We consider model representations from a synthetic story with well-defined event boundaries. (b) Computing pairwise cosine similarity of latent codes extracted using different protocols, we see the predictive code of Temporal Feature Analysis organizes in hierarchical block structures that seem to align with (sub)event boundaries in the analyzed story. (c) We confirm the alignment of predictive codes with event boundaries by computing average pairwise similarity of token latent codes for tokens that span the same event (‘within’) versus not (‘across’). Results clearly show high within-event similarity scores for the predictive code. (d) and (e) The results above are further corroborated by running a noising process on the latent codes: we add Gaussian noise of scale $\sigma$ to the input before computing latent codes, defining the similarity maps and computing explained variance of un-noised data. This process elicits coarser grained clusters from the similarity maps for the predictive code, suggesting the multi-scale temporal structure of stories is reflected in predictive codes (see App. [D](https://arxiv.org/html/2511.01836v3#A4) for other latent codes). (f) Our interactive interface generates the similarity matrix for predictive codes live. Detect event boundaries in your own stories: [https://www.neuronpedia.org/gemma-2-2b/12-temporal-res](https://www.neuronpedia.org/gemma-2-2b/12-temporal-res).
Refer to caption: x5.png

###### Predictive Component Captures Local Event Boundaries.

The results above demonstrate Temporal Feature Analyzers’ predictive component qualitatively align with event boundaries in a story.
To investigate this result more quantitatively, we use GPT-5 to create a synthetic dataset of 50 stories with well-defined event boundaries (see Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7)a for an example).
We extract latent codes for these stories’ tokens, center them by subtracting the mean to remove any globally shared information, and compute the cosine similarity of token to token latent codes.
If the latent codes reflect local event structure of a story, the cosine similarity (on average) will be high between token pairs that come from the same event and low (if not zero) between pairs sampled across events; see Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7)b for an example similarity map corresponding to the story shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7)a.
Results are shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7) (c) and corroborate our qualitative findings: we see predictive components from Temporal Feature Analyzers show substantially higher similarity of codes if tokens are sampled from within an event, while the novel component and SAEs generally show low similarity between two tokens.
These results are further supported by the robustness of Temporal Feature Analysis to noise.
Specifically, we see that when we add noise to the input data, which, on average, will lead to turning off of latents with small magnitudes (due to the encoding nonlinearity), the temporal structure of the data, if it is present, will be amplified.
We see precisely this effect in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7)d: the predictive components’ cosine similarity map under Gaussian noised input maps elicits coarser block structures with increasing noise scale; this is reminiscent of percolation or heat diffusion perspectives on graph clustering, wherein noise diffuses only within a connected component and hence community structure is elicited (von2007tutorial).
Correspondingly, we see Temporal Feature Analyzers respond most gracefully to noise: variance explained reduces slower than SAEs’, which in fact drops to $\sim$0 at some critical noise scale.

Figure: Figure 8: Hierarchically clustering SAE codes for garden path sentences. (a) Pairwise similarity maps of the predictive code from Temporal Feature Analysis link long-distance heads and dependents that define the ultimately correct parse in garden path sentences, while standard SAEs (e.g., BatchTopK) emphasize only local, transient relations, falling for the misleading cues. (b, c) Comparing the cosine similarity of average latent code extracted from the subject phrase (SP), verb phrase (V), and object phrase (OP), we see across ambiguous garden path sentences and unambiguous control variants thereof, only the predictive component of Temporal Features Analyzers shows consistent similarity scores (as expected if the SP and V ambiguity is reflect in the latent codes).
Refer to caption: x6.png

#### 6.2 Garden Path Sentences: Ambiguities Resolved via Temporal Structure

Garden-path sentences—e.g., The old man the boat—initially cue an incorrect local parse before a later token forces reanalysis.
Language models have been shown to be able to correctly parse such ambiguous sentences, offering in fact a predictive account of human per-token surprisals (li2024incremental; hanna2024incremental; oh2023transformer).
Interestingly, when using SAE codes to assess whether LM activations offer a valid parse of the sentence, we find hierarchical clustering of SAE codes yields a parse that is suggestive of the misleading cue; meanwhile, Temporal Feature Analysis recovers the correct parse by separating the predictable, slow-moving component of the representation from the novel, fast-changing residual.
Specifically, in Fig. [8](https://arxiv.org/html/2511.01836v3#S6.F8), we see the predictive codes link long-distance heads and dependents that reflect the correct parse (e.g., man as verb), producing coherent similarity structure over the full span, whereas standard SAEs emphasize only transient, local changes and miss these cross-temporal constraints (see App. [D.4](https://arxiv.org/html/2511.01836v3#A4.SS4) for more examples).

The results above suggest Temporal Feature Analyzers encode syntactic structure that unfolds over time when evidence to collapse the correct constituent parse emerges.
To make these results more quantitative in nature, we use GPT-5 to synthetically generate a set of 50 garden path sentences where the subject is ambiguous.
We create 50 control variants of these sentences such that the controls do not possess ambiguity with respect to typical parse of the sentence constituents: e.g., changing the subject in the sentence The old man the boat from old to sailors, yielding The sailors man the boat.
We then divide all sentences into their respective subject phrase (SP), verb phrase (V), and object phrase (OP): e.g., The old (SP), man (V), and the road (OP).
We compute the average latent code for tokens from these three constituent phrases, under the hypothesis that if the sentence ambiguity is reflected in the latent code, then until the OP shows up, the correct parse of prior words cannot be identified. Correspondingly, all valid parses must be stored in the same representation.
This suggests the cosine similarity of latent codes of V and SP tokens should be of a similar order in both the garden path and control sentences; meanwhile, the similarity between V and OP should be much higher than V and SP.
Results are reported in Fig. [8](https://arxiv.org/html/2511.01836v3#S6.F8) (b,c).
We clearly see extreme sensitivity in similarity values of SAE latent codes and the novel component of Temporal Feature Analyzers, but the predictive component is essentially invariant across sentence type, suggesting it captures the temporal dynamics likely relevant for a LM to parse garden path sentences.

Figure: Figure 9: In-context representations defined by temporal relations. (a) Schematic of the synthetic “cognitive map” task defined by park2025iclrincontextlearningrepresentations: a random walk over a small graph is concatenated into a string of tokens, which when fed into a Gemma-2-2B model leads to model next-token predictions that align with the graph structure. This behavior is exhibited in the model’s activations, since a mere 2D PCA shows the graph structure. (b) Using latent codes for low-dimension projection, we see only the *predictive* component of Temporal Feature Analysis recovers the graph’s geometry, whereas standard SAEs and the novel component only partially reflect it. (c) Counterfactual editing: projecting the final-token representation onto the PCA basis and replacing it with a target node’s code reliably shifts next-token predictions in a graph-consistent manner.
Refer to caption: x7.png

#### 6.3 In-Context Representations defined by entirely temporal relations

We finally use the recently described phenomenon of in-context representations by park2025iclrincontextlearningrepresentations as a domain to compare SAEs and Temporal Feature Analyzers.
In particular, inspired by experiments on cognitive maps (behrens2018cognitive), park2025iclrincontextlearningrepresentations propose to take a structured graph whose nodes are assigned entities a pretrained LM is likely to have seen before during training (e.g., apple, car, etc.); see Fig. [9](https://arxiv.org/html/2511.01836v3#S6.F9)a for an example.
A random walk of length 500 tokens is performed on this graph and the sampled nodes during the walk are concatenated together into a string.
When this is inputted into a pretrained language model, provided enough context, one finds (see Fig. [9](https://arxiv.org/html/2511.01836v3#S6.F9)a): (i) the model begins to produce next-token predictions inline with the graph structure and (ii) a 2D PCA projection of the model activations reflects this structure.
That is, the graph structure, which is randomly defined and hence impossible for the model to have seen during pretraining, is an entirely temporal construct that is inferred by the model in-context.

Can latent codes from SAEs or Temporal Feature Analyzers capture the above phenomenon?
To analyze this, we repeat the low-dimensional projection experiment with model activations, but use the latent codes instead.
We find that SAEs’ and the novel component from Temporal Feature Analyzers partially (but not entirely) reflects the geometry of the graph.
Meanwhile, the geometrical structure of the graph is vividly observed in the predictive component of the Temporal Feature Analyzers.
To further confirm of the utility of this representation, we perform a simple counterfactual analysis: we sample walks of different lengths, project the model activations for the last token onto the PCA basis defined in Fig. [9](https://arxiv.org/html/2511.01836v3#S6.F9)b, and finally replace it with the representation of a ‘target’ node.
Ideally, this intervention changes the model’s next-token predictions as expected by the neighborhood of the target node.
Results show for $\nicefrac{{4}}{{9}}$ nodes, the predictive component yields a counterfactual result as expected by the graph structure; meanwhile, no other latent representation offers the ability to counterfactually edit model beliefs (see Fig. [9](https://arxiv.org/html/2511.01836v3#S6.F9)c for an example).

### 7 Further Investigation of Temporal Feature Analysis

In the results above, we showed across three domains that Temporal Feature Analysis elicits intricate in-context geometry inline with our expectations based on experiments with LM activations in Sec. [3](https://arxiv.org/html/2511.01836v3#S3).
Next, to make a case for following such more structured approaches for interpretability, we add to the results above to show that Temporal Feature Analysis (i) offers a new way of interpreting temporally structured domains (e.g., user–model chats) via the context-dependent predictive component, and (ii) information that can be identified via SAEs continues to remain available within the novel component of Temporal Feature Analysis.

Figure: Figure 10: Population and trajectory geometry in OOD dialogue. UMAPs of raw activations and SAE codes on Ultrachat (first 350 tokens per conversation). Top: population embeddings colored by sequence position (gray$\to$color) and marked by role (user=$\circ$, assistant=$\times$); only the *predictive* code cleanly separates speaker roles while preserving a smooth temporal gradient. Bottom: a single conversation overlaid on the same UMAP; the predictive code follows a smooth near-geodesic with low tortuosity $\tau$, in contrast to jagged paths from standard SAEs/novel codes.
Refer to caption: x8.png

#### 7.1 Geometry of latent codes in an OOD dialogue domain

We analyze a multi-turn chat domain to test whether temporal feature analysis continues to reveal slow-moving structure.
We emphasize the activations in this experiment are sampled from a Gemma-2-2B-IT model, although all analyzed interpretability protocols were trained on the base (non-instruction tuned) model.
Thus, in a sense, the experiments in this section also gauge the ability of different protocols to generalize out-of-distribution (OOD)—a property SAEs have been argued to partially possess when transferring between base and the corresponding instruction-tuned models (saetransfer).
Specifically, we use the Ultrachat dataset, where we take the first 350 tokens from 1,000 conversations (mostly single-turn within this window, with up to four turns).
For each token, we compute a shared 2D UMAP embedding from (i) raw model activations and (ii) latent codes from ReLU, TopK, and BatchTopK SAEs, as well as the novel and predictive components of Temporal Feature Analyzers.
To visualize population structure clearly, we display only 100 conversations at a time to avoid clutter (Fig. [10](https://arxiv.org/html/2511.01836v3#S7.F10)), though fitting is applied to all 1,000.

At the population level, the predictive component of Temporal Feature Analysis is the only representation that cleanly organizes the data along *two* interpretable axes: (1) a *speaker-role separation*, with user and assistant tokens forming distinct yet smoothly adjoining manifolds, and (2) a *temporal gradient*, with positions flowing continuously from early to late tokens along the manifold. By contrast, standard SAEs and the novel code show diffuse clouds with weaker role separability and no coherent positional gradient (Fig. 10, top row).
To assess within-sequence geometry, we overlay a single conversation’s trajectory onto the population embedding (Fig. 10, bottom row). Predictive codes trace a *smooth trajectory* that remains locally coherent within each role segment and transitions gracefully at speaker switches. In contrast, standard SAEs and the novel code produce jagged, zig–zagging paths. We again quantify this with *tortuosity* $\tau$ (bullitt2003measuring) (lower is straighter): predictive codes achieve $\tau\approx 2.0$, whereas SAEs and the novel code yield $\tau\approx 25$–$30$, indicating substantially more local turning (Fig. 10, panel headers). This straightening mirrors the story-domain results (Sec. [6.1](https://arxiv.org/html/2511.01836v3#S6.SS1)), now in an OOD conversational chat regime.

Figure: Figure 11: Interpretability of novel codes. We qualitatively compare the interpretability of novel codes extracted using Temporal Feature Analysis with latent codes extracted using GemmaScope SAEs across four diverse categories. Example feature cards are shown across four categories (Optimism, Manipulation, University, Slang), produced with the same activation-triggered examples plus direct logit attribution workflow (GemmaScope / Neuronpedia) used for standard SAEs. Browse the full list of features [on Neuronpedia](https://www.neuronpedia.org/list/cmh9dj4cq0001i9h).
Refer to caption: x9.png

#### 7.2 Automatic Interpretability of Novel Codes

A central benefit of SAEs is that their latents admit automatic interpretability pipelines (“autointerp”) that attach human-readable semantics to features via activation-triggered examples, token/phrase saliency, and lightweight dashboards (e.g., GemmaScope / Neuronpedia). Our Temporal Feature Analyzer adds a new, temporally aware predictive axis, but crucially it does not sacrifice these established interpretability affordances: the novel code remains compatible with standard SAE autointerp workflows. In other words, Temporal Feature Analysis augments the toolbox rather than replacing it.
To support this point, we apply a conventional autointerp loop to the novel codes: (i) collect high-activation snippets and nearest-neighbor contexts for each latent; (ii) compute direct logit attribution to inspect which vocabulary items a latent tends to support or suppress; and (iii) render compact, browsable “feature cards” that aggregate examples and scores (as in GemmaScope / Neuronpedia). While we do not perform an exhaustive causal-intervention study here, nothing in the pipeline is specific to standard SAEs; the same probes and dashboards operate unchanged on the novel codes, and we leave a systematic intervention suite to future work.
Qualitative inspection confirms that the novel codes recover the kinds of monosemantic, language-level features typically reported for SAEs. Figure 11 (p. 13) shows representative examples across diverse categories—Optimism, Manipulation, University, and Slang—where the novel latents consistently activate on intuitively relevant cue phrases and contexts, and where their attribution profiles align with the expected lexical sets.
Overall, when standard SAEs produce readable feature cards, the Temporal Feature Analysis’s novel stream does as well. This complements our earlier results: the predictive stream captures slow, contextual structure (events, roles, long-range constraints), whereas the novel stream concentrates the fast, stimulus-driven information that matches inferences possible via existing SAEs, including autointerp pipelines.

### 8 Discussion

A longstanding view in computational and cognitive neuroscience treats perception as predictive and time-coupled: slowly varying, context-invariant structure coexists with fast-changing, surprising residuals (levy2008expectation; millidge2024predictive; shain2020fmri).
We find that this structure is reflected in LM activations (Sec. [3](https://arxiv.org/html/2511.01836v3#S3)): activations are strongly correlated with recent context and can be decomposed accordingly.
In contrast, SAEs assume independence over time; this is misaligned with the identified non-stationary structure in language model activations.
Our protocol—Temporal Feature Analysis—relaxes this assumption, incorporating empirically observed temporal correlations to surface structure that i.i.d. priors obscure.
We showed that Temporal Feature Analysis is able to highlight temporal structures in its predictable codes that SAEs are unable to capture due to their prior-data mismatch.
Overall, our work adds to recent findings that inductive biases about the specific application under study are necessary to build improved interpretability tools (hindupur2025projecting; costa2025flat; sutter2025the).

#### 8.1 Future Work

Our findings provide indirect support for alternative perspectives on the structure of concepts in language models. Here, we highlight these perspectives, and emphasize directions that we believe will have significant impact.

###### Features as Low-Dimensional Manifolds.

We believe our work brings interpretability of language models one step closer to recent advances in interpretability of neural networks designed to simulate computational neuroscience tasks (sussillo2013opening; cotler2023analyzing; mante2013context; maheswaranathan2019universality; maheswaranathan2020recurrent).
Specifically, rather than isolated directions, as is assumed in both sparse coding (olshausen1996emergence; olshausen1997sparse) and SAEs (elhage2022superposition; bricken2023monosemanticity), “features” are better viewed as low-dimensional manifolds that evolve over time—as has also been stated in other recent work (modell2025origins; fel2025into; gurnee2025when).
In particular, we see in our results that predictive codes trace smooth, stable trajectories (events, roles, long-horizon constraints), whereas novel codes capture transient excursions.
Under this framing, high reconstruction with poor temporal structure in standard SAEs reflects a prior that fragments local manifold geometry (as formalized in Sec. [4](https://arxiv.org/html/2511.01836v3#S4)); adding the temporal prior helps preserve the geometry that yields temporally coherent representations (berkes2005slow; olshausen2007learning).

###### Temporal Straightening in Language Model Representations.

Across narrative and dialogue settings, the Temporal Feature Analyzer’s predictable code organizes representations into hierarchical blocks and smooth paths—an effect reminiscent of “temporal straightening”  (henaff2019perceptual; hosseini2023large; xutemporal), where the representational trajectory of an input sequence becomes linearized to support extrapolation toward future states of the sequence. In stories, predictive codes align with (sub)event boundaries and show robustness under noise; in OOD dialogue, they cleanly separate roles and produce near-geodesic single-conversation paths. Kernel spectra and CKA further support the slow/fast split: predictive codes align with slow components; novel codes and standard SAEs align with fast ones. We believe this dynamic warrants further investigation, and may serve fertile ground to develop computational hypotheses about temporal straightening itself.

###### Priors in Temporal Feature Analysis.

The encoding process in Temporal Feature Analysis realizes the predictable component with a single-layer self-attention module, which may bias the kinds of dependencies it can capture. Future work could place the inductive bias inside the sparsity regularizer (retaining SAE encoders) or adopt richer temporal encoders. More broadly, we assume i.i.d. residuals in the novel stream; relaxing this to allow controlled temporal correlations is an important direction. Finally, scaling causal evaluations—intervening on codes during generation—will help move from qualitative alignment to mechanistic claims.

###### Training Dynamics and Possible Variants of Temporal Feature Analysis

Since the goal of proposing Temporal Feature Analysis was to identify and highlight the benefits of accommodating the temporal structure of model activations, we did not perform significant tuning of the proposed protocol. That is, we strongly emphasize that our proposed protocol is just an example and we hope much better variants will likely follow as the community explores this research direction—as, e.g., happened for SAEs, which started with vanilla ReLU variants (cunningham2023sparse; bricken2023monosemanticity), eventually leading to much more sophisticated approaches eventually (rajamanoharan2024jumpingaheadimprovingreconstruction; bussmann2024batchtopksparseautoencoders; costa2025flat). To assist this, we highlight the following.

Figure: Figure 12: Competition Dynamics. We see while the overall learning curve looks smooth, the predictive and novel code in fact compete with each other to explain the input signal.
Refer to caption: x10.png

1. Competition dynamic in the learning process.
The precise way the training objective is defined in the current version of Temporal Feature Analysis (see Eq. [4](https://arxiv.org/html/2511.01836v3#S5.E4)), there is a competition dynamic imposed between the predictive and novel component for explaining model representations. Specifically, we found that learning dynamics underwent three phases (see Fig. [12](https://arxiv.org/html/2511.01836v3#S8.F12)). First, since the objective of predicting a future direction is more challenging than reconstruction to begin with, we found that the novel part, which is defined using a standard SAE and can in fact reconstruct a bulk of the training signal (since SAEs are trainable in the first place), will dominate the learning dynamics. Then, once the dictionary quality improves, the ability to predict starts to improve and the learning dynamics goes through a chaotic phase with successful training resulting in the predictive component taking over the learning dynamics. Finally, we saw that the novel component keeps conceding its dominance over the reconstructed signal, such that its NMSE finally becomes very high and the predictive component’s NMSE becomes very low. Unsuccessful training, however, can also happen here: at times we saw a fourth phase occurs where the novel component, relatively suddenly, takes over the learning dynamics. This would make the predictive component futile. This was rarely seen when working with Gemma models, but a frequent occurrence in Llama models; we resolved this problem by switching from using TopK to BatchTopk activation for the novel component, which resulted in perfectly stable training for Llama. We believe there are principled mechanisms to address this challenge, however. For example, Temporal Feature Analysis, in its structure, is very similar to Kalman filters (welch1995introduction)—latter generally involves Markov assumption, but Temporal Feature Analysis uses the entire context. Seeking inspiration from training objectives for Kalman filters, it can be useful to decompose the loss into two halves: (i) a term that uses only the predictive component to reconstruct the signal, and (ii) a term that uses the novel component to reconstruct just the residual. This breaks the competition dynamic by introducing an asymmetry in the training objective.

Figure: Figure 13: Attention Patterns. Attention weights for single sequences across three domains show low-rank structure in similarities.
Refer to caption: x11.png

2. Rank sparsity.
While the predictive component is dense in the sense used in interpretability literature, i.e., its $L_{0}$ norm is not constrained and can be high, in practice we found the learned attention maps and cosine similarity matrices (e.g., in Sec. [6.1](https://arxiv.org/html/2511.01836v3#S6.SS1)) to be very low-rank (e.g., see Fig. [13](https://arxiv.org/html/2511.01836v3#S8.F13)). In fact, we note in the broader field of dictionary learning and sparse coding (elad2010sparse), an alternative notion of sparsity is rank-sparsity. Specifically, in this paradigm, one constrains the rank of a matrix to be low: in the Euclidean sense, the latent code to reconstruct the target signal will still be dense, but the number of directions used to achieve that, in an alternative space, will be small. While one can optimize for this property using a nuclear norm like regularization objective, this can be expensive in practice (due to the need of an SVD). An alternative is thus to constrain the dimensions of the factors used for a signal’s reconstruction, i.e., what is often called imposing a “rank bottleneck”. When using this methodology on the predictive component, i.e., reducing the dimensions in the K, Q, V matrices of the attention layer, we find the resulting Temporal Feature Analyzers to achieve similar reconstruction as the unconstrained ones. While we did not explore this thread further, we believe it can be promising and is related to recent work on interpreting attention layers (he2025towards; kamath2025tracing).

Figure: Figure 14: Llama Learning Curves. Temporal Feature Analysis works best in middle layers of the model, with training on deeper layers often showing a learning dynamic whereby the novel part eventually takes over and starts to explain a bulk of the signal.
Refer to caption: x12.png

3. Layer choice.
For all our experiments, we trained SAEs and Temporal Feature Analyzers at layers found around 50% model depth (Layer 12 for Gemma and Layer 15 for Llama; both 0-indexed). When we tried to train Temporal Feature Analyzers on a deeper layer—specifically, layer 26 ($\sim$81% depth) of Llama—we found the predictive component would not be dominate the norm of the latent code (see Fig. [14](https://arxiv.org/html/2511.01836v3#S8.F14)). We believe this is a feature, not a bug. In particular, deeper layers have to unembed the extracted contextual signal to make a next-token prediction, and correspondingly a model is likely to discard unnecessary information in the deeper layer that can interfere with its ability to confidently produce next-token predictions. This claim is corroborated by block structured similarities seen in CKA and other representation similarity analysis literature (kornblith2019similarity; raghu2017svcca). We believe this thread warrants further investigation, as it may hint at a deeper result on temporal dynamics be varied across different layers, such that one could argue SAEs have the right inductive biases for interpreting deeper layers, but not middle or earlier layers.

### Acknowledgments

ESL thanks members of the Mechanisms team at Goodfire AI, particularly Tom McGrath, Owen Lewis, Jack Merullo, and Atticus Geiger for helpful discussions. ESL additionally thanks Jack Lindsey for useful comments that helped formalize the intrinsic dimensionality arguments in Sec. [4](https://arxiv.org/html/2511.01836v3#S4), and Raphael Sarfati for suggesting the Tortuosity metric used in Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5).
The authors also thank David Bau for a helpful discussion about the generalization ability of SAEs, and David Klindt for several useful comments on an earlier draft of the paper.
ESL and SSRH further thank Noor Sajid and members of the CRISP lab at Harvard for helpful conversations.
CR is supported by a MATS extension grant.
AM’s work is partially supported by the National Science Foundation (Grant No. 2530728) and Binational Science Foundation.
This work has been made possible in part by a gift from the Chan Zuckerberg Initiative Foundation to establish the Kempner Institute at Harvard University.

## Appendix

### Appendix A Experimental details

Below, we discuss broad details of the experimental setup and measures used for corresponding evaluations.
For experiments with LM representations, we primarily analyze Gemma-2-2B and Llama-3.1-8B models in this paper.
Specifically, we precache 10K activations for the analyzed datasets for our experiments on dynamics on language model representations.

#### A.1 Autocorrelation Between Activations

We compute autocorrelation by selecting evenly spaced tokens across the sequence and measuring the cosine similarity between each token and tokens at various lags in the past. Specifically, for tokens at position $t$, we compute similarities to tokens at $t-w$ where lag $w$ ranges from 5 to 20. This creates a heatmap where rows represent lag offsets and columns represent token positions.

For a stationary process, we expect the autocorrelation pattern to remain consistent across time—that is, the relationship between a token and its historical context should be similar regardless of position in the sequence. This would manifest as similar autocorrelation patterns repeating horizontally across token positions. In contrast, for a non-stationary process where representations evolve over time, we expect the autocorrelation patterns to vary systematically across positions, with columns showing different temporal dependency structures as the sequence progresses.

#### A.2 Projection Analysis

This analysis quantifies how much of the representation $\bm{x}_{t}\in\mathbb{R}^{D}$ at token position $t$ can be reconstructed from its preceding context $\{\bm{x}_{i}\}_{i\in W}$, where the context window $W=[t-w,t-1]$ contains the previous $w$ tokens.

For a population of $B$ sentence samples, we compute the projection of each target representation onto the subspace spanned by its context. Let $\mathbf{X}_{W}^{(b)}=[\bm{x}_{t-w}^{(b)},\ldots,\bm{x}_{t-1}^{(b)}]^{T}\in\mathbb{R}^{w\times D}$ denote the matrix of context representations for sample $b$, where each row is a context vector.
Prior to projection, we center all representations by subtracting the mean computed over the target positions across samples. The projection of $\bm{x}_{t}^{(b)}$ onto the span of the context is:

$$ $\displaystyle\mathbf{c}_{t,w}^{(b)}={\text{span}(\mathbf{X}_{W}^{(b)})}\cdot\bm{x}_{t}^{(b)}$ (5) $$

where $\text{span}(\mathbf{X}_{W}^{(b)})$ is the row space of $\mathbf{X}_{W}^{(b)}$, that can be computed via SVD.
The variance explained by the context is:

$$ $\displaystyle\text{expvar}(t,w)=\frac{\sum_{d=1}^{D}\text{var}(\mathbf{c}_{t,w,d})}{\sum_{d=1}^{D}\text{var}(\mathbf{x}_{t,d})}$ (6) $$

where $\text{var}(\mathbf{c}_{t,w,d})$ and $\text{var}(\bm{x}_{t,d})$ denote the variance of the $d$-th dimension across the $B$ samples. This ratio measures the fraction of total representational variance that can be linearly reconstructed from the preceding context. Values approaching 1 indicate high predictability from context; values near 0 indicate representations largely orthogonal to their context subspace.

#### A.3 U-statistic

We initially experimented with more well-known measures of intrinsic dimensionality, but found them to either be too computationally expensive [costa2005estimating] or saturate due to inappropriate assumptions (e.g., data satisfying a Euclidean geometry) [carter2009local].
We thus defined an intrinsic dimensionality measure for a specific generative process for activations that is motivated by the linear representation hypothesis [park2023linear, elhage2022superposition, costa2025flat, arora2018linear].
Specifically, assume that each observation (activation) $x_{t}\in\mathbb{R}^{D}$ is a unit-norm vector generated from an (unknown) overcomplete dictionary $V=[v_{1},\dots,v_{N}]\in\mathbb{R}^{D\times N},$ where $D<N$, whose columns are approximately orthonormal, in the sense that $\langle v_{i},v_{j}\rangle$ is small for $i\neq j$ and $\|v_{i}\|_{2}=1$.
At time $t$, the observation is $x_{t}=Va_{t}$ and $\|x_{t}\|_{2}=1$, where $a_{t}\in\mathbb{R}^{N}$ is a random coefficient vector with a time-dependent distribution $P_{t}(a)$.
Thus the process is nonstationary through the evolution of $P_{t}$ (and hence $a_{t}$), while the dictionary $V$ is fixed, matching the phenomenology observed in Sec. [3](https://arxiv.org/html/2511.01836v3#S3).

Under the assumptions above, we measure the intrinsic dimensionality of LM activations based on pairwise cosine similarities. Specifically, let $\mathbf{X}_{t}=[\bm{x}_{t}^{(1)},\dots,\bm{x}_{t}^{(M)}]$ be $M$ samples of normalized activations at time $t$ (from different timeseries). Define the gram matrix $\mathbf{G}_{t}=\mathbf{X}_{t}^{T}\mathbf{X}_{t}$. Then, a U-statistic of intrinsic dimension can be defined as follows:

$$ $\displaystyle\text{U-stat}(t)=\frac{M^{2}-M}{\|\mathbf{G}_{t}\|_{F}^{2}-M},$ (7) $$

where $\|\mathbf{G}_{t}\|_{F}^{2}$ is the squared Frobenius norm of the Gram matrix.
As we show in the following subsection, this quantity is an estimator of the effective rank $\nicefrac{{1}}{{\text{tr}(\mathbf{C}_{t}^{2})}}$, where $\mathbf{C}_{t}=\mathbb{E}[\bm{x}_{t}\bm{x}_{t}^{T}]$ is the second moment matrix and $\bm{x}_{t}$ is the activation vector at time $t$.
Under stationarity, U-stat remains constant; when representations evolve over time, U-stat increases systematically as more orthogonal directions become active.

##### A.3.1 Why is U-stat an intrinsic dimensionality measure

Let $P_{t}$ denote the distribution of $x_{t}$ at time $t$ and define the (time-local) second-moment matrix $C_{t}\;:=\;\mathbb{E}_{x\sim P_{t}}[xx^{\top}]\in\mathbb{R}^{D\times D}$.
Assuming $\|x\|_{2}=1$, we have $\operatorname{tr}(C_{t})=1$.
Given $M$ samples $\{x_{t,1},\dots,x_{t,M}\}$ drawn from $P_{t}$ (e.g., from a time window or an ensemble of trajectories at time $t$), we form the Gram matrix $G_{t}\in\mathbb{R}^{M\times M}$ with entries $(G_{t})_{ij}=x_{t,i}^{\top}x_{t,j}$.
The corresponding U-statistic

$$ $\hat{U}_{t}\;:=\;\frac{1}{M(M-1)}\sum_{i\neq j}(x_{t,i}^{\top}x_{t,j})^{2}\;=\;\frac{\|G_{t}\|_{F}^{2}-M}{M^{2}-M}$ $$

is a consistent estimator of

$$ $U_{t}\;:=\;\mathbb{E}_{x,x^{\prime}\sim P_{t}}[(x^{\top}x^{\prime})^{2}]\;=\;\operatorname{tr}(C_{t}^{2}),$ $$

where $x$ and $x^{\prime}$ are independent draws from $P_{t}$.
Thus $\hat{U}_{t}$ provides a basis-free estimate of $\operatorname{tr}(C_{t}^{2})$.

Now, a natural, rotation-invariant notion of intrinsic dimensionality at time $t$ is the (order-2) effective rank of $C_{t}$, $d_{\mathrm{eff}}(t)\;:=\;\frac{(\operatorname{tr}C_{t})^{2}}{\operatorname{tr}(C_{t}^{2})}$.
For unit-norm data $\operatorname{tr}(C_{t})=1$, so

$$ $d_{\mathrm{eff}}(t)\;=\;\frac{1}{\operatorname{tr}(C_{t}^{2})}\;=\;\frac{1}{U_{t}},\qquad\hat{d}_{\mathrm{eff}}(t)\;:=\;\frac{1}{\hat{U}_{t}}\;=\;\frac{M^{2}-M}{\|G_{t}\|_{F}^{2}-M}.$ $$

Let $\lambda_{1}(t),\dots,\lambda_{D}(t)$ be the eigenvalues of $C_{t}$.
Then $\operatorname{tr}(C_{t}^{2})=\sum_{i=1}^{D}\lambda_{i}(t)^{2}$. This yields the following.

$$ $d_{\mathrm{eff}}(t)=\frac{1}{\sum_{i=1}^{D}\lambda_{i}(t)^{2}}.$ $$

Now if the process at time $t$ spreads its energy equally over exactly $k$ orthogonal directions, then $\lambda_{1}(t)=\dots=\lambda_{k}(t)=1/k$ and $\lambda_{k+1}(t)=\dots=0$, and, hence, $d_{\mathrm{eff}}(t)=k$.
However, if most of the energy lies in a single direction, then $\sum_{i}\lambda_{i}(t)^{2}$ is close to $1$ and $d_{\mathrm{eff}}(t)$ is close to $1$.
Thus, $d_{\mathrm{eff}}(t)$ behaves like the “number of active directions” in which the process varies at time $t$.
This is especially appropriate as a measure for our assumed generative process, i.e., $x_{t}=Va_{t}$, which yields

$$ $C_{t}=\mathbb{E}[x_{t}x_{t}^{\top}]=\mathbb{E}[Va_{t}a_{t}^{\top}V^{\top}]=VS_{t}V^{\top},\qquad S_{t}:=\mathbb{E}[a_{t}a_{t}^{\top}].$ $$

If $V$ is close to orthonormal and $S_{t}$ is close to diagonal, with

$$ $e_{i}(t):=\mathbb{E}[a_{t,i}^{2}]$ $$

denoting the average energy of coefficient $i$, then

$$ $\operatorname{tr}(C_{t}^{2})\approx\sum_{i=1}^{N}e_{i}(t)^{2},\qquad\sum_{i=1}^{N}e_{i}(t)\approx 1.$ $$

Hence

$$ $d_{\mathrm{eff}}(t)=\frac{1}{\operatorname{tr}(C_{t}^{2})}\approx\frac{1}{\sum_{i=1}^{N}e_{i}(t)^{2}},$ $$

which is the usual “effective number” of coefficients that carry energy
at time $t$.
In particular, if the energy is spread evenly across $k$ dictionary atoms,
then $d_{\mathrm{eff}}(t)\approx k$.
Therefore $d_{\mathrm{eff}}(t)$, estimated from the Gram U-statistic,
is a simple and natural measure of intrinsic dimensionality for this
generative process.

#### A.4 Surrogate

For the U-statistic and Autocorrelation metrics, we compare LLM activations to surrogate distributions that preserve certain statistical properties while removing temporal structure. We operate on representation vectors $\mathbf{X}\in\mathbb{R}^{B\times T\times d}$, where $B$ denotes batch size, $T$ denotes sequence length, and $d$ denotes the model dimension.

U-statistic surrogate (Fig. [2](https://arxiv.org/html/2511.01836v3#S3.F2) a, e): For each sequence $i\in\{1,\ldots,B\}$, we construct the surrogate $\tilde{\mathbf{X}}_{i}$ by applying a random permutation $\pi_{i}:\{1,\ldots,T\}\to\{1,\ldots,T\}$ to the temporal positions:

$$ $\tilde{\mathbf{X}}_{i,t,:}=\mathbf{X}_{i,\pi_{i}(t),:}\quad\forall t\in\{1,\ldots,T\}$ $$

This preserves the marginal distribution of activations within each sequence while destroying temporal dependencies.

Autocorrelation surrogate (Fig. [2](https://arxiv.org/html/2511.01836v3#S3.F2) c, g): Given the similarity matrix $\mathbf{S}\in\mathbb{R}^{T\times T}$ where $S_{ij}=\text{sim}(\mathbf{X}_{\cdot,i,\cdot},\mathbf{X}_{\cdot,j,\cdot})$, we construct the surrogate similarity matrix $\tilde{\mathbf{S}}$ by replacing each diagonal with its mean:

$$ $\tilde{S}_{ij}=\bar{S}_{k}\quad\text{where }k=|i-j|,\quad\bar{S}_{k}=\frac{1}{T-k}\sum_{t=1}^{T-k}S_{t,t+k}$ $$

This preserves the average correlation structure at each lag while removing position-specific temporal patterns.

#### A.5 Analysis of Stories: UMAP Projections, Dendrograms, Similarity Heatmaps, and Event Boundary Detection

###### Data.

For experiments in Sec. [6.1](https://arxiv.org/html/2511.01836v3#S6.SS1), we used either stories sampled from the TinyStories dataset [eldan2023tinystories] (for Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5), [6](https://arxiv.org/html/2511.01836v3#S6.F6)) or synthetically sampled stories using GPT-5 (for Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7)).
For the former, we sampled stories that were up to a 100 tokens long, allowing us to characterize compute UMAP for both Gemma and Llama models without hitting memory bottlenecks.
For consistency, we then used these stories for all experiments.
For synthetically defined stories, we defined 3 in-context examples (one of which is shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7)a) and prompted GPT-5 to produce 50 stories with similar such suddenly changing events.
Stories varied in size, but were generally less than 150 tokens long.
Following georgiou2023using, the generated stories’ events were labeled using GPT-5.
These labels served as the ‘ground truth’ event boundaries.
We qualitatively analyzed and confirmed the event boundaries overlap with our intuitively expected event structure in the stories.

###### Analysis Details.

To isolate the low-dimensional geometry of latent codes, we perform a 3D UMAP analysis using the open-source package [umap].
Similarly, Dendrograms in Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5) are computed using the hierarchical clustering package in SciPy dendrogram.
Branches are colored based on proximity, which in our case was between 0–$1$ (since we use cosine similarity as the clustering measure).
We put a proximity bound of $0.2$ as the distance under which branches are colored within a cluster.
For Fourier Analysis, as mentioned in Sec. [6.1](https://arxiv.org/html/2511.01836v3#S6.SS1), we performed a Fourier transform of the model activations for a given story, isolated its lower frequency components—defined as $0.1\times$ the Nyquist rate, which turns out to possess $\sim$50% of the signal norm—and compute the cosine similarity kernel of the low / high frequency spectra. These kernels are then compared to kernels of latent codes derived using Temporal Feature Analysis or SAEs.

#### A.6 Analyzing Garden Path Sentences: Dendrograms and Phrase Similarity

###### Data.

We used GPT-5 to sample 50 garden path sentences and control variants thereof (1 corresponding to each sentence).
Sentences were 20–40 tokens long and had a structure such that the observation of the object phrase resolved ambiguity.
The precise ambiguity structure was were varied in type, i.e., the ambiguity could be resolved by altering the subject (e.g., old $\to$ sailors in The old man the road) or by other mechanisms such as adding punctuations to elicit a pause (e.g., adding a comma, such as The old train the young fight $\to$ The old train, the young fight).
Control variants had a mixture of such resolutions applied.

###### Analysis Details.

For assessing whether tokens in verb phrase relate more with the subject phrase, as would be expect by the typical parse in our used garden path sentences, or with the object phrase, as would be necessary for the correct parse, we computed Dendrograms using the hierarchical clustering package in SciPy [dendrogram].
The similarity between subject phrase (SP), verb phrase (V), and object phrase (OP) was computed by taking the set of tokens $T_{P}$ that belong to a phrase $P\in\{$SP, V, OP$\}$, and computing the average latent code: $\bar{c}(P)=\frac{1}{|P|}\sum_{t\in T_{P}}c_{t}$, where $c_{t}$ denotes the latent code extracted for token $t$ using either SAEs or the predictive or novel component of Temporal Feature Analysis.
This protocol is similar to popularly used strategies for computing sentence embeddings (see, e.g., work using BERT embeddings [koroteev2021bert]).
Cosine similarity between these phrase-averaged garden path / control sentences yields the tables shown in Fig. [8](https://arxiv.org/html/2511.01836v3#S6.F8).

#### A.7 Analyzing In-Context Representations: Visualization and Counterfactual Experiments

###### Data.

We followed the pipeline of park2025iclrincontextlearningrepresentations and defined grid graphs where random 1-token words are associated to the nodes of a 3$\times$3 grid.
A random walk of length 500 tokens is performed on the grid, with the visited nodes stringed together with a whitespace (e.g., stringing apple, car, house yields the string ‘apple car house’).
These walks were inputted into our analyzed, pretrained language models (Gemma-2-2B in Fig. [9](https://arxiv.org/html/2511.01836v3#S6.F9) and Llama-3.1-8B in Fig. [54](https://arxiv.org/html/2511.01836v3#A5.F54)) to extract corresponding activations.

###### Analysis Details.

A 3D PCA of activations was performed to check which two subsequent dimensions (i.e., PC1/2 or PC2/3) best reflect the graph structure, which was then used to plot the figures.
These activations were then passed through both Temporal Feature Analyzers and SAEs, resulting in latent codes that were again visualized using PCA.

### Appendix B Details on Training Temporal Feature Analyzers and SAEs

#### B.1 Training, Harvesting, and Inference for Temporal Feature Analysis and SAEs

###### Compute.

All experiments were performed using 1 H100, both for extracting representations, analyzing them, and training SAEs and Temporal Feature Analyzers.
To save training costs, we precache activations on a local server and train in an “online manner”, i.e., sampling a new batch of activations every training iteration.

###### SAEs’ training.

All analyzed SAEs were trained from scratch on 1B precached activations from the Pile-Uncopyrighted dataset [monology2021pile-uncopyrighted].
While we did try to use existing, off-the-shelf SAEs, we found the results to be wildly inconsistent depending on where we borrowed the SAE from. To enable a consistent and fair evaluation, we thus preferred to train all SAEs from scratch.
For Gemma models, activations were extracted from Layer 12; for Llama models, from layer 15 (all 0-indexed).

###### Training procedure.

We use Adam optimizer with standard hyperparameters. We initialize training with a warmup of 200 steps to a learning rate of $10^{-3}$, following from thereon to a minimum of $9\times 10^{-4}$, i.e., the learning rate remains essentially constant throughout training.

###### Activations normalization.

Following best practices, we normalize activations to unit expected norm [bricken2023monosemanticity, costa2025flat]. This helps put different SAEs on the same training scale, and especially for ReLU SAEs, makes training substantially easier by reducing the need for tuning regularization strength. On a related note, we emphasize our ReLU SAEs have a slightly higher $L_{0}$, i.e., are less sparse, than almost all other SAEs analyzed in this work.

### Appendix C Further Results: Language Model Representations (U-Statistic and Autocorrelation)

#### C.1 U-Statistic across domains for Gemma-2-2B

Figure: Figure 15: U-Statistic across domains for LLM activations, surrogate
Refer to caption: x13.png

#### C.2 Autocorrelation across domains for Gemma-2-2B

Figure: Figure 16: Autocorrelation of language model activations and a stationary surrogate across webtext, simple stories and code domains.
Refer to caption: final_figures/autocorrelation_datasets_llm.png

### Appendix D More Gemma Results

#### D.1 Hierarchical clustering of codes from story tokens

##### D.1.1 Story 1

Figure: Figure 17: Geometry. Repeating results of Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5), we again find smooth trajectories in UMAP projections for Temporal Feature Analysis.
Refer to caption: appendix_pngs/story_trajectory_1.png

Figure: Figure 18: Dendrograms of Predictive and Novel Components from Temporal Feature Analysis. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but including the novel component’s dendrogram as well.
Refer to caption: appendix_pngs/story_temporal_1.png

Figure: Figure 19: Dendrograms of Standard SAEs’ Latents Codes. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but for latent codes extracted using standard SAEs.
Refer to caption: appendix_pngs/story_standard_1.png

##### D.1.2 Story 2

Figure: Figure 20: Geometry. Repeating results of Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5) on a different story, we again find smooth trajectories in UMAP projections for Temporal Feature Analysis.
Refer to caption: appendix_pngs/story_trajectory_2.png

Figure: Figure 21: Dendrograms of Predictive and Novel Components from Temporal Feature Analysis. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but on a different story and with both the predictive and novel component.
Refer to caption: appendix_pngs/story_temporal_2.png

Figure: Figure 22: Dendrograms of Standard SAEs’ Latents Codes. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but for latent codes extracted using standard SAEs on a different story.
Refer to caption: appendix_pngs/story_standard_2.png

##### D.1.3 Story 3

Figure: Figure 23: Geometry. Repeating results of Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5) on a different story, we again find smooth trajectories in UMAP projections for Temporal Feature Analysis.
Refer to caption: appendix_pngs/story_trajectory_3.png

Figure: Figure 24: Dendrograms of Predictive and Novel Components from Temporal Feature Analysis. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but on a different story and with both the predictive and novel component.
Refer to caption: appendix_pngs/story_temporal_3.png

Figure: Figure 25: Dendrograms of Standard SAEs’ Latents Codes. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but for latent codes extracted using standard SAEs on a different story.
Refer to caption: appendix_pngs/story_standard_3.png

#### D.2 Code: Analyzing Another Domain

Figure: Figure 26: Code. Reproducing the results of Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5), but on a different domain, i.e., code. We again see a temporally disentangled, smoothly running trajectory for latent codes extracted using the predictive component of Temporal Feature Analysis.
Refer to caption: appendix_pngs/ind_code_geometry.png

Figure: Figure 27: Dendrograms of Predictive and Novel Components from Temporal Feature Analysis. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but on a different domain shows similar results as with narrative-driven text (stories) for both the predictive and novel component.
Refer to caption: appendix_pngs/code_temporal_story_graphs.png

Figure: Figure 28: Dendrograms of Standard SAE Latent Codes. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but on a different domain shows similar results as with narrative-driven text (stories) for standard SAEs.
Refer to caption: appendix_pngs/code_standard_story_graphs.png

#### D.3 Further Results: Similarity Maps on Stories

Figure: Figure 29: Temporal (Pred) Similarity Maps Elicit Multi-Scale Structure with Noise. Repeating the analysis shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7), we find the coarsening of temporal blocks is a robust result that continues to hold for different inputs.
Refer to caption: appendix_pngs/temporal_sim_and_noise.png

Figure: Figure 30: Temporal (Novel) Similarity Maps under Noise. Repeating the analysis shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7), we find the novel component is only able to capture minimal local similarities, which when analyzed via dendrograms, show clustering based on lexical information.
Refer to caption: appendix_pngs/novel_sim_and_noise.png

Figure: Figure 31: ReLU Codes’ Similarity Maps under Noise. Repeating the analysis shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7) on ReLU SAEs, we find the ReLU latent code has high similarity across the board, suggesting lack of meaningful temporal information. This similarity is entirely removed when noise scale increases too much, which, as shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7)c, corresponds to the point that variance explained by ReLU SAEs drops to $\sim$0.
Refer to caption: appendix_pngs/relu_sim_and_noise.png

Figure: Figure 32: TopK Codes’ Similarity Maps under Noise. Repeating the analysis shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7), we find TopK SAE’s latent codes are only able to capture minimal local similarities, which when analyzed via dendrograms, show clustering based on lexical information. Akin to ReLU SAEs, we see this similarity map approximately turns into an identity matrix when the noise scale increases too much, which, as shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7)c, corresponds to the point that variance explained by TopK SAEs drops to $\sim$0.
Refer to caption: appendix_pngs/topk_sim_and_noise.png

Figure: Figure 33: BatchTopK Codes’ Similarity Maps under Noise. Repeating the analysis shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7), we find BatchTopK SAE’s latent codes are only able to capture minimal local similarities, which when analyzed via dendrograms, show clustering based on lexical information. Akin to ReLU SAEs, we see this similarity map approximately turns into an identity matrix when the noise scale increases too much, which, as shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7)c, corresponds to the point that variance explained by BatchTopK SAEs drops to $\sim$0.
Refer to caption: appendix_pngs/batchtop_sim_and_noise.png

#### D.4 Further Results: Dendrograms and Evaluations on Garden Path Sentences

##### D.4.1 Dendrograms

Figure: Figure 34: Example Sentence 1. Repeating the results of Fig. [8](https://arxiv.org/html/2511.01836v3#S6.F8)a on a different garden path sentence.
Refer to caption: appendix_pngs/gemma_garden_path_dendrograms_1.png

Figure: Figure 35: Example Sentence 2. Repeating the results of Fig. [8](https://arxiv.org/html/2511.01836v3#S6.F8)a on a different garden path sentence.
Refer to caption: appendix_pngs/gemma_garden_path_dendrograms_2.png

Figure: Figure 36: Example Sentence 3. Repeating the results of Fig. [8](https://arxiv.org/html/2511.01836v3#S6.F8)a on a different garden path sentence.
Refer to caption: appendix_pngs/gemma_garden_path_dendrograms_3.png

#### D.5 Further Results: Counterfactual Generation with In-Context Representations

Figure: Figure 37: Counterfactuals. Counterfactual analysis of performing belief update intervention for all latent codes. We see essentially no SAE offers the ability to alter model beliefs, i.e., altering where on the graph it currently believes it is and hence what next tokens it should produce. Temporal Feature Analysis shows success in this case: we find $\nicefrac{{4}}{{9}}$ nodes can be intervened upon using the predictive component, out of which, interestingly, 2 also respond to an intervention on the novel component.
Refer to caption: appendix_pngs/all_iclr_counterfactuals.png

### Appendix E Further Results: Temporal Feature Analysis on LLama-3.1-8B

We replicate a subset of the experiments evaluating Temporal Feature Analysis on Llama-3.1-8B. The training protocol remains the same as that of Gemma-2-2B model: train on 1B token activations with similar normalization schema, but activations are harvested now from Layer 15 (i.e., at $\sim$50% of model depth). We note that the trained SAEs are not finished training, and hence the following results are solely meant to be an impression of whether the qualitative trends observed with Gemma models generalize to another model class.

#### E.1 Geometry, Dendrograms, and Spectra

##### E.1.1 Story 1

Figure: Figure 38: Geometry. Repeating results of Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5), we again find smooth trajectories in UMAP projections for Temporal Feature Analysis.
Refer to caption: appendix_pngs/llama_story_trajectory_1.png

Figure: Figure 39: Dendrograms of Predictive and Novel Components from Temporal Feature Analysis. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but including the novel component’s dendrogram as well.
Refer to caption: appendix_pngs/llama_story_temporal_1.png

Figure: Figure 40: Dendrograms of Standard SAEs’ Latents Codes. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but for latent codes extracted using standard SAEs.
Refer to caption: appendix_pngs/llama_story_standard_1.png

##### E.1.2 Story 2

Figure: Figure 41: Geometry. Repeating results of Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5), we again find smooth trajectories in UMAP projections for Temporal Feature Analysis.
Refer to caption: appendix_pngs/llama_story_trajectory_2.png

Figure: Figure 42: Dendrograms of Predictive and Novel Components from Temporal Feature Analysis. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but including the novel component’s dendrogram as well.
Refer to caption: appendix_pngs/llama_story_temporal_2.png

Figure: Figure 43: Dendrograms of Standard SAEs’ Latents Codes. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but for latent codes extracted using standard SAEs.
Refer to caption: appendix_pngs/llama_story_standard_2.png

##### E.1.3 Story 3

Figure: Figure 44: Geometry. Repeating results of Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5), we again find smooth trajectories in UMAP projections for Temporal Feature Analysis.
Refer to caption: appendix_pngs/llama_story_trajectory_3.png

Figure: Figure 45: Dendrograms of Predictive and Novel Components from Temporal Feature Analysis. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but including the novel component’s dendrogram as well.
Refer to caption: appendix_pngs/llama_story_temporal_3.png

Figure: Figure 46: Dendrograms of Standard SAEs’ Latents Codes. Reproduction of the results from Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5)b, but for latent codes extracted using standard SAEs.
Refer to caption: appendix_pngs/llama_story_standard_3.png

##### E.1.4 Comparing Eigenspectrum with Slow vs. Fast Features

Figure: Figure 47: Kernel spectrum for latent codes and model representations. Kernels defined using novel code from Temporal Feature Analysis and standard SAEs both align well with the fast-changing part of model representations; meanwhile, only the predictive code shows strong similarity to the slow changing part.
Refer to caption: appendix_pngs/llama_spectra.png

#### E.2 Event Boundaries and Noise Stability

##### E.2.1 Event Boundaries

Figure: Figure 48: Analysis of Event Boundaries. Reproducing results from Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7) on Llama-3.1-8B, we see similar behavior as the Gemma analysis done previously.
Refer to caption: appendix_pngs/llama_event_boundaries.png

Figure: Figure 49: Temporal (Pred) Similarity Maps Elicit Multi-Scale Structure with Noise. Repeating the analysis shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7), we find the coarsening of temporal blocks is a robust result that continues to hold for different inputs.
Refer to caption: appendix_pngs/llama_pred_sim_and_noise.png

Figure: Figure 50: Temporal (Novel) Similarity Maps under Noise. Repeating the analysis shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7), we find the novel component is only able to capture minimal local similarities, which when analyzed via dendrograms, show clustering based on lexical information.
Refer to caption: x14.png

Figure: Figure 51: ReLU Codes’ Similarity Maps under Noise. Repeating the analysis shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7) on ReLU SAEs, we find the ReLU latent code has high similarity across the board, suggesting lack of meaningful temporal information.
Refer to caption: appendix_pngs/llama_relu_sim_and_noise.png

Figure: Figure 52: TopK Codes’ Similarity Maps under Noise. Repeating the analysis shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7), we find TopK SAE’s latent codes are only able to capture minimal local similarities, which when analyzed via dendrograms, show clustering based on lexical information.
Refer to caption: appendix_pngs/llama_topk_sim_and_noise.png

Figure: Figure 53: BatchTopK Codes’ Similarity Maps under Noise. Repeating the analysis shown in Fig. [7](https://arxiv.org/html/2511.01836v3#S6.F7), we find BatchTopK SAE’s latent codes are only able to capture minimal local similarities, which when analyzed via dendrograms, show clustering based on lexical information.
Refer to caption: appendix_pngs/llama_batchtopk_sim_and_noise.png

#### E.3 In-Context Representations

Figure: Figure 54: In-Context Representations. Reproducing the results of Fig. [9](https://arxiv.org/html/2511.01836v3#S6.F9) on Llama-3.1-8B, we see similar results as the Gemma analysis.
Refer to caption: appendix_pngs/llama_iclr_iclr.png

### Appendix F Further Results: Temporal Feature Analysis with Only Predictive Component

In this section, we reproduce a subset of the experiments on an ablated version of Temporal Feature Analysis wherein only the predictive component is used for reconstructing the signal, i.e., the training objective is entirely based on prediction of the next-token representation. As we showed in Tab. [1](https://arxiv.org/html/2511.01836v3#S5.T1), this ablation does not achieve low-enough error. Correspondingly, the results in this section are a mere qualtitative artifact to show that Temporal Feature Analysis is not over-powered by its expressivity.

Figure: Figure 55: Geometry and Heatmaps from the Prediction-only Ablation. We see block and event structures do emerge, but they are substantially more wound than the smooth trajectories seen in Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5).
Refer to caption: appendix_pngs/pred_only_heatmaps.png

Figure: Figure 56: Dendrograms from the Prediction-only Ablation. Dendrograms reflect the event structures to some extent, but the results is Fig. [5](https://arxiv.org/html/2511.01836v3#S6.F5) are substantially more crisp.
Refer to caption: appendix_pngs/pred_only_dendro.png

Figure: Figure 57: Attention Patterns from the Prediction-only Ablation. Attention weights for single sequences from three domains. For reference see the more coherent attention patterns of TemporalSAE, where the attention layer is trained jointly with a subsequent SAE in Figure [13](https://arxiv.org/html/2511.01836v3#S8.F13).
Refer to caption: x15.png

### Appendix G Further Results: Emergent Separation of Predictive and Novel Code in Temporal Feature Analysis

Figure: Figure 58: Separation of Predictive and Novel Dictionaries. Predictive and novel codes for sequences drawn from three datasets—SimpleStories (top), Webtext (middle), and Code (bottom)—are binarized to zero (black) and non-zero (white). The dictionary elements are sorted by the sum of non-zero activations in the predictive code across the sequence. The ordering obtained from the predictive code is also applied to the novel code. The red dashed line marks the separation of 90% of non-zero activation counts; only 10% are overlapping into the other subset. Additionally, the effective rank of the predictive codes is two orders of magnitudes lower than the average number of non-zero dictionary elements (Mean L0).
Refer to caption: x16.png

The Temporal Feature Analyzers studied in the main paper share a dictionary for both predictive and novel codes.
We now investigate if there is any shared structure in the two dictionaries.
In particular, given the predictive and novel codes play a different role, it is plausible the SAE learns to approximately split the dictionary into two parts: one responsible for computing the predictive code and the other for novel one.
Specifically, we investigate a Temporal Feature Analyzer trained on Gemma-2-2B Layer 12 residual stream activations.
Results are shown in Fig. [58](https://arxiv.org/html/2511.01836v3#A7.F58) and show a separation of dictionary elements for sequences drawn from three datasets: SimpleStories, Webtext, and Code.
Specifically, we find $\sim$2K dictionary elements participate in defining the predictive code, while the remaining $\sim$8K primarily participate in defining the novel code.
However, the absolute count is merely indicative of the L0 sparsity of a code, which may not reflect how many directions are actually used to define the code—estimating latter requires computing the rank of the code (in fact, we note that rank-sparsity is a well-known alternative to L0 sparsity in dictionary learning literature [elad2010sparse]).
As we show in Fig. [58](https://arxiv.org/html/2511.01836v3#A7.F58), the effective rank of the predictive codes ($\sim$20–30) is substantially lower than both the effective rank ($\sim$390–410) and absolute L0 sparsity of novel codes.

Overall, the posthoc analysis above shows that predictive and novel codes largely use separate subsets of dictionary elements.
This emergent disentanglement of the two components motivates one to preemptively split the dictionary into two components—one responsible for the predictive code and other for novel one.
Our preliminary experiments show a Temporal SAE trained with this split-dictionary architecture is similarly performant as the tied dictionary one, resulting in predictive / novel codes with effective ranks $\sim$20–30 / $\sim$390–420, and a slightly better overall loss.
It is possible optimizing hyperparameters (e.g., having different expansion factors for the two components) can make this architecture more performant than the tied dictionary one, but we leave a further characterization to future work.

### Appendix H Proofs

#### H.1 Priors on the Sparse Code for various SAEs

We restate and prove the proposition on independence priors of SAEs over time (Proposition [2](https://arxiv.org/html/2511.01836v3#S4.E2)) below.

###### Proposition H.1 (Independence priors over time) .

Consider the SAE maximum aposteriori (MAP) objective for ReLU, JumpReLU, TopK and BatchTopK SAEs. The sparsity constraints for these SAEs are additive over time, resulting in:

$$ $\begin{split}\operatorname*{arg\,min}_{\bm{D},\bm{z}}\frac{1}{T}&\sum_{i=1}^{T}\|\bm{x}_{i}-\bm{D}\bm{z}_{i}\|_{2}^{2}+\lambda\mathcal{R}(\bm{z}_{i}),\\ \text{s.t. }\bm{z}_{k}&=f_{\mathtt{SAE}}(\bm{x}_{k})\;\forall k,\;\tilde{g}(\bm{z}_{1},\dots,\bm{z}_{T})=\sum_{i}\tilde{g}(\bm{z}_{i})=0.\;\end{split}$ (8) $$

This MAP objective has an independent and identically distributed (i.i.d.) prior over time i.e.,

$$ $\displaystyle P(\bm{z}_{1},\dots,\bm{z}_{T})$ $\displaystyle\propto\prod_{t=1}^{T}\exp\left(-\lambda\mathcal{R}(\bm{z}_{i})-\tilde{\lambda}\tilde{g}(\bm{z}_{i})\right)=\prod_{i}P(\bm{z}_{i}),$ $$

###### Proof.

The sparsity constraints and sparsity-promoting regularizers for the SAEs under study are specified in the table below.

**Table 5: Sparsity constraints and regularizers for SAEs**
| SAE | Regularizer<br>$\mathcal{R}(\bm{z}_{i})$ | Sparsity Constraint<br>$\tilde{g}(\bm{z}_{1},\dots,\bm{z}_{T})=0$ |
| --- | --- | --- |
| ReLU | $\|\bm{z}_{i}\|_{1}$ | 0 |
| JumpReLU | $\|\bm{z}_{i}\|_{0}$ | 0 |
| TopK | 0 | $\sum_{i=1}^{T}(\|\bm{z}_{i}\|_{0}-K)^{2}$ |
| BatchTopK | 0 | $\frac{1}{T}\sum_{i=1}^{T}\|\bm{z}_{i}\|_{0}-K$ |

Note that TopK imposes a pointwise hard sparsity constraint, which has been restated using sum-of-squares above for convenience. While BatchTopK imposes the fixed mean sparsity for each mini batch, we take the batch to capture the entire timeseries in the above formulation. The above table shows us that the sparsity constraint is additive over time in all cases:

$$ $\displaystyle\tilde{g}(\bm{z}_{1},\dots,\bm{z}_{T})$ $\displaystyle=\sum_{i=1}^{T}\tilde{g}(\bm{z}_{i})=0,$ (9) $\displaystyle\text{where }\tilde{g}(\bm{z}_{i})$ $\displaystyle=\begin{cases}(\|\bm{z}_{i}\|_{0}-K)^{2}&\text{TopK}\\ \frac{1}{T}(\|\bm{z}_{i}\|_{0}-K)&\text{BatchTopK}\\ 0&\text{ReLU, JumpReLU}\end{cases}$ (10) $$

Recall that SAEs solve the following constrained optimization problem (restated from Eq. [1](https://arxiv.org/html/2511.01836v3#S2.E1)).

$$ $\begin{split}\operatorname*{arg\,min}_{\bm{D},\bm{z}}\frac{1}{N}&\sum_{i=1}^{T}\|\bm{x}_{i}-\bm{D}\bm{z}_{i}\|_{2}^{2}+\lambda\mathcal{R}(\bm{z}_{i}),\\ \text{s.t. }\bm{z}_{k}&=f_{\mathtt{SAE}}(\bm{x}_{k})\;\forall k,\;\tilde{g}(\bm{z}_{1},\dots,\bm{z}_{T})=0\;.\end{split}$ (11) $$

We rewrite the above problem using Lagrange multipliers on the sparsity constraints, and further simplify using the above result on constraints being additive over time, as:

$$ $\begin{split}\operatorname*{arg\,min}_{\bm{D},\bm{z}}\frac{1}{N}&\sum_{i=1}^{T}\|\bm{x}_{i}-\bm{D}\bm{z}_{i}\|_{2}^{2}+\lambda\mathcal{R}(\bm{z}_{i})+\tilde{\lambda}\tilde{g}(\|\bm{z}_{i}\|_{0}),\\ \text{s.t. }\bm{z}_{k}&=f_{\mathtt{SAE}}(\bm{x}_{k})\;\forall k.\end{split}$ (12) $$

Note that we don’t use Lagrange multipliers on the SAE architecture constraint $\bm{z}=f_{\mathtt{SAE}}(\bm{x})$ since we only care about $\bm{z}$-specific constraints (which don’t include the data $\bm{x}$) for the prior.

###### Proposition H.1 (Independence priors over time) .

###### Proof.

###### Bayesian Interpretation.

The objective function above (sans the SAE architecture constraint) can be thought of as minimizing the negative log posterior, which is proportional to log prior added to log likelihood:

$$ $\displaystyle-\log P(\bm{z}_{1},\dots,\bm{z}_{T}\mid\bm{x}_{1},\dots,\bm{x}_{T})$ $\displaystyle\propto\underbrace{\frac{1}{N}\sum_{i=1}^{T}\|\bm{x}_{i}-\bm{D}\bm{z}_{i}\|_{2}^{2}}_{-\log P(\bm{x}_{1},\dots,\bm{x}_{T}\mid\bm{z}_{1},\dots,\bm{z}_{T})}+\underbrace{\frac{1}{N}\sum_{i=1}^{T}\lambda\mathcal{R}(\bm{z}_{i})+\tilde{\lambda}\tilde{g}(\|\bm{z}_{i}\|_{0})}_{-\log P(\bm{z}_{1},\dots,\bm{z}_{T})}$ (13) $$

The prior over latents $\bm{z}$ is:

$$ $\displaystyle\log P(\bm{z}_{1},\dots,\bm{z}_{T})$ $\displaystyle=-\frac{1}{N}\sum_{i=1}^{T}\lambda\mathcal{R}(\bm{z}_{i})+\tilde{\lambda}\tilde{g}(\|\bm{z}_{i}\|_{0}),$ (14) $\displaystyle\implies P(\bm{z}_{1},\dots,\bm{z}_{T})$ $\displaystyle=\prod_{i=1}^{T}\exp\left(-\lambda\mathcal{R}(\bm{z}_{i})+\tilde{\lambda}\tilde{g}(\|\bm{z}_{i}\|_{0})\right)=\prod_{i}P(\bm{z}_{i}).$ (15) $$

Therefore, the prior is multiplicative over time, implying independence, and the distribution at each time $t$ has the same form, implying that the prior is independent and identically distributed (i.i.d.). This completes the proof.

∎

#### H.2 Sparsity over time and implications

The i.i.d. prior over time leads to similar i.i.d. assumptions about the sparsity of representations over time for SAEs. The following corollary of Prop. [2](https://arxiv.org/html/2511.01836v3#S4.E2) states this assumption.

###### Corollary H.1.1 (Assumptions of time-invariant sparsity) .

As a consequence of the i.i.d. priors over time from Prop. [2](https://arxiv.org/html/2511.01836v3#S4.E2), standard SAEs assume that sparsity of representations emerges from a fixed distribution independently over time (i.i.d.), i.e.,

$$ $\displaystyle P(\|\bm{z}_{1}\|_{0},\dots,\|\bm{z}_{T}\|_{0})$ $\displaystyle=\prod_{t}P(\|\bm{z}_{t}\|_{0}),$ (16) $\displaystyle\implies P(\|\bm{z}_{t}\|_{0})$ $\displaystyle=P(\|\bm{z}_{t^{\prime}}\|_{0})\;\;t\neq t^{\prime}$ (17) $$

###### Proof.

This result essentially follows from Prop. [2](https://arxiv.org/html/2511.01836v3#S4.E2). Since the prior over latents $\bm{z}$ is i.i.d., and sparsity is a function of the latents, sparsity must also be i.i.d. over time. As a consequence, the distributions at different times are identical.
∎

Due to the observation of increase in number of concepts over time in language (Fig. [1](https://arxiv.org/html/2511.01836v3#S1.F1)), in the following proposition, we show that time invariant sparsity priors in SAEs lead to SAEs losing out on important structural information in model activations.

###### Proposition H.2 (Restrictive Sparsity Budget Leads to Support Switching in SAEs) .

Suppose data $\bm{x}$ lies on a $C^{1}$ manifold $\mathcal{M}\subset\mathbb{R}^{d}$. For $\bm{x}\in\mathcal{M}$, let $T_{\bm{x}}\mathcal{M}$ denote the tangent space at $\bm{x}$, and $m(\bm{x})=\dim T_{\bm{x}}\mathcal{M}$ its dimension. Suppose a Sparse Autoencoder (SAE) has a sparsity budget $|S(\bm{x})|=K$. . If $K<m(\bm{x})$, under the assumption of low mean-squared error (MSE) of the SAE, in some neighborhood $\mathcal{N}_{\bm{x}}$ of $\bm{x}$, $\exists\bm{x}_{1},\bm{x}_{2}\in\mathcal{N}_{\bm{x}}$ s.t. $S(\bm{x}_{1})\neq S(\bm{x}_{2})$, i.e., support switching occurs in the SAE latents.

###### Proof.

Suppose not: this implies that for every neighborhood $\mathcal{N}_{\bm{x}}$ of $\bm{x}$, the SAE has the same support at all points: $\bm{x}_{1},\bm{x}_{2}\in\mathcal{N}_{\bm{x}}\implies S(\bm{x}_{1})=S(\bm{x}_{2})$. Since the SAE decoder is linear (from latents $\bm{z}$ to output $\hat{\bm{x}}$ conditioned on the support $S(\bm{x})$), fixed support with size $|S(\bm{x})|=K$ in the neighborhood $\mathcal{N}_{\bm{x}}$ corresponds to a $K$-dimensional region $\mathcal{R}_{S(\bm{x})}$ in $\mathbb{R}^{d}$ in the SAE outputs (which aim to reconstruct the input $\bm{x}$). Since $m(x)=\dim T_{\bm{x}}$, $\mathcal{N}_{\bm{x}}$ spans an $m(x)$-dimensional region in Euclidean space (using a local chart of the manifold $\mathcal{M}$ at $\bm{x}$). Since $m(\bm{x})=\dim\mathcal{N}_{\bm{x}}>K=\dim\mathcal{R}_{S(\bm{x})}$, the data (inputs to SAE) and SAE reconstructions belong to regions with different dimensions—with the reconstructions $\hat{\bm{x}}$ belonging to a lower dimensional region than inputs $\bm{x}$—which implies that the reconstruction error of the SAE (MSE) is high. This contradicts the assumption that SAE reconstruction error is low, which implies that our assumption about same support within a neighborhood of $\bm{x}$ is false. This implies that there is a neighborhood of $\bm{x}$ where the support of the SAE switches. This completes the proof.
∎

The implications of support switching have been discussed in the main text: the local geometry in data $\bm{x}$ is not preserved in SAE latents $\bm{z}$, so concepts that correspond to local geometry (over time, which indicates contextual information) may not be observed in the SAE latents.

###### Corollary H.1.1 (Assumptions of time-invariant sparsity) .

###### Proof.

###### Proposition H.2 (Restrictive Sparsity Budget Leads to Support Switching in SAEs) .

###### Proof.

### Appendix I Further Theoretical Results: SAE-wise Characterization of Priors over concepts and time

We can further think of the priors of each SAE over concepts as well as over time in a generative fashion. In some cases, this mainfests as a hierarchical latent variable model $n\rightarrow S\rightarrow\bm{z}$, where $n=\|\bm{z}\|_{0}$ is the sparsity, $S=\mathrm{supp}(\bm{z})=\{k:z^{k}>0\}$ is the support, and $\bm{z}$ is the SAE latent code.

###### Proposition I.1 (SAE Priors on Sparse Code) .

Let $S_{t}=\mathrm{supp}(\bm{z}_{t})=\{k:z^{k}_{t}>0\}$ be the set of active latents in the sparse code $\bm{z}$ at time $t$, and $n_{t}=|S_{t}|$ be the cardinality of $S_{t}$ (the number of active latents). Each SAE imposes a prior distribution on the sparse code $\bm{z}$, arising from its sparsity penalty $\mathcal{R}(\bm{z})$ or implicit conditions imposed on the sparse code. These conditions are highlighted in Table [6](https://arxiv.org/html/2511.01836v3#A9.T6).

**Table 6: Priors over concept interactions and dynamics for various SAEs**
| $f_{\mathtt{SAE}},\mathcal{R}(\bm{z})$ | Across-Concept Prior (interaction) | Across-time Prior (dynamics) |
| --- | --- | --- |
| ReLU,<br>$L_{1}$-norm | $z^{1}_{t},\dots,z^{M}_{t}\overset{\text{i.i.d.}}{\sim}\text{Laplace}(0,\cdot)$ | $\bm{z}_{1},\dots,\bm{z}_{t}\overset{\text{i.i.d.}}{\sim}P_{\bm{z}}$ |
| TopK | $z^{i_{1}}_{t},\ldots,z^{i_{K}}_{t}\mid S_{t}\;\overset{\text{i.i.d.}}{\sim}U(0,\cdot)\;\forall i_{\cdot}\in S_{t},$<br>$S_{t}\sim U([M]^{K})$ | $(\bm{z}_{1},S_{1}),\dots,(\bm{z}_{t},S_{t})\overset{\text{i.i.d.}}{\sim}P_{S}P_{\bm{z}\mid S},$<br>$S_{1},\dots,S_{t}\overset{\text{i.i.d.}}{\sim}U([M]^{K})$ |
| JumpReLU,<br>$L_{0}$-norm | $z^{i_{1}}_{t},\ldots,z^{i_{n_{t}}}_{t}\mid S_{t}\;\overset{\text{i.i.d.}}{\sim}U(0,\cdot)\;\forall i_{\cdot}\in S_{t},$<br>$S_{t}\mid n_{t}\sim U([M]^{n_{t}})$ | $(\bm{z}_{1},S_{1},n_{1}),\dots,(\bm{z}_{t},S_{t},n_{t})\overset{\text{i.i.d.}}{\sim}P_{n}P_{S\mid n}P(\bm{z}\mid S),$<br>$n_{1},\dots,n_{t}\overset{\text{i.i.d.}}{\sim}P_{n}$ |
| BatchTopK | $z^{i_{1}}_{t},\ldots,z^{i_{n_{t}}}_{t}\mid S_{t}\;\overset{\text{i.i.d.}}{\sim}U(0,\cdot)\;\forall i_{\cdot}\in S_{t},$<br>$S_{t}\mid n_{t}\sim U([M]^{n_{t}})$ | $(\bm{z}_{1},S_{1},n_{1}),\dots,(\bm{z}_{t},S_{t},n_{t})\overset{\text{i.i.d.}}{\sim}P_{n}P_{S\mid n}P(\bm{z}\mid S),$<br>$n_{1},\cdots,n_{t}\overset{\text{i.i.d.}}{\sim}P_{n},\;\mathbb{E}[n_{t}]=K$ |

###### Proposition I.1 (SAE Priors on Sparse Code) .

##### I.0.1 ReLU SAE

The vanilla ReLU SAE (bricken2023monosemanticity, cunningham2023sparse) is trained with the $L_{1}$-norm penalty:

$$ $\displaystyle\mathcal{R}(\bm{z})$ $\displaystyle=\|\bm{z}\|_{1}.$ (18) $$

The prior over $\bm{z}$ for the above case is:

$$ $\displaystyle\log P(\bm{z}_{1},\dots,\bm{z}_{N})$ $\displaystyle\propto-\sum_{i=1}^{N}\sum_{k=1}^{M}|z_{i}^{k}|,$ (19) $\displaystyle\implies P(\bm{z}_{1},\dots,\bm{z}_{N})$ $\displaystyle\propto\prod_{i=1}^{N}\Bigg(\prod_{k=1}^{M}\exp-\nu|z_{i}^{k}|\Bigg).$ (20) $$

This joint distribution implies that for each sample $i$, different indices $k$ are sampled i.i.d. from the same distribution:

$$ $\displaystyle z_{i}^{1},\dots,z_{i}^{M}\overset{\text{i.i.d.}}{\sim}\text{Laplace}(0,1/\nu),$ (21) $$

and different samples are all independently sampled from the same product Laplace distribution:

$$ $\displaystyle\bm{z}_{1},\dots\bm{z}_{N}\overset{\text{i.i.d.}}{\sim}\text{Laplace}^{M}(0,1/\nu).$ (22) $$

This concludes the proof for priors of ReLU SAE trained with $L_{1}$ norm sparsity penalty. $\square$

##### I.0.2 TopK SAE

The TopK SAE (makhzani2013k, gao2024scaling) directly controls the sparsity of the representation $\bm{z}$ by fixing it at $\|\bm{z}\|_{0}=K$, instead of imposing an explicit sparsity penalty $\mathcal{R}(\bm{z})$ in the loss function. The objective function for TopK SAE is:

$$ $\displaystyle\operatorname*{arg\,min}_{\bm{D},\bm{z}}\sum_{i=1}^{N}\frac{1}{N}\|\bm{x}_{i}-\bm{D}\bm{z}_{i}\|_{2}^{2},$ (23) $\displaystyle\text{s.t. }\forall j,\bm{z}_{j}=f_{TopK}(\bm{x}_{j}),\;\|\bm{z}_{j}\|_{0}=K.$ (24) $$

Since the fixed sparsity is a hard constraint that depends on $\bm{z}$ alone (and not the data $\bm{x}$), it can further be simplified as a sum-of-squares constraint: $\sum_{j}(\|\bm{z}_{j}\|_{0}-K)^{2}=0$. We can use Lagrange multipliers to reformulate it as an effective prior:

$$ $\displaystyle\operatorname*{arg\,min}_{\bm{D},\bm{z},\{\lambda_{i}\}}\sum_{i=1}^{N}\frac{1}{N}\Bigg(\|\bm{x}_{i}-\bm{D}\bm{z}_{i}\|_{2}^{2}+\lambda\left(\big|\|\bm{z}_{i}\|_{0}-K\big|\right)^{2}\Bigg),$ (25) $\displaystyle\text{s.t. }\forall j,\bm{z}_{j}=f_{TopK}(\bm{x}_{j}).$ (26) $$

The prior over $\bm{z}$ for the above (effective) regularizer is:

$$ $\displaystyle\log P(\bm{z}_{1},\dots,\bm{z}_{N})$ $\displaystyle\propto-\sum_{i=1}^{N}\lambda\left(\big(\|\bm{z}_{i}\|_{0}-K\big)^{2}\right)$ (27) $\displaystyle\implies P(\bm{z}_{1},\dots,\bm{z}_{N})$ $\displaystyle\propto\prod_{i=1}^{N}\exp\big(-\lambda\big(\|\bm{z}_{i}\|_{0}-K\big)^{2}\big)$ (28) $$

Note that the above prior is finite for finite values of $\lambda$, but the overall objective optimizes over $\lambda$, resulting in a hard prior peaked at $\|\bm{z}_{i}\|_{0}=K$ for each sample $i$.

The factorization over samples $i$ implies mutual independence of $\bm{z}_{1},\dots,\bm{z}_{n}$: $P(\bm{z}_{1},\dots,\bm{z}_{n})=\prod_{i=1}^{N}P(\bm{z}_{i})$.

As defined in Theorem [I.1](https://arxiv.org/html/2511.01836v3#A9.Thmproposition1) (and restated here for convenience), let $S_{i}=\mathrm{supp}(\bm{z}_{i})=\{k:z_{i}^{k}>0\},n_{i}=|S_{i}|=\|\bm{z}_{i}\|_{0}$ denote the active indices and their number (sparsity) respectively.

For individual samples $\bm{z}_{i}$, if we condition on the set of active indices $S_{i}$, the sparsity gets fixed since $\|\bm{z}_{i}\|_{0}=|S_{i}|=n_{i}$, and the distribution becomes constant:

$$ $\displaystyle P(\bm{z}_{i}\mid S_{i})$ $\displaystyle=C$ (29) $\displaystyle\implies z_{i}^{\mu}\mid S_{i}\sim$ $\displaystyle\begin{cases}U(0,\kappa)\;&\mu\in S_{i}\\ \delta_{0}\;&\mu\notin S_{i}\end{cases},\text{and}$ (30) $\displaystyle z_{i}^{\mu_{1}},\dots,z_{i}^{\mu_{|S_{i}|}}\mid S_{i}$ $\displaystyle\overset{\text{i.i.d.}}{\sim}U(0,\kappa)\;\text{ for }\mu_{\cdot}\in S_{i}$ (31) $$

where $C,\kappa$ are appropriate constants.

Since $\{\bm{z}_{i}\}_{i}$s are mutually independent, any measurable function of each is also independent. The indices of nonzero entries of $\bm{z}_{j}$, i.e., $S_{j}$ is a measurable function since it is a map $S:\mathbb{R}_{+}^{M}\rightarrow 2^{M}$ which is discrete valued, and pre images of each value—a set of nonzero indices—are measurable since they equal the cartesian products of the measurable sets $\{z=0\},\{z>0\}$ over all indices. Hence, $S_{1},\dots,S_{n}$ are also independent.

Since $S_{i}=g(\bm{z}_{i})$ and the distribution of $\bm{z}_{i}$ depends only on $n_{i}=\|\bm{z}_{i}\|_{0}$ (Eq. [27](https://arxiv.org/html/2511.01836v3#A9.E27)), the distribution of $S_{i}$ will also depend only on $n_{i}$, becoming uniform when conditioned on $n_{i}$. In TopK SAE, $n_{i}=K$ is a constant. Therefore, each $S_{i}\sim U([M]^{K})$, and together with independence argued above,

$$ $\displaystyle S_{1},\dots,S_{N}\overset{\text{i.i.d.}}{\sim}U([M]^{K})$ (32) $$

This completes the proof for the priors of TopK SAE. $\square$

##### I.0.3 BatchTopK SAE

BatchTopK SAE (bussmann2024batchtopksparseautoencoders) is a modification of the TopK SAE. Instead of fixing sparsity like TopK, BatchTopK allows variable sparsity per input while fixing the mean sparsity over a batch at $K$. The objective function for BatchTopK SAE can equivalently be written as:

$$ $\displaystyle\operatorname*{arg\,min}_{\bm{D},\bm{z}}\sum_{i=1}^{N}\frac{1}{N}\|\bm{x}_{i}-\bm{D}\bm{z}_{i}\|_{2}^{2},$ (33) $\displaystyle\text{s.t. }\forall j,\bm{z}_{j}=f_{TopK}(\bm{x}_{j}),\;\frac{1}{N}\sum_{j=1}^{N}\|\bm{z}_{j}\|_{0}=K.$ (34) $$

While BatchTopK imposes a mean sparsity per batch, for simplicity, we use the batch size to match the size of the entire dataset (WLOG). Smaller batch sizes can easily be incorporated by adding separate constraints, each over the entire batch (only leads to a change in constants—lagrange multipliers—in the analysis).

Following similar analysis as for TopK SAE (App. [I.0.2](https://arxiv.org/html/2511.01836v3#A9.SS0.SSS2)), we can derive an equivalent prior over $\bm{z}$ for BatchTopK SAE:

$$ $\displaystyle P(\bm{z}_{1},\dots,\bm{z}_{N})\propto\prod_{i=1}^{N}\exp\big(-\lambda\big|\|\bm{z}_{i}\|_{0}-K\big|\big)$ (35) $$

The sparse codes for different samples $\{\bm{z}_{i}\}_{i}$ are thus sampled i.i.d. from a distribution that only depends on the sparsity penalty.
While this prior looks very similar to the prior of TopK SAE, the difference is that in TopK, the fixed sparsity constraint is imposed per sample, leading to a different Lagrange multiplier $\lambda_{i}$ per sample to optimize over, while in BatchTopK, we have a common multiplier $\lambda$ over all examples in a batch (with multiple batches, we will have one multiplier per batch), which is then optimized over to ensure that average sparsity per batch constraint is met.

Similar to the analysis for the TopK SAE, we get the following prior over different latents per sample:

$$ $\displaystyle z_{i}^{\mu}\mid S_{i}\sim$ $\displaystyle\begin{cases}U(0,\kappa)\;&\mu\in S_{i}\\ \delta_{0}\;&\mu\notin S_{i}\end{cases},\text{and}$ (36) $\displaystyle z_{i}^{\mu_{1}},\dots,z_{i}^{\mu_{|S_{i}|}}\mid S_{i}$ $\displaystyle\overset{\text{i.i.d.}}{\sim}U(0,\kappa)\;\text{ for }\mu_{\cdot}\in S_{i}$ (37) $$

The active indices $S_{i}$ are sampled uniformly conditioned on the number of active indices $n_{i}$:

$$ $\displaystyle S_{i}\mid n_{i}\sim U([M]^{n_{i}})$ (38) $$

The number of active latents $n_{i}$ are themselves sampled i.i.d. (since $n_{i}=\tilde{g}(\bm{z}_{i})$ and $\{\bm{z}_{i}\}_{i}$ are i.i.d.) from a distribution whose mean is fixed:

$$ $\displaystyle n_{1},\dots,n_{N}\overset{\text{i.i.d.}}{\sim}P,\;\text{s.t.}\;\mathbb{E}[n_{\cdot}]=K$ (39) $$

This completes the derivation for the BatchTopK prior. $\square$

##### I.0.4 JumpReLU SAE

JumpReLU SAE (rajamanoharan2024jumpingaheadimprovingreconstruction) is trained with the $L_{0}$ (pseudo-)norm regularizer. This leads to the following optimization problem:

$$ $\displaystyle\operatorname*{arg\,min}_{\bm{D},\bm{z}}\sum_{i=1}^{N}\frac{1}{N}\left(\|\bm{x}_{i}-\bm{D}\bm{z}_{i}\|_{2}^{2}+\lambda\|\bm{z}_{i}\|_{0}\right)$ $\displaystyle\text{s.t. }\forall k,\bm{z}_{k}=f_{JumpReLU}(\bm{x}_{k})$ $$

This objective is equivalent to the following prior over $\bm{z}$:

$$ $\displaystyle P(\bm{z}_{1},\dots,\bm{z}_{N})$ $\displaystyle\propto\prod_{i=1}^{N}\exp\Big(-\eta\|\bm{z}_{i}\|_{0}\Big)$ (40) $$

Noting the similarity with the TopK/ BatchTopK cases, we use the same analysis to derive the following conditions:

$$ $\displaystyle z_{i}^{\mu}\mid S_{i}\sim$ $\displaystyle\begin{cases}U(0,\kappa)\;&\mu\in S_{i}\\ \delta_{0}\;&\mu\notin S_{i}\end{cases},\text{and}$ (41) $\displaystyle z_{i}^{\mu_{1}},\dots,z_{i}^{\mu_{|S_{i}|}}\mid S_{i}$ $\displaystyle\overset{\text{i.i.d.}}{\sim}U(0,\kappa)\;\text{ for }\mu_{\cdot}\in S_{i}$ (42) $\displaystyle S_{i}\mid n_{i}$ $\displaystyle\sim U([M]^{n_{i}})$ (43) $$

The number of active latents $n_{i}$ are again i.i.d., but there is no constraint on the mean of the distribution (unlike BatchTopK which constrained the mean of $n_{i}$ to equal $K$):

$$ $\displaystyle n_{1},\dots,n_{N}\overset{\text{i.i.d.}}{\sim}P,$ (44) $$

which completes the analysis for JumpReLU SAE. $\square$