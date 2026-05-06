---
title: "Literature review -- SAEs"
author: Han Xuanyuan
date: 2026-02-28
type: document
---

## Time aware SAEs

#### [Bhalla et al. Temporal Sparse Autoencoders (T-SAEs). arXiv:2511.05541 (2025)](https://arxiv.org/abs/2511.05541)
[[references#bhalla2025tsae]]

Temporal Sparse Autoencoders modify standard SAEs for language-model activations by adding an explicit temporal contrastive term that encourages higher-level features to remain consistent across adjacent tokens. The paper motivates this with a distinction between slow semantic variables and fast syntactic variables, and argues that vanilla per-token SAE training overemphasizes shallow/token-local features. Empirically, T-SAEs aim to recover smoother, more semantically coherent features without sacrificing reconstruction quality.

**Key methods**: Matryoshka-style partition into high-level vs low-level features + temporal contrastive loss on adjacent-token high-level codes; probing-based analyses for semantics/syntax separation.

**Main results**: improves semantic coherence and temporal smoothness of learned features relative to standard SAE baselines while maintaining competitive reconstruction.

#### [Lubana et al. Priors in Time: Temporal Feature Analysis for Interpreting Neural Representations. arXiv:2511.01836 (2025)](https://arxiv.org/abs/2511.01836)
[[references#lubana2025tfa]]

This work argues that standard SAE objectives implicitly impose an i.i.d. prior over time on latent codes, which clashes with the strong temporal structure and non-stationarity observed in language-model activations. It introduces Temporal Feature Analysis (TFA), which decomposes each activation into a predictable component (inferred from context) and a residual novel component, thereby building temporal inductive bias directly into the interpretability method. The paper demonstrates that this decomposition can capture meaningful temporal structure in language phenomena such as event boundaries and garden-path resolution.

**Key methods**: learned projection/attention from past context to produce predictable code; residual code as novelty; empirical analysis of autocorrelation, intrinsic dimension, and context-explained variance.

**Main results**: predictable codes capture temporally structured linguistic phenomena (e.g., event boundaries, ambiguity resolution) where standard SAEs show pitfalls; highlights non-stationarity and time-varying correlations in LM activations.


#### [Oublal et al. TimeSAE: Sparse Decoding for Faithful Explanations of Black-Box Time Series Models. arXiv:2601.09776 (2026)](https://arxiv.org/abs/2601.09776)
[[references#oublal2026timesae]]

TimeSAE adapts SAE-style sparse concept decompositions to explain black-box time-series predictors, emphasizing faithfulness under distribution shift. It learns sparse concepts and a decoder that maps concept activations back into time/feature attributions, with additional machinery aimed at counterfactual-style faithfulness. The paper reports results on synthetic and real-world time-series benchmarks and claims improved explanation faithfulness and robustness relative to baselines.

**Key methods**: sparse concept learning for time series; decoder-based attribution; faithfulness/counterfactual evaluation protocols.

**Main results**: reports improved faithfulness/robustness of explanations relative to competing explainers.


#### [Baccouche et al. Spatio-Temporal Convolutional Sparse Auto-Encoder for Sequence Classification. BMVC (2012)](https://www.bmva.org/bmvc/2012/Baccouche_57/Baccouche_57.pdf)
[[references#baccouche2012spatiotemporal]]

This work proposes an unsupervised spatio-temporal convolutional sparse autoencoder that learns sparse, shift-invariant features over local video blocks. It then models the temporal evolution of these features with an LSTM for sequence classification tasks such as action recognition and facial expression recognition. The paper is an early example of combining sparse feature learning with explicit temporal modeling.

**Key methods**: convolutional SAE over spatiotemporal patches; sparsifying nonlinearity; LSTM trained on feature trajectories for classification.

**Main results**: reports competitive (sometimes state-of-the-art) classification performance using learned sparse features.




#### [Gao et al. Sequential sparse autoencoder for dynamic heading representation in ventral intraparietal area. Computers in Biology and Medicine (2023)](https://doi.org/10.1016/j.compbiomed.2023.107114)
[[references#gao2023sequential]]

This paper uses a sequential sparse autoencoder to learn spatiotemporal features from neural population recordings and decode heading direction. The model treats temporal dynamics as a representation-learning problem with a sparse latent and evaluates utility via downstream decoding performance. While not about LLMs, it provides a concrete example of sparse autoencoding adapted for temporally structured biological signals.

**Key methods**: sequential SAE architecture over time + classifier/decoder for heading prediction.

**Main results**: reports high decoding accuracy and emphasizes efficiency/robustness for dynamic decoding.



#### [Demircan et al. Sparse Autoencoders Reveal Temporal Difference Learning in Large Language Models. arXiv:2410.01280 (2024)](https://arxiv.org/abs/2410.01280)
[[references#demircan2024tdlearning]]

This paper uses SAEs to analyze an LLM performing in-context reinforcement learning tasks and reports SAE features matching **temporal difference (TD) errors**. It then performs interventions indicating these features are causally involved in computing TD errors and Q-values.

**Key methods**: SAE training on residual-stream activations; identification of latents correlated with TD error/Q-values; causal interventions to test functional involvement.

**Main results**: finds interpretable SAE features corresponding to TD-learning variables and validates their causal role in the model’s in-context RL-like computation.



## Other SAE related papers


#### [Chanin et al. Sparse but Wrong: When Sparsity Leads to Misrecovered SAE Features. arXiv:2508.16560 (2025)](https://arxiv.org/abs/2508.16560)
[[references#chanin2025sparsewrong]]

Sparse but Wrong shows that even when ground-truth features are linear and known, SAEs can learn the “wrong” latents when underlying factors are correlated, because mixing correlated directions can improve reconstruction under a constrained sparsity budget. The paper uses toy models with controlled feature correlations and validates analogous patterns in LLM SAEs, arguing that common “sparsity–reconstruction tradeoff” plots can be misleading. It proposes proxy metrics based on decoder projections to help detect when L0 is mis-specified.

**Key methods**: synthetic toy data with controlled correlation matrices; analysis of feature mixing under low/high L0; decoder-projection proxy metrics; comparisons of BatchTopK and JumpReLU-style training.

**Main results**: demonstrates systematic feature mixing at mis-specified L0 (both too low and too high), undermining monosemanticity even when reconstruction improves.


#### [Nanda et al. Sparse Crosscoders (Transformer Circuits). Web resource (2024)](https://transformer-circuits.pub/2024/crosscoders/index.html)
[[references#nanda2024crosscoders]]

Crosscoders extend sparse dictionary learning beyond single-layer decomposition by learning sparse mappings that align or translate representations across layers (and, in some applications, across model variants). In interpretability workflows, this can support tracking “the same” feature across depth and diagnosing where it is computed or transformed. For temporal/persistent-feature projects, crosscoders provide scaffolding for linking token-level persistence to multi-layer computation.

**Key methods**: sparse linear mappings / dictionary learning objectives that encourage alignment between representations in different layers or models; evaluation via feature tracking and interpretability analyses.

**Main results**: enables feature correspondence across depth and model changes, supporting deeper mechanistic analyses than single-layer SAEs alone.

#### [Li, T. Ed; Ren, Junyu. Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training. arXiv:2510.08855 (2025)](https://arxiv.org/pdf/2510.08855)
[[references#li2025atm]]

This work targets training-time instability in SAEs—specifically “feature absorption” where features collapse/merge under sparsity pressure. It proposes Adaptive Temporal Masking (ATM), which tracks activation statistics over time (magnitudes, frequencies, reconstruction contributions) to compute evolving importance scores and then probabilistically masks features based on those statistics. Experiments on Gemma‑2‑2B claim substantially reduced absorption compared to TopK/JumpReLU baselines while maintaining reconstruction quality.

**Key methods**: time-varying importance scores from activation statistics + probabilistic masking schedule; comparisons against standard SAE training baselines.

**Main results**: reduces absorption/feature-collapse indicators while keeping reconstruction competitive.



#### [Engels, Smith, Tegmark. Decomposing the Dark Matter of Sparse Autoencoders. arXiv:2410.14670 (2024)](https://arxiv.org/abs/2410.14670)
[[references#engels2024darkmatter]]

This paper studies SAE “dark matter”—the residual variance not captured by sparse reconstructions—and argues that the residual is structured rather than pure noise. A key empirical claim is that much of the error is linearly predictable from the original activation and shows systematic scaling behavior with SAE size. The authors propose a decomposition of error sources (e.g., unlearned linear features, dense components, nonlinear errors introduced by the SAE) and explore interventions to reduce harmful nonlinear error.

**Key methods**: empirical predictability tests for residuals; scaling analyses; inference-time pursuit or transformations intended to reduce nonlinear reconstruction artifacts.

**Main results**: shows substantial structure in SAE residuals and that certain components behave like scaling-limited error floors; partial improvements via pursuit/transformations.

#### [Muhamed, Diab, Smith. Decoding Dark Matter: Specialized Sparse Autoencoders for Interpreting Rare Concepts in Foundation Models. arXiv:2411.00743 (2024)](https://arxiv.org/abs/2411.00743)
[[references#muhamed2024specialsae]]

This paper argues that general-purpose SAEs underrepresent rare but important concepts and frames these as a “dark matter” problem linked to data scarcity and long-tail structure. It proposes Specialized SAEs trained on curated subdomain data (e.g., via retrieval-based selection) and uses risk-sensitive objectives to prioritize recall of tail concepts. The main message is that training distribution and objective design can strongly affect which concepts are recoverable.

**Key methods**: retrieval/subdomain data selection; specialized SAE training; tail-focused risk objectives (e.g., tilted empirical risk).

**Main results**: improves capture of rare subdomain concepts and reports robustness gains relative to general SAEs.


#### [Engels, Michaud, Liao, Gurnee, Tegmark. Not All Language Model Features Are One-Dimensionally Linear. arXiv:2405.14860 (ICLR 2025)](https://arxiv.org/pdf/2405.14860)
[[references#engels2025multidim]]

This paper argues that some language-model “features” are **irreducibly multi-dimensional**, meaning they cannot be decomposed into a set of independent or non-co-occurring one-dimensional features. It proposes formal definitions for multi-dimensional features and introduces a scalable method that uses sparse autoencoders plus clustering/analysis to automatically discover such features in real LMs (e.g., GPT-2 and Mistral 7B). The headline empirical examples are strikingly interpretable **circular manifolds** representing concepts like days of the week and months of the year, which the authors claim are causally used for modular arithmetic-like computations in larger models.

**Key methods**: definitions for irreducible multi-dimensional features; SAE-based discovery pipeline (including clustering of dictionary elements into low-dimensional subspaces/manifolds); intervention experiments to test causal role in tasks.

**Main results**: discovers interpretable circular subspaces for temporal/cyclic concepts; provides evidence these representations are used causally in relevant computation tasks; argues multi-dimensional features are necessary to explain some behaviors.

#### [Song, Zhong. Uncovering Hidden Geometry in Transformers via Disentangling Position and Context. arXiv:2310.04861 (2023)](https://arxiv.org/abs/2310.04861)
[[references#song2023geometry]]

This paper proposes a simple mean-based decomposition of transformer hidden states into **global mean**, **positional basis**, **context basis**, and a **residual** component. Across multiple architectures and datasets, it reports pervasive geometric structure: positional bases form low-dimensional, smooth, often spiral-like trajectories across layers; context bases cluster by topic/document; and positional and context components are nearly orthogonal. The authors argue that smoothness is a robust and beneficial property in language-trained transformers and that the decomposition improves interpretability and diagnostic analysis.

**Key methods**: mean-effects decomposition across contexts and positions; PCA/Gram-matrix visualization; spectral and Fourier/DCT analyses of positional structure; quantitative measures of low-rankness, smoothness, clustering, and incoherence.

**Main results**: positional components exhibit robust low-rank, low-frequency (smooth) geometry across layers and even under distribution shifts; context components cluster semantically; positional and context bases are nearly orthogonal, yielding a clean disentangling lens.


#### [Rolfe, LeCun. Discriminative Recurrent Sparse Auto-Encoders (DrSAE). arXiv:1301.3775 (2013)](https://arxiv.org/abs/1301.3775)
[[references#rolfe2013drsae]]

DrSAE interprets the encoder as an **unrolled iterative inference process** (akin to ISTA dynamics) and trains the resulting recurrent network end-to-end with backpropagation through time. Over multiple inference steps, representations are refined and can exhibit hierarchical structure, with “part” units and more categorical units emerging through recurrent interactions. This provides an explicit temporal/unrolled computation view of sparse inference, where the “time” dimension is the inference trajectory rather than the data sequence.

**Key methods**: unrolled ISTA-like dynamics; recurrent architecture trained via BPTT; joint reconstruction and discriminative objectives.

**Main results**: recurrent inference yields improved discriminative performance and emergent hierarchical representations compared to static sparse encoders in certain settings.
