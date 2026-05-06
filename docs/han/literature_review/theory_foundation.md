---
title: "Literature review -- theory / foundational works"
author: Han Xuanyuan
date: 2026-02-28
type: document
---

## sparse coding


#### [Chen, Paiton, Olshausen. The Sparse Manifold Transform. arXiv:1806.08887 (2018)](https://arxiv.org/abs/1806.08887)
[[references#chen2018smt]]

The Sparse Manifold Transform (SMT) is a representation framework that unifies **sparse coding**, **manifold learning**, and **slow feature analysis**. It proposes a two-stage view: first map inputs to an overcomplete sparse code (discrete, part-like features), then learn a low-dimensional embedding of these sparse coefficients that *linearizes* the underlying transformations by enforcing temporal smoothness (notably using a second-order temporal derivative penalty in the embedding space). The overarching goal is an approximately invertible representation that captures both the sparse “dictionary” structure and the continuous manifold structure present in natural signals.


**Key methods**: sparse coding front-end; manifold embedding trained to minimize temporal variation (especially second-order temporal derivative) of embedded sparse coefficients; approximate invertibility/stackability.

**Main results**: shows that temporal continuity can linearize nonlinear transformations in the embedding space while keeping the representation approximately invertible; provides a unified conceptual framework tying together sparsity and slowness.


#### [Chalk, Marre, Tkačik. Toward a Unified Theory of Efficient, Predictive, and Sparse Coding. PNAS (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5776796/)
[[references#chalk2018unified]]

This paper provides a theoretical framework unifying **efficient coding**, **predictive coding**, and **sparse coding** for temporal sensory data. It analyzes how optimal codes depend on stimulus statistics, internal constraints, and the trade-off between representing past inputs efficiently versus predicting the future. The work characterizes regimes in which optimal coding becomes temporally extended or smoothing-like, and presents “phase diagram”-style results describing transitions between coding behaviors as constraints vary.

**Key methods**: normative/theoretical derivations for optimal encoding under constraints; analysis of coding regimes as a function of stimulus statistics and capacity constraints.

**Main results**: derives conditions under which predictive vs efficient objectives dominate; shows how sparsity and temporal correlations shape optimal temporal codes.

## HMMs

#### [Ghahramani & Jordan. Factorial Hidden Markov Models. Machine Learning (1997)](https://mlg.eng.cam.ac.uk/pub/pdf/GhaJor97a.pdf)
[[references#ghahramani1997fhmm]]

Factorial HMMs replace a single discrete latent chain with **multiple parallel Markov chains** whose combined state produces observations, yielding a “distributed state representation”. Exact inference becomes intractable (state space exponential in number of chains), so the paper derives variational (mean-field and structured) approximations and compares them to Gibbs sampling and exact methods where feasible.

**Key methods**: EM (Baum–Welch-style) learning with approximate E-steps; completely factorised and structured variational approximations; comparisons to Gibbs sampling; complexity analysis.

**Main results**: shows variational methods can provide efficient approximate inference/learning that scales to large factorial state spaces, often outperforming Gibbs sampling in time-to-quality; formalises why naïve “convert to a giant HMM” is exponential-time.

#### [Fox et al. A Sticky HDP-HMM with Application to Speaker Diarization. Annals of Applied Statistics (2011)](https://arxiv.org/abs/1010.3004)
[[references#fox2011stickyhmm]]

The sticky HDP-HMM is a Bayesian nonparametric HMM variant that adds a “sticky” self-transition bias to reduce unrealistic rapid state switching and address over-segmentation. The paper motivates stickiness, develops inference, and demonstrates improvements in applications such as speaker diarization, where persistence of speaker identity is a key inductive bias.

**Key methods**: hierarchical Dirichlet process HMM; sticky self-transition augmentation; Bayesian inference for state sequences under persistence bias; diarization evaluation framing.

**Main results**: stickiness mitigates pathological rapid switching / state fragmentation and improves interpretability and practical performance in persistent-regime sequence tasks like diarization.

#### [Johnson et al. Composing Graphical Models with Neural Networks for Structured Representations and Fast Inference. NeurIPS (2016)](https://arxiv.org/abs/1603.06277)
[[references#johnson2016svae]]

Structured Variational Autoencoders (“SVAE”) combine graphical-model latent structure (e.g., HMM/LDS family components) with neural-network observation models while retaining efficient structured inference (e.g. message passing) inside a variational objective. The core pitch is: keep the parts that make inference tractable (structured latent) while using neural nets for flexible emissions.

**Key methods**: variational objectives with embedded exponential-family message passing; hybrid graphical model + neural-network emissions; amortised approximations that preserve conjugacy where possible.

**Main results**: demonstrates that retaining structured inference inside a VAE-like framework can provide fast inference and structured latent representations while allowing flexible observation models.



## nonlinear ICA

#### [Hyvärinen, Morioka. Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA. NeurIPS (2016). arXiv:1605.06336](https://arxiv.org/abs/1605.06336)
[[references#hyvarinen2016tcl]]

This paper proposes **time-contrastive learning (TCL)** as a method for unsupervised feature extraction using temporal nonstationarity. By training a classifier to discriminate between time segments, the learned features can (under assumptions) identify underlying independent sources in a nonlinear ICA setting.

**Key methods**: discriminating time segments via classification; theoretical link to nonlinear ICA identifiability under nonstationarity; deep feature learning via surrogate supervised task.

**Main results**: provides identifiability-style theory and empirical demonstrations that temporal structure can enable robust unsupervised feature learning.


#### [Hyvärinen, Morioka. Nonlinear ICA of Temporally Dependent Stationary Sources. AISTATS (2017)](https://proceedings.mlr.press/v54/hyvarinen17a.html)
[[references#hyvarinen2017nica]]

This work provides identifiability results for nonlinear ICA when latent sources are temporally dependent even if stationary. It shows temporal dependencies can be leveraged to recover the mixing function, complementing nonstationarity-based identifiability results.

**Key methods**: theoretical analysis of nonlinear ICA under temporal dependence; identifiability conditions based on temporal structure.

**Main results**: establishes identifiability regimes where temporal dependence enables recovery of nonlinear mixtures, strengthening theoretical support for temporal-prior learning.





## language modelling


#### [Shannon. A Mathematical Theory of Communication. Bell System Technical Journal (1948)](http://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
[[references#shannon1948information]]

Shannon uses Markov-style approximations to illustrate how increasing context length yields progressively more English-like text generation and frames language modelling in terms of entropy and prediction. This is a core, highly-cited early reference connecting language statistics, Markov processes, and information-theoretic evaluation.

**Key methods**: information-theoretic analysis of communication; Markov approximations for text generation; entropy-based evaluation framing.

**Main results**: establishes entropy as a central lens for language predictability and uses Markov approximations as a concrete modelling illustration.


#### [Katz. Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer. IEEE TASSP (1987)](https://www.cis.upenn.edu/~danroth/Teaching/CS598-05/Papers/Katz87.pdf)
[[references#katz1987backoff]]

This paper presents what became known as **Katz back-off smoothing**, explicitly addressing the sparse-data problem in n-gram language modelling by discounting low counts (using Good–Turing style ideas) and recursively backing off to lower-order conditionals. It was developed for IBM speech recognisers but is presented as generally applicable wherever sparse discrete sequence statistics arise.

**Key methods**: Good–Turing-based discounting; recursive back-off to lower-order n-grams; probability mass redistribution for unseen events.

**Main results**: provides a computation- and memory-efficient smoothing method that avoids zero probabilities for unseen n-grams and improves practical language modelling under sparse data.

#### [Kneser & Ney. Improved Backing-off for M-gram Language Modeling. ICASSP (1995)](https://www-i6.informatik.rwth-aachen.de/publications/download/951/KneserNey_ICASSP1995.pdf)
[[references#kneser1995smoothing]]

This short but influential paper introduces what is now widely known as **Kneser–Ney smoothing**, improving back-off n-gram estimates using continuation counts and a principled redistribution of probability mass. It is a cornerstone method for strong n-gram baselines and is frequently referenced in later empirical smoothing surveys.

**Key methods**: improved back-off smoothing with continuation probabilities; discounting and principled back-off weighting.

**Main results**: improves performance over prior back-off schemes by using continuation statistics that better estimate lower-order distributions in back-off contexts.

#### [Chen & Goodman. An Empirical Study of Smoothing Techniques for Language Modeling. ACL (1996)](https://aclanthology.org/P96-1041.pdf)
[[references#chen1996smoothing]]

This paper provides a systematic empirical comparison of major **n-gram smoothing** methods (including Jelinek–Mercer, Katz, Church–Gale, and others), studying how performance depends on training size, corpus choice (e.g., Brown vs Wall Street Journal), and n-gram order, and introducing new simple interpolation variants that perform strongly. It is a go-to empirical guide for “what works” in classic Markov language modelling.

**Key methods**: extensive smoothing comparisons; cross-entropy/perplexity evaluation; parameter selection procedures; new interpolation variants.

**Main results**: shows that the relative ranking of smoothing methods depends strongly on training size and n-gram order, and reports strong performance from simple tuned interpolation variants.

#### [Brown et al. Class-Based n-gram Models of Natural Language. Computational Linguistics (1992)](https://aclanthology.org/J92-4003.pdf)
[[references#brown1992classngram]]

This work introduces **class-based n-gram language models** where words are mapped to latent classes, reducing sparsity and capturing syntactic/semantic regularities. It frames language modelling in a noisy-channel style and uses information-theoretic tools (entropy/mutual information) to motivate and derive clustering criteria and algorithms for assigning words to classes.

**Key methods**: class-based n-gram parameterisation; mutual information objectives for clustering; heuristic algorithms for suboptimal class assignments; perplexity evaluation framing.

**Main results**: demonstrates that word-class structure can reduce sparsity, yield more coherent predictive distributions, and produce classes with syntactic/semantic flavour depending on the statistics used.

#### [Bengio et al. A Neural Probabilistic Language Model. JMLR 3 (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
[[references#bengio2003neurallm]]

This landmark paper introduces a neural language model that learns distributed word representations and uses a neural network to predict the next word from a fixed-size context window (a neural generalisation of n-grams). It reports experiments on the Brown corpus and AP News, comparing perplexity to strong modified Kneser–Ney n-gram baselines and showing gains, especially when mixing neural predictions with trigram outputs.

**Key methods**: distributed word embeddings + feed-forward neural net for next-word prediction; perplexity evaluation; mixture of neural and n-gram outputs.

**Main results**: achieves lower perplexity than strong n-gram baselines and shows that mixing neural and trigram predictions further improves perplexity, suggesting complementary error patterns.


#### [Hsu et al. A Spectral Algorithm for Learning Hidden Markov Models. COLT (2012)](https://www.cs.columbia.edu/~djhsu/papers/hmm.pdf)
[[references#hsu2012spectralhmm]]

This paper develops a **provably correct spectral/moment-based** algorithm for learning HMM sequence distributions under rank/separation conditions, avoiding local optima typical of EM/Baum–Welch. The approach is framed as particularly relevant when the observation alphabet is large (explicitly noting NLP-style “words in a language”), since sample complexity depends on spectral properties rather than directly on vocabulary size.

**Key methods**: method-of-moments / spectral learning; SVD-based subspace identification; observable operator representations; sample complexity analysis.

**Main results**: gives a polynomial-time algorithm with correctness guarantees under spectral separation assumptions, and avoids EM’s local optima pathologies for the learnt sequence distributions/prediction tasks.


#### [Rabiner. A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. Proceedings of the IEEE (1989)](https://www.cs.cmu.edu/~cga/behavior/rabiner1.pdf)
[[references#rabiner1989hmm]]

This tutorial is the canonical accessible reference for HMM definitions, inference (forward-backward), decoding (Viterbi framing), and parameter estimation via reestimation / Baum–Welch (EM) style updates. It also provides historical context: core HMM theory from Baum and colleagues (late 1960s/early 1970s) and early speech processing implementations at CMU and IBM in the 1970s.

**Key methods**: forward-backward inference; Baum–Welch (EM) reestimation; decoding; practical modelling considerations for speech.

**Main results**: consolidates the core HMM toolkit and explains why reestimation improves likelihood (up to local maxima), providing practical steps for applying HMMs to real sequence problems.

#### [Baum et al. A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains. Annals of Mathematical Statistics (1970)](https://www.biostat.wisc.edu/~kbroman/teaching/statgen/2004/refs/baum.pdf)
[[references#baum1970baumwelch]]

This paper is a foundational source for the likelihood-increasing reestimation technique underlying what became widely known as the **Baum–Welch algorithm**, central to training HMMs via EM-style iterations. It formalises an iterative maximisation procedure for probabilistic functions of Markov chains and is one of the classic “Baum and colleagues” references highlighted in later tutorials.

**Key methods**: iterative maximisation technique for Markov-chain probabilistic functions (the core of Baum–Welch style reestimation).

**Main results**: establishes an iterative procedure that (under appropriate conditions) improves likelihood for Markov-chain-based probabilistic models, underpinning practical HMM training.

#### [Cutting et al. A Practical Part-of-Speech Tagger. ANLP (1992)](https://aclanthology.org/A92-1018.pdf)
[[references#cutting1992postagger]]

This paper presents an HMM-based part-of-speech tagger designed to be robust and practical, illustrating a historically important way HMMs were used in NLP: **sequence labelling** rather than pure next-word language modelling. It’s a good snapshot of how Markov assumptions (tag bigrams/trigrams, emissions as word likelihoods) can be “good enough” for strong real tasks.

**Key methods**: HMM sequence labelling for POS; practical tagging methodology; decoding with standard HMM routines.

**Main results**: demonstrates robust, accurate tagging with a practical HMM-based implementation, reinforcing HMMs as useful for NLP sequence labelling.

#### [Brants. TnT: A Statistical Part-of-Speech Tagger. ANLP (2000)](https://aclanthology.org/A00-1031.pdf)
[[references#brants2000tnt]]

TnT is a trigram HMM POS tagger that became a widely used strong baseline, combining Markov tag transitions with practical smoothing/unknown-word handling. It is a key “HMM done right” reference for sequence labelling in language.

**Key methods**: trigram HMM for POS; practical smoothing and unknown-word handling; efficient decoding for tagging.

**Main results**: strong practical tagging performance with a comparatively simple HMM framework, helping establish HMM taggers as baselines for years.

#### [Lafferty et al. Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data. ICML (2001)](https://daiwk.github.io/assets/Conditional%20random%20fields%20Probabilistic%20models%20for%20segmenting%20and%20labeling%20sequence%20data.pdf)
[[references#lafferty2001crf]]

CRFs are discriminative sequence models that retain a Markov-structured label dependency (chain structure) but condition on rich, overlapping observation features without the strong independence assumptions required by generative HMMs. The paper also discusses MEMMs and the label bias problem, positioning CRFs as a principled fix while keeping dynamic-programming tractability.

**Key methods**: conditional random fields; gradient-based parameter estimation; dynamic programming for normalization/inference; discussion of label bias vs MEMMs.

**Main results**: CRFs provide a more expressive, discriminative alternative to HMMs for sequence labelling while retaining tractable inference, and address label bias issues in directed discriminative Markov models.

#### [Bourlard & Morgan. Connectionist Speech Recognition: A Hybrid Approach. Kluwer (1994)](https://publications.idiap.ch/downloads/papers/2013/Bourlard_KLUWER_1994.pdf)
[[references#bourlard1994hybrid]]

This book documents hybrid **HMM/MLP** approaches to large-vocabulary continuous speech recognition, combining HMM temporal structure (state sequencing) with neural networks for estimating probabilities useful for recognition. It is a key historical reference for what became one of the most practically successful uses of HMMs: as the temporal scaffold in speech recognition systems with increasingly powerful discriminative components.

**Key methods**: hybrid HMM/MLP (connectionist) speech recognition systems; practical training and system integration considerations.

**Main results**: documents that HMM/NN hybrids can yield strong practical ASR performance and offers detailed system-level lessons about combining temporal Markov structure with neural estimators.

#### [Young. Statistical Modelling in Continuous Speech Recognition (CSR). UAI (2001)](https://arxiv.org/pdf/1301.2318)
[[references#young2001csr]]

This review surveys the evolution of statistical speech recognition, focusing on the HMM + n-gram framework and its refinements, and explicitly credits IBM’s 1970s work as foundational. It explains modelling assumptions, mitigation techniques, and directions beyond classical HMM/n-gram pipelines.

**Key methods**: survey of HMM and n-gram modelling in CSR; discussion of refinements and mitigation strategies; historical framing of modelling choices.

**Main results**: synthesises why the HMM+n-gram framework worked and how successive refinements addressed limitations, serving as a “state of the art as of early-2000s” snapshot.

