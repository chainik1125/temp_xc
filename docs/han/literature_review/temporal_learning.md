---
title: "Literature review -- temporal learning"
author: Han Xuanyuan
date: 2026-02-28
type: document
---

## Sparse coding / dictionary learning with temporal objectives

#### [Hu, Chklovskii. Online Sparse Coding with Temporal Smoothness: Leaky Linearized Bregman Iteration (LLBI). IEEE Transactions on Neural Networks and Learning Systems (2015)](https://ieeexplore.ieee.org/document/7178319)
[[references#hu2015llbi]]

This paper studies **online sparse coding** for time-varying stimuli and proposes Leaky Linearized Bregman Iteration (LLBI), an optimization algorithm that explicitly exploits temporal correlations in consecutive inputs. The key idea is to add a Bregman-divergence penalty between successive sparse codes, creating an update rule that balances sparsity with attraction to the previous code via a “forgetting factor.” The authors motivate the algorithm with a generative model in which each coefficient is the product of a Markov-evolving support variable and an AR(1)-evolving magnitude, closely matching common autocorrelation toy models for sparse latents.

**Key methods**: LLBI updates using a linearized gradient step plus a Bregman divergence term tying successive codes together; sparsity induced by an elastic-net-style regularizer; forgetting factor controlling temporal coupling strength.

**Main results**: exploiting temporal correlation improves sparse-code tracking and reconstruction in online settings; performance depends strongly on matching the forgetting factor to the true temporal correlation timescale.


#### [Olshausen. Learning Sparse, Overcomplete Representations of Time-Varying Natural Images. ICIP (2003)](https://www.rctn.org/bruno/papers/icip03.pdf)
[[references#olshausen2003spatiotemporal]]

This paper extends classic sparse coding to **video** by learning overcomplete dictionaries of *space–time* basis functions whose coefficients are sparse over time. The model represents image sequences as a superposition of spatiotemporal basis elements (e.g., localized, oriented, bandpass filters that translate or evolve across time), and the inferred coefficient signals tend to be temporally spike-like. A key contribution is showing that when trained on natural movies, sparse coding yields spatiotemporal receptive fields reminiscent of early visual neurons, helping connect sparse coding to biological plausibility in temporal sensory streams.

**Key methods**: overcomplete spatiotemporal dictionary learning; sparse inference for coefficient time series; training on natural image sequences.

**Main results**: learned basis functions are spatially localized and oriented with coherent temporal evolution (e.g., translation), and inferred coefficients form sparse, spike-like temporal codes.






#### [Cadieu, Olshausen. Learning Intermediate-Level Representations of Form and Motion from Natural Movies. Neural Computation (2012)](https://pubmed.ncbi.nlm.nih.gov/22168556/)
[[references#cadieu2012formmotion]]

This work learns intermediate-level visual representations from natural movies using a two-stage sparse coding approach. The first stage learns a sparse representation that factorizes information into temporally persistent components (interpretable as “form” amplitude) and dynamic components (capturing motion via phase-like variables). A second stage then learns higher-order representations by exploiting statistical dependencies in these amplitude signals and in temporal derivatives (e.g., phase changes), yielding units that capture richer invariances across form and motion.

**Key methods**: multi-layer sparse coding on natural movies; complex-valued / phase-amplitude style factorization; learning higher-order dependencies using amplitude features and temporal derivatives.

**Main results**: learns intermediate representations that factorize form and motion components and capture higher-order invariances from temporal statistics.




#### [Charles et al. GraFT: Graph Filtered Temporal Dictionary Learning for Functional Neural Imaging. IEEE Transactions on Image Processing 31 (2022)](https://ieeexplore.ieee.org/document/9789909)
[[references#charles2022graft]]

GraFT frames functional imaging (e.g., calcium imaging) as **temporal dictionary learning**, where dictionary elements are time traces and spatial maps serve as sparse coefficients. It introduces graph filtering based on shared temporal activity to robustly extract neural components across modalities and handle variability.

**Key methods**: temporal dictionary learning with sparse spatial coefficients; graph filtering/regularization across spatial units; robust component extraction across imaging conditions.

**Main results**: improves recovery of neural components and temporal traces under challenging imaging conditions, supporting robust extraction across sessions/modalities.








#### [Springenberg, Riedmiller. Learning Temporal Coherent Features through Life-Time Sparsity. ICONIP (2012)](https://dblp.org/rec/conf/iconip/SpringenbergR12.html)
[[references#springenberg2012temporal]]

This paper combines a **temporal coherence** objective with a “**lifetime sparsity**” constraint. Lifetime sparsity encourages each hidden unit to be active rarely across its lifetime (dataset), while temporal coherence encourages consecutive frames to produce similar activations. The combination aims to learn features that are both selective (sparse) and temporally stable, and the authors report consistent recognition improvements from adding the temporal coherence term.

**Key methods**: temporal coherence loss on consecutive frames; lifetime sparsity constraint across the dataset; evaluation via downstream recognition.

**Main results**: reports accuracy improvements from temporal coherence beyond sparsity-only baselines; supports the claim that temporal regularization is broadly helpful.





#### [Zou, Ng, Yu. Unsupervised Learning of Visual Invariance with Temporal Coherence. NeurIPS Workshop (2011)](https://ai.stanford.edu/~wzou/nipswshop_ZouNgYu11.pdf)
[[references#zou2011temporal]]

This workshop paper trains a two-layer model on unlabeled natural videos with an explicit **temporal coherence** regularizer encouraging representations to change smoothly between consecutive frames. The first layer learns low-level features; higher-layer pooling units learn to associate first-layer features across time, yielding more invariant representations. The paper reports consistent improvements on object recognition benchmarks and argues that temporal coherence can be a powerful alternative or complement to sparsity-driven unsupervised learning.

**Key methods**: two-layer unsupervised feature learning; temporal coherence regularization; pooling/association across time to build invariances.

**Main results**: temporal coherence improves recognition performance relative to sparsity-only or non-temporal baselines, suggesting time-based priors are broadly beneficial.



## Autoencoders with slowness/temporal coherence regularization

#### [van den Oord et al. Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748 (2018)](https://arxiv.org/abs/1807.03748)
[[references#oord2018cpc]]

CPC learns representations by **predicting future samples in latent space** using an autoregressive context model and a contrastive (InfoNCE) objective that estimates a density ratio and lower-bounds mutual information. The paper argues that predicting further into the future encourages “slow features” spanning many time steps, and demonstrates performance across multiple modalities including speech, images, text, and reinforcement learning environments.

**Key methods**: encoder + autoregressive context model; InfoNCE contrastive loss; mutual-information lower bound interpretation; multi-step prediction in latent space.

**Main results**: shows that a unified contrastive predictive objective can learn transferable representations and achieve strong downstream performance across distinct domains, supporting the “slow/shared latent” framing.


#### [Klindt et al. Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding (SlowVAE). ICLR (2021). arXiv:2007.10930](https://arxiv.org/abs/2007.10930)
[[references#klindt2021slowvae]]

This work introduces **SlowVAE**, a variational autoencoder with a *temporal sparse prior* designed to achieve nonlinear disentanglement in realistic sequential data. The key empirical claim is that in segmented natural movies, latent factors tend to undergo mostly small changes with occasional large jumps; SlowVAE captures this with a sparse prior on differences between consecutive latent states (often described as Laplacian / generalized Laplace behavior). The paper provides a theoretical identifiability result showing that, under this temporally sparse transition assumption, latent variables can be recovered (up to permutation and sign flips), and demonstrates strong disentanglement performance on standard benchmarks as well as more natural video-style datasets.

**Key methods**: VAE with a sparse prior on temporal differences of latents; theory connecting sparse temporal transitions to identifiability; benchmark evaluation including new natural-dynamics datasets.

**Main results**: achieves strong disentanglement scores and transfers to more natural temporal dynamics; provides identifiability guarantees under the temporal sparse transition assumptions.


#### [Goroshin et al. Unsupervised Learning of Spatiotemporally Coherent Metrics. ICCV (2015)](https://openaccess.thecvf.com/content_iccv_2015/papers/Goroshin_Unsupervised_Learning_of_ICCV_2015_paper.pdf)
[[references#goroshin2015spatiotemporal]]

This paper studies unsupervised representation learning from video using a combination of **reconstruction**, **sparsity**, and a **temporal coherence / slowness-type** penalty, explicitly showing that maximising temporal coherence *alone* can yield weakly discriminative features. It evaluates temporal coherence on a large YouTube-derived dataset and tests transfer to CIFAR-10 via nearest-neighbour precision in feature space.

**Key methods**: convolutional autoencoder-style pipelines with sparsity and slowness; metric learning comparisons (e.g., DrLIM); greedy layerwise training; evaluation via temporal nearest neighbours and CIFAR-10 transfer.

**Main results**: shows that directly maximising temporal coherence can produce poor semantic neighbourhoods, while combining temporal penalties with reconstruction and sparsity yields more discriminative and temporally consistent features; reports detailed precision/recall style evaluations.

#### [Struckmeier, Tiwari, Kyrki. Autoencoding Slow Representations for Semi-supervised Data-efficient Regression. Machine Learning (2023)](https://link.springer.com/article/10.1007/s10994-022-06299-1)
[[references#struckmeier2023slowrep]]

This paper proposes a semi-supervised learning approach that improves data efficiency by learning **slow (temporally coherent) latent representations** with an autoencoder and then training a regression model using only a small amount of labeled data. It compares multiple slowness mechanisms, including Lp-norm penalties between adjacent latents, the SlowVAE-style temporal sparse prior, and a Brownian-motion-based temporal similarity constraint. The results suggest that encouraging slowness in the representation can dramatically improve regression performance when labels are scarce.

**Key methods**: autoencoder/variational autoencoder with slowness regularization; comparison of Lp smoothness, temporal sparse priors, and Brownian similarity constraints; semi-supervised regression pipeline.

**Main results**: slow representations substantially improve regression accuracy and data efficiency relative to non-slow baselines; different slowness terms perform similarly in many regimes.


#### [Struckmeier et al. Learning Slow Features with a Variational Autoencoder (Slowness-Regularized VAE). arXiv:2012.06279 (2020)](https://arxiv.org/pdf/2012.06279)
[[references#struckmeier2020slowvae]]

This paper presents a general framework for incorporating **slowness regularization** into variational autoencoders by augmenting a β-VAE-style objective with an explicit temporal loss. The resulting objective combines reconstruction, an information-bottleneck/KL term, and a slowness term that encourages adjacent latent states to change slowly. The authors compare multiple concrete slowness formulations—simple L1/L2 penalties between consecutive latents, a sparse temporal prior (SlowVAE-style), and a Brownian-motion prior—and report that adding slowness generally improves downstream performance and data efficiency, while no single slowness term dominates across tasks.

**Key methods**: β-VAE objective augmented with a temporal slowness regularizer; comparative study of multiple slowness penalties/priors including a Brownian motion latent prior.

**Main results**: slowness regularization typically yields equal or better downstream task performance and improved sample efficiency relative to the β-VAE baseline; choice of slowness term matters, but the presence of slowness is often the main driver.


#### [Wehmeyer & Noé. Time-lagged autoencoders: Deep learning of slow collective variables for molecular kinetics. arXiv:1710.11239 (2017)](https://arxiv.org/abs/1710.11239)
[[references#wehmeyer2017tae]]

Time-lagged autoencoders (TAEs) modify the autoencoder objective so that the network predicts a **future frame** \(x_{t+\tau}\) from \(x_t\) rather than reconstructing \(x_t\), aligning representation learning with the identification of **slow variables** in (approximately) Markovian dynamical systems. The paper connects linear TAEs to time-lagged CCA/TICA under assumptions and positions TAEs as tools for building or aiding Markov state models in molecular dynamics.

**Key methods**: time-lagged prediction autoencoder objective; theoretical links to TCCA/TICA in linear cases; nonlinear encoders/decoders for slow-coordinate discovery.

**Main results**: argues and demonstrates that TAEs can learn embeddings that capture slow coordinates better than linear baselines, supporting improved kinetic modelling and long-timescale structure discovery.


#### [Hernández et al. Variational Encoding of Complex Dynamics (Variational Dynamics Encoder, VDE). Physical Review E 97, 062412 (2018). arXiv:1711.08576](https://arxiv.org/abs/1711.08576)
[[references#hernandez2018vde]]

This paper introduces the **Variational Dynamics Encoder (VDE)**, a time-lagged variational autoencoder designed to learn low-dimensional coordinates that capture the **slow modes** of complex dynamics. Instead of reconstructing the current input, the model is trained to reconstruct a future state \(x_{t+	au}\), and it augments the time-lagged reconstruction with an explicit **latent autocorrelation objective** intended to maximize the slowness (timescale) of the learned coordinate. The autocorrelation term is motivated by the variational approach to conformational dynamics (VAC), framing representation learning as a way to approximate optimal slow collective variables.

**Key methods**: time-lagged VAE reconstruction \(x_t 	o x_{t+	au}\); latent autocorrelation loss (VAC-inspired) to favor slow coordinates; evaluation via free-energy landscapes and Markov state model timescales.

**Main results**: on Brownian dynamics and protein-folding examples, the VDE coordinate captures slow conformational structure and can outperform linear baselines (e.g., tICA) in separating metastable states and yielding longer implied timescales.




## Temporal dynamics in latent space

#### [Krishnan et al. Structured Inference Networks for Nonlinear State Space Models. AAAI (2017)](https://arxiv.org/abs/1609.09869)
[[references#krishnan2017dmm]]

This paper reframes “VAE-style amortised inference” for time series as learning a *compiled* inference network for latent **Gaussian state space models** whose transition and emission functions may be neural networks. It motivates **Deep Markov Models (DMMs)** as “HMM-like” (Markovian latent) sequence generative models with deep parameterisations, and argues that inference should use *future* information (smoothing-like) rather than only past context (filtering-like) for better variational bounds and held-out likelihood.

**Key methods**: variational learning with structured inference networks (RNN-parameterised) for nonlinear state-space models; DMM framing; comparisons of mean-field vs structured variational approximations. 

**Main results**: structured inference (including use of future information) improves held-out likelihood; DMMs compare favourably to RNN and other stochastic-RNN baselines on polyphonic music, and support “querying”/counterfactual-style questions in EHR settings. 

#### [Karl et al. Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data. ICLR (2017)](https://openreview.net/forum?id=HyTqHL5xg)
[[references#karl2017dvbf]]

DVBF explicitly targets **latent Markovian state-space model identification** from high-dimensional observations by combining a latent transition model with stochastic-gradient variational Bayes and emphasising *backpropagation through the transition dynamics* as a way to enforce state-space assumptions. The paper’s experiments are geared toward whether the learned latent actually captures the underlying physical state (not merely reconstructing frames). 

**Key methods**: SGVB/ELBO training for latent state-space models; explicit latent transition learning; analyses of latent “information content” and long-horizon generative stability. 

**Main results**: DVBF learns latents that encode full system state (e.g., both angle and angular velocity) and supports stable long-term prediction beyond training-horizon length; contrasts with DKF on recovering velocity information in the pendulum setting.

#### [Fraccaro et al. A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning. arXiv:1710.05741 (2017)](https://arxiv.org/abs/1710.05741)
[[references#fraccaro2017kvae]]

The **Kalman Variational Auto-Encoder (KVAE)** splits sequential modelling into (a) a per-frame recognition model producing a latent “encoding” and (b) a **linear Gaussian state-space model** for dynamics with *exact* Kalman filtering/smoothing, while allowing nonlinearity by letting dynamics parameters vary over time via a separate network. It targets missing-frame imputation and interpretable latent dynamics in video-like settings (bouncing balls, pendulum, Pong-like environment variants).

**Key methods**: VAE encoder/decoder + LGSSM (Kalman filtering/smoothing); time-varying dynamics parameters via an auxiliary network; evaluation via imputation and generation. 

**Main results**: learns interpretable latent encodings for position/velocity-like factors; improves missing-frame imputation using smoothing, and produces realistic rollouts across several environments; reports comparative ELBOs against alternatives (including DVBF/DMM-style baselines). 

#### [Krishnan et al. Deep Kalman Filters. arXiv:1511.05121 (2015)](https://arxiv.org/abs/1511.05121)
[[references#krishnan2015dkf]]

Deep Kalman Filters provide a variational-learning framework for a broad class of Kalman-filter-like generative models where non-linear transition/emission functions are parameterised by deep networks, motivated by counterfactual inference and high-dimensional sequential modelling. It introduces the “Healing MNIST” dataset and applies the approach to EHR-style patient trajectories with actions/interventions. 

**Key methods**: variational lower bound training for state-space models; deep parameterisations of transitions/emissions; counterfactual inference motivation; experiments comparing inference parameterisations.

**Main results**: demonstrates feasibility of learning nonlinear Kalman-filter-like temporal generative models with deep nets; illustrates modelling of action effects and application to EHR counterfactual-style questions. 

#### [Chung et al. A Recurrent Latent Variable Model for Sequential Data. arXiv:1506.02216 (2015)](https://arxiv.org/abs/1506.02216)
[[references#chung2015vrnn]]

The Variational Recurrent Neural Network (VRNN) injects latent random variables into an RNN’s hidden dynamics, combining RNN sequence modelling with VAE-style latent variables to represent sequence variability. The paper evaluates on multiple speech datasets and a handwriting dataset, highlighting that latents help capture structured variability beyond deterministic RNNs.

**Key methods**: RNN sequence model with per-time-step latent variables trained via variational objectives; empirical comparisons to related sequence models.

**Main results**: demonstrates improved modelling of structured sequential variability via latent variables integrated into RNN dynamics.

#### [Bayer & Osendorfer. Learning Stochastic Recurrent Networks. arXiv:1411.7610 (2014)](https://arxiv.org/abs/1411.7610)
[[references#bayer2014storn]]

STORNs (“Stochastic Recurrent Networks”) extend deterministic RNNs by introducing latent random variables trained with SGVB-style estimators, targeting more expressive conditional distributions at each time step. The paper evaluates on multiple polyphonic music datasets and motion capture, positioning STORNs as a generalisation that captures multi-modality and uncertainty better than deterministic RNNs.

**Key methods**: SGVB-trained stochastic RNN with per-step latent variables; likelihood estimation; evaluations on structured sequence benchmarks.

**Main results**: demonstrates practical training of stochastic RNNs with latent variables and improved expressivity for sequential data.

#### [Srivastava et al. Unsupervised Learning of Video Representations using LSTMs. arXiv:1502.04681 (2015)](https://arxiv.org/abs/1502.04681)
[[references#srivastava2015lstmae]]

This is an influential “sequence autoencoder” line: an encoder LSTM maps an input sequence to a fixed-length representation, then decoder LSTMs reconstruct the input and/or predict future frames. It evaluates representations by fine-tuning on action recognition tasks (e.g., UCF-101 and HMDB-51) and uses both raw patches and higher-level percept inputs.

**Key methods**: LSTM encoder–decoder; reconstruction vs future-prediction training objectives; representational evaluation via downstream supervised fine-tuning.

**Main results**: learned sequence representations can improve downstream recognition performance, especially in low-label regimes, suggesting that temporal structure provides useful self-supervision.


#### [Li, Mandt. Disentangled Sequential Autoencoder. ICML (2018). arXiv:1803.02991](https://arxiv.org/abs/1803.02991)
[[references#li2018dsequential]]

This paper proposes a sequential autoencoder that disentangles **static (content)** factors from **dynamic** factors. The dynamic component is modeled with a stochastic RNN and Markovian latent transitions, while the static component captures time-invariant information across a sequence. The approach is motivated by learning representations that separate persistent identity-like variables from time-varying motion/action variables.

**Key methods**: split latent space into static and dynamic parts; stochastic RNN transition model for dynamics; sequence-level inference encouraging content persistence.

**Main results**: improves disentanglement of content vs dynamics on sequential benchmarks, yielding representations with clearer factor separation.



#### [Wu, Mardt, Pasquali, Noé. Deep Generative Markov State Models (DeepGenMSM). NeurIPS (2018). arXiv:1805.07601](https://arxiv.org/abs/1805.07601)
[[references#wu2018deepgenmsm]]

DeepGenMSM is a probabilistic framework for learning metastable dynamics from time series by combining (i) a probabilistic encoder mapping configurations to soft metastable-state memberships, (ii) a Markov chain modeling transitions between metastable states, and (iii) a generative decoder that samples the next-step configuration distribution. A key emphasis is that the model can be run **recursively** to generate trajectories and make larger effective time steps than typical MD integration, while still respecting learned long-time kinetics.

**Key methods**: encoder to metastable membership vectors; learned Markov transition matrix in latent space; conditional generative decoder for \(x_{t+	au}\); recursive rollout for trajectory generation.

**Main results**: on MD benchmark systems, DeepGenMSM estimates long-time kinetics accurately and generates plausible configurations, including realistic structures in regions not directly observed during training.

#### [Otto, Rowley. Linearly Recurrent Autoencoder Networks for Learning Dynamics. SIAM Journal on Applied Dynamical Systems 18(1) (2019)](https://epubs.siam.org/doi/10.1137/18M1218189)
[[references#otto2019koopmanae]]

This paper combines a nonlinear autoencoder with **linear recurrent dynamics** in latent space to learn a Koopman-invariant subspace. The model is trained on sequential pairs \((x_t, x_{t+1})\) to jointly learn the nonlinear embedding and the linear transition matrix that governs latent evolution. The emphasis is on stable learning of a globally linear latent model that supports prediction and system identification.

**Key methods**: autoencoder + latent linear recurrence; training on one-step pairs; joint learning of encoder/decoder with linear transition.

**Main results**: shows that linearly recurrent latent models can capture and predict nonlinear dynamics while yielding an interpretable latent linear operator.

#### [Champion et al. Data-Driven Discovery of Coordinates and Governing Equations. PNAS 116(45) (2019)](https://www.pnas.org/doi/10.1073/pnas.1906995116)
[[references#champion2019sindy]]

This paper introduces the **SINDy Autoencoder**, combining a deep autoencoder with Sparse Identification of Nonlinear Dynamics (SINDy) to learn both (i) intrinsic latent coordinates and (ii) parsimonious governing equations in that latent space. Temporal dynamics enter via a SINDy loss that enforces sparse dynamical equations, with time derivatives computed using the chain rule through the encoder. The result is a hybrid model that is both data-driven and equation-discovering.

**Key methods**: autoencoder for coordinate learning; SINDy sparse regression in latent space; derivative matching via chain rule; joint loss combining reconstruction with sparse-dynamics constraints.

**Main results**: discovers compact latent coordinates and sparse governing equations on benchmark systems, yielding interpretable dynamics models rather than black-box forecasts.

#### [Berman, Naiman, Azencot. Multifactor Sequential Disentanglement via Structured Koopman Autoencoders. ICLR (2023). arXiv:2303.17264](https://arxiv.org/abs/2303.17264)
[[references#berman2023koopman]]

This paper proposes structured Koopman autoencoders aimed at **multifactor disentanglement** in sequential data. It uses a spectral/structure-inducing loss to encourage a **block-diagonal** Koopman matrix, so different latent factor groups evolve with minimal coupling. The resulting model separates dynamics into nearly independent components, enabling unsupervised disentanglement driven by linear temporal evolution in latent space.

**Key methods**: Koopman autoencoder; structured (block-diagonal) constraints/regularizers on the Koopman operator; spectral losses to enforce disentangled temporal evolution.

**Main results**: demonstrates improved unsupervised disentanglement of sequential factors and more interpretable latent dynamics compared to unstructured Koopman AE baselines.

#### [Takeishi, Kawahara, Yairi. Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition. NeurIPS (2017)](https://arxiv.org/abs/1710.04340)
[[references#takeishi2017koopman]]

This work introduces neural-network feature learning for approximating the Koopman operator in **Dynamic Mode Decomposition (DMD)**. Instead of applying DMD to fixed observables, it learns observables from time-lagged data pairs so that the resulting linear operator better captures the underlying transition structure. The approach can be seen as a precursor to Koopman autoencoders, emphasizing learned observables optimized for linear evolution.

**Key methods**: neural feature map trained from sequential pairs; Koopman/DMD-style linear operator fitting in learned feature space; objectives aligned with Markov transition modeling.

**Main results**: demonstrates improved Koopman/DMD approximations and forecasting by learning observables rather than using hand-designed features.

#### [Azencot et al. Forecasting Sequential Data Using Consistent Koopman Autoencoders. ICML (2020). arXiv:2004.04080](https://arxiv.org/abs/2004.04080)
[[references#azencot2020ckae]]

This paper proposes **consistent Koopman autoencoders (cKAE)** for forecasting sequential data. In addition to standard KAE reconstruction and latent linear evolution, it enforces a **consistency constraint** requiring that forward and backward Koopman operators compose approximately to the identity. This exploits reversibility-like structure (or approximate invertibility) to stabilize prediction and reduce drift in long rollouts.

**Key methods**: Koopman autoencoder with forward/backward linear operators; cycle/consistency regularization \(K_f K_b pprox I\); multi-step rollout losses.

**Main results**: improves long-horizon stability and forecasting accuracy compared to standard KAEs, especially where compounding error is a concern.

#### [Naiman et al. Generative Modeling of Regular and Irregular Time Series Data via Koopman VAEs. ICLR (2024). arXiv:2310.02619](https://arxiv.org/abs/2310.02619)
[[references#naiman2024koopmanvae]]

Koopman VAEs incorporate Koopman theory into a variational generative model for both regularly and irregularly sampled time series. The model uses a **linear dynamical prior** in latent space represented by a Koopman map, with spectral constraints on eigenvalues to promote stability. The framework aims to unify uncertainty-aware generative modeling with linear latent propagation.

**Key methods**: VAE with Koopman-structured latent prior; spectral constraints for stability; handling irregular sampling through the latent dynamics prior.

**Main results**: supports generative modeling and forecasting for irregular time series with coherent latent dynamics and uncertainty estimates.

#### [Morton et al. Deep Variational Koopman Models. IJCAI (2019). arXiv:1902.09742](https://arxiv.org/abs/1902.09742)
[[references#morton2019dvkm]]

This paper proposes a variational model that infers distributions over Koopman “observations” (latent variables) that can be propagated linearly in time. By making the Koopman embedding probabilistic, the model provides **uncertainty-aware** predictions for dynamical systems while retaining the rollout advantages of linear latent evolution.

**Key methods**: variational inference over Koopman embeddings; linear latent propagation; decoding to observation space; training with multi-step prediction objectives.

**Main results**: produces uncertainty-aware dynamical forecasts and improves robustness compared to deterministic Koopman embeddings in noisy or limited-data regimes.

#### [Nayak et al. Temporally consistent Koopman autoencoders for forecasting dynamical systems. Scientific Reports (2025). arXiv:2403.12335](https://arxiv.org/abs/2403.12335)
[[references#nayak2025tckae]]

This paper introduces **temporally consistent Koopman autoencoders** designed to be robust under limited and noisy training data. The main addition is a temporal-consistency regularization that enforces prediction coherence across different time steps in Koopman latent space, reducing drift and improving generalization in long-horizon rollouts.

**Key methods**: Koopman autoencoder with temporal-consistency regularizer; multi-step prediction alignment across time steps; robustness-focused training under noisy/limited data.

**Main results**: improves long-term forecasting stability and accuracy relative to standard KAEs, particularly under data scarcity and observation noise.
















## Neuroscience

#### [Pandarinath et al. Inferring Single-Trial Neural Population Dynamics Using Sequential Auto-Encoders (LFADS). Nature Methods 15(10) (2018)](https://www.nature.com/articles/s41592-018-0109-9)
[[references#pandarinath2018lfads]]

LFADS is a landmark **variational sequential autoencoder** for neural spike trains, built around a dynamical-systems perspective. An encoder RNN compresses time-binned spike counts into a latent code, and a generator RNN produces low-dimensional temporal factors that model underlying neural population dynamics. The method is widely used for de-noising and inferring latent dynamics in neuroscience.

**Key methods**: sequential VAE with RNN encoder and generator dynamics; latent factors driving spike generation; inference of trial-specific inputs and dynamical trajectories.

**Main results**: recovers smooth latent neural dynamics from noisy single-trial spike trains and improves downstream decoding/analysis compared to conventional baselines.



#### [Franzius, Sprekeler, Wiskott. Slowness and Sparseness Lead to Place, Head-Direction, and Spatial-View Cells. PLoS Computational Biology (2007)](https://www.ini.rub.de/PEOPLE/wiskott/Reprints/FranziusSprekelerEtAl-2007d-PLoSCompBiol-PlaceCells.pdf)
[[references#franzius2007slowsparse]]

This paper combines **slowness** (via Slow Feature Analysis) and **sparseness** (via sparse coding) in a hierarchical model trained on quasi-natural visual input streams to explain the emergence of hippocampal representations. The model first extracts slowly varying features that capture stable underlying variables, then applies sparse coding to produce localized, interpretable cell-like responses. It demonstrates that the joint optimization of temporal slowness and sparsity can yield place cells, head-direction cells, and spatial-view cells, providing a compelling computational account of how navigational representations might self-organize.


**Key methods**: hierarchical Slow Feature Analysis; sparse coding to localize distributed slow variables into cell-like units; analysis of emergent response properties.

**Main results**: emergence of place, head-direction, and spatial-view cell-like units from unsupervised learning; demonstrates sensitivity of learned representations to input statistics and movement patterns.

## Molecular dynamics

#### [Mardt et al. VAMPnets: Deep learning of molecular kinetics. arXiv:1710.06012 (2017)](https://arxiv.org/abs/1710.06012)
[[references#mardt2017vampnets]]

VAMPnets use the **variational approach for Markov processes (VAMP)** to learn an end-to-end mapping from high-dimensional trajectories to low-dimensional Markovian models, effectively learning both a representation and a kinetic model. The paper emphasises interpretable few-state kinetic models and compares learned Koopman/Markov models to traditional pipelines; it includes toy Brownian dynamics and protein folding (NTL9) experiments.

**Key methods**: VAMP-1/VAMP-2 objectives; Koopman operator approximation; end-to-end neural mapping to Markov states; extensive repeated training runs and validation by lag-time consistency checks.

**Main results**: learns compact Markov state decompositions that match or outperform traditional MSM pipelines while yielding more interpretable few-state models; demonstrates meaningful state decompositions and timescale recovery in protein folding.


#### [Wu, Noé. Variational approach for learning Markov processes from time series data (VAMP). arXiv:1707.04659 (2017)](https://arxiv.org/abs/1707.04659)
[[references#wu2017vamp]]

VAMP frames representation learning for dynamical systems as learning features that best approximate the Koopman operator, offering variational score functions that can be estimated from time series. It provides a principled objective for choosing representations that preserve time-evolution structure, and it supports model selection via variational criteria. For temporal interpretability, it supplies theoretical language for “representations that preserve dynamics.”

**Key methods**: Koopman/operator approximation; VAMP-r and related variational scores; cross-validation/model selection via temporal criteria.

**Main results**: provides theoretical guarantees/interpretations and practical scoring procedures for learning dynamical representations.









#### [Chen, Sidky, Ferguson. Nonlinear Discovery of Slow Molecular Modes using State-Free Reversible VAMPnets (SRVs). J. Chemical Physics 150, 214114 (2019)](https://pubmed.ncbi.nlm.nih.gov/31176319/)
[[references#chen2019srv]]

This paper introduces **State-Free Reversible VAMPnets (SRVs)**, a Siamese/twin-network architecture that learns nonlinear approximations to the **leading slow eigenfunctions** of the transfer operator directly from time-lagged data pairs \((x_t, x_{t+	au})\). SRVs are closely related to “time-lagged representation learning” approaches but are trained to satisfy operator-theoretic constraints (reversibility and orthogonality) in a way that targets slow modes explicitly. The method produces differentiable collective variables that can be plugged into enhanced sampling.

**Key methods**: Siamese network trained on time-lagged pairs; reversible/VAMP-style objective to approximate leading transfer-operator eigenfunctions; orthogonality constraints handled within training.

**Main results**: recovers accurate slow-mode coordinates on toy systems (with known eigenfunctions) and on MD benchmarks including alanine dipeptide and the WW-domain protein, yielding parsimonious nonlinear representations of slow dynamics.



#### [Bonati, Piccini, Parrinello. Deep Learning the Slow Modes for Rare Events Sampling (Deep-TICA). PNAS 118(44):e2113533118 (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8612227/)
[[references#bonati2021deeptica]]

This work develops **Deep-TICA**, which learns slow collective variables by combining neural-network feature learning with a time-lagged, transfer-operator-inspired objective. Conceptually, it uses an encoder to map configurations to a low-dimensional space and then optimizes a criterion related to **time-lagged covariance/eigenvalues** (in the spirit of tICA / transfer-operator methods) to extract slow modes. The learned slow coordinates are then integrated into enhanced sampling (notably with on-the-fly probability-enhanced sampling) to accelerate rare events.

**Key methods**: neural encoder for feature compression; time-lagged objective that maximizes slow-mode eigenvalues (tICA/transfer-operator inspired); coupling to enhanced sampling to iteratively improve collective variables.

**Main results**: demonstrates improved rare-event sampling by learning and accelerating slow modes across diverse systems, from small-molecule conformational transitions to miniprotein folding and materials crystallization.

#### [Gupta, Mukherjee. Prediction of good reaction coordinates and future evolution of MD trajectories using Regularized Sparse Autoencoders. arXiv:2208.10962 (2022)](https://arxiv.org/abs/2208.10962)
[[references#gupta2022mdsae]]

This paper models molecular dynamics trajectories as multivariate time series and uses a **regularized sparse autoencoder** to (i) discover a small set of important reaction-coordinate-like latent variables and (ii) predict future trajectory evolution. The sparsity regularization is framed as selecting a concise set of salient coordinates, aligning with sparse-coding intuitions, while the forecasting component tests whether the learned latents retain dynamics-relevant information.

**Key methods**: sparse autoencoder with regularization to select a small latent set; multi-step time series prediction using learned latents; evaluation of reaction-coordinate quality via ability to separate states and forecast dynamics.

**Main results**: reports that sparsity regularization helps identify compact, informative reaction coordinates and supports multi-step prediction of MD trajectory evolution on benchmark systems (including alanine dipeptide and a proflavine–DNA intercalation setting).





