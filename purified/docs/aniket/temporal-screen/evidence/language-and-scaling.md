# Language statistics and scaling evidence

Full-text evidence cards for the first six papers in the core packet and every
paper in Direction 1 of the annotated reading list. Duplicate appearances were
merged, leaving **14 unique primary sources**. All 14 full texts were accessed;
there were no inaccessible sources.

The audit keeps three claims separate:

1. a sequence has statistically detectable temporal dependence;
2. the dependence is useful for a declared downstream target beyond an anchor
   token or an order-invariant summary;
3. a TXC can exploit that dependence better than matched alternatives.

Most papers below support only the first claim. The predictive
\(\mathcal V\)-information papers provide the cleanest machinery for the
second. None alone establishes the third.

## Cagnetta, Raventós, Ganguli, and Wyart (2026), *Deriving Neural Scaling Laws from the Statistics of Natural Language*

- **Primary source:** [arXiv:2602.07488](https://arxiv.org/abs/2602.07488),
  full text, version 3.
- **Scientific question and observable:** The paper asks whether
  data-limited next-token scaling can be predicted from two corpus statistics:
  the decay of finite-context conditional entropy \(H_n-H_\infty\) and the
  operator norm of a lagged one-hot token covariance matrix
  \(\|C(n)\|_{\mathrm{op}}\).
- **Temporal object, estimand, estimator, and null:** The temporal object is a
  token pair separated by lag \(n\). The proposed signal is the strongest
  singular mode of the empirical lagged covariance. The null is the
  \(O(P^{-1/2})\) finite-sample fluctuation expected from \(P\) independent
  examples; its intersection with a fitted \(n^{-\beta}\) decay defines a
  resolvable horizon \(n^\star(P)\).
- **Assumptions:** The derivation assumes a fixed vocabulary,
  \(\|C(n)\|_{\mathrm{op}}\sim n^{-\beta}\),
  \(H_n-H_\infty\sim n^{-\gamma}\), and a learner that rapidly acquires every
  dependency inside the resolvable horizon. The statistic is linear,
  stationary, unconditional, and direction-insensitive in magnitude because
  time reversal transposes \(C(n)\) without changing its singular values.
- **Target alignment:** It is aligned to next-token prediction only. It is not
  conditioned on a backtracking, misalignment, or other local task label.
- **Dependence units, splits, and uncertainty:** Appendix B treats lagged
  example pairs as independent when invoking entrywise central-limit and
  matrix-Bernstein arguments. The empirical pairs overlap and share
  documents, so those units are not independent for an LM activation screen.
  Appendix C bootstraps fitted lag points and varies fit windows; it does not
  perform a grouped document bootstrap. WikiText's central \(\gamma\) fit uses
  only three context lengths and its \(\beta\) fit uses six lags up to 32.
- **Full-text result:** Equations (3)--(9) in Section 3 derive
  \(n^\star(P)\sim P^{1/(2\beta)}\) and
  \(L_{\mathrm{AR}}(P)-H_\infty\sim P^{-\gamma/(2\beta)}\). Section 4.1 and
  Figure 3 report \(\beta=0.88\pm0.06\) on TinyStories and
  \(0.94\pm0.16\) on WikiText; the latter uses a short-lag fit because of a
  broken power law and a local peak near lag 10. Figures 1 and 4 show the
  predicted scaling collapse across several architectures. Section 5 and
  Figure 6 show that within-horizon learning can decay faster than the
  horizon-limited asymptote, confirming that the fast-learning assumption is
  architecture-dependent.
- **Author-stated limitations and additional failure modes:** The limitations
  section on page 9 says the largest empirically resolved horizons are only
  tens of tokens and explicitly questions extrapolation to trillion-token
  training and \(10^5\)-token contexts. A corpus-wide topic process can create
  the same operator-norm tail without making order useful for a local label.
- **Screen decision this evidence can support:** Use a grouped,
  finite-sample resolvability curve as a prior on which lags are worth testing,
  and distinguish “signal exists above noise” from “the learner exploited it.”
- **What it cannot support:** It cannot certify a task's ordered horizon,
  distinguish forward from reversed history, or predict that a TXC will beat
  an SAE.
- **Direct-method adaptation:** Inputs would be grouped activation histories
  and labels; outputs would include a target-conditioned lag curve, a
  group-bootstrap noise floor, and \(n^\star\). Fit windows and lag grids must
  be preregistered, with documents or rollouts—not token pairs—as resampling
  units. A decisive synthetic falsification is a label generated from the
  anchor token while an independent latent-topic process gives activations a
  matched power-law covariance: the generic horizon should be long, but the
  target-aligned ordered gain should be zero.
- **Exact evidence pointers:** Section 2 and Equations (1)--(2), pages 2--3;
  Section 3 and Equations (3)--(9), pages 3--5; Section 4.1 and Figures 2--3,
  pages 5--7; Section 5 and Figure 6, page 9; Appendices B--C, pages 19--21.

## O'Connor and Andreas (2021), *What Context Features Can Transformer Language Models Use?*

- **Primary source:** [ACL 2021, long paper 70](https://aclanthology.org/2021.acl-long.70/),
  full conference paper.
- **Scientific question and observable:** The paper asks which lexical and
  ordering properties of mid- and long-range context remain usable to a
  language model after controlled ablations.
- **Temporal object, estimand, estimator, and null:** The temporal object is a
  word-level context. The estimand is normalized ablated predictive
  \(\mathcal V\)-information: the held-out log-likelihood contribution lost
  when a predictor is retrained on an ablated context, normalized between
  full-context and no-context predictors. Separate matched models are trained
  for each ablation. The null is the no-information padded context.
- **Assumptions:** The conclusion is relative to the declared GPT-2 predictor
  family and the train-time ablation distribution. No stationarity theorem is
  required, but the experiment pools token targets and interprets random
  permutations as removing particular structural information.
- **Target alignment:** It is directly aligned to next-token prediction and
  therefore supplies a method template, not evidence about TXC probe labels.
- **Dependence units, splits, and uncertainty:** Models are trained from
  scratch on 103M WikiText-103 words and evaluated on 217,646 held-out words.
  Each condition is averaged over two random initializations. The paper does
  not report document-clustered confidence intervals.
- **Full-text result:** Section 3.1 and Figure 2 report that shuffling every
  word removes 41% of mid-range and 84% of long-range usable information;
  global trigram shuffling removes 31% and 50%; sentence shuffling removes
  17% and 14%; and within-sentence shuffling removes 26% and 55%. Section 3.2
  and Figure 3 find that content words carry much of the usable signal.
  Section 3.3 and Figure 4 show that fixed-model test-time ablations give
  materially different answers from retrained ablations because the former
  are out of distribution.
- **Author-stated limitations and additional failure modes:** Section 5 notes
  roughly 100 training runs and leaves open whether these findings generalize
  beyond the chosen LM and context windows. Some ablations improve held-out
  likelihood, producing negative “removed information,” which the authors
  attribute to finite-model approximation or overfitting.
- **Screen decision this evidence can support:** The screen should compare
  separately refit anchor-only, order-invariant, ordered, local-shuffle, and
  global-shuffle predictors. One omnibus shuffle does not isolate what kind
  of temporal information is useful.
- **What it cannot support:** The reported percentages do not transfer to
  activation classification, and a fixed-probe shuffle cannot be interpreted
  as the same estimand as a refit ablation.
- **Direct-method adaptation:** Hold probe family, training budget, label
  cohort, and held-out groups fixed while retraining on each representation.
  Return paired held-out log-loss gains and group-bootstrap intervals. A
  synthetic bag-generated target should make the refit ordered and bag models
  tie even if a fixed ordered model breaks under shuffling.
- **Exact evidence pointers:** Definition 1 and Equations (6)--(10), Sections
  2.1--2.4, pages 851--854; Figures 2--4 and Sections 3.1--3.3, pages 854--857;
  Discussion, page 858; ablation details in Appendices A--B, pages 859--864.

## Xu, Zhao, Song, Stewart, and Ermon (2020), *A Theory of Usable Information Under Computational Constraints*

- **Primary source:** [arXiv:2002.10689](https://arxiv.org/abs/2002.10689),
  full text.
- **Scientific question and observable:** The paper asks how to define the
  information in \(X\) about \(Y\) that a restricted predictive family
  \(\mathcal V\) can actually use.
- **Temporal object, estimand, estimator, and null:** The framework is not
  intrinsically temporal. It defines
  \(I_{\mathcal V}(X\!\to\!Y)=H_{\mathcal V}(Y)-
  H_{\mathcal V}(Y\mid X)\), estimated as the reduction in held-out
  log-loss between an ignorance predictor and a predictor using \(X\). The
  null is the best input-ignoring member of the same optional-ignorance
  family.
- **Assumptions:** The predictive family must satisfy optional ignorance for
  nonnegativity and independence properties. The quantity is directed and
  family-relative. Unlike Shannon mutual information, it need not obey data
  processing: a transformation can make information easier for
  \(\mathcal V\) to use.
- **Target alignment:** Alignment is exact when \(Y\) is the downstream label
  and \(X\) is the representation being screened.
- **Dependence units, splits, and uncertainty:** Definition 4 is a held-out
  empirical risk estimator. Theorem 1 gives a PAC bound using Rademacher
  complexity and bounded log-loss for independent samples; grouped dependent
  sequences require group-level splitting or a different bound. The paper
  does not prescribe bootstrap intervals.
- **Full-text result:** Definitions 1--3 in Section 2 formalize predictive
  families, predictive entropy, and \(\mathcal V\)-information. Proposition 1
  recovers Shannon mutual information for an unrestricted family and a scaled
  linear-Gaussian \(R^2\)-like quantity for a linear family. Proposition 2
  gives nonnegativity under optional ignorance. Sections 3.2--3.3 establish
  failure of data processing and asymmetry. Definition 4 and Theorem 1 in
  Section 4 provide the empirical estimator and generalization bound.
- **Author-stated limitations and additional failure modes:** Appendix F
  notes that general \(\mathcal V\)-information lacks Shannon's additive chain
  rule. A probe that is too weak gives false negatives; one that is too rich
  can overfit or smuggle in a task-specific architecture.
- **Screen decision this evidence can support:** Declare a small,
  architecture-external observer family and measure ordered history's
  cross-fitted log-loss contribution to the actual target.
- **What it cannot support:** The resulting value is not probe-independent
  Shannon information, and scores from unrelated predictive families are not
  directly comparable.
- **Direct-method adaptation:** Use the same ridge and shallow-MLP families for
  anchor, bag, and ordered representations; nest optional ignorance; split by
  rollout or document; report both families and paired intervals. An XOR
  synthetic target is the essential capacity falsifier: a linear observer
  should report no usable information while a sufficiently small nonlinear
  observer recovers it.
- **Exact evidence pointers:** Definitions 1--3 and Propositions 1--2,
  Section 2, pages 2--4; data-processing and directional properties,
  Sections 3.2--3.3, pages 5--6; Definition 4 and Theorem 1, Section 4, pages
  6--8; limitations in Appendix F.

## Hewitt, Ethayarajh, Liang, and Manning (2021), *Conditional Probing: Measuring Usable Information Beyond a Baseline*

- **Primary source:** [EMNLP 2021, main paper 122](https://aclanthology.org/2021.emnlp-main.122/),
  full conference paper.
- **Scientific question and observable:** The paper asks how much a
  representation makes predictable beyond information already supplied by a
  baseline representation.
- **Temporal object, estimand, estimator, and null:** The method is general,
  not intrinsically temporal. Conditional probing compares predictors using
  \([B;\phi(X)]\) and \([B;0]\), so the baseline \(B\) is present in both
  conditions. The estimand is conditional multivariable
  \(\mathcal V\)-information,
  \(H_{\mathcal V}(Y\mid B)-H_{\mathcal V}(Y\mid B,\phi(X))\).
- **Assumptions:** The theoretical family satisfies a multivariable optional
  ignorance condition. Empirical conclusions remain relative to the chosen
  affine-softmax probe and representation. The setup is directed toward
  \(Y\), but does not itself establish causal use.
- **Target alignment:** It is exactly the desired target alignment if
  \(Y\) is the local label, \(B\) is the anchor plus an order-invariant
  summary, and \(\phi(X)\) is the ordered residual history.
- **Dependence units, splits, and uncertainty:** The experiments use standard
  task train/dev splits across five NLP tasks with RoBERTa and ELMo layers.
  They do not center grouped sequence dependence or uncertainty estimation,
  so those are required additions for rollout data.
- **Full-text result:** Equation (2) in Section 2.3 defines the practical
  concatenated conditional probe. Section 2.6 gives the information
  interpretation. Section 4.1, Figure 1, and Table 1 show that conditioning on
  input-layer information changes across-layer conclusions, demonstrating
  that standalone probe accuracy can credit redundant information to a
  representation.
- **Author-stated limitations and additional failure modes:** The authors do
  not claim one universally correct predictive family and explicitly treat
  probe complexity as a theory-external design choice. If the augmented model
  does not contain the baseline model as a submodel, optimization and capacity
  differences can masquerade as negative or positive conditional information.
- **Screen decision this evidence can support:** Test whether ordered history
  contributes predictive information after the anchor and bag-of-history
  baseline have already been exposed to the same observer.
- **What it cannot support:** A positive conditional probe gap is not evidence
  of causal model use, nor does it imply that TXC will learn the corresponding
  feature.
- **Direct-method adaptation:** Use exact nested features and shared
  regularization grids, cross-fit by sequence, and retain a zero-vector path
  so optional ignorance is realizable. A duplicate-feature synthetic case
  should yield zero conditional gain even though the added feature alone is
  predictive.
- **Exact evidence pointers:** Equation (2), Section 2.3, pages 1628--1629;
  conditional \(\mathcal V\)-information, Section 2.6, page 1630; Figure 1
  and Table 1, Section 4.1, pages 1632--1634; multivariable definitions and
  nonnegativity, Appendix A, pages 1637--1639.

## Jiao, Courtade, Venkat, and Weissman (2015), *Justification of Logarithmic Loss via the Benefit of Side Information*

- **Primary source:** [arXiv:1403.4679](https://arxiv.org/abs/1403.4679),
  full text.
- **Scientific question and observable:** The paper asks which loss makes the
  optimal reduction in prediction risk from side information obey a data
  processing axiom.
- **Temporal object, estimand, estimator, and null:** There is no intrinsic
  temporal object. Equation (2) defines the population benefit of side
  information \(Y\) as optimal Bayes risk without \(Y\) minus optimal risk
  with \(Y\). Under log-loss, common side information yields conditional
  mutual information.
- **Assumptions:** The principal uniqueness theorem assumes a finite target
  alphabet of size at least three and Bayes-optimal prediction over all
  distributions. It is a population result, not a guarantee for a learned
  finite probe.
- **Target alignment:** It is target-aligned by construction because the
  benefit is defined for the predicted variable.
- **Dependence units, splits, and uncertainty:** The paper is theoretical; it
  has no empirical split or uncertainty protocol. Applying it to dependent
  rollouts therefore requires cross-fitting and group-aware uncertainty.
- **Full-text result:** Theorem 1 shows that for output alphabet size
  \(n\ge3\), the data processing axiom uniquely gives mutual information up
  to a positive scale. Corollary 1 extends the result to conditional mutual
  information with common side information. Section III connects sequential
  conditional log-loss improvements to directed information and transfer
  entropy for stationary finite-alphabet processes.
- **Author-stated limitations and additional failure modes:** Theorem 2 shows
  that binary targets do **not** uniquely select log-loss; any symmetric
  convex generator in the stated class can satisfy the axiom. This matters
  because many proposed Temporal Screen labels are binary.
- **Screen decision this evidence can support:** Prefer paired held-out
  log-loss reduction as a proper measure of predictive side-information
  value and, at the unrestricted population optimum, an information-theoretic
  quantity.
- **What it cannot support:** A finite probe's gain is not exact mutual
  information, AP and accuracy do not inherit this identity, and the
  uniqueness claim cannot be invoked for binary labels.
- **Direct-method adaptation:** Report raw paired test log-loss differences
  with group-bootstrap intervals rather than relabeling them as exact bits.
  A calibrated finite-alphabet synthetic process with analytic conditional
  mutual information should show convergence as data and probe capacity
  increase.
- **Exact evidence pointers:** Benefit of side information and data-processing
  axiom, Equations (2)--(4), pages 2--3; Theorem 1 and Corollary 1, pages
  4--6; binary exception, Theorem 2, pages 6--8; sequential applications,
  Section III, pages 9--12.

## Altmann, Cristadoro, and Degli Esposti (2012), *On the Origin of Long-Range Correlations in Texts*

- **Primary source:** [arXiv:1207.0658](https://arxiv.org/abs/1207.0658),
  full PNAS paper and supplement.
- **Scientific question and observable:** The paper asks which hierarchical
  linguistic levels generate long-range correlations observed in lower-level
  character and word observables.
- **Temporal object, estimand, estimator, and null:** A text becomes a binary
  time series for a chosen local observable. Equation (1) defines lagged
  autocorrelation; Equation (3) estimates long-range dependence through the
  anomalous growth exponent of an integrated random walk. Null A1 shuffles
  all bits, while A2 shuffles inter-event intervals, preserving their
  marginal distribution but destroying interval-to-interval correlation.
- **Assumptions:** The formal setup assumes a stationary ergodic source and a
  carefully chosen observable. It is a symmetric two-point statistic, linear
  in the indicator process, and not aligned to a downstream label.
- **Target alignment:** None; observables are vowels, letters, words, or
  topical categories rather than task labels.
- **Dependence units, splits, and uncertainty:** The empirical corpus is ten
  Gutenberg novels with 41 binary observables per book. Reported variation is
  across observables and books, not held-out prediction or group-bootstrap
  uncertainty.
- **Full-text result:** Section I, Equations (1)--(4), separates power-law
  autocorrelation from broad inter-event distributions. Section II and
  Figure 3 introduce M1, which randomly relocates word tokens while
  preserving within-word structure, and M2, which consistently recodes word
  identities while preserving token positions. M1 reduces the average
  letter-level exponent from \(1.26\pm0.11\) to \(1.07\pm0.08\), whereas M2
  retains \(1.20\pm0.09\), supporting a cascade from higher semantic levels.
- **Author-stated limitations and additional failure modes:** Section III
  says the observable and finite fitting range are decisive, notes that
  literary text may be nonstationary, and warns that two-point functions
  cannot capture all higher-order or entropy structure. Topic recurrence and
  rare-word burstiness can create low-frequency tails without useful local
  order.
- **Screen decision this evidence can support:** Require a hierarchy of nulls
  that separately preserves marginals, burst distributions, bags, and
  within-segment structure. Stratify by document and task phase instead of
  fitting one global slope.
- **What it cannot support:** It cannot determine a task's useful context
  horizon, infer directionality, or show that a temporal representation is
  better than an order-invariant one.
- **Direct-method adaptation:** Build event indicators from labels or
  preregistered activation projections; calculate within-group curves; compare
  bit shuffle, interval shuffle, within-window permutation, and group-block
  shuffle. A latent-topic block process with labels drawn independently is the
  essential false-positive test.
- **Exact evidence pointers:** Equations (1)--(4) and hierarchy argument,
  Section I, pages 1--2; nulls A1/A2, Section II and Figure 2, pages 2--3;
  M1/M2 and Figure 3, pages 3--4; limitations, Section III, page 4;
  word-shuffle analysis, Supplementary Section VII.

## Cagnetta and Wyart (2024), *Towards a Theory of How the Structure of Language Is Acquired by Deep Neural Networks*

- **Primary source:** [arXiv:2406.00048](https://arxiv.org/abs/2406.00048),
  full text, version 3.
- **Scientific question and observable:** The paper asks how a hierarchical
  generative process produces correlation plateaus, sample-complexity
  transitions, and context-length scaling during next-token learning.
- **Temporal object, estimand, estimator, and null:** The main object is a
  token at distance \(t\) from a masked final token in a random hierarchy
  model. Equations (5)--(9) use an RMS token co-occurrence covariance and a
  sampling-noise scale of order \((v^2P)^{-1/2}\); their intersection defines
  the effective context \(t^\star(P)\).
- **Assumptions:** The theory uses a fixed-length, uniform, unambiguous
  probabilistic context-free hierarchy and a stationary corpus ensemble.
  Dependence is an unconditional two-token statistic and has no directional
  task label.
- **Target alignment:** The synthetic analysis is aligned to its masked-token
  target. The real-text statistic remains corpus-global and next-character
  oriented.
- **Dependence units, splits, and uncertainty:** The theory treats sampled
  sentences as independent. Synthetic neural results average finite runs;
  the real-text experiments fit Tiny Shakespeare and Wikipedia character
  statistics and do not provide document-clustered uncertainty.
- **Full-text result:** Section 3 and Figure 1 show hierarchical covariance
  plateaus approximated by a power law and a finite-data resolution horizon.
  Section 4, Equations (10)--(13), and Figure 2 connect successively learned
  hidden levels to step-like loss improvements. Section 5, Equation (16), and
  Figure 4 report an approximate real-text correlation exponent
  \(\beta\approx1.4\) and a context-dependent scaling collapse.
- **Author-stated limitations and additional failure modes:** Section 6 says
  the context-free, fixed-tree, uniform, and unambiguous grammar is
  unrealistic; it does not prove the stated sample complexity for
  gradient-trained networks, and architecture-dependent discrepancies remain.
  The same lag covariance can arise from nonhierarchical topic mixtures.
- **Screen decision this evidence can support:** Use synthetic hierarchies to
  calibrate whether a proposed estimator recovers a known finite-data horizon
  and known sample-complexity transitions.
- **What it cannot support:** It does not make unconditional correlation a
  valid screen for arbitrary labels or select TXC over competing architectures.
- **Direct-method adaptation:** Replace the masked-token covariance with
  target-conditioned activation/label statistics, estimate noise by grouped
  resampling, and compare recovered horizons with known synthetic latent
  levels. A same-covariance construction whose label depends only on the bag
  must fail the ordered screen.
- **Exact evidence pointers:** Model assumptions, Section 2 and Section 2.1,
  pages 2--4; correlation and resolution threshold, Equations (5)--(9),
  Section 3 and Figure 1, pages 4--6; learning transitions, Section 4 and
  Figure 2, pages 6--8; real-text scaling, Section 5 and Figure 4, pages
  8--10; limitations, Section 6, page 10.

## Belletti, Chen, and Chi (2019), *Quantifying Long Range Dependence in Language and User Behavior to Improve RNNs*

- **Primary source:** [arXiv:1905.09414](https://arxiv.org/abs/1905.09414),
  full KDD paper.
- **Scientific question and observable:** The paper asks whether a scalable
  spectral estimate of long-range dependence (LRD) in embedded symbolic
  sequences can guide the distribution of recurrent-model capacity over
  temporal distances.
- **Temporal object, estimand, estimator, and null:** The temporal object is a
  sequence of learned symbol embeddings. Definition 2 parameterizes
  univariate LRD by
  \(\gamma(h)=L(h)h^{2d-1}\), equivalently a low-frequency spectrum
  \(f(\lambda)=L(\lambda)\lambda^{-2d}\). Algorithm 1 estimates each embedding
  dimension's \(d\) by removing the DC component and applying OLS to the
  log-periodogram at low Fourier frequencies. Random word shuffling is the
  empirical null.
- **Assumptions:** The estimator assumes second-order stationarity and detects
  only linear dependence in the chosen embedding coordinates. The Gaussian
  argument connecting covariance decay to mutual information is presented
  only for a simple case. Frequency magnitude is direction- and
  phase-insensitive.
- **Target alignment:** The statistic is unconditional and embedding-relative;
  it is not aligned to next-token errors or another downstream label.
- **Dependence units, splits, and uncertainty:** The language experiment uses
  100 MB of Wikipedia, split into 2,048-word sequences, represented with
  300-dimensional GloVe embeddings trained on 2014 Wikipedia. Figure 3
  reports OLS slopes and nominal p-values per dimension. The paper acknowledges
  that the residual-distribution assumptions behind those p-values are
  violated and does not provide a document bootstrap.
- **Full-text result:** Figure 3 shows positive estimated LRD slopes across
  the real Wikipedia embedding dimensions and near-zero slopes after random
  word shuffling. Sections 4--5 map the fitted power or exponential decay to
  an EvoRNN compute schedule. Figure 7 reports comparable or better language
  modeling perplexity with fewer multiply-adds than the authors' baseline.
- **Author-stated limitations and additional failure modes:** Section 3.5 says
  the estimator cannot detect nonlinear or nonstationary LRD. The user
  sequences have irregular timestamps that the estimator ignores. The
  architecture comparison changes the capacity schedule and compute budget,
  so it does not isolate the causal effect of matching a measured exponent.
  Lexical semantics, topic blocks, and the DC neighborhood can dominate the
  low-frequency slope.
- **Screen decision this evidence can support:** A log-periodogram is a cheap
  unconditional diagnostic and a candidate prior over relevant scales, but it
  should be gated by target-conditioned prediction and phase/order controls.
- **What it cannot support:** It cannot determine whether a label needs order,
  whether the useful direction is past-to-future, or whether spectral matching
  explains the EvoRNN gain.
- **Direct-method adaptation:** Input grouped activation sequences; remove or
  report DC separately; estimate spectra per group with a preregistered
  frequency band and a less biased low-frequency method such as multitaper;
  aggregate through group bootstraps. Return exponent stability and a
  target-conditioned residual spectrum. A topic-block process with labels
  independent of topic should retain a strong low-frequency slope while
  failing the target-aligned screen.
- **Exact evidence pointers:** Definitions 2--3 and frequency-domain
  equivalence, Sections 2--3, pages 2--3; Gaussian information discussion and
  Algorithm 1, Sections 3.3--3.4, pages 3--4; Wikipedia setup, caveats, and
  Figure 3, Section 3.5, pages 4--5; architecture mapping, Sections 4--5 and
  Figure 7, pages 5--8.

## Khandelwal, He, Qi, and Jurafsky (2018), *Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context*

- **Primary source:** [ACL 2018, long paper P18-1027](https://aclanthology.org/P18-1027/),
  full conference paper.
- **Scientific question and observable:** The paper asks how much context an
  LSTM LM uses and whether nearby and distant context contribute through
  word order, exact identity, or rough semantic content.
- **Temporal object, estimand, estimator, and null:** At test time the authors
  truncate, shuffle, reverse, drop, or replace portions of a 300-token prefix
  and measure the target word's loss. The implicit null is the model's loss
  after a particular perturbation relative to the intact prefix.
- **Assumptions:** The model is fixed and was trained only on intact order, so
  perturbation loss combines information removal with distribution shift.
  Conclusions are model- and target-subgroup-specific, not stationary process
  identities.
- **Target alignment:** The loss is aligned to next-token prediction, but not
  to the local labels proposed for TXC evaluation.
- **Dependence units, splits, and uncertainty:** A standard AWD-LSTM is
  evaluated on Penn Treebank and WikiText-2 development data with three model
  seeds; the paper reports standard deviations and says test trends agree.
  It does not cluster intervals by document.
- **Full-text result:** Section 4 and Figure 1 place the 1%-perplexity
  effective context near 150 tokens on Penn Treebank and 250 on WikiText-2,
  with infrequent words benefiting beyond 200 tokens but determiners largely
  saturating near 10. Section 5, Equation (2), and Figure 2 show that order
  matters mainly in the most recent 20--50 tokens; beyond roughly 50,
  shuffling and reversal have little effect while replacing words still
  hurts. Section 6 attributes part of distant benefit to copying and compares
  an external cache.
- **Author-stated limitations and additional failure modes:** Section 3
  explicitly describes test-time ablation losses as upper bounds because
  models retrained on the perturbations could adapt. Section 7 says results
  are tightly coupled to the model, vocabulary, and dataset. Aggregate curves
  hide large target-class heterogeneity.
- **Screen decision this evidence can support:** Report separate curves for
  total history, exact/order-invariant content, ordered local history, and
  copy-positive subgroups.
- **What it cannot support:** A fixed-model shuffle cost cannot establish
  retrained usable information or an architecture-independent temporal
  horizon.
- **Direct-method adaptation:** Pair fixed-probe stress tests with separately
  refit nested probes on anchor, bag, and ordered histories; preserve document
  groups; preregister target strata. A bag-generated label with a canonical
  train order should produce an apparent fixed-model shuffle penalty but no
  refit ordered advantage.
- **Exact evidence pointers:** Test-time intervention caveat and setup,
  Section 3, pages 286--287; truncation and target heterogeneity, Section 4
  and Figure 1, pages 287--289; shuffle/reversal/replacement results, Section
  5, Equation (2), and Figure 2, pages 289--291; cache analysis, Section 6,
  pages 291--293; limitations, Section 7, page 293.

## Sun et al. (2021), *Do Long-Range Language Models Actually Use Long-Range Context?*

- **Primary source:** [EMNLP 2021, main paper 62](https://aclanthology.org/2021.emnlp-main.62/),
  full conference paper.
- **Scientific question and observable:** The paper asks which target tokens
  actually benefit from thousands of tokens of context and whether a
  long-range Transformer's aggregate perplexity gain reflects sophisticated
  use or simpler repetition and topic mechanisms.
- **Temporal object, estimand, estimator, and null:** For each target chunk,
  the authors truncate or perturb parts of an 8,000-token prefix and compare
  token loss. Perturbations include shuffling, replacing the prefix with a
  sequence from another book, and dropping selected tokens. Five random
  perturbations are averaged per condition.
- **Assumptions:** These are fixed-checkpoint, test-time interventions. They
  assume the perturbations selectively remove order, identity, or lexical
  evidence, but distribution shift remains possible. Aggregate token averages
  need not represent all events.
- **Target alignment:** It is aligned to next-token prediction and explicitly
  stratified by token mechanisms, but not aligned to TXC task labels.
- **Dependence units, splits, and uncertainty:** The study uses 49 PG-19
  validation books after removing one artifact, approximately 220,000 sampled
  target tokens, 10-token target chunks, and 8K prefixes. The public
  Transformer checkpoint is not retrained. Results are averaged across
  targets and perturbations rather than reported with book-clustered
  intervals.
- **Full-text result:** Section 3 and Figure 1 show little aggregate
  perplexity gain beyond 2K tokens, while infrequent, subword, repeated, and
  continuous-fiction targets benefit farther. Section 4, Figures 7 and 9,
  finds random replacement more harmful than shuffling and sometimes finds
  shuffling slightly better than intact distant context, suggesting weak use
  of distant order in aggregate. Figures 10--12 reveal greater
  identity/order sensitivity for copy and subword subsets. Section 5 and
  Figure 13 find no improvement beyond 2K on suffix identification and
  sequence copying.
- **Author-stated limitations and additional failure modes:** The conclusions
  concern the tested Transformer and PG-19 subsets; the authors characterize
  much of the long-range gain as “superficial” copying but do not rule out
  mechanisms absent from their probes. Large non-temporal strata can wash out
  a real minority temporal effect.
- **Screen decision this evidence can support:** Require both group-average
  and event/subgroup screen results, including an explicit copy/lexical
  shortcut audit.
- **What it cannot support:** Aggregate saturation at 2K does not define a
  universal language horizon, and fixed-checkpoint perturbations do not
  measure the gain a refit observer could extract.
- **Direct-method adaptation:** Split and bootstrap by rollout or document,
  report preregistered event strata, and pair order interventions with
  identity-preserving and cross-group replacement controls. A synthetic
  mixture containing a rare copy-positive temporal minority and a large
  anchor-only majority should be accepted only if the screen reports both
  rather than averaging the minority away.
- **Exact evidence pointers:** PG-19 cohort and checkpoint protocol, Section
  2.2, pages 808--809; context-length and subgroup curves, Section 3 and
  Figure 1, pages 809--812; perturbation definitions and Figures 7, 9--12,
  Section 4, pages 812--818; explicit retrieval tests, Section 5 and Figure
  13, pages 818--820; conclusion, Section 7, page 821.

## Lin and Tegmark (2017), *Critical Behavior in Physics and Probabilistic Formal Languages*

- **Primary source:** [Entropy 19(7):299](https://www.mdpi.com/1099-4300/19/7/299),
  full journal article.
- **Scientific question and observable:** The paper asks which classes of
  stochastic grammars can generate power-law mutual information between
  symbols separated by distance \(t\).
- **Temporal object, estimand, estimator, and null:** The theoretical
  observable is pairwise mutual information \(I(X_0;X_t)\). Finite-state
  Markov and hidden-Markov generators are the comparison class; recursive
  probabilistic context-free constructions provide an existence example.
  The empirical estimator uses finite-sample entropy/MI corrections described
  in Appendix D.
- **Assumptions:** The exponential-decay theorems require irreducibility,
  aperiodicity, and finite state. Pairwise MI is symmetric and
  reversal-invariant, so it contains no temporal direction. Power-law decay is
  an existence result for some recursive grammars, not a characterization of
  all hierarchy.
- **Target alignment:** None; the statistic is corpus-global and untargeted.
- **Dependence units, splits, and uncertainty:** Figure 1 analyzes character
  sequences from text, DNA, code-like Wikipedia, and music. Appendix D
  discusses finite-sample entropy estimators and approximate variance, but
  the empirical curves are not grouped by document and corpus artifacts are
  visible.
- **Full-text result:** Theorem 1 proves asymptotically exponential MI decay
  for irreducible aperiodic finite Markov chains; Theorem 2 extends this to
  finite-state hidden Markov models. Theorem 3 constructs probabilistic
  context-free grammars with power-law MI because tree distance grows
  logarithmically with surface distance. Figure 3 shows LSTM-generated text
  approximately reproducing a power law but with less long-range MI than its
  training text.
- **Author-stated limitations and additional failure modes:** Figure 1 notes
  that Wikipedia's long tail is influenced by markup-like syntax and a French
  corpus by concatenated poems. Reducible or periodic finite-state processes
  can plateau, and finite data make MI estimation delicate. A power law
  therefore does not uniquely diagnose hierarchical semantics.
- **Screen decision this evidence can support:** Treat nonexponential generic
  dependence as evidence against a simple mixing finite-state null and as a
  reason to include hierarchical synthetic tests.
- **What it cannot support:** It cannot show that history helps a declared
  label, that order rather than a bag matters, or that the relevant
  architecture should be a TXC.
- **Direct-method adaptation:** Use pairwise MI only as a background
  diagnostic and demand agreement with target-conditioned/refit tests. A
  reversal test is an immediate falsifier of directional overinterpretation:
  reversed data have the same pairwise MI curve. Topic-block and repeated
  markup processes are false-positive controls.
- **Exact evidence pointers:** Corpus curves and artifact discussion, Figure
  1, page 2; finite Markov result, Theorem 1, pages 4--6; hidden-Markov
  result, Theorem 2, pages 6--7; recursive-grammar existence result, Theorem
  3, page 9; LSTM-generation comparison, Figure 3, page 12; estimator details,
  Appendix D, page 24.

## Bialek, Nemenman, and Tishby (2001), *Predictability, Complexity, and Learning*

- **Primary source:** [arXiv:physics/0007070](https://arxiv.org/abs/physics/0007070),
  full text, version 3.
- **Scientific question and observable:** The paper asks whether the mutual
  information between an observed past and a future can quantify the
  predictable structure and learning complexity of a stationary process.
- **Temporal object, estimand, estimator, and null:** The temporal object is a
  contiguous past block \(x_{\mathrm{past}}\) and future block
  \(x_{\mathrm{future}}\). Equations (4)--(9) define predictive information
  as their mutual information and show that, for a stationary source, it is
  the subextensive entropy term \(S_1(T)\). An extensive entropy-rate process
  with only finite excess entropy is the finite-predictability comparison.
- **Assumptions:** The core entropy identity assumes time-translation
  invariance. Mutual information is symmetric between prediction and
  postdiction and is neither directional nor target-conditioned. Later
  asymptotic results assume Bayesian model classes and controllable
  fluctuations.
- **Target alignment:** None; “future” is the whole future stream rather than
  a declared task label.
- **Dependence units, splits, and uncertainty:** This is a population theory
  paper with illustrative synthetic processes, not an empirical estimator
  study. It provides no finite-corpus split or confidence-interval protocol.
- **Full-text result:** Section 3 derives
  \(I_{\mathrm{pred}}(T)=S_1(T)\) and the learning-curve relation
  \(\Lambda(N)\approx \partial I_{\mathrm{pred}}/\partial N\). Sections
  4.1--4.3 show logarithmic growth
  \(S_1(N)\sim(K/2)\log_2N\) for finite-parameter learning under regularity
  conditions. Equation (69) makes an especially important nonidentifiability
  point: subextensive entropy does not distinguish intrinsic long-range
  interaction from a global unknown parameter. Sections 4.5--4.6 derive
  power-law predictive-information growth for a regularized nonparametric
  density example.
- **Author-stated limitations and additional failure modes:** The analysis
  repeatedly conditions asymptotic claims on controllable fluctuations and
  regularity of the model density. The authors state that predictive
  information measures the amount of structure but does not exhibit its
  concrete form. A global latent parameter can therefore mimic intrinsic
  temporal dependence.
- **Screen decision this evidence can support:** Plot cumulative useful
  information against history length and distinguish saturation,
  logarithmic, and stronger growth regimes as descriptive signatures.
- **What it cannot support:** Generic past-future predictability cannot
  identify the task-relevant observable, separate topic from order, or
  certify a TXC advantage.
- **Direct-method adaptation:** Replace the whole future with a local target
  \(Y_t\), condition on anchor and bag baselines, and estimate a cross-fitted
  information-gain curve with grouped uncertainty. A sequence with one
  document-level latent topic and a topic-independent local label should
  exhibit large generic predictive information but zero conditional task
  gain.
- **Exact evidence pointers:** Predictive-information definition and
  subextensive identity, Equations (4)--(14), Section 3, pages 8--11;
  finite-parameter logarithmic class, Sections 4.1--4.3, pages 13--24;
  intrinsic-versus-parameter nonidentifiability, Equations (65)--(69), pages
  23--24; nonparametric power-law class, Sections 4.5--4.6 and Equations
  (80)--(97), pages 29--34; compression and order-parameter discussion,
  Section 6, pages 43--44.

## Dębowski and Wieczyński (2025), *Long-Range Dependence in Word Time Series: The Cosine Correlation of Embeddings*

- **Primary source:** [Entropy 27(6):613](https://www.mdpi.com/1099-4300/27/6/613),
  full journal article.
- **Scientific question and observable:** The paper asks whether a
  computationally cheap cosine-correlation statistic on word embeddings can
  provide evidence of word-level LRD across languages and distinguish human
  from LLM-generated text.
- **Temporal object, estimand, estimator, and null:** The temporal object is a
  pair of pooled embedding windows separated by lag \(n\). Equation (28)
  defines their centered cosine correlation \(C(n\mid k)\); Equations
  (29)--(30) give an \(O(Nd)\) estimator. Power-law and stretched-exponential
  decays are fit over \(k\le n\le1000\). Noise-dominated LLM text is used as a
  comparison, but there is no permutation null in the reported experiment.
- **Assumptions:** The theoretical lower bound is distributional and uses
  discrete random vectors; the empirical fit assumes that a static word2vec
  geometry is a meaningful semantic observable and that pooled text segments
  can be averaged across position. The statistic is linear after
  normalization, unconditional, symmetric, and directionless.
- **Target alignment:** None; no downstream label is present.
- **Dependence units, splits, and uncertainty:** The study uses up to 100
  Gutenberg books in each of 17 languages and 1,000 texts from each of six
  LLM sources plus 1,000 human comparison texts. It reports means and standard
  deviations of per-text fits, Kruskal--Wallis tests across sources, and
  Bonferroni-corrected Dunn tests. It does not hold out groups for prediction.
- **Full-text result:** Theorems 1--3 show
  \(I(U;V)\ge CC(U;V)^2/2\), then use data processing to lower-bound word MI
  by embedding cosine correlation. Section 3.2 uses pooling orders
  \(k\in\{1,3,9,27\}\), logarithmically spaced lags, SciPy `curve_fit`, and
  log-residual fit scores. Figure 1 and Tables 3--8 find a stable slow decay
  in human books out to roughly 1,000 words but noise-dominated, unstable fits
  in the shorter LLM corpus.
- **Author-stated limitations and additional failure modes:** Section 3.4 and
  the conclusion say source differences are not yet explained and admit that
  short LLM texts may cause the apparent human/LLM gap. Japanese is an
  outlier with very short texts and low embedding coverage. Power-law versus
  stretched-exponential selection is unstable, and static lexical embeddings
  can turn topic coherence into apparent temporal dependence.
- **Screen decision this evidence can support:** Cosine correlation is a
  scalable activation-side descriptive check and its squared value has a
  formal, though potentially loose, MI lower-bound interpretation.
- **What it cannot support:** It cannot identify useful direction, downstream
  target relevance, or an architecture advantage; a fitted decay family is
  not stable enough to be a sole acceptance criterion.
- **Direct-method adaptation:** Use within-rollout normalized activations,
  preserve group boundaries, report pooling-order sensitivity, include
  within-group permutation and topic-block nulls, and bootstrap whole groups.
  The authors' implementation is linked in the article's Data Availability
  statement. A matched topic-block/random-label generator should trigger the
  unconditional statistic but fail the target-conditioned gate.
- **Exact evidence pointers:** Cosine-correlation definition and MI lower
  bound, Theorems 1--3 and Equations (14)--(26), pages 5--7; corpora, Tables
  1--2, pages 7--8; estimator, lag grid, fits, and tests, Equations (27)--(35),
  Section 3.2, pages 8--10; Figure 1 and Tables 3--8, pages 10--14;
  limitations and conclusions, pages 14--16.

## Takahashi and Tanaka-Ishii (2017), *Do Neural Nets Learn Statistical Laws Behind Natural Language?*

- **Primary source:** [PLOS ONE 12(12):e0189326](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0189326),
  full journal article.
- **Scientific question and observable:** The paper asks whether
  character-level neural LMs reproduce Zipf, Heaps, and long-range dependence
  statistics of their training corpora, with special attention to conflicting
  conclusions from different LRD estimators.
- **Temporal object, estimand, estimator, and null:** One LRD observable is
  character-pair mutual information at separation \(s\). The second turns the
  intervals between the rarest \(1/16\) of word types into a scalar sequence
  and computes its autocorrelation \(C(s)\), testing for a power-law decay.
  Generated pseudo-text is compared with the source text; supplementary
  controls include word-shuffled and document-shuffled corpora.
- **Assumptions:** Pairwise MI suffers sparse-count bias as alphabet size
  grows. Rare-word interval autocorrelation assumes that this event encoding
  is the relevant observable and treats the resulting series as sufficiently
  stationary for a power-law fit. Both primary statistics are symmetric and
  untargeted.
- **Target alignment:** None; the target is corpus-statistical fidelity, not a
  local task label.
- **Dependence units, splits, and uncertainty:** The model is a
  three-layer, 256-unit character LSTM trained with Adam and context length
  128; it generates 2M characters, or 20M for one WSJ figure. The main
  corpora are 4.12M-character Shakespeare and 4.78M-character WSJ, with
  additional Wikipedia and Chinese examples. The paper does not report
  multi-seed or document-bootstrap uncertainty for the LRD curves.
- **Full-text result:** Section 4.1 and Figure 6 reproduce a Wikipedia
  character-MI power law but find exponential decay followed by a
  low-frequency plateau after about 10 characters on Shakespeare and WSJ.
  The authors show that larger alphabets reach this estimation floor almost
  immediately. Section 4.2, Equations (6)--(7), and Figure 7 instead finds
  rare-word interval autocorrelation decaying to about \(10^3\) lags in
  Shakespeare and \(10^2\) in WSJ, while LSTM-generated text stays near zero.
  Supplementary Figure S3 reports the same failure across several neural
  architectures.
- **Author-stated limitations and additional failure modes:** The paper says
  LRD measurement in language is controversial and observable-dependent.
  Wikipedia markup and repeated corpus artifacts can create misleading MI
  tails; rare-word results depend on the event threshold. The authors also
  note that context length 128 may itself explain the LSTM failure.
- **Screen decision this evidence can support:** Require robustness across
  observables, corpus slices, preprocessing, fit ranges, and matched nulls
  before interpreting any decay family. Estimator disagreement is a failed
  diagnostic, not evidence to select whichever curve helps the hypothesis.
- **What it cannot support:** The rare-event power law does not show a task
  needs ordered history, and failure of a 128-character LSTM does not imply a
  TXC will succeed.
- **Direct-method adaptation:** Predeclare several activation observables and
  label-event encodings, establish their null floors with within-group
  permutations, and demand qualitative agreement or report a contradiction.
  A sparse-alphabet simulation should reproduce the false MI plateau, while a
  rare-event burst process with random downstream labels should reproduce the
  untargeted autocorrelation false positive.
- **Exact evidence pointers:** LSTM architecture, context, and generation
  protocol, Section 2.1, pages 1--3; character-MI estimator dispute and Figure
  6, Section 4.1, pages 8--10; rare-event autocorrelation, Equations (6)--(7)
  and Figure 7, Section 4.2, pages 10--12; supplementary shuffle and
  architecture controls, Figures S2--S4; conclusion, pages 12--13.

## Cross-source implication for the Temporal Screen

The sources jointly support a staged screen, but not an unconditional
power-law gate:

1. **Target utility comes first.** Use conditional predictive
   \(\mathcal V\)-information to ask whether history lowers held-out log-loss
   beyond the anchor and an order-invariant summary. O'Connor and Andreas,
   Xu et al., Hewitt et al., and Jiao et al. supply the formal and experimental
   basis.
2. **Order is a separate intervention.** Retrain matched observers on ordered,
   bagged, locally shuffled, reversed, and cross-group-replaced histories.
   Khandelwal et al. and Sun et al. show why fixed-model perturbations and
   aggregate targets are insufficient.
3. **Generic scaling diagnoses mechanism, not eligibility.** Correlation,
   periodogram, predictive-information, and embedding-cosine curves can
   estimate resolvable scales or expose hierarchy. Cagnetta et al., Cagnetta
   and Wyart, Belletti et al., Lin and Tegmark, Bialek et al., and Dębowski and
   Wieczyński motivate these diagnostics.
4. **Every decay claim needs competing nulls.** Altmann et al. and Takahashi
   and Tanaka-Ishii show that topic cascades, burst distributions, corpus
   artifacts, observable choice, and estimator floors can each manufacture or
   erase an apparent long-range tail.

The most important falsification case is consequently simple: create
activations with a robust power-law or stretched-exponential correlation
through an independent latent topic process, while making the downstream
label depend only on the anchor or on an order-invariant bag. A valid Temporal
Screen must reject ordered temporal eligibility despite the impressive
unconditional spectrum.
