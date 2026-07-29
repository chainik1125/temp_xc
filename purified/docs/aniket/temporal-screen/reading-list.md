# Temporal Screen: annotated reading list and research synthesis

**Recorded:** 2026-07-29

**Purpose:** reading packet for Aniket and Dmitry's post-rebuttal Temporal
Screen project

**Status:** literature synthesis and proposed screen; no new TXC result is
claimed here

## Bottom line

The literature does not support using a power-law fit, an autocorrelation
length, or low-frequency activation power as a pass/fail Temporal Screen.
Those quantities establish that a sequence has memory, but they do not show
that a particular target depends on ordered history, that the dependence is
available before the target event, or that an order-sensitive dictionary
should beat a last-token or order-invariant model.

The strongest screen suggested by the literature is instead:

> **A task is TXC-motivated when ordered local activation history adds
> cross-fitted held-out predictive information about a local target beyond
> the anchor activation and an order-invariant summary of the same history,
> and that gain disappears under controls that destroy alignment or order
> while preserving easier content statistics.**

Power-law and spectral measurements remain useful after this gate. They can
estimate the horizon of measured target-aligned second-order dependence,
distinguish short-memory from multiscale structure, and suggest a frequency
bias for a later Spectral Cross-Coder. That horizon must still agree with
held-out predictive saturation; spectra cannot establish task temporality on
their own.

This conclusion is also consistent with our existing
[correlation audit](../neurips-rebuttal/theory-experiment-log.md#e0-results):
the activation trajectories exhibit slow, heterogeneous, finite-range
dependence with a persistent component, but no centering/layer condition
selected a robust pure-power covariance law. A screen should therefore not
depend on a single fitted exponent.

### Map of the literature

| Direction | Start with | Decision it informs |
|---|---|---|
| Language scaling and long memory | Cagnetta et al.; Altmann et al. | Whether generic lag structure exists and where finite-data noise hides it |
| Target-aligned information | O'Connor and Andreas; Xu et al.; Hewitt et al. | Whether ordered history predicts the actual label beyond anchor and bag |
| Spectral mechanism | Thomson; Geweke; Schreiber and Schmitz | Which timescales, phase relations, and directions carry passed target signal |
| Activation timescales | Skrill and Norman-Haignere; Chien et al. | Whether the frozen LM retains earlier context and resets at structure boundaries |
| Benchmark validity | CheckList; Contrast Sets; HANS; Benchmark Lottery | Whether the task inventory and controls resist shortcuts and task cherry-picking |

## The proposed object

Let:

- \(Y_t\) be a local state, transition, onset, or event-aligned label;
- \(A_t\) be the anchor activation available at prediction time;
- \(H_t^{(T)}=(A_{t-T},\ldots,A_{t-1})\) be the ordered pre-event history;
- \(B_t^{(T)}=\{A_{t-T},\ldots,A_{t-1}\}\) be the multiset of exactly the
  same history, represented by a sufficiently expressive
  permutation-invariant encoder rather than only its mean;
- \(C_t\) contain preregistered, pre-target, deployment-available nuisance
  variables such as position, length, task, or model.

Document, writer, rollout, story, and conversation identifiers normally
define group splits and bootstrap clusters; they should not be fed to the
predictor as nuisance features. Any variable in \(C_t\) needs a causal and
deployment justification so conditioning does not introduce a collider,
erase the scientific signal, or leak the label.

The population quantity we want is

\[
S_{\mathrm{order}}(T)
=I\!\left(Y_t;H_t^{(T)}
\mid A_t,B_t^{(T)},C_t\right).
\]

Under log loss, estimate this using genuinely nested cross-fitted
predictors. The larger model contains the anchor-plus-multiset model as an
exact submodel and adds an ordered residual branch that it can choose to
ignore:

\[
\widehat S_{\mathrm{order}}(T)
=\frac{1}{n}\sum_i
\left[
\log q_1(y_i\mid h_i,a_i,b_i,c_i)
-\log q_0(y_i\mid a_i,b_i,c_i)
\right].
\]

At Bayes-optimal prediction this difference is conditional mutual
information. With a fixed predictor family it is better described as
**conditional usable information**: the part of the ordered history that a
declared, capacity-matched class of probes can exploit beyond the baselines.
That qualification is desirable because raw Shannon information in a
high-dimensional activation window is neither reliably estimable at our
sample sizes nor necessarily accessible to a sparse dictionary.

If the larger model does not contain the baseline as a submodel with
optional ignorance, report only a **cross-fitted log-loss gap**; it is not a
valid conditional usable-information estimate.

Two controls answer different questions and should not be conflated:

1. A **refit bag or independently randomized-order model** asks how much
   target information remains after the window's presented slot order is
   removed. It is the appropriate information-content comparison because
   each model is trained on its own input distribution. A fixed reversal is
   invertible and belongs with orientation tests, not information-removal
   controls.
2. A **fixed-model perturbation** applies a trained ordered predictor to
   shuffled or reversed test inputs. It tests whether that predictor relies
   on order, but the loss may include ordinary distribution shift. It is a
   useful falsification test, not by itself an estimate of information lost.

All splits and uncertainty estimates must operate at the highest dependence
unit—rollout, problem, document, story, writer, or conversation—rather than
at the overlapping-window level.

## Read first: the core packet

The following twelve papers are the shortest route from Dmitry's scaling-law
question to a defensible screen. They are ordered by the decision they
resolve, not by publication date.

### 1. Cagnetta, Raventós, Ganguli, and Wyart (2026)

[*Deriving Neural Scaling Laws from the statistics of natural
language*](https://arxiv.org/abs/2602.07488)

**What it establishes.** The paper relates data-limited LM scaling to two
dataset-level quantities: lagged token-token covariance estimated from
counts, and the decay of next-token conditional entropy approximated using
the converged losses of large neural models restricted to \(n\)-token
contexts. The comparison between a decaying correlation signal and a
\(P^{-1/2}\) sampling floor under their fixed-vocabulary setting gives a
finite-data prediction horizon.

**Why we need it.** It gives us a clean distinction among available temporal
structure, the horizon at which it is statistically resolvable, and a
learner's ability to exploit that structure. The natural extension is to
replace corpus-global token covariance with target-aligned
\(\operatorname{Cov}(A_{t-\tau},Y_t)\) or conditional usable information.
Our noise floor must be estimated with grouped document resampling because
token imbalance, overlapping lag pairs, and document dependence change its
constants.

**What it does not establish.** Its statistics are corpus-global and its
target is the next token. These exponents do not specify a downstream
label's horizon or order dependence. For an ideal stationary process,
reversal transposes the lagged covariance and preserves its singular values,
although finite-document boundary handling can break exact equality. The
statistics therefore cannot decide whether backtracking, EM, or another
label requires ordered history, nor whether a TXC should beat an SAE.

### 2. O'Connor and Andreas (2021)

[*What Context Features Can Transformer Language Models
Use?*](https://aclanthology.org/2021.acl-long.70/)

**What it establishes.** The authors retrain matched predictors on
systematically ablated contexts and use predictive
\(\mathcal V\)-information to separate the amount of context from the kind
of usable information it contains. The result depends strongly on the
ablation: sentence-order shuffle removes only about 14% of long-range usable
information, within-sentence shuffle removes roughly 55%, and global word
shuffle roughly 84%. Some lexical deletions also preserve substantial
usable information.

**Why we need it.** This is the closest existing template for the Temporal
Screen. Substitute a local task label for the next token and compare
last-token, order-invariant, ordered, and independently randomized-order
histories under the same predictor family, with fixed-model reversal as an
orientation stress test. Its graduated ablations also show why one
undifferentiated “shuffle control” is inadequate.

**Caveat.** Retrained ablations are more expensive than a fixed-probe shuffle,
and their empirical result concerns language modeling rather than activation
classification. The methodological separation is the important part.

### 3. Xu, Zhao, Song, Stewart, and Ermon (2020)

[*A Theory of Usable Information Under Computational
Constraints*](https://arxiv.org/abs/2002.10689)

**What it establishes.** Predictive \(\mathcal V\)-information replaces an
unrestricted observer with an explicit predictor family, making
informativeness dependent on what the observer can compute and making the
quantity estimable in high dimensions through held-out prediction.

**Why we need it.** “Architecture-independent” cannot mean independent of
every possible observer. It should mean that the screen is frozen before TXC
training and uses a small declared family—such as ridge logistic regression
and one shallow MLP—equally across all tasks and controls.

**Caveat.** Different probe classes can rank tasks differently. The screen
must publish both the probe class and sensitivity across the two
preregistered capacities.

### 4. Hewitt, Ethayarajh, Liang, and Manning (2021)

[*Conditional probing: measuring usable information beyond a
baseline*](https://aclanthology.org/2021.emnlp-main.122/)

**What it establishes.** Conditional probing measures what a representation
makes predictable beyond information already present in a baseline, rather
than merely comparing two standalone probe scores.

**Why we need it.** The correct question is not whether a window predicts the
label. It is whether the ordered window contributes information that is not
already available in the anchor token and a bag of the same activations.
This paper supplies the probing analogue of that exact comparison.

**Caveat.** Probe selectivity and dataset leakage still matter. Grouped
splits, control tasks, and capacity matching remain necessary.

### 5. Jiao, Courtade, Venkat, and Weissman (2015)

[*Justification of Logarithmic Loss via the Benefit of Side
Information*](https://arxiv.org/abs/1403.4679)

**What it establishes.** Under logarithmic loss, the population reduction in
optimal prediction risk supplied by additional side information is mutual
information; with common side information it yields the corresponding
conditional quantity.

**Why we need it.** It provides the formal reason to report cross-fitted
log-loss gain rather than only accuracy or average precision. The gain has a
direct information-theoretic interpretation at the population optimum and
remains a proper held-out predictive score for finite probes.

**Caveat.** The identity does not make a finite learned predictor Bayes
optimal. Report raw paired log-loss differences and confidence intervals
rather than overclaiming an exact number of bits.

### 6. Altmann, Cristadoro, and Degli Esposti (2012)

[*On the origin of long-range correlations in
texts*](https://arxiv.org/abs/1207.0658)

**What it establishes.** Semantic burstiness and hierarchical topical
organization can generate long-range dependence at word and character
levels. Carefully chosen shuffles distinguish broad recurrence-time
distributions from correlations among recurrence intervals.

**Why we need it.** This is the main warning against treating low-frequency
power as ordered task signal. Topic persistence or rare-word bursts can
produce a strong correlation tail even when an order-invariant bag contains
nearly everything useful for the target.

**Caveat.** The result depends on the chosen observable and literary texts
are nonstationary. That dependence is a reason to stratify by sequence and
task phase, not to fit one corpus-wide slope.

### 7. Skrill and Norman-Haignere (2023)

[*Large language models transition from integrating across position-yoked,
exponential windows to structure-yoked, power-law
windows*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/020ad0ac6a1974e6748e4a5a48110a07-Abstract-Conference.html)

**What it establishes.** A black-box word-swap intervention estimates how a
token influences later LM representations. Earlier layers show more
position-yoked exponential windows, while later layers increasingly show
longer, structure-yoked windows fit by an exponential/power-law mixture.

**Why we need it.** The work gives a practical activation-side complement to
correlation analysis: perturb an earlier token and measure the influence on
the anchor representation across distance and structural boundaries.

**Caveat.** An integration window describes the model representation, not
whether a chosen target needs that history. It belongs beside, not in place
of, the target-aligned screen.

### 8. Chien, Zhang, and Honey (2021)

[*Mapping the Timescale Organization of Neural Language
Models*](https://openreview.net/forum?id=J3OUycKwz-)

**What it establishes.** Two runs receive different prefixes followed by an
identical continuation, and the decay of their activation difference maps
how long prior context remains distinguishable. Fewer than 15% of units were
assigned long timescales under the authors' criterion.

**Why we need it.** Paired-prefix relaxation transfers directly to frozen
transformers and detects rare long-timescale units or subspaces that a layer
average can hide.

**Caveat.** A relaxation window measures retained context, not whether the
retained information predicts a chosen task target.

### 9. Keshishian, Norman-Haignere, and Mesgarani (2021)

[*Understanding Adaptive, Multiscale Temporal Integration in Deep Speech
Recognition
Systems*](https://proceedings.neurips.cc/paper/2021/file/ccce2fab7336b8bc8362d115dec2d5a2-Paper.pdf)

**What it establishes.** Temporal Context Invariance embeds an identical
speech segment in different contexts and measures how much shared segment is
needed before representations converge. Later-layer windows dilate and
contract as speech is time-stretched or compressed, consistent with
adaptation to linguistic structure.

**Why we need it.** It supplies a second intervention-based window
measurement and motivates testing resets at sentence or reasoning-state
boundaries rather than assuming a fixed token decay.

**Caveat.** Continuous speech provides a natural time-dilation intervention
with no exact text-token analogue, and generic integration remains distinct
from target relevance.

### 10. Thomson (1982)

[*Spectrum Estimation and Harmonic
Analysis*](https://doi.org/10.1109/PROC.1982.12433)

**What it establishes.** Multitaper estimation averages orthogonal
DPSS-tapered eigenspectra to reduce leakage and variance in finite
sequences; the same framework supports cross-spectra, coherence, and line
tests.

**Why we need it.** Short reasoning traces and event-aligned windows make raw
periodograms unreliable. If we add a spectral screen, multitaper estimates
should be the default rather than a single FFT of concatenated rollouts.

**Caveat.** Approximate stationarity is still required within each analyzed
segment. A power spectrum also cannot distinguish a trajectory from its
reversal.

### 11. Geweke (1982)

[*Measurement of Linear Dependence and Feedback Between Multiple Time
Series*](https://doi.org/10.1080/01621459.1982.10477803)

**What it establishes.** A joint linear time-series model can decompose the
reduction in target prediction error attributable to past source values by
frequency, with the frequency-domain measure integrating to time-domain
Granger feedback.

**Why we need it.** This is much closer to the TXC question than activation
power: do past activations predict a future local task state, and in which
frequency bands? It retains direction and target alignment.

**Caveat.** The measure is linear, sensitive to lag order, and vulnerable to
omitted confounders. We should call it frequency-resolved predictive
dependence, not causal influence.

### 12. Schreiber and Schmitz (1996, 2000)

[*Surrogate time series*](https://doi.org/10.1016/S0167-2789%2899%2900181-4)
and the associated
[*Improved Surrogate Data for Nonlinearity
Tests*](https://arxiv.org/abs/chao-dyn/9909041)

**What it establishes.** Iterative amplitude-adjusted Fourier-transform
surrogates alter phases while iteratively matching a sequence's marginal
distribution and autocorrelation under a
Gaussian-linear-process-plus-static-transform null. The iterations do not
guarantee independent uniform phases.

**Why we need it.** A control hierarchy—full shuffle, block shuffle, circular
target shift, reversal, and phase-randomized surrogate—can triangulate
marginals/DC, local proximity, target alignment, direction, and nonlinear
phase coupling under different nulls rather than identify any one mechanism
uniquely. This is stronger than calling every shuffle a temporal ablation.

**Caveat.** IAAFT is approximate and tests a particular
linear-process-plus-static-nonlinearity null. Residual spectrum and
autocorrelation mismatch must be reported, and rejection can reflect
nonstationarity, surrogate mismatch, higher-order dependence, or residual
phase structure.

## Direction 1: language statistics, hierarchy, and useful context

### Cagnetta and Wyart (2024)

[*Towards a theory of how the structure of language is acquired by deep
neural networks*](https://arxiv.org/abs/2406.00048)

A probabilistic hierarchy produces progressively longer-range correlations,
and finite data resolves them only until signal meets sampling noise. This
supports defining a resolvable target horizon rather than a metaphysical
“true context length,” but the real-text evidence is limited and the theory
does not select an architecture for labeled tasks.

### Belletti, Chen, and Chi (2019)

[*Quantifying Long Range Dependence in Language and User Behavior to improve
RNNs*](https://arxiv.org/abs/1905.09414)

The paper maps large-alphabet sequences into vectors, estimates long-range
dependence from a low-frequency log-periodogram slope, and uses the estimate
to choose distance-dependent RNN capacity. It is the closest precedent for
turning a spectral statistic into an architectural prior, but its statistic
is unconditional, embedding-dependent, and susceptible to DC/topic
structure.

### Khandelwal, He, Qi, and Jurafsky (2018)

[*Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use
Context*](https://aclanthology.org/P18-1027/)

Test-time truncation, shuffling, reversal, deletion, and replacement show
that an LSTM uses roughly 200 tokens of context, while order matters mainly
in the recent 20–50 tokens and distant context behaves more like topic or a
rough semantic field; a separate external cache supplies distant copying.
This motivates separate curves for total history gain, ordered gain, and
order-invariant gain. Because the perturbations occur only at test time,
distribution shift can inflate the apparent cost.

### Sun et al. (2021)

[*Do Long-Range Language Models Actually Use Long-Range
Context?*](https://aclanthology.org/2021.emnlp-main.62/)

Prefix-length and perturbation sweeps find little aggregate gain beyond
2,000 tokens for the tested models, with gains concentrated in rare,
copyable tokens and continuous fiction. The screen should therefore report
event- and sequence-level heterogeneity rather than only an average, while
also checking whether apparent long-range gains reduce to copying.

### Lin and Tegmark (2017)

[*Critical Behavior in Physics and Probabilistic Formal
Languages*](https://www.mdpi.com/1099-4300/19/7/299)

Finite-state Markov and hidden-Markov processes have asymptotically
exponential mutual-information decay, while recursive context-free
constructions can produce power-law decay. This connects hierarchy and
scale-free dependence, but it does not imply that every hierarchical
language task has a power law or that power-law mutual information is
order-sensitive.

### Bialek, Nemenman, and Tishby (2001)

[*Predictability, Complexity, and
Learning*](https://arxiv.org/abs/physics/0007070)

Predictive information \(I(X_{\text{past},T};X_{\text{future}})\) separates
processes with finite, logarithmic, and power-law growth of predictable
structure. It motivates a history-length curve, but is untargeted: syntax,
topic, and generic autocorrelation may all make the future predictable
without helping the task label.

### Dębowski and Wieczyński (2025)

[*Long-Range Dependence in Word Time Series: The Cosine Correlation of
Embeddings*](https://www.mdpi.com/1099-4300/27/6/613)

The authors relate squared cosine correlation to a lower bound on mutual
information and find stretched-exponential embedding correlations over long
lags in literary text. The vector-valued estimator is computationally
attractive for activations, but static embeddings bake lexical semantics
into the observable and topic structure can dominate.

### Takahashi and Tanaka-Ishii (2017)

[*Do Neural Nets Learn Statistical Laws behind Natural
Language?*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0189326)

Different estimators give sharply different conclusions about long-range
structure in text and generated samples: sparse-count character mutual
information plateaus or appears exponential on Shakespeare and WSJ, while
rare-event autocorrelation reveals long-range decay that generated LSTM text
fails to reproduce. The screen should require agreement across
preprocessing, corpus slices, observables, and matched nulls before
interpreting a decay family.

## Direction 2: target-aligned and conditional information

### Watson and Wright (2021)

[*Testing Conditional Independence in Supervised Learning
Algorithms*](https://arxiv.org/abs/1901.09917)

Conditional predictive impact compares held-out loss on real features and
conditional knockoffs given baseline variables. It naturally supports
arbitrary predictors and losses, but a credible sampler for activation
history conditional on anchor, bag, topic, and position is hard. Use it if we
can build matched synthetic or learned conditional surrogates.

### Berrett, Wang, Barber, and Samworth (2020)

[*The Conditional Permutation Test for Independence While Controlling for
Confounders*](https://doi.org/10.1111/rssb.12340)

Conditional, nonuniform permutations formalize the correct null when a
global shuffle would also destroy relationships with topic, anchor, or
position. In natural-language activation windows we rarely know
\(P(H\mid A,B,C)\) well enough for exact calibration, so structured
permutations should usually be reported as falsification controls rather
than exact \(p\)-values.

### Runge (2018)

[*Conditional Independence Testing Based on a Nearest-Neighbor Estimator of
Conditional Mutual Information*](https://arxiv.org/abs/1709.01447)

A shared-radius nearest-neighbor CMI estimator with local permutations
preserves conditioning relationships better than global shuffling. It is not
credible directly on a \(dT\)-dimensional raw activation window with only a
few thousand grouped examples; any use should follow a fixed, label-blind
projection and be treated as a sensitivity analysis.

### Goldfeld and Greenewald (2021)

[*Sliced Mutual Information: A Scalable Measure of Statistical
Dependence*](https://proceedings.neurips.cc/paper_files/paper/2021/hash/92c4661685bf6681f6a33b78ef729658-Abstract.html)

Sliced MI averages dependence across one-dimensional random projections and
has a conditional extension, making high-dimensional dependence more
estimable. It is a distinct projected dependence measure rather than a
numerically faithful estimate of ordinary CMI, so it should triangulate the
cross-fitted predictive screen rather than replace it.

### Schreiber (2000)

[*Measuring Information Transfer*](https://arxiv.org/abs/nlin/0001042)

Transfer entropy
\(I(X_{\text{past}};Y_{t+1}\mid Y_{\text{past}})\) asks whether one process's
history predicts another process's next state beyond its own persistence.
It is well matched to evolving labels such as backtracking pressure or
misalignment onset. It is less natural for a single event-aligned label, and
directional statistical transfer is not automatically intervention-level
causality.

### Papapetrou and Kugiumtzis (2013)

[*Markov Chain Order Estimation with Conditional Mutual
Information*](https://arxiv.org/abs/1301.0148)

The paper adds successively older lags while conditioning on intervening
history. Adapted to a task, the incremental curve

\[
\delta_k=I(Y_t;A_{t-k}\mid A_t,B_t,C_t,A_{t-k+1:t-1})
\]

would estimate how much history is actually needed. The original estimator
assumes finite alphabets and becomes sample-hungry as order grows, so we
should implement the idea through nested cross-fitted probes.

### Shah and Peters (2020)

[*The Hardness of Conditional Independence Testing and the Generalised
Covariance Measure*](https://doi.org/10.1214/19-AOS1857)

Their impossibility result shows that no universally valid finite-sample
conditional-independence test exists for unrestricted continuous
distributions. This means the screen must state its probe model and validate
power on matched synthetic controls; “model-free CMI test” would be an
overclaim.

### McAllester and Stratos (2020)

[*Formal Limitations on the Measurement of Mutual
Information*](https://proceedings.mlr.press/v108/mcallester20a.html)

Distribution-free, high-confidence MI lower bounds from \(N\) samples cannot
generally exceed \(O(\log N)\). We should report paired held-out predictive
gains and confidence intervals, not headline neural-estimator claims about
an absolute number of information bits.

## Direction 3: spectra, phase, direction, and nonstationarity

### Robinson (1995)

[*Gaussian Semiparametric Estimation of Long Range
Dependence*](https://doi.org/10.1214/aos/1176324317)

Local Whittle estimation fits the low-frequency behavior
\(f(\lambda)\simeq G\lambda^{-2d}\) without specifying the full short-memory
process. Its result can move with the frequency bandwidth, trends, and
sequence length, so any estimate must be accompanied by a preregistered
bandwidth-stability curve.

### Veitch and Abry (1999)

[*A Wavelet-Based Joint Estimator of the Parameters of Long-Range
Dependence*](https://doi.org/10.1109/18.761330)

Wavelet log-scale diagrams estimate memory from energy across octaves and
use vanishing moments to suppress polynomial trends. Agreement with local
Whittle over preregistered scales is stronger evidence than either estimate
alone; coarse octaves have few coefficients and substantial boundary
effects.

### Dahlhaus (2012)

[*Locally Stationary Processes*](https://arxiv.org/abs/1109.4174)

An evolutionary spectrum \(f(u,\omega)\) allows the frequency distribution
to change over sequence time. Reasoning traces contain state changes and
event onsets, so local or event-conditioned spectra are more appropriate
than concatenating traces into one stationary process. The cost is a sharp
time-resolution versus frequency-resolution tradeoff.

### Grinsted, Moore, and Jevrejeva (2004)

[*Application of the Cross Wavelet Transform and Wavelet
Coherence*](https://doi.org/10.5194/npg-11-561-2004)

Cross-wavelet coherence localizes coupling in both time and scale, while
complex phase gives local relative phase. A lead-lag interpretation requires
a coherent narrowband relationship, stable phase, and an unambiguous
phase-to-delay mapping. For language, their AR(1) red-noise null should be
replaced with sequence-level circular shifts or block surrogates; the cone of
influence and multiple time-scale testing also need explicit handling.

### Diebold and Inoue (2001)

[*Long Memory and Regime Switching*](https://www.nber.org/papers/t0264)

Rare regime switches can mimic long memory, including in very large samples.
Backtracking, misalignment onset, and reasoning-mode changes are themselves
regime-like, so compare long-memory fits with breakpoint and state-mixture
models and repeat estimates within sequences or stable phases.

### Nolte et al. (2008)

[*Robustly Estimating the Flow Direction of Information in Complex Physical
Systems*](https://doi.org/10.1103/PhysRevLett.100.234101)

The phase-slope index uses the slope of complex coherency to recover
lead-lag direction under delayed linear-coupling assumptions while reducing
sensitivity to instantaneous mixtures. It is a useful check on “activation
changes before onset,” but it needs enough frequency resolution and is not a
general causal-direction estimator; nonlinear dependence may remain
invisible.

### Sun, Li, Kuceyeski, and Basu (2018)

[*Large Spectral Density Matrix Estimation by
Thresholding*](https://arxiv.org/abs/1812.00532)

Thresholded averaged periodograms can estimate high-dimensional spectral
matrices under approximate frequency-domain sparsity. Transformer neurons
need not be sparse in the native basis, so this belongs only after a frozen,
label-blind PCA or random projection; otherwise regularization can create
apparently clean timescales.

### Besserve, Logothetis, and Schölkopf (2013)

[*Statistical Analysis of Coupled Time Series with Kernel Cross-Spectral
Density
Operators*](https://papers.nips.cc/paper_files/paper/2013/hash/ae5e3ce40e0404a45ecacaaf05e5f735-Abstract.html)

Kernel cross-spectral operators can detect nonlinear dependence between
structured time series when linear cross-correlation is zero. They are
computationally expensive, kernel-sensitive, stationary, and non-directional,
so they are a later robustness check rather than a v0 screen statistic.

## Direction 4: activation integration windows and timescale heterogeneity

### Chien, Zhang, and Honey (2021)

[*Mapping the Timescale Organization of Neural Language
Models*](https://openreview.net/forum?id=J3OUycKwz-)

Two model runs receive different prefixes and then an identical continuation;
the decay of their activation difference measures how long the earlier
context remains in each unit. Fewer than 15% of units were assigned long
timescales under the authors' mapping criterion, so a layer average can erase
the rare subspace relevant to a task. The method transfers naturally to
transformers, although relaxation time still does not prove target relevance.

### Keshishian, Norman-Haignere, and Mesgarani (2021)

[*Understanding Adaptive, Multiscale Temporal Integration in Deep Speech
Recognition
Systems*](https://proceedings.neurips.cc/paper/2021/file/ccce2fab7336b8bc8362d115dec2d5a2-Paper.pdf)

Their Temporal Context Invariance method embeds an identical speech segment
in different surrounding contexts and asks how much shared context is needed
before representations converge. Later-layer integration windows dilate and
contract as speech is stretched and compressed, which is consistent with
structure-yoking rather than a fixed clock-time window. Continuous speech
has a natural dilation intervention with no exact text-token analogue; for
reasoning traces, boundary-reset tests are the closer translation.

### Sun and Hsieh (2025)

[*How much do contextualized representations encode long-range
context?*](https://aclanthology.org/2025.findings-naacl.90/)

Keeping a suffix fixed while shuffling its long prefix reveals strong
layer-, architecture-, complexity-, and length-dependent differences in
context encoding. Anisotropy-corrected geometry is useful for activation
comparisons, but prefix shuffling changes order, meaning, and compressibility
together and therefore needs matched interventions.

### Mahto, Vo, Turek, and Huth (2020)

[*Multi-timescale Representation Learning in LSTM Language
Models*](https://arxiv.org/abs/2009.12727)

A bank of exponential-memory LSTM units with an appropriate distribution of
time constants can produce aggregate power-law memory. This gives a
principled alternative to one scale-free mechanism: a TXC or Spectral
Cross-Coder may learn a mixture of finite timescales. The derivation depends
on LSTM gates and does not by itself diagnose useful transformer history.

### Lakretz et al. (2019)

[*The emergence of number and syntax units in LSTM language
models*](https://aclanthology.org/N19-1002/)

Long-range number agreement is carried mainly by a tiny specialized
subnetwork. Ablating two number units drives several difficult,
long-distance incongruent agreement conditions toward chance, while easier
or local conditions retain distributed backup and effects differ by number.
This is a clear warning that bulk persistence and task relevance differ: a
small long-timescale subspace may matter even when most directions look
short-range.

### Khajehabdollahi et al. (2024)

[*Emergent mechanisms for long timescales depend on training curriculum and
affect performance in memory
tasks*](https://proceedings.iclr.cc/paper_files/paper/2024/file/54e45765d0f0b797027948bfe8be6446-Paper-Conference.pdf)

Controlled networks obtain long memory either through long single-unit
constants or through recurrent population dynamics while optimized
single-neuron time constants remain roughly fixed. A screen based only on
neuron autocorrelation can miss a distributed memory mechanism, so paired
population interventions and low-rank subspace analyses are necessary.

### Toneva and Wehbe (2019)

[*Interpreting and improving natural-language processing (in machines) with
natural language-processing (in the
brain)*](https://proceedings.neurips.cc/paper/2019/file/749a8e6c231831ef7756db230b4359c8-Paper.pdf)

Sweeping available context across layers and model families produces
layer-specific and sometimes nonmonotonic context benefits. This supports
layer-by-layer window sweeps; brain-prediction results remain ancillary
because they do not show that the same context is useful for our target.

### Chien and Honey (2020)

[*Constructing and Forgetting Temporal Context in the Human Cerebral
Cortex*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7244383/)

Human cortical responses converge slowly after different preceding contexts
but can separate rapidly after an experimentally imposed context switch.
Their successful model explains this construction/reset asymmetry through
prediction-error-gated updating. The model-side analogue is to measure both
persistence within an event and reset across a reasoning-state boundary.

### Regev et al. (2024)

[*Neural populations in the language network differ in the size of their
temporal receptive
windows*](https://www.nature.com/articles/s41562-024-01944-2)

Interleaved neural populations show average estimated receptive windows of
about one, four, and six words. The biological setting limits direct LM
claims, but it reinforces the methodological point: report a mixture or
distribution of timescales rather than one layer exponent.

### Lerner et al. (2011)

[*Topographic Mapping of a Hierarchy of Temporal Receptive Windows Using a
Narrated Story*](https://doi.org/10.1523/JNEUROSCI.3684-10.2011)

Word-, sentence-, and paragraph-level scrambling map a hierarchy of context
integration in cortex. The direct lesson for the screen is a ladder of
structure-preserving interventions rather than one arbitrary shuffle:
within-phrase, within-sentence, across-sentence, and across-event controls
test different hypotheses.

### Activation-side diagnostic module

These papers suggest two label-free measurements that can accompany, but
must not replace, the ordered-information gate:

1. **Paired-prefix relaxation.** Give two runs controlled alternative
   prefixes followed by an identical suffix beginning at \(b\), whiten
   activations on the training split, and measure

   \[
   R_\ell(\tau)=
   \frac{\mathbb E\,d(\tilde h^\ell_{b+\tau}(x),
   \tilde h^\ell_{b+\tau}(x'))}
        {\mathbb E\,d(\tilde h^\ell_b(x),\tilde h^\ell_b(x'))}.
   \]

   Compare exponential, power, mixture, cutoff, and nonparametric curves on
   held-out documents, reporting the timescale distribution and
   threshold-crossing times rather than assuming one family.

2. **Local input-influence kernel.** Replace or paraphrase the token at lag
   \(k\), matching embedding distance and surprisal where possible, and
   measure

   \[
   I_\ell(k)=\mathbb E\,d\!\left(
   \tilde h^\ell_t(x),\tilde h^\ell_t(x^{(t-k)})
   \right).
   \]

   Stratify by layer, subspace, and whether source and destination cross a
   declared sentence or event boundary. An event-yoked state should show a
   boundary-aligned reset that a stationary topic/DC component does not.

An input-influence kernel asks whether earlier inputs affect the anchor; a
relaxation curve asks whether different pasts remain distinguishable;
autocorrelation asks how slowly the trajectory varies. None of these asks
whether the label needs the ordered joint configuration. Only the
target-aligned predictive gate answers the TXC suitability question.

## Direction 5: benchmark design and shortcut resistance

### Ribeiro, Wu, Guestrin, and Singh (2020)

[*Beyond Accuracy: Behavioral Testing of NLP Models with
CheckList*](https://aclanthology.org/2020.acl-main.442/)

CheckList organizes evaluation as a capability-by-test-type matrix with
minimum-functionality, invariance, and directional-expectation tests. The
Temporal Screen should similarly enumerate mechanisms—onset, overwrite,
delayed consequence, accumulation, and persistent state—against controls
before running models. The matrix itself does not prevent favorable
selection, so both cells and tasks must be frozen.

### Dehghani et al. (2021)

[*The Benchmark Lottery*](https://arxiv.org/abs/2107.07002)

Algorithm rankings can change when the selected benchmark tasks change. This
directly motivates a declared task universe, a fixed aggregation rule, and
reporting every attempted task rather than replacing screen failures with
new candidates.

### Gardner et al. (2020)

[*Evaluating Models' Local Decision Boundaries via Contrast
Sets*](https://aclanthology.org/2020.findings-emnlp.117/)

Small, meaningful edits that typically change the gold label expose model
failures hidden by ordinary IID test sets. Temporal tasks should include
content-matched pairs in which changing order or event location flips the
label, alongside label-preserving lexical and writer perturbations. Human
edits can introduce their own artifacts, so pair labels and reduced-input
baselines require independent audit.

### McCoy, Pavlick, and Linzen (2019)

[*Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural
Language Inference*](https://aclanthology.org/P19-1334/)

HANS balances examples where named heuristics agree and disagree with the
true label. Our analogue should explicitly name last-token, best-offset, bag,
length, absolute-position, writer, source, and lexical shortcuts, then
construct slices where each shortcut succeeds and fails.

### Kaushik and Lipton (2018)

[*How Much Reading Does Reading Comprehension
Require?*](https://aclanthology.org/D18-1546/)

Question-only, passage-only, and last-sentence baselines remain surprisingly
competitive on several reading-comprehension datasets. Long inputs therefore
do not establish long dependencies. Every Temporal Screen task needs
truncation, best-offset, anchor-only, and metadata-only baselines before an
ordered model is informative.

### Gururangan et al. (2018)

[*Annotation Artifacts in Natural Language Inference
Data*](https://aclanthology.org/N18-2017/)

Hypothesis-only models exploit label-correlated lexical patterns introduced
by annotation. The corresponding temporal audit must test length, position,
rollout source, task template, writer, and local lexical cues. These baselines
only catch anticipated shortcuts, which is why paired counterfactuals remain
necessary.

### Sinha et al. (2021)

[*UnNatural Language
Inference*](https://aclanthology.org/2021.acl-long.569/)

Many NLI predictions survive severe word-order corruption, offering direct
precedent for testing whether a nominally structured task actually needs
order. Arbitrary permutations are unnatural and can produce distribution
shift, so natural multiset-matched order counterfactuals should be primary
and random shuffles should be stress tests.

### Miralles-González et al. (2025), auditing Tay et al. (2021)

[*On the Locality Bias and Results in the Long Range
Arena*](https://arxiv.org/abs/2501.14850), with the original
[*Long Range Arena*](https://openreview.net/forum?id=qVyeW-grC2k)

Bounded-receptive-field models recover much of Long Range Arena performance,
showing directly that a benchmark with long sequences may still be dominated
by short-range or positional cues. The critique is a recent preprint and its
result is benchmark-specific, but the bounded-window sweep is exactly the
audit our task inventory needs.

### van Miltenburg et al. (2021)

[*Preregistering NLP
Research*](https://aclanthology.org/2021.naacl-main.51/)

The paper recommends timestamping hypotheses, outcomes, conditions,
exclusions, sample sizes, analyses, and stopping rules. For us that means
committing task and split hashes, label definitions, controls, metrics,
probe classes, windows, seeds, thresholds, and qualitative predictions
before confirmatory TXC runs. Exploration remains allowed, but it must be
labeled and kept separate.

### Ye et al. (2021)

[*CrossFit: A Few-shot Learning Challenge for Cross-task Generalization in
NLP*](https://aclanthology.org/2021.emnlp-main.572/)

CrossFit standardizes seen and unseen task partitions and shows that upstream
task selection affects transfer. The Temporal Screen should be developed on
one set of mechanism families and evaluated on sealed tasks from those
families. Random task-ID splits are insufficient when tasks share a source
dataset, format, or labeling rule.

### Magar and Schwartz (2022)

[*Data Contamination: From Memorization to
Exploitation*](https://aclanthology.org/2022.acl-short.18/)

The paper separates mere memorization of contaminated examples from actual
downstream exploitation. Every task card should record release dates,
duplicate checks, likely pretraining exposure, and newly collected
counterfactuals where possible. For closed pretraining corpora this becomes a
risk assessment, not a provable exclusion.

### Kiela et al. (2021)

[*Dynabench: Rethinking Benchmarking in
NLP*](https://aclanthology.org/2021.naacl-main.324/)

Human-and-model-in-the-loop collection finds verified failure cases and can
help discover new temporal shortcuts. A complete discovery round should then
be frozen: a continually changing model-targeted benchmark is unsuitable as
the confirmatory metric and may encode artifacts of the adversary model.

## Proposed Temporal Screen v0

The literature suggests a staged screen rather than one scalar “temporality
score.” Each stage removes a distinct easier explanation. Exploratory
candidates may stop after a preregistered data-quality or screen failure, but
the frozen benchmark must still train matched dictionaries on a declared set
of screen-negative tasks; otherwise the screen's specificity is never
tested.

### Stage 0: audit the target

A target must be local enough that it can switch within a sequence and must
be aligned to a declared operational prediction time. For an onset task,
only activations strictly available before onset are inputs; this establishes
temporal precedence, not causality. A rollout-level EM label, for example, is
not a suitable local target if a short misaligned span labels all otherwise
neutral windows as positive.

Record the sampling unit, label construction, prediction point, base rate,
event count, median sequence length, and possible lexical or positional
leaks. Reject tasks whose positive label is assigned only at the sequence
level unless the scientific target is genuinely a sequence-global property.

### Stage 1: establish that history helps

Fit grouped, cross-validated predictors for:

1. nuisance variables only;
2. the anchor activation plus nuisances;
3. the best single historical offset plus anchor and nuisances;
4. anchor, full multiset, and the ordered residual history.

Best-offset selection, feature preprocessing, calibration, and all
hyperparameter tuning occur inside each outer group fold. Otherwise the
best-offset baseline receives an unfair multiple-comparisons advantage.

The first curve is

\[
S_{\mathrm{history}}(T)
=L(q_{\mathrm{anchor}})-L(q_{\mathrm{ordered},T}),
\]

where lower log loss is better. A positive value says that some history is
usable beyond the anchor. It does not yet say that order matters, because the
ordered predictor may use averaging, repetition, topic, or one especially
informative offset.

### Stage 2: isolate order from content and denoising

Fit a strong permutation-invariant encoder over the full multiset, capable of
representing more than its mean. The primary ordered predictor contains this
anchor-plus-multiset model as a frozen or jointly trained submodel and adds
an ordered residual branch with optional ignorance. The primary order
statistic is

\[
S_{\mathrm{order}}(T)
=L(q_{\mathrm{anchor+bag},T})
-L(q_{\mathrm{ordered},T}).
\]

Also report the raw best-single-offset gap. Because an invariant encoder and
an ordered encoder have different inductive biases, “exact function-class
matching” is impossible; run a separate sensitivity comparison matched on
parameter count, regularization, optimization steps, and search budget, and
validate it on synthetic equal-signal controls.

Choose one primary probe before screening and treat the second probe class as
a sensitivity analysis. Calibrate a smallest effect of interest and a
group-bootstrap interval during synthetic/discovery work, then freeze them.
Define group stability explicitly, for example through a leave-one-group-out
sign and effect-size audit, rather than informally claiming that no document
drives the result.

An independently randomized window with the permutation hidden from the
model removes the externally presented slot order. A fixed reversal does
**not**: reversal is a bijective encoding, so an expressive refit model can
learn to undo it. Reversal is still valuable as a frozen-model orientation
test and as an equal-magnitude-spectrum control, but it must not be described
as deleting information.

There is one activation-specific limitation: every contextual activation can
already encode earlier tokens and absolute or relative position. A bag or
shuffle removes the arrangement of activation vectors presented to the
probe, not temporal information already stored inside each vector. This is
the correct conservative question for TXC—what does ordered cross-position
composition add beyond T=1 and invariant pooling—but it should be stated
explicitly. Condition on absolute position, audit position decodability, and
pair the activation screen with a task-side token/text screen when possible.

For a refit shuffle, draw a fresh hidden within-window permutation for every
example or training epoch and average evaluation over multiple fresh
permutations. A single fixed permutation is invertible and learnable. As a
sanity check, a sufficiently expressive model trained and evaluated on a
fixed reversal should recover ordered performance; only applying reversal
to a fixed ordered model tests orientation reliance, with ordinary
distribution shift as a caveat.

### Estimation contract

Use nested cross-fitting: preprocessing, probe and regularization selection,
best-offset search, early stopping, and probability calibration all occur
inside each outer group fold. State whether losses weight examples, sequences,
or groups equally; the default should prevent longer sequences from silently
dominating. Preserve natural class prevalence for the primary information
estimand, state whether log loss is reported in nats or bits, and treat
class-balanced analyses as named sensitivities.

### Stage 3: estimate the useful horizon

Report the full preregistered window curve rather than the best \(T\).
Incremental lag gains can be approximated by nested predictors,

\[
\delta_k=
L(q_{1:k-1})-L(q_{1:k}),
\]

but separately refitted increments need not be additive or monotone. Keep
the fixed-\(T\) curve primary and define the useful horizon as the earliest
\(T\) whose score and confidence interval fall within a preregistered
tolerance of the best score on the fixed grid. Freeze the tolerance during
synthetic/discovery calibration and report its sensitivity. Compare this
horizon with the TXC saturation or peak predicted before dictionary
training.

Use sequence-level bootstraps and a common cohort across all \(T\). If
longer windows reduce the available cohort, report both a fixed-cohort curve
and the maximal-data sensitivity curve.

### Stage 4: diagnose the temporal mechanism

For continuous or densely labeled state paths, residualize the target and
activation projections against anchor, bag, and nuisance baselines using
cross-fitting. Then estimate:

\[
G(\tau)=\operatorname{Cov}(z_{t-\tau},y_t),\qquad
S_{zy}(\omega)=\sum_\tau G(\tau)e^{-i\omega\tau},
\]

and, where sample size permits, non-DC multiple coherence

\[
\kappa^2(\omega)=
\frac{S_{yz}(\omega)S_{zz}^{-1}(\omega)S_{zy}(\omega)}
     {S_{yy}(\omega)}.
\]

This is a population expression. Estimate it only after a frozen,
label-blind low-dimensional projection and use a training-selected ridge,
shrinkage estimator, or pseudoinverse for \(S_{zz}\). Calibrate its
finite-sample upward bias with effective degrees of freedom and matched
surrogates.

Report positive-versus-negative lag asymmetry, complex phase or another
directional statistic, integrated non-DC coherence, and the frequency/lag
range containing most target-aligned energy. Use multitaper estimates,
whole-sequence bootstraps, and local/event-conditioned analyses rather than
concatenating unrelated rollouts.

These are predictive-dependence statistics, not causal estimates:
transformer activations already summarize earlier context, and shared slow
state or overlapping receptive fields can produce activation–target
cross-spectra.

For sparse event labels, event-triggered lag profiles and nested predictive
gains are primary; a spectrum of a handful of impulses is usually less
stable and less interpretable.

Compare at least:

- independently resampled within-window permutations relative to the
  unchanged target, which remove presented slot assignment;
- a within-block shuffle, which preserves coarse state and local marginals;
- several target shifts beyond the frozen horizon, performed within a
  sequence and stable phase while excluding wrapped regions;
- a fixed reversal, which tests orientation while preserving information;
  real univariate power is preserved, and jointly reversing both series
  preserves coherence magnitude while conjugating their cross-spectrum;
- an IAAFT or other phase surrogate when the spectral null is scientifically
  relevant.

These controls triangulate explanations under different declared nulls; no
single surrogate uniquely identifies DC, proximity, alignment, direction, or
nonlinear phase coupling. Treat their empirical distributions as
falsification tests unless the exchangeability assumptions for a calibrated
test are justified.

### Stage 5: test whether the screen predicts architecture results

After the screen, thresholds, task inventory, and window predictions are
frozen, train matched SAE, T-SAE, ordered TXC, and order-destroyed controls.
The confirmatory claim is not merely “TXC wins on screen-positive tasks.”
It is:

1. higher preregistered ordered-information scores predict a larger
   TXC-minus-SAE/T-SAE gap across both positive and negative tasks; and
2. the screen's predicted useful horizon correlates with the window at which
   the TXC gain saturates or peaks.

Before training, freeze one primary architecture contrast, task-score
normalization, task weighting, association statistic, uncertainty method,
and missing-run policy. SAE, T-SAE, and shuffled-TXC comparisons that are not
the primary contrast remain separately named secondary results; do not take
their post-hoc maximum. Keep the declared screen-negative tasks and all
failures in the architecture analysis. Otherwise the screen becomes a new
way to cherry-pick tasks rather than a scientific explanation of when
temporal dictionaries help.

## Validation matrix before real tasks

The first implementation should prove that its controls separate mechanisms
we already understand:

| Synthetic process | History gain | Ordered gain | Spectrum | Expected screen result |
|---|---:|---:|---|---|
| IID label from current token | none beyond anchor | none | arbitrary background | reject |
| Window mean / count | positive | none beyond bag | DC/low frequency | reject TXC-specific claim |
| One informative named old offset | positive | may beat bag, but zero beyond best offset | lag spike | reject distributed-history claim |
| Directed two-stage motif | positive | positive | phase-bearing cross-spectrum | pass |
| Motif versus its reversal | positive | positive | identical magnitude power | pass; phase/direction distinguishes |
| Multiscale background, label independent | background only | none | apparent long memory | reject |
| Sparse onset preceded by a ramp | positive pre-onset | positive if path shape matters | low-frequency target coherence | pass |
| Regime mixture with no within-regime memory | apparent long memory | task dependent | low-frequency slope | require breakpoint explanation |

This matrix should be calibrated before inspecting new TXC results. It also
tests the most important falsification from the spectral literature: strong
long-range power can coexist with zero target-relevant ordered information.

## Anti-cherry-picking and benchmark protocol

1. Build the task inventory and label definitions without looking at new
   TXC performance.
2. Divide tasks into screen-development and untouched task-holdout sets,
   stratified by hypothesized mechanism rather than random examples.
   Backtracking, EM, RLHF, sparse probing, deletion, and every other task
   whose TXC result has already been inspected are retrospective development
   evidence, never untouched confirmation.
3. Validate the screen on exact synthetic positives and negatives, then
   freeze probe classes, capacity, splits, window grid, controls, effect-size
   summaries, and rejection rules.
4. Run the cheap task- and activation-side screen on every inventoried task,
   preserving both passes and failures.
5. Write the predicted sign and useful window for each architecture
   comparison before training any dictionary.
6. Train matched representations and test the screen-to-architecture
   relationship on held-out tasks, not only held-out examples from tasks used
   to tune the screen.
7. Maintain an append-only registry of screen decisions and experiments so a
   negative task cannot silently disappear from the benchmark.

Within each task, use disjoint sequence groups to estimate the screen score
and the architecture gap, or model their shared uncertainty explicitly.
Reusing the same noisy evaluation examples for both axes can create a
spurious screen–TXC correlation.

This protocol makes the screen dictionary-independent in the sense needed
for the paper: no SAE, T-SAE, or TXC result is used to decide whether a task
is temporal. It does not claim observer-free information, because the raw
activation probe class is explicit and finite.

## What to read this week

### Four-hour minimum packet

1. **Cagnetta et al.** for the lag-decay, conditional-entropy, and finite-data
   horizon picture.
2. **O'Connor and Andreas** for refit ablations and usable context
   information.
3. **Xu et al. plus Hewitt et al.** for the probe-class-dependent and
   conditional-information formulation.
4. **Altmann et al.** for the topical-burstiness counterexample.
5. **Skrill and Norman-Haignere** for an intervention-based activation
   timescale measurement.
6. **Chien et al. and Keshishian et al.** for paired-prefix relaxation and
   Temporal Context Invariance.

### Second pass

Read Jiao et al., Watson and Wright, Berrett et al., Khandelwal et al.,
Belletti et al., Dahlhaus, Diebold and Inoue, and the benchmark-design papers
below. Read Thomson, Geweke, and Schreiber/Schmitz when implementing the
spectral diagnostic; use Robinson/Veitch-Abry only when implementing a
long-memory estimator. None of the classical spectral methods should delay
the first predictive screen.

## Questions for Surya, Francesco, Allan, and Matthieu

1. Does the finite-data horizon argument extend cleanly from the operator
   norm of token-token covariance to a target-conditioned cross-covariance or
   conditional-entropy reduction?
2. Their covariance norm is invariant to reversal. What statistic would they
   use when the scientific question specifically requires temporal order and
   direction?
3. How should the sampling floor be estimated with dependent, grouped
   sequences rather than effectively IID token pairs?
4. On WikiText, how do they choose the fitting range around broken scaling,
   and what prediction would falsify the claimed exponent rather than merely
   shift the chosen range?
5. Would they view a cross-fitted conditional usable-information curve as a
   principled downstream analogue of \(H_n-H_\infty\)?
6. Are they interested in collaborating on a study that tests whether
   target-conditioned horizons predict the relative performance and optimal
   window of temporal representation learners?

## Claims this literature would support

- Natural language and LM activations contain multiscale temporal
  dependencies, but their form, horizon, and estimator stability vary.
- A target-aligned, grouped, cross-fitted ordered-information curve is a
  defensible pretraining screen for whether a task merits a TXC experiment.
- Spectra, cross-spectra, and phase can diagnose the timescale and direction
  of a passed task, provided stationarity and finite-sample limits are
  handled explicitly.
- A frozen screen that predicts architecture gaps on untouched tasks would
  turn the project from task cherry-picking into a testable account of when
  temporal representations help.

## Claims this literature would not support

- A corpus or activation power law implies that TXCs should work.
- More low-frequency power means that order matters.
- A power spectrum distinguishes a motif from its reversal.
- One fitted decay exponent defines a task's useful window.
- A fixed test-time shuffle estimates information content without
  distribution-shift effects.
- A positive raw-window probe guarantees that an unsupervised TXC dictionary
  will recover the available signal.
