# Activation-timescale and benchmark evidence

Full-text evidence cards for the activation integration-window and benchmark
design papers in the annotated reading list.

**Audit status.** This ledger contains 24 unique primary sources: 11 on
activation integration windows and 13 on benchmark design. Chien, Zhang, and
Honey (2021) and Keshishian, Norman-Haignere, and Mesgarani (2021) each occur
twice in the annotated reading list and are intentionally represented by one
card. All 24 full texts were accessible; none of the cards below was prepared
from an abstract alone. Stable paper links and the shorter annotations are in
the parent [annotated reading list](../reading-list.md).

The activation papers answer whether a model or biological system retains,
constructs, or resets context. With the exception of task-specific causal
ablations, they do **not** answer whether an NLP label depends on that context.
The benchmark papers answer how to keep a task inventory, split, and control
suite from rewarding shortcuts or post-hoc selection. They do **not** supply a
temporal statistic. These two roles remain separate throughout the ledger.

## Activation integration windows and timescale heterogeneity

### Skrill and Norman-Haignere (2023), *Large language models transition from integrating across position-yoked, exponential windows to structure-yoked, power-law windows*

**Question, object, and estimator.** The paper asks how a word at position
\(k\) influences later hidden states. It replaces that word with one of the
five most probable alternatives from BigBird-RoBERTa, reruns the sequence, and
defines unit influence at lag \(\Delta\) as the mean absolute activation
difference
\(\theta[\Delta]=\mathbb E|a_i[k+\Delta]-a_i^{\mathrm{swap}}[k+\Delta]|\),
normalized by \(\theta[0]\). Nonlinear least squares compares exponential,
power-law, and convex exponential/power-law curves; the main mixture is
\(c(\Delta+1)^{-a}+(1-c)e^{-b\Delta}\). A second, structure-yoked experiment
uses five 12-word sentences with punctuation and boundary-token cues removed
and measures whether influence follows sentence boundaries rather than
absolute distance. The relevant null is a position-yoked short exponential
response and, for learned structure, the same analyses in an untrained model.

**Sampling, assumptions, and uncertainty.** The principal experiment uses
GPT-2-small on 600-word Brown Corpus sequences, removing punctuation,
capitalization, and boundary tokens; LLaMA and RoBERTa provide architecture
checks. Held-out sequences determine test MSE, with BIC and Kolmogorov-Smirnov
checks reported as robustness analyses. The estimator assumes that replacement
magnitude is comparable across lag, that absolute unit changes are meaningful
without whitening, and that the chosen curve family is an adequate summary.
The structure experiment additionally assumes the embedding-distance-matched
swaps control intervention size. Sequence rather than overlapping token should
be the resampling unit, but the paper does not present the grouped
cross-validation and cluster bootstrap that our screen needs.

**Relevant result.** The exponential/power-law mixture increasingly dominates
in later layers. In GPT-2, the median mixture weight \(c\) rises from 0.108 in
layer 1 to 0.274 in layer 11; fitted power exponents range from about 1.02 in
layer 3 to 0.51 in layer 12, while exponential rates slow in middle layers.
The structure-yoked analysis shows that later-layer effects align more closely
with sentence structure, and the corresponding patterns are absent or much
weaker in the untrained network.

**Screen role and direct implementation.** This is a practical, label-free
input-influence module. Inputs are token sequences plus a controlled
replacement distribution; outputs are layer/unit/subspace influence curves,
mixture fits, and boundary-alignment scores. A grouped LM adaptation should
whiten activations on the training fold, match replacements on token
probability and embedding distance, fit curves per held-out document or
rollout, and bootstrap documents. The authors release an implementation at
[`dskrill/TemporalIntegration`](https://github.com/dskrill/TemporalIntegration).
The computation is two forward passes per intervention, multiplied by sampled
positions and replacements; no dictionary training is required. A useful
synthetic falsification is a process whose target is entirely
order-invariant but whose activations retain a long nuisance token: the
influence curve should be long while the target-aligned order gate remains
zero.

**Failure modes and allowed decision.** Word substitution changes meaning,
surprisal, token identity, and sometimes syntax together. Absolute activation
differences are sensitive to anisotropy, residual-stream scale, intervention
distribution, and curve-family choice; a finite mixture of exponentials can
look power-law over the measured range. The paper also averages away sparse
task-relevant subspaces. It supports the decision “this frozen model retains
and structurally gates prior-token influence at these layers and horizons.” It
cannot support “this task needs order,” “the tail is a pure power law,” or
“TXC should beat an anchor or bag baseline.”

**Primary-source pointer.** Sections 3.1–3.2 and Equation 3, PDF pp. 4–6;
Figures 2–3, PDF pp. 5–8; untrained-model control in Figure 5 and limitations
in Section 6, PDF p. 10.

### Chien, Zhang, and Honey (2021), *Mapping the Timescale Organization of Neural Language Models*

**Question, object, and estimator.** The paper asks how long a recurrent
language-model unit remains sensitive to preceding context. One run receives
prefix \(A\), another a random prefix \(X\), and both then receive the
identical suffix \(B\). For each suffix position, the authors measure the
activation difference between runs, fit a decreasing logistic curve, and
define the unit's timescale as its time to half-maximum relaxation. The null is
immediate convergence after the shared suffix begins; random-context
resampling supplies the counterfactual prefix.

**Sampling, assumptions, and uncertainty.** The main data are 77 long
sentences from *Anna Karenina*, with 30 random contexts per sentence, applied
to word-level and character-level LSTMs. The authors discard poor logistic
fits—15 of 650 units in the word LSTM and 12 of 1,024 in the character
LSTM—and report confidence intervals across contexts. Wikipedia and an
alternative text split provide replication. The estimator assumes a monotone
logistic relaxation, comparable activation scale across units, and random
prefixes that do not create qualitatively different suffix processing. Unit
timescales can miss a distributed subspace whose coordinates individually
relax quickly, and sentence samples are not a substitute for
document-grouped held-out inference.

**Relevant result.** Higher layers converge more slowly. About 70% of
word-LSTM units have timescales under three words and 13% exceed seven words;
over 63% of character-LSTM units are under three characters and fewer than 15%
exceed ten. Timescales correlate across *Anna Karenina* and Wikipedia
(\(r=.82\)) and across another split (\(r=.83\)). Full stops sharply reset
many units, and structured versus shuffled prefixes produce different
responses. Causal ablations separate fast controller-like units, whose removal
harms general prediction, from slow integrator-like units, whose removal most
harms sentence-final prediction. The reported connectivity-timescale
relationship does not reproduce across every tested recurrent architecture.

**Screen role and direct implementation.** This is the cleanest
paired-prefix-relaxation diagnostic. Inputs are controlled alternative
prefixes and a shared suffix; outputs are whitened unit and low-rank-subspace
distance curves, relaxation half-lives, and boundary-reset ratios. For grouped
LM activations, construct pairs within the same task/template, begin the
identical suffix at a known boundary, estimate whitening and any subspace on
training groups only, and bootstrap documents, stories, or rollouts. Report a
nonparametric threshold-crossing horizon beside the logistic fit. The public
implementation is
[`sherrychien/LSTM_timescales`](https://github.com/sherrychien/LSTM_timescales).
A falsification should rotate a persistent low-rank state across coordinates:
unitwise half-lives will be short while subspace distance remains long.

**Failure modes and allowed decision.** Relaxation establishes retained
prefix dependence, not label relevance, direction-specific order information,
or sparse-dictionary accessibility. Random prefixes may cause an unnatural
distribution shift, and half-maximum depends on the asymptote and activation
normalization. The method supports selecting candidate layers and maximum
history horizons and testing whether boundaries reset state. It cannot pass a
task into a TXC benchmark without a separate target-aligned ordered-history
gain.

**Primary-source pointer.** Sections 3 and 5, PDF pp. 4–7; Figures 2–3, PDF
pp. 5 and 7; ablations in Section 6.3 and Figure 4, PDF pp. 8–9.

### Keshishian, Norman-Haignere, and Mesgarani (2021), *Understanding Adaptive, Multiscale Temporal Integration in Deep Speech Recognition Systems*

**Question, object, and estimator.** Temporal Context Invariance asks how much
shared acoustic history a speech representation needs before responses to an
identical segment become invariant to the preceding context. The same speech
segment is placed in multiple random contexts; for each segment duration and
layer, the estimator is the across-context Pearson correlation of
segment-aligned response matrices. The integration window is interpolated at
the duration where the peak correlation reaches 0.75 of its asymptote. A
second intervention uniformly stretches or compresses speech by 20% and
defines an adaptation index by the proportional change in the inferred window
divided by the proportional change in stimulus duration. Randomly initialized
networks and early layers are the principal nulls.

**Sampling, assumptions, and uncertainty.** Models receive natural speech
sampled at 100 Hz and repeated segments embedded in independently sampled
contexts. The estimator assumes that context placements are exchangeable,
within-segment response similarity is a valid invariance measure, the
asymptotic correlation is identifiable, and interpolation at 0.75 is stable.
The authors report that 98.9% of asymptotic correlations exceed 0.9 and check
threshold choices, but lagged samples and context embeddings are dependent.
Uniform time dilation also changes acoustics in a way that tokenized text
cannot reproduce exactly.

**Relevant result.** Training creates a roughly ten-fold hierarchy, from a
median 58 ms in the first convolutional layer to 592 ms in the fifth LSTM;
random recurrent layers cluster around 167–204 ms. Later integration windows
dilate and contract with speech rate: the adaptation indices in LSTM3 are
about 0.84 for stretching and 0.63 for compression, compared with about
0.19–0.21 in LSTM1. DeepSpeech2, a seven-layer bidirectional LSTM, and
QuartzNet all show a hierarchy at comparable word-error rates, although
recurrent models adapt more strongly.

**Screen role and direct implementation.** TCI is a second label-free
intervention diagnostic and a precedent for asking whether a window follows
linguistic structure rather than clock time. For text, use identical suffixes
after matched alternative prefixes and compare resets at sentence,
reasoning-state, or turn boundaries; do not imitate continuous dilation by
duplicating tokens. Inputs are paired contexts and shared segments; outputs
are convergence curves, threshold windows, and boundary-conditioned
adaptation indices. Computation is forward-pass-only. Group all embeddings
from one utterance/document/rollout in the same split. A synthetic
falsification is a fixed-token-memory system with variable event duration:
token half-life stays fixed even though an event-yoked statistic appears to
dilate.

**Failure modes and allowed decision.** Correlation is insensitive to some
scale changes, the 0.75 threshold is conventional, and a context-invariant
asymptote need not exist. Uniform speech dilation is unnatural, only a few
architectures/tasks/stimuli are tested, and no downstream target is involved.
The evidence supports “integration adapts across layers and speech rate” and
motivates boundary-conditioned relaxation. It cannot establish target
temporality or a benefit for TXC.

**Primary-source pointer.** Equations 3–5 and TCI construction, PDF p. 5;
Figure 3, PDF p. 6; adaptation analysis and Figure 4, PDF pp. 7–8;
cross-architecture robustness in Figure 5, PDF p. 9; limitations in Section
9, PDF p. 10.

### Sun and Hsieh (2025), *How much do contextualized representations encode long-range context?*

**Question, object, and estimator.** The paper asks whether a suffix
representation changes when the content and order of a long prefix are
destroyed. For an intact prefix \(x\) and suffix \(y\), the authors shuffle
\(x\), run both \(xy\) and \(\pi(x)y\), and compute cosine self-similarity for
corresponding suffix positions. Because raw cosine is distorted by
anisotropy, they subtract the expected cosine between unrelated hidden states;
the resulting anisotropy-calibrated contextualization score (ACCS) is low when
the suffix strongly encodes the prefix. The null is the cross-example
anisotropy level, not a target-conditioned no-history model.

**Sampling, assumptions, and uncertainty.** Six 0.5B models are pretrained
for matched iterations on OpenWebText with context 1,024, then compared with
Llama 8B/70B on PG19 and synthetic periodic sequences. Roughly 100,000 suffix
pairs estimate self-similarity, and 500 million token pairs estimate
anisotropy. Prefix shuffling preserves unigram counts but changes syntax,
meaning, local \(n\)-grams, and compressibility; the analysis is therefore not
a clean order-only intervention. ACCS assumes a global anisotropy subtraction
adequately calibrates local representation geometry.

**Relevant result.** RoPE models continue contextualizing with long or noisy
prefixes and can “overcontextualize,” while recurrent and ALiBi models plateau
more locally and hybrids accumulate context gradually. In a periodic
synthetic sequence of period 200 and stride 56, recurrent models need repeated
accumulation whereas attention/hybrid models integrate more continuously.
Raw self-similarity is strongly confounded by prefix incompressibility
measured with LZMA, which is why the anisotropy correction matters.

**Screen role and implementation.** ACCS is useful as a representation-geometry
sanity check and as a warning to whiten or calibrate comparisons. A screen
adaptation should replace arbitrary shuffles with multiple refit controls:
unigram-preserving, local-\(n\)-gram-preserving, within-structure, and
event-boundary shuffles. Inputs are intact/perturbed prefix pairs plus shared
suffixes; outputs are layerwise calibrated similarity curves. The forward
cost is two passes per perturbation. Grouped documents must define splits and
bootstrap units. A falsification is a model that encodes a shuffled nuisance
prefix strongly while the task label depends only on the anchor.

**Failure modes and allowed decision.** ACCS can reward encoding irrelevant
noise and cannot distinguish order from content/compressibility changes.
Position encodings and history-dependent normalization can also lower
similarity without recoverable information. The authors explicitly limit
their study to ten models, a few domains, one main perturbation family, and
simple synthetic regularity. This supports comparing architecture-level
context encoding and choosing geometric calibration; it cannot screen a task
as target-temporal.

**Primary-source pointer.** ACCS definition in Section 2, PDF pp. 2–3;
anisotropy/compressibility analysis in Figure 3, PDF pp. 5–6; architecture
comparison in Figure 4, PDF p. 7; synthetic periodic experiment in Figure 5,
PDF p. 8; limitations in Section 7, PDF pp. 9–10.

### Mahto, Vo, Turek, and Huth (2020), *Multi-timescale Representation Learning in LSTM Language Models*

**Question, object, and estimator.** The paper asks whether explicitly
distributing LSTM memory time constants improves language and synthetic
memory tasks. Under a free-input approximation with selected recurrent terms
removed, an LSTM cell decays exponentially with
\(T=-1/\log f_0\). If cell time constants follow an inverse-gamma
distribution, the population mean of those exponentials decays as a power
law. Experiments compare fixed forget biases drawn from this distribution
against standard trainable-bias LSTMs; the null is a single/narrow learned
timescale population.

**Sampling, assumptions, and uncertainty.** The derivation assumes an LSTM,
no new input after the event, and omitted hidden-to-cell/forget connections
and cell bias. Empirical tests use three-layer 1,150/1,150/400 LSTMs on PTB,
WikiText-2, a Markov version of PTB, and Dyck-2. The inverse-gamma shape is
tuned—0.56 is a design setting and the PTB empirical best is around 1.4—so
the claimed form is not a parameter-free discovery. Runs compare language
perplexity, rare-word bins, and generalization to longer Dyck sequences;
uncertainty across data seeds is less central than controlled ablations.

**Relevant result.** Multi-timescale initialization reduces PTB perplexity by
about 1.60 and helps rare words disproportionately. On Dyck-2 it reaches
93.82% versus 91.66% and gains roughly 5–10 points for test lengths above 75.
A routing ablation shows longer-timescale units contribute more to rare-word
prediction. Markov-PTB favors a narrower Gaussian timescale distribution,
linking the useful prior to task statistics.

**Screen role and failure modes.** This supplies a mechanistic alternative to
a literal scale-free process: a heterogeneous finite mixture can generate an
aggregate power-law tail. It therefore motivates fitting exponential
mixtures and reporting timescale distributions before interpreting a slope.
The derivation does not transfer directly to transformers, nonlinear
population averages need not follow the free-input equation, and better
perplexity is not evidence that a downstream target needs order. It supports
a candidate model class for activation decay, not a screen pass or a
Spectral Cross-Coder advantage.

**Primary-source pointer.** Sections 2.1–2.3 and Equations 1–4, PDF pp. 3–4;
PTB/Markov results and Figure 1, PDF p. 5; Dyck result and Figure 2, PDF pp.
6–7; rare-word routing and Figure 3, PDF pp. 8–9.

### Lakretz et al. (2019), *The emergence of number and syntax units in LSTM language models*

**Question, object, and estimator.** This paper asks which LSTM components
causally support long-distance subject–verb number agreement. The observable
is agreement accuracy and the logit difference for correct versus incorrect
verb number under controlled attractor conditions. Each unit or selected
group is ablated and the performance change is compared with the intact
network and random-unit ablations. Tree depth is separately decoded by
five-fold ridge regression.

**Sampling, assumptions, and uncertainty.** The main model is a pretrained
two-layer 650-unit Wikipedia LSTM. Controlled lexical templates provide 600
sentences per condition, including singular/plural subjects, congruent and
incongruent intervening nouns, and long dependencies; a natural corpus
analysis complements them. The causal interpretation assumes zeroing a unit
does not create arbitrary off-manifold effects and that template lexicons
cover the intended agreement phenomenon. A footnote reports replication
across seeds/hyperparameters, but the exact sparse units can be seed-specific.

**Relevant result.** Ablating unit 776 or 988 selectively drives difficult
long-distance incongruent agreement conditions toward chance, with one unit
specialized for plural and the other for singular. A separate group of
short-range units carries a distributed signal that is overwritten by an
intervening noun. A syntax-tracking unit and associated gate connections help
explain how the sparse long-range circuit is controlled; tree-depth decoding
reaches \(R^2\approx .85\).

**Screen role and failure modes.** This is target-aligned and causal within
the studied LSTM task. It demonstrates why layer-mean correlation or median
unit half-life can miss the small subspace that matters. For the screen, any
activation relaxation analysis should include low-rank and sparse
target-aligned subspaces and, after screening, causal ablation or steering.
The result is architecture- and task-specific, and single-unit ablation can
overstate modularity when distributed backups exist. It supports the claim
that rare long-timescale directions can matter; it does not establish that
such directions exist in arbitrary transformer tasks or that TXC discovers
them.

**Primary-source pointer.** Experimental setup in Section 3, PDF pp. 3–4;
single-unit ablations in Section 4.1 and Table 2, PDF pp. 4–5; short-range and
syntax subnetworks in Sections 4.2–4.3 and Figures 2–4, PDF pp. 6–8.

### Khajehabdollahi et al. (2024), *Emergent mechanisms for long timescales depend on training curriculum and affect performance in memory tasks*

**Question, object, and estimator.** The paper asks whether long task memory
comes from long intrinsic neuron constants or recurrent population dynamics.
In a 500-unit leaky RNN, the trainable intrinsic constant \(\tau\) is compared
with an effective network constant \(\tau_{\mathrm{net}}\), estimated by
fitting one- or two-exponential curves to each neuron's autocorrelation and
selecting the slow component. Single-head curricula train one memory horizon;
multi-head curricula train nested horizons simultaneously. Causal ablations
of units binned by \(\tau\) are compared with random perturbations.

**Sampling, assumptions, and uncertainty.** Autocorrelations use
uncorrelated Bernoulli inputs of length \(10^5\), ten trials, nonlinear
least-squares fits, and AIC model selection; BIC agrees in over 95% of cases.
Four networks are trained for delayed-match-to-sample and parity tasks with
sequence lengths sampled from \([N+2,4N]\), up to \(N=101\). A GTX 2080 Ti
run takes up to three days per network. The estimator assumes stationary
responses to Bernoulli drive, exponential decay, and a slow fitted component
that reflects useful memory. Those assumptions need not hold in natural
language.

**Relevant result.** Single-head training succeeds to roughly \(N=90\) for
delayed match and \(N=35\) for parity by increasing intrinsic constants.
Multi-head training with fixed \(\tau=1\) reaches at least \(N=100\) by
developing longer effective population timescales. In both curricula the mean
and spread of \(\tau_{\mathrm{net}}\) grow with task horizon, but intrinsic
\(\tau\) grows only in single-head models. Ablations confirm that long
intrinsic-\(\tau\) neurons matter under single-head training while short
intrinsic-\(\tau\) neurons participate critically in the recurrent
multi-head solution.

**Screen role and direct implementation.** The result mandates both unitwise
and population/subspace relaxation. Inputs for the diagnostic are held-out
activation trajectories under controlled drive; outputs are fitted
timescales, subspace persistence, and target-performance effects after
ablation. In LM data, estimate autocorrelation within documents, stratify by
state, and compare it with paired-prefix subspace relaxation and
target-aligned prediction. A synthetic falsification should reproduce the
paper's multi-head regime: a unit-autocorrelation-only screen should fail
while population relaxation and the ordered target gate pass.

**Failure modes and allowed decision.** The tasks are deliberately simple,
multi-head curricula expose the entire nested task family, Bernoulli drive is
unlike language, and fitted exponentials can absorb oscillatory or
nonstationary dynamics. The evidence supports rejecting a neuron-only
timescale screen and including causal task tests. It cannot tell us which
natural-language task is temporal or which dictionary architecture will win.

**Primary-source pointer.** Model and estimator in Sections 2–3 and Equation
3, PDF pp. 2–4; curriculum/task results in Figures 4–5, PDF pp. 5–7; causal
ablation in Figure 7, PDF p. 8; limitations/discussion, PDF p. 9.

### Toneva and Wehbe (2019), *Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)*

**Question, object, and estimator.** The authors ask which LM layers and
context lengths best predict human brain responses during language
comprehension. ELMo, BERT, USE, and Transformer-XL representations computed
with different preceding-context lengths are mapped to fMRI voxels by nested
ridge regression; held-out brain-prediction and MEG classification are the
observables. Uniform-attention interventions test whether changing a model
layer in the direction favored by brain alignment transfers to NLP tasks.
The null is short/no context or an alternative model layer, not an
order-invariant history.

**Sampling, assumptions, and uncertainty.** Eight fMRI participants read a
45-minute *Harry Potter* passage under four-fold cross-validation; three MEG
participants provide a second modality. The analysis assumes cross-validated
linear encoding scores reflect shared representational content and that
adding words principally changes usable context. It does not isolate order
from more content, and subject count, one naturalistic narrative, spatial
autocorrelation, and model-family confounds limit generalization. The authors
also caution against reverse inference and direct score comparisons across
different encoding models.

**Relevant result.** Middle layers tend to predict brain responses best once
more than about 15 words are available, while the deepest layers are often
nonmonotonic; Transformer-XL improves more continuously with added context.
Making shallow BERT attention more uniform improves some brain-encoding
scores up to roughly 25 words and changes downstream syntactic behavior,
whereas the same intervention in later layers can hurt.

**Screen role and failure modes.** The study supports layer-by-layer window
sweeps and warns against assuming a monotone benefit from more context. Its
target is human neural activity, not a local NLP state, and context-length
effects mix history, model truncation, and representation quality. It can
motivate stratification by layer and horizon; it cannot establish a Temporal
Screen criterion or justify TXC on an NLP target.

**Primary-source pointer.** Data and nested encoding method in Sections 2–3,
PDF pp. 2–5; context-by-layer result in Figure 4, PDF p. 7; attention
intervention and Figure 6, PDF pp. 8–9; interpretive cautions in Section 2,
PDF p. 3.

### Chien and Honey (2020), *Constructing and Forgetting Temporal Context in the Human Cerebral Cortex*

**Question, object, and estimator.** This fMRI study separates construction
from forgetting of event context. In the construction condition, different
narratives converge on an identical sentence; in the forgetting condition,
the same narrative context is followed by different sentences. Inter-subject
spatial pattern correlation measures convergence or separation over time.
Fixed linear integrators and a hierarchical accumulation-to-bound model are
compared with a prediction-error-gated hierarchical model.

**Sampling, assumptions, and uncertainty.** Independent subjects hear matched
narratives and sentences, so participants rather than time points are the
dependence units. The analysis assumes inter-subject pattern correlation
tracks a shared context state and that hemodynamic smoothing is comparable
across regions. Stimulus matching limits lexical confounds, and control
analyses argue that hemodynamic delay alone cannot explain regional
differences of about eight seconds. The proposed gated model remains a
low-dimensional explanatory model with within-layer recurrence and local
prediction-error gates.

**Relevant result.** Sensory regions align quickly after the identical
sentence begins, while higher-order regions can retain different preceding
contexts for over ten seconds. After an explicit context switch, separation
can be much faster than that slow construction. A simple signal-gain account
and fixed linear integrators fail to explain the asymmetry; prediction-error
gating reproduces it within the tested model family.

**Screen role and failure modes.** The useful design is a two-sided
intervention: measure persistence within an event and reset after a
state-changing boundary. In an LM, compare paired-prefix convergence under an
identical continuation with divergence after matched overwrite or correction
events, using document/rollout grouping. Brain-region hierarchy and fMRI
times cannot be transferred numerically to transformers, and the model
comparison is not causal identification. This supports a boundary-reset
diagnostic; it cannot establish target relevance or a power-law mechanism.

**Primary-source pointer.** Experimental logic and stimuli in Sections 2–3,
PDF pp. 2–4; construction/forgetting results in Figures 2–5, PDF pp. 4–8;
model comparison and limitations in Discussion, PDF pp. 8–10.

### Regev et al. (2024), *Neural populations in the language network differ in the size of their temporal receptive windows*

**Question, object, and estimator.** The paper asks whether interleaved
electrode populations in human language cortex integrate over distinct
numbers of words. Intracranial electrodes are clustered from concatenated
responses to sentences, word lists, Jabberwocky, and nonwords. A toy temporal
receptive-window estimator convolves word onsets with truncated Gaussian
kernels spanning roughly one third to eight words, then chooses the width
whose predicted trajectory best correlates with each response profile. The
null is one homogeneous population or a continuous, unclustered distribution.

**Sampling, assumptions, and uncertainty.** Dataset 1 includes six patients
and 177 language-responsive electrodes; Dataset 2 includes 16 patients and
362 electrodes. \(k\)-medoids uses an elbow criterion for three clusters;
trial-half permutations and electrode subsampling test stability, and a
second dataset replicates the main profiles. The window estimator assumes a
fixed, symmetric Gaussian integration kernel, stimulus-locked response, and
commensurate word durations. Electrode coverage is clinical and uneven, and
patient/electrode nesting constrains population inference.

**Relevant result.** Three reproducible profiles emerge: a slow
sentence-building population, an intermediate population that plateaus after
roughly a few words, and a word-locked population. The fitted windows in the
replication are approximately 4.5, 2.1, and 1 word, respectively. The authors
show robustness across several kernels but explicitly leave open whether
timescales form clusters or a continuum and whether words or information
content are the right unit.

**Screen role and failure modes.** This argues for reporting a distribution
of unit and subspace horizons rather than a layer-wide exponent. For LM
adaptation, fit candidate kernels only after grouped held-out relaxation
curves are available and compare mixture and continuous models. The
biological response, fixed-kernel toy fit, correlated linguistic features,
short/common stimuli, and electrode sampling do not establish a transformer
mechanism. The paper supports heterogeneity analysis, not target screening.

**Primary-source pointer.** Datasets and clustering in Results/Methods, PDF
pp. 3–8 and pp. 20–23; temporal-window convolution and kernel sweep in
Methods, PDF pp. 25–27; robustness and qualifications in Discussion, PDF pp.
10–12.

### Lerner et al. (2011), *Topographic Mapping of a Hierarchy of Temporal Receptive Windows Using a Narrated Story*

**Question, object, and estimator.** The study asks at what structural scale
human cortical responses remain reliable. Participants hear a seven-minute
story intact, reversed, or scrambled at word, sentence, or paragraph scale.
The estimator is inter-subject Pearson correlation of BOLD time courses per
voxel; phase-randomized bootstraps provide a null for response reliability.
An independent story localizer defines regions of interest.

**Sampling, assumptions, and uncertainty.** Fifteen subjects participate,
with 11 assigned to each condition. The story contains 608 words, 69
sentences, and 11 paragraphs; mean segment durations are about 0.7, 7.7, and
38.1 seconds. Condition order is pseudorandomized. Inter-subject correlation
assumes a common time-locked response, and shuffling changes coherence,
predictability, boundary placement, and intelligibility together. Hemodynamic
blur prevents reliable interpretation of very short boundaries; the authors
exclude sentences under six seconds in one control.

**Relevant result.** Early auditory cortex is reliable under every
condition, while progressively higher regions require word-, sentence-,
paragraph-, or intact-story structure. Precuneus responses integrate on the
order of tens of seconds and medial prefrontal response is reliable mainly
for the intact narrative. Unscrambling controls preserve local acoustic
moments and support a gradual rather than perfectly discrete hierarchy.

**Screen role and failure modes.** The direct benchmark lesson is an
intervention ladder: within-phrase, within-sentence, across-sentence, and
across-event shuffles test different structural hypotheses. For LM tasks,
each perturbation must preserve the same label and content statistics or be
refit as a distinct input distribution, with documents as split/bootstrap
units. Neural reliability is generic, not task-aligned; scrambling changes
many properties and a receptive window does not identify function. The paper
supports hierarchical controls, not a numeric LM horizon or a TXC claim.

**Primary-source pointer.** Stimuli and analysis in Methods, PDF pp. 2–4;
hierarchy in Figures 3 and 5, PDF pp. 4–7; scrambling and hemodynamic controls
in Results/Discussion, PDF pp. 6–9.

## Benchmark design and shortcut resistance

### Ribeiro, Wu, Guestrin, and Singh (2020), *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*

**Question, object, and estimator.** CheckList asks whether a model exhibits
specified behaviors rather than merely obtaining a high aggregate test score.
Its object is a capability-by-test-type matrix. Minimum Functionality Tests
(MFTs) check simple expected behavior, INVariance tests (INVs) apply
label-preserving perturbations, and DIRectional expectation tests (DIRs)
require a predicted score to move in a specified direction. The estimator is
the failure rate over generated cases or perturbation pairs; the null is that
the behavior satisfies the declared oracle. There is no temporal process,
stationarity assumption, or direction-of-time estimand in this paper.

**Sampling, assumptions, and uncertainty.** Tests are constructed with
templates, lexicons, masked-LM suggestions, and hand-written examples for
sentiment, duplicate-question detection, and reading comprehension. The main
evidence is diagnostic rather than population inference: generated cases are
not IID samples, and confidence intervals are not the focus. A commercial
team brainstormed about 30 tests in one day. In a second study, 18
participants had two hours and were assigned unaided, capability-only, or
capability-plus-template interfaces. The latter group produced
\(13.5\pm3.4\) tests and \(198\pm96\) cases per test, versus
\(5.8\pm1.1\) tests and \(7.3\pm5.6\) cases unaided. The validity of every
failure depends on the human oracle and template coverage.

**Relevant result.** The matrix exposes severe failures in production and
research systems on elementary negation, robustness, temporal language,
semantic roles, and logic despite good benchmark scores. Templates greatly
increase case count, while capability prompts broaden what people test.
Because many generated examples share a template, the case count must not be
treated as the effective sample size.

**Temporal Screen role and implementation.** CheckList supplies the registry
form, not the temporal statistic. Rows should be preregistered mechanisms
(onset, overwrite, delayed consequence, accumulation, persistence, reset);
columns should include local MFTs, label-preserving content invariances,
order-changing directional pairs, and shortcut-failure slices. Inputs are a
task card, oracle, templates, and frozen predictors; outputs are per-cell
failure rates and examples. Group uncertainty by source
document/writer/rollout and template, and seal a task-level holdout. A
synthetic falsification is a template family where a lexical cue perfectly
predicts the label: ordinary MFT accuracy passes, while a counterbalanced
lexical INV must fail.

**Failure modes and allowed decision.** The matrix does not prevent users
from selecting favorable cells, writing weak or invalid oracles, or generating
near-duplicate cases. MFT/INV/DIR coverage is neither exhaustive nor a
population accuracy estimate. The evidence supports freezing a systematic
test inventory and reporting cellwise failures. It cannot show that a task is
temporal or that a TXC advantage generalizes.

**Primary-source pointer.** Framework and Figure 1 in Sections 2.1–2.2, PDF
pp. 2–4; model case studies in Section 3 and Tables 1–3, PDF pp. 4–8; user
studies in Sections 4.1–4.2 and Table 4, PDF pp. 8–9.

### Dehghani et al. (2021), *The Benchmark Lottery*

**Question, object, and estimator.** The paper asks how benchmark task choice,
aggregation, community convention, and repeated test access change judgments
about methods. The principal estimand is ranking stability under alternative
subsets of a benchmark suite. SuperGLUE scores for 55 models are
mean-aggregated over every subset of eight tasks, and unique top-\(k\)
rankings are counted for \(k\in\{1,3,5,10\}\). VTAB and RL Unplugged use
Kendall rank correlation between full-suite, category, and individual-task
rankings; LRA compares top-three identities across subsets. The null is that a
method ranking is stable to reasonable task selection and aggregation.

**Sampling, assumptions, and uncertainty.** The analysis reuses public
leaderboards, so models are a selected and historically test-adapted sample,
not IID draws. It assumes reported scores are comparable and treats the
chosen suite/task subset as the perturbation. There is no temporal object or
target label, and statistical uncertainty over new tasks is unavailable.
Community uptake and repeated holdout access make the benchmark explicitly
stateful.

**Relevant result.** Among the 70 ways to choose four of eight SuperGLUE
tasks, six different models can rank first and almost 60 subsets yield
different top-three or top-five orderings. Individual VTAB tasks have mean
Kendall correlation about 0.60 with the aggregate ranking, including negative
correlations; RL suite correlations average about 0.49–0.54. LRA's top three
also change frequently with task subset. These are direct demonstrations that
an apparently neutral task menu can select the winner.

**Temporal Screen role.** The screen must declare its task universe,
mechanism strata, aggregation rule, exclusion/stopping policy, compute and
hyperparameter budget, and every attempted task before confirmatory model
comparison. It should report task-level outcomes and leave-one-family-out
rankings rather than only a pooled score. Sealed task families and versioned
evaluation rounds reduce stateful overfitting. This is a governance procedure,
so its inputs are registry metadata and per-task scores and its output is a
sensitivity analysis, not a temporality scalar.

**Failure modes and allowed decision.** The case studies show fragility in
specific public suites but do not estimate the probability that a new task
reverses a ranking. Alternative aggregation cannot remove value judgments,
and a fixed suite can still be unrepresentative. The evidence supports
anti-cherry-picking, full disclosure, and ranking-sensitivity audits. It
cannot prescribe which temporal tasks belong in the universe or prove that
one architecture is generally better.

**Primary-source pointer.** Task-selection argument in Section 3, PDF pp.
6–10; SuperGLUE Figure 1, PDF p. 6; VTAB Figure 2 and LRA Table 1, PDF pp.
7–8; benchmark statefulness in Section 5, PDF pp. 11–13; guidelines in
Section 7.1, PDF pp. 14–17.

### Gardner et al. (2020), *Evaluating Models' Local Decision Boundaries via Contrast Sets*

**Question, object, and estimator.** Contrast Sets ask whether a model's local
decision boundary agrees with the intended task boundary near an existing
test example. An expert makes a small, meaningfully label-changing edit to a
pivot example. Evaluation reports ordinary performance on the contrast
instances and contrast consistency—the fraction of sets for which every pivot
and perturbation is correct. The null is local label invariance or reliance on
an IID artifact rather than the edited phenomenon. This is not a temporal
estimator, but ordered event relocation can instantiate the edit.

**Sampling, assumptions, and uncertainty.** Experts, often original dataset
authors, create up to about 1,000 examples for each of ten NLP datasets,
averaging one to five perturbations per pivot. Most edits take one to three
minutes; structured-output examples can take around 15. Models are trained
only on original training sets. At least 100 original/contrast pairs in four
datasets are independently answered by authors who did not create those
pairs. The design assumes expert edits stay in the intended local
neighborhood, labels are correct, and editor/style artifacts do not themselves
identify the new label. Pairs and pivots—not individual edits—are the
dependence units.

**Relevant result.** Near-SOTA models drop substantially: for example BoolQ
falls from 86.1 to 71.1 accuracy and MC-TACO from 38.0 to 14.0 exact match;
dependency-parse contrast consistency is only 17.3 despite 95.7 UAS on the
standard benchmark. Human performance changes much less on IMDb
(\(-0.4\)), PERSPECTRUM (\(-1.2\)), QUOREF (\(-6.8\)), and ROPES
(\(-3.0\)), arguing against mere noise as the whole explanation.

**Temporal Screen role and implementation.** Each task should include
content-matched pairs where changing event order, onset location, overwrite,
or delayed consequence changes the local label, plus label-preserving lexical
and writer edits. Inputs are pivots, edit specifications, and independent
labels; outputs are paired loss differences and set consistency. Split all
members of a pivot, writer, document, and template together. A synthetic
falsification is a pair whose label changes with order but whose editor inserts
a unique cue; a lexical-only baseline should reveal the invalid pair family.

**Failure modes and allowed decision.** The authors emphasize that contrast
sets have negative predictive power: failure falsifies a claimed capability,
but passing a finite local neighborhood does not confirm the full boundary.
Post-hoc or model-in-the-loop editing can bias examples toward one system,
automatic edits may not cross the intended boundary, and expert construction
is dataset-specific. The evidence supports paired temporal counterfactuals
and pair-grouped evaluation; it cannot by itself certify a task or mechanism.

**Primary-source pointer.** Definition and construction in Sections 2.2–2.5,
PDF pp. 3–6; ten-dataset protocol in Section 4.2, PDF p. 8; model results in
Table 2 and Section 4.3, PDF pp. 8–9; human control in Table 3 and Section
4.4, PDF p. 9.

### McCoy, Pavlick, and Linzen (2019), *Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference*

**Question, object, and estimator.** HANS asks whether high-scoring NLI models
use lexical-overlap, subsequence, or constituent heuristics. For each
heuristic, five templates make the heuristic agree with the gold entailment
label and five make it disagree. One thousand examples per template produce
10,000 examples per heuristic and 30,000 total. The estimator is accuracy in
the six heuristic-by-label slices; the null is balanced behavior across
supporting and contradicting cases rather than blanket entailment.

**Sampling, assumptions, and uncertainty.** A manually curated vocabulary
keeps every noun plausible as subject/object of the relevant verbs, and verbs
must appear at least 50 times in each required MNLI frame. Templates are
constructed so lexical-overlap cases are not subsequences and subsequence
cases are not constituents. DA, ESIM, SPINN, and BERT are trained on MNLI and
evaluated without HANS training; 95 human participants answer a subset.
Template families, not their 1,000 lexical instantiations, are the meaningful
generalization units. Generated-language plausibility and a two-label mapping
from MNLI's three labels remain assumptions.

**Relevant result.** All four models score well on MNLI and almost always
predict entailment in HANS. They are near perfect when a heuristic supports
entailment and below 10% in most non-entailment slices, despite 50% chance.
Augmenting MNLI with 30,000 HANS-like examples improves the exposed behavior,
but held-out template categories show incomplete transfer. Human average is
76% for Mechanical Turk participants and 90% for expert annotators, with a
different error pattern.

**Temporal Screen role and implementation.** Name the candidate shortcuts
before collection—anchor/last token, best offset, order-invariant bag, length,
absolute position, writer/source, and lexical cue—and construct slices where
each agrees and disagrees with the true temporal label. Inputs are a formal
shortcut rule and counterbalanced templates; outputs are per-slice
performance and transfer to held-out templates. Group by template family and
hold out lexical inventories. A synthetic falsification can make a bag cue
99% accurate in training but anti-correlated in the held-out slice.

**Failure modes and allowed decision.** HANS diagnoses only three named
heuristics, templated sentences differ from natural MNLI, and model failure
does not uniquely identify the causal heuristic. Training on the diagnostic
can teach its templates rather than the general rule. It supports
shortcut-agree/disagree stratification and family holdouts. It cannot
guarantee absence of unknown shortcuts or establish temporal dependence
without temporal counterfactuals.

**Primary-source pointer.** Heuristics in Section 2 and Table 1, PDF pp. 2–3;
construction/controls in Sections 3–3.1, PDF pp. 3–4; model results in Figure
1 and Section 5, PDF pp. 5–6; human control and augmentation in Sections 6–7,
PDF pp. 7–9.

### Kaushik and Lipton (2018), *How Much Reading Does Reading Comprehension Require?*

**Question, object, and estimator.** The paper asks whether reading
comprehension benchmarks actually require both passage and question. Existing
architectures are retrained on question-only, passage-only, or, for CBT, last
sentence inputs while keeping the task loss and answer space. The estimand is
the held-out performance retained after removing an ostensibly necessary
input component; the full-input model and majority/random baselines are the
comparisons. There is no temporal stochastic-process assumption: truncation
is a functional necessity test.

**Sampling, assumptions, and uncertainty.** The study covers bAbI, SQuAD,
CBT, CNN, and Who-did-What with KV-MemNet, Gated Attention Reader, and QANet
implementations. It uses each dataset's existing train/test split and metric.
The reduced input must not leak the omitted component through preprocessing or
candidate construction, and equal architecture quality across variants is
assumed. Dataset examples are the nominal units; shared stories/templates
mean group-aware uncertainty would be stricter than the original aggregate
scores.

**Relevant result.** Passage-only KV-MemNet exceeds 50% on 14 of 20 bAbI
tasks and matches the full model on several. On CBT, question-only models
approach or exceed full performance in multiple parts of speech, and the last
sentence often matches the complete passage. CNN and especially SQuAD fare
better under the audit: SQuAD Q-only and P-only F1 are 4.0 and 14.8 versus
79.1 full. The result shows that nominal input length does not imply usable
long-range dependence.

**Temporal Screen role and implementation.** Before an ordered model is
trained, every task needs anchor-only, metadata-only, truncation, best-offset,
and expressive order-invariant baselines. Inputs and outputs are the same task
examples and held-out metric, with one component removed or bounded. Match
capacity and tuning budget and split by source document/rollout. A synthetic
falsification is a long sequence whose final token encodes the label: the
anchor-only baseline must reject it despite strong long-window model scores.

**Failure modes and allowed decision.** A weak reduced-input model can create
a false necessity claim, while candidate-answer structure can make a strong
reduced baseline for accidental reasons. Low reduced-input performance does
not prove order rather than unordered content is necessary. The evidence
supports reduced-input rejection gates; it cannot by itself distinguish TXC
from an order-invariant history encoder.

**Primary-source pointer.** Datasets and corrupted-input design in Sections
2–3, PDF pp. 2–3; bAbI/CBT/CNN/SQuAD results in Tables 1–4, PDF pp. 4–5;
benchmark recommendations in Section 5, PDF pp. 5–6.

### Gururangan et al. (2018), *Annotation Artifacts in Natural Language Inference Data*

**Question, object, and estimator.** The authors ask whether an NLI label is
predictable from the crowd-written hypothesis without its premise. A fastText
bag-of-words/bigrams classifier is trained on hypotheses alone, and
word–label pointwise mutual information with add-100 smoothing identifies
lexical cues. Examples correctly classified by the premise-oblivious model
form an “Easy” split; the rest form a “Hard” split for full NLI models. The
null is majority-class performance and equal full-model performance across
Easy/Hard items.

**Sampling, assumptions, and uncertainty.** SNLI and MultiNLI supply their
official splits, including matched and mismatched MultiNLI genres. Logistic
regression and a premise-oblivious decomposable-attention model provide model
checks. The interpretation assumes hypothesis-only accuracy reflects
annotation artifacts rather than legitimate task priors, and the Easy/Hard
partition is model-relative. Examples share annotators, genres, and prompt
conventions, but the original analysis does not use annotator-grouped
uncertainty.

**Relevant result.** Hypothesis-only fastText reaches 67.0% on SNLI, 53.9%
on matched MultiNLI, and 52.3% on mismatched MultiNLI, far above majority.
Negation words cue contradiction, generic category words cue entailment, and
length distributions differ by class. Full NLI models score substantially
worse on the Hard partitions, showing that ordinary aggregate performance is
inflated by exploitable collection artifacts.

**Temporal Screen role and implementation.** Audit labels using lexical
window, length, position, template, writer, rollout source, and metadata-only
predictors before interpreting a window advantage. Cross-fit artifact models
with writers/templates/source documents held out, then report easy/hard
slices without selecting the final task on them. Pairwise order
counterfactuals remain necessary because a finite artifact inventory catches
only anticipated cues. A synthetic falsification is a task with writer IDs
correlated with label in random splits and independent in writer-held-out
splits.

**Failure modes and allowed decision.** Hypothesis-only cues do not make an
individual example invalid, the Hard split can contain remaining or new
artifacts, and filtering easy examples can distort the population. The
evidence supports reduced-input artifact audits and group splits. It cannot
prove that an unflagged task is clean or that any surviving information is
temporally ordered.

**Primary-source pointer.** Premise-oblivious model and Table 2 in Section 2,
PDF pp. 2–3; PMI/length artifacts in Section 3 and Figures/Tables 3–4, PDF pp.
3–4; Easy/Hard re-evaluation in Section 4 and Table 5, PDF pp. 4–5.

### Sinha et al. (2021), *UnNatural Language Inference*

**Question, object, and estimator.** The paper asks how often an NLI model
retains a correct label under severe word-order corruption. For each correctly
classified premise–hypothesis pair, the authors sample random permutations,
including derangements, and define permutation-acceptance quantities such as
the probability that at least one or a threshold fraction of permutations
retains the gold prediction. BLEU-\(n\), POS mini-tree overlap, entropy, and
human judgments quantify residual local structure. Random-label behavior and
out-of-domain datasets are comparisons.

**Sampling, assumptions, and uncertainty.** BERT, RoBERTa, ALBERT, DistilBERT,
and BART are trained on MNLI and evaluated on matched/mismatched MNLI, SNLI,
ANLI, and OCNLI. The number of possible permutations is approximated by a
finite sample, typically 100, and 200 permuted MNLI examples receive expert
human judgments. The intervention preserves token multiset but changes
grammar, meaning, local \(n\)-grams, and position distribution, so retained
model predictions do not imply semantic equivalence. Example-level
permutations are dependent and should be clustered by original pair.

**Relevant result.** For RoBERTa, at least one sampled permutation yields the
gold prediction for 98.7% of MNLI examples that the model originally gets
right, compared with original accuracy 90.6%; strong acceptance persists
under stricter thresholds and across models, but falls out of distribution.
Humans perform much worse than RoBERTa on the 200 permuted examples. A
maximum-entropy objective on randomized samples reduces acceptance with
little ordinary MNLI loss, showing the behavior is trainable rather than
inevitable.

**Temporal Screen role and implementation.** This is precedent for a
fixed-model order stress test, not an information-removal estimator. The
primary screen should retrain an order-invariant multiset model on its own
distribution and use natural, label-audited order counterfactuals; random
permutations can then stress-test the frozen ordered predictor. Record
permutation count, derangement/local-\(n\)-gram constraints, and cluster
uncertainty by source example. A falsification is a perfectly
order-invariant target under an order-sensitive model: fixed-model shuffling
may hurt from distribution shift even though a refit bag loses nothing.

**Failure modes and allowed decision.** Arbitrary shuffles are unnatural,
acceptance measures search over many lottery tickets, and local word patterns
can survive. The result diagnoses prediction invariance, not causal
insensitivity or retained label information. It supports including both
refit-order-removal and fixed-model perturbation controls and keeping them
conceptually separate. It cannot establish temporality from a shuffle gap
alone.

**Primary-source pointer.** Permutation definitions and Figure 1 in Sections
3–4, PDF pp. 4–6; main acceptance results in Tables 2–3 and Figures 2–3, PDF
pp. 6–8; syntax/human audits in Sections 6–7 and Table 4, PDF pp. 9–11;
entropy intervention in Section 8 and Table 5, PDF pp. 11–12.

### Miralles-González et al. (2025), *On the Locality Bias and Results in the Long Range Arena*

**Question, object, and estimator.** This audit asks whether LRA performance
actually requires long-range dependencies. A residual convolutional model is
trained with kernels \(K\in\{5,7,11,21,31,61\}\). With \(L\) layers, its
maximum dependency range is explicitly bounded by
\(L\lfloor K/2\rfloor\); accuracy as a function of that bound is compared
with MEGA. Transformer positional-encoding and training-procedure ablations
separate locality/position bias from optimization. The null is that a small
bounded receptive field cannot approach full-range state of the art.

**Sampling, assumptions, and uncertainty.** The authors reuse LRA's
CIFAR-10, Pathfinder, IMDB byte text, ACL retrieval, and ListOps splits,
excluding Path-X in some comparisons for compute. Models are trained on a
single RTX 3090 or 3080 Ti with published hyperparameters and source code at
[`pablomiralles22/paper-LRA-source`](https://github.com/pablomiralles22/paper-LRA-source).
Accuracy is largely point-estimated; random-seed uncertainty and
hyperparameter parity with every published baseline are limited. A bounded
convolution can compose local steps across layers, so the bound is the full
receptive field, not kernel width alone.

**Relevant result.** Kernel 61 (30 positions per side per layer) obtains
92.46% of MEGA on CIFAR-10, 75.28% on Pathfinder, 94.45% on text
classification, 97.58% on retrieval, and 83.51% on ListOps. More strikingly,
kernel 5 reaches 98.37% and 98.98% of MEGA on the two text tasks. With better
training and rotary embeddings, a vanilla Transformer reaches 85.69 average
versus MEGA's 86.25; removing those training techniques drops it to 64.98.

**Temporal Screen role and implementation.** This is a directly instantiable
bounded-history necessity audit. Train identical-capacity predictors with
strict pre-target windows \(T\), plot grouped held-out loss versus \(T\), and
compare saturation with the proposed TXC window. Inputs are task examples and
a preregistered window grid; outputs are performance/saturation curves and
cost. Use nested cross-fitting, group splits, matched tuning budgets, and a
sealed maximum window. A synthetic falsification should include one task with
a long sequence but a local sufficient statistic and one with an exact
delayed dependency; only the latter should have a long saturation horizon.

**Failure modes and allowed decision.** Strong local models can exploit
different shortcuts, optimization quality can dominate architectural
capacity, and near-SOTA is an application-dependent threshold. The preprint
is one benchmark-specific audit and leaves design of a genuine long-range
benchmark open. It supports rejecting tasks solved by bounded history and
requiring a window-saturation curve. It does not prove that residual
performance at large \(T\) is ordered rather than bag-like.

**Primary-source pointer.** Local-convolution design and Table 1, PDF pp.
2–3; LRA task definitions in Section 2.2, PDF pp. 3–4; training/position
ablations in Sections 4–7 and Tables 2–4, PDF pp. 6–8; bounded-range audit in
Section 8, PDF pp. 7–8; compute/code details in Appendix A, PDF pp. 10–12.

### Tay et al. (2021), *Long Range Arena: A Benchmark for Efficient Transformers*

**Question, object, and estimator.** LRA asks how efficient sequence
architectures trade task quality, speed, and memory on long inputs without
pretraining. It standardizes six tasks: 2K-token ListOps, 1K–4K-byte IMDB,
4K-byte document retrieval, 1,024-pixel CIFAR-10, 1,024-pixel Pathfinder, and
16,384-pixel Path-X. Accuracy, runtime, and memory are compared under shared
implementations; attention-span visualizations are supplementary. The
implicit null is that restricted/efficient attention can match a full
Transformer across diverse long-sequence tasks.

**Sampling, assumptions, and uncertainty.** Each task retains its own
train/dev/test distribution and metric. Most models use six layers,
fixed batch sizes/hyperparameter grids, and no pretraining, but the exact
training schedule differs by task; the authors openly note implementation
limitations. Selecting the best of several IMDB sequence lengths and tuning
architectures on the suite make scores stateful. Long sequence length is
treated as a proxy for long dependency, an assumption later challenged by
bounded receptive-field models.

**Relevant result.** No architecture wins every task. All tested models fail
to learn Path-X, image models show large train–test gaps, and the best
Pathfinder accuracy is about 77.05 in the reported initial suite. The
benchmark successfully standardized performance and resource comparisons,
but its results establish difficulty and efficiency, not that every task
requires distant ordered information.

**Temporal Screen role.** LRA is a useful warning and registry precedent:
record target definition, sequence length, declared required dependency,
performance, speed/memory, and exact training protocol separately. A temporal
benchmark must add a necessity certificate—bounded-history, bag, and
order-counterfactual controls—rather than infer temporality from input length.
Tasks and examples that share source datasets or generation rules must remain
in the same family split.

**Failure modes and allowed decision.** Suite aggregation hides task-specific
tradeoffs, training instability can look like inability, and long serialized
images/text contain strong local or positional cues. Attention visualization
does not establish causal dependence. LRA supports standardized multi-domain
evaluation and resource reporting; it cannot serve as evidence that a model
uses long temporal context unless the individual task passes additional
controls.

**Primary-source pointer.** Desiderata in Section 2.1, PDF pp. 3–4; task
definitions in Sections 2.2.1–2.2.6, PDF pp. 4–7; required-span analysis in
Section 2.3 and Figure 2, PDF p. 7; main performance in Section 3 and Tables
1–2, PDF pp. 8–11; implementation limitations and task hyperparameters in the
appendix, PDF pp. 13–16.

### van Miltenburg et al. (2021), *Preregistering NLP Research*

**Question, object, and estimator.** This position/methods paper asks how NLP
can separate exploratory from confirmatory analysis. Its object is a
timestamped preregistration specifying hypotheses, dependent variables,
conditions, exclusions, sample size, analyses, stopping rules, software, and
anticipated error analysis before outcomes are observed. It presents forms
for computational linguistic analysis, NLP engineering experiments,
resources, surveys, human evaluation, and error analysis. There is no
temporal estimator; the null is governance without outcome-contingent design
choices.

**Sampling, assumptions, and uncertainty.** The paper synthesizes open-science
practice rather than estimating an effect from a participant sample. It
assumes an honest timestamp, sufficiently precise plans, and transparent
deviation reporting. It explicitly allows exploratory work and recommends
labeling it rather than pretending every decision was confirmatory. NLP's
often-public datasets complicate “before seeing data,” so the appendix asks
authors to record prior data access and split exploration.

**Relevant result.** The contribution is a concrete question set: state the
main hypothesis and assumptions, how variables are measured, exact analyses,
libraries, data and sample-size decisions, error-analysis goals/categories,
and deviations. Registered reports move design review before results. The
authors identify three limitations: residual analytic flexibility, inability
to prevent fraud/multiple registrations, and imperfect fit for all
qualitative/theory work.

**Temporal Screen role and implementation.** Before confirmatory TXC runs,
commit task/split hashes, dependence units, local label definition,
availability time, nuisance variables, control models, window grid, metrics,
probe capacities, seeds, thresholds, exclusions, stopping rule, and
qualitative outcome predictions. Keep discovery and sealed confirmation task
families separate and append, never silently overwrite, deviations. The
output is a versioned protocol and decision log.

**Failure modes and allowed decision.** A vague preregistration can preserve
all researcher degrees of freedom, and timestamping cannot validate a bad
metric or prevent hidden parallel analyses. Preregistration reduces
cherry-picking risk but does not eliminate it. The evidence supports a frozen
screen contract and explicit exploratory/confirmatory labels; it cannot make
the chosen screen scientifically valid on its own.

**Primary-source pointer.** Rationale and workflow in Sections 1–2 and Table
1, PDF pp. 1–4; NLP-specific questions in Section 3 and Tables 2–3, PDF pp.
4–6; registered reports in Section 4, PDF pp. 6–7; limitations in Section 7,
PDF p. 9; detailed engineering/error-analysis forms in Appendix A, PDF pp.
10–11.

### Ye et al. (2021), *CrossFit: A Few-shot Learning Challenge for Cross-task Generalization in NLP*

**Question, object, and estimator.** CrossFit asks whether upstream learning
over seen NLP tasks improves few-shot performance on unseen tasks. It
standardizes all tasks as text-to-text, defines disjoint
\((\mathcal T_{\mathrm{train}},\mathcal T_{\mathrm{dev}},
\mathcal T_{\mathrm{test}})\) partitions, and reports average relative gain
(ARG) over direct fine-tuning. Partitions include random task splits,
classification/non-classification shifts, and held-out subfamilies such as
NLI and machine reading comprehension. The null is direct task-specific
few-shot fine-tuning with no upstream tasks.

**Sampling, assumptions, and uncertainty.** NLP Few-shot Gym contains 160
tasks. The random partition uses 120/20/20 train/dev/test tasks; controlled
partitions reserve ten development and ten test classification tasks, while
held-out-NLI includes eight NLI test tasks. Classification tasks provide 16
examples per class and other tasks 32 examples. Hyperparameters are tuned on
task-dev data. The ontology treats format/goal as task relatedness and some
source datasets or label rules may cross apparent task boundaries; the
authors later call this ontology imperfect.

**Relevant result.** Across partitions and multi-task/meta-learning methods,
51.47% of test tasks improve by more than 5% relative to direct fine-tuning,
35.93% change within \(\pm5\%\), and 12.60% degrade. Upstream task
selection materially changes transfer. Increasing each upstream task's data
up to eightfold changes aggregate gain by only about 4%, and no simple task
category consistently selects the best sources.

**Temporal Screen role and implementation.** Develop thresholds and probe
choices on a declared set of temporal mechanism families, tune them on
separate development families, and evaluate once on sealed families. Split at
the highest shared source—dataset, generator, writer, label rule, and
mechanism—not random task ID. Inputs are versioned task cards and family
metadata; outputs are pass/fail calibration, sensitivity, and architecture
ranking on unseen families. A falsification should clone one generator under
different task names; a proper family split keeps all clones together.

**Failure modes and allowed decision.** The text-to-text conversion changes
tasks, task ontologies are subjective, few-shot subsampling is noisy, and
relative gains depend on the base model and tuning budget. Cross-task transfer
is not temporal dependence. The evidence supports sealed task-family
holdouts and reporting negative transfer. It cannot specify the correct
Temporal Screen threshold or prove extrapolation beyond represented
mechanisms.

**Primary-source pointer.** Challenge definition in Sections 2–3, PDF pp.
2–4; 160-task Gym and partition construction in Section 4 and Figure 3, PDF
pp. 4–7; exact partitions and ARG in Table 1, PDF p. 8; transfer results and
ontology limitations in Section 6 and Figures 4–6, PDF pp. 9–13.

### Magar and Schwartz (2022), *Data Contamination: From Memorization to Exploitation*

**Question, object, and estimator.** The paper asks whether seeing labeled
downstream test examples during pretraining improves their post-finetuning
performance. Each downstream test set is randomly divided into “seen”
examples inserted, with gold labels, into an MLM pretraining corpus and
“unseen” examples withheld. Explicit memorization is the pretrained MLM
accuracy gap for masked labels; exploitation is the downstream
post-finetuning performance gap between seen and unseen halves. The null is
zero seen–unseen gap under otherwise exchangeable sampling.

**Sampling, assumptions, and uncertainty.** BERT-base/large are pretrained on
up to 600M Wikipedia tokens mixed with SST-5, SST-2, or SNLI examples, then
fine-tuned on 1,000 downstream training samples. Main pretraining uses one
epoch, batch 32, learning rate \(5\times10^{-5}\) with 10% warmup; ten
fine-tuning trials are averaged. Experiments vary contamination frequency,
model size, corpus position, learning-rate schedule, batch size, and seed.
The seen/unseen random split supports a causal contamination comparison in
this controlled corpus but may not generalize to web-scale causal LMs.

**Relevant result.** Memorization and exploitation generally grow with
repeated contamination and model size, but memorization does not guarantee
exploitation. Early contamination can yield high exploitation even when the
explicit masked-label memory measure is low; timing, learning rate, batch
size, and fine-tuning seed all matter. The authors therefore separate data
presence, explicit retrieval, and downstream use.

**Temporal Screen role and implementation.** Record dataset/task release
dates, exact duplicate and near-duplicate searches, benchmark appearances in
likely pretraining corpora, and whether labels appear near the examples.
Whenever possible, pair public examples with newly collected
counterfactuals and report performance separately. Inputs are corpus
provenance and duplicate indices; outputs are contamination risk and
seen/novel slices. For closed corpora this is an auditable risk assessment,
not a guaranteed clean split.

**Failure modes and allowed decision.** The explicit memory metric is a lower
bound, contamination in the controlled experiment is more direct than
natural web overlap, and random test halves can still differ. The small
pretraining regime and MLM objective limit scale/model transfer. The evidence
supports separating exposure from exploitation and documenting provenance.
It cannot infer that a particular LM has seen a closed-source task or that a
temporal effect is uncontaminated without direct evidence.

**Primary-source pointer.** Seen/unseen design and definitions in Section 2,
PDF pp. 2–3; frequency/model-size results in Figures 2–3, PDF pp. 3–4;
timing/batch/seed effects in Figures 4–5, PDF pp. 4–5; full hyperparameters in
Appendix D, PDF pp. 8–9.

### Kiela et al. (2021), *Dynabench: Rethinking Benchmarking in NLP*

**Question, object, and estimator.** Dynabench asks whether human-and-model
interaction can continually expose failures that static benchmarks miss.
Annotators create or edit examples while receiving target-model feedback;
independent humans validate labels and model errors. The validated model error
rate (vMER) is verified model-fooling examples divided by validated attempts,
and performance is tracked by collection round and target model. The null is
that later human-generated rounds add no verified failures beyond a static
test.

**Sampling, assumptions, and uncertainty.** The initial platform covers NLI,
QA, sentiment, and hate speech. NLI begins from three ANLI rounds with
BERT-Large then RoBERTa-Large ensembles; QA and hate-speech use their own
round-specific targets and mixtures of original/prior-round data. At the
snapshot reported in Table 1, NLI has four rounds and 170,294 examples and QA
two rounds and 36,406; vMER is 33.24% and 33.74%, respectively. Annotators
adapt to visible systems, examples are round- and model-dependent, and
independent validation controls label correctness but not natural-distribution
representativeness.

**Relevant result.** Later rounds remain capable of finding many verified
errors, while models trained on accumulated rounds improve on earlier
rounds. The paper explicitly observes increasingly adversarial content and
warns that dynamic rounds are not directly interchangeable with a stable
natural test distribution. It recommends evaluation across all rounds and
high-quality static sets, retaining earlier tests to detect forgetting.

**Temporal Screen role and implementation.** Use model-in-the-loop collection
only for exploratory shortcut discovery: show screen-negative candidates or
baseline errors to annotators, validate resulting order/onset/overwrite
pairs, record the target model and round, then freeze a complete discovery
round. Confirmatory evaluation must use a sealed, independently validated
set, stratified by round and collection mode, plus natural non-adversarial
data. Inputs are a target model, annotation interface, oracle, and validators;
outputs are versioned failure sets and vMER.

**Failure modes and allowed decision.** Annotators can overfit one model,
adversarial examples can be unnatural, dynamic scores change meaning across
rounds, older capabilities can be forgotten, and collection is expensive.
An ensemble helps only if its models are genuinely diverse. The evidence
supports iterative discovery and permanent regression tests; it cannot serve
as a single confirmatory leaderboard or prove a task family is naturally
representative.

**Primary-source pointer.** Platform loop and validation in Section 3, PDF
pp. 3–5; initial task rounds in Section 3.2, PDF pp. 5–8; vMER and Table 1 in
Section 3.3, PDF p. 8; distribution, comparability, overfitting, and cost
caveats in Section 4, PDF pp. 8–11.

## Cross-paper decision rules for the Temporal Screen

The full-text evidence supports five distinct gates. Their order matters
because each removes a different easier explanation:

1. **Registry and holdout gate.** Freeze the task universe, mechanism
   taxonomy, aggregation rule, probe/tuning budget, and task-family split
   before confirmatory architecture runs. Random task IDs are insufficient;
   shared generators, source datasets, writers, templates, and label rules
   define families.
2. **Reduced-input and bounded-history gate.** Reject a candidate when
   metadata, anchor-only, best-offset, truncation, or a short bounded
   receptive field recovers the claimed effect. Long sequences and long
   activation tails do not waive this gate.
3. **Order and local-boundary gate.** Compare a refit expressive
   order-invariant model with the nested ordered model, and audit natural
   multiset-matched label-changing pairs. Keep fixed-model shuffle/reversal
   results as distribution-shift-sensitive stress tests.
4. **Shortcut and contamination gate.** Report preregistered slices where
   every named shortcut agrees and disagrees with the label; hold out
   templates/writers/sources; document pretraining exposure and newly
   collected counterfactuals. Passing known shortcuts does not certify the
   absence of unknown ones.
5. **Activation diagnostic, after target passage.** Use paired-prefix
   relaxation, boundary resets, unit/subspace heterogeneity, and calibrated
   spectra to select layer and horizon and to propose mechanism. These
   diagnostics explain a task that has already passed; they never substitute
   for target-aligned held-out predictive gain.

The benchmark papers consistently support negative evidence more strongly
than positive certification. A failed contrast, reduced-input, or held-out
family test can disqualify a claim. Passing a finite audit means “no failure
was found under this frozen protocol,” not “the model learned the intended
mechanism.” That asymmetry should appear in both the preregistration and the
language used to report screen outcomes.
