# Temporal Screen contradiction and failure-mode matrix

**Recorded:** 2026-07-29

**Evidence base:** 58 deduplicated primary sources were audited. Fifty-seven
full texts were accessible; Robinson (1995) remains the only abstract-level
entry and is excluded from design claims. The paper-level cards are in:

- [language and scaling](language-and-scaling.md), 14 full texts;
- [information and spectra](information-and-spectra.md), 19 full texts plus
  one explicitly inaccessible source;
- [activations and benchmarks](activations-and-benchmarks.md), 24 full texts.

The matrix resolves apparent contradictions by distinguishing four claims:

1. the activation process has memory;
2. that memory predicts the declared target;
3. the prediction requires ordered, distributed history;
4. an unsupervised TXC learns and exposes that information better than its
   matched baselines.

Evidence for an earlier claim does not establish a later one.

## The central contradictions

| Attractive inference | Evidence that makes it tempting | Counterevidence or missing step | Screen consequence |
|---|---|---|---|
| A slow correlation tail means a task is temporal. | Cagnetta et al., Belletti et al., Lin and Tegmark, Dębowski and Wieczyński, and the FutureLens observation all report slow or multiscale dependence under particular observables. | Altmann et al. show that topical burstiness and interval structure can manufacture long memory. A label-independent topic process can give activations the same tail while the target depends only on the anchor. | Unconditional autocorrelation is a background diagnostic, never an eligibility gate. |
| More low-frequency power means order matters. | Long-memory processes concentrate power near zero frequency, and a stationary block covariance is approximately Fourier diagonal. | A window mean, count, persistent state, or bag statistic is low frequency but permutation invariant. Reversal preserves univariate power exactly. | Compare the ordered view with a strong full-multiset encoder; report non-DC target-aligned signal only after this comparison. |
| One fitted decay length gives the useful window. | Exponential fits supply an intuitive \(\xi\), while a fitted power law plus a sampling floor supplies a resolvable horizon. | Broken power laws, floors, cutoffs, regime mixtures, structural resets, and estimator noise can all move the fitted scale. Our earlier audit changes its selected family under centering and removal choices. | Estimate a held-out target-usefulness curve over a frozen \(T\) grid. Treat correlation fits as mechanism hypotheses and require their horizon to agree with predictive saturation. |
| The operator norm is the cleanest multivariate summary. | It gives the strongest singular mode and appears in the Cagnetta et al. token-covariance argument. | A single persistent direction can dominate it while the decaying bulk behaves differently; it also discards the rest of the spectrum. Our audit of the FutureLens estimator indicated this failure. | Report the signed matrix operator, its eigenvalue/effective-rank curve, and target alignment; no single matrix norm is primary. |
| A Frobenius-norm power law is an activation power spectrum. | The norm is positive and visually smooth, so Fourier language is tempting. | A norm of signed lag-covariance matrices is nonlinear, loses sign and phase, and need not be a positive-definite covariance sequence. | Construct a Hermitian lag-window spectral matrix from \(\Gamma(-k)=\Gamma(k)^\top\), then verify Hermiticity and nonnegative eigenvalues. |
| Averaging many sequences reveals the universal temporal law. | Ensemble averaging gives stable aggregate curves and underlies scaling-law derivations. | Sequence-specific persistent directions or event kernels can cancel in the mean. Conversely, random group effects can create an aggregate tail with no within-state memory. | Report a shared lag operator and a noise-corrected between-group deviation operator. A heterogeneous signal only motivates a common TXC if its subtype is observable before prediction. |
| A fixed reversal destroys order. | Performance often drops when an ordered model is evaluated on reversed inputs. | Reversal is bijective. A refit expressive model can learn to undo it, and a fixed-model drop also contains distribution shift. | Use fresh hidden per-example permutations for the information comparison. Keep fixed reversal as an orientation stress test with explicit caveats. |
| A shuffle result measures temporal information removed. | Shuffling breaks the slot sequence and is operationally simple. | A fixed permutation is learnable; contextual vectors already contain past and positional information; and different shuffle scales destroy different content statistics. | Refit on independently randomized order, audit position decodability, and use block, interval, and target-shift controls under separately named nulls. |
| More context use implies distributed temporal computation. | Long-context LM ablations show large likelihood gains from history. | Khandelwal et al. and Sun et al. show that nearby lexical overlap or one bounded context can recover much of the effect. A single named old offset can also beat a bag. | Include anchor-only, best-single-offset, bounded-history, and full-multiset baselines before crediting a distributed window. |
| A high-capacity probe estimates the information in a representation. | Flexible probes reduce held-out error and can approximate Bayes prediction. | Usable information is predictor-family relative; unrestricted conditional mutual information is not estimable here, and flexible observers can overfit or import a task-specific architecture. | Freeze a small nested probe family, use inner group folds for all tuning, and repeat with one declared nonlinear sensitivity family. |
| Conditional independence can be tested without modeling choices. | Conditional randomization and residual covariance tests have exact or asymptotic results under their assumptions. | Shah and Peters show that general conditional-independence testing is impossible without assumptions; conditional permutation requires a good \(X\mid Z\) model. | Describe a conditional usable-information estimate, not an observer-free independence certificate. Use multiple falsifiers rather than one omnibus \(p\)-value. |
| A target-aligned cross-correlation captures all temporal relevance. | The residual lag operator is signed, directed toward the label, and has a spectral decomposition. | XOR, variance changes, phase-amplitude coupling, and higher-order motifs can have zero linear cross-covariance. | Keep nonlinear ordered predictive gain as the umbrella gate. Add class-conditional covariance or kernel cross-spectral analysis only as a prespecified sensitivity. |
| Frequency-local dependence is causal. | Geweke measures, transfer entropy, phase-slope indices, and cross-spectra can be directional under specific models. | Common drivers, instantaneous mixing, receptive-field overlap, and nonstationarity can all produce direction-like statistics. | Use “predictive dependence” language. Causal claims require a content-preserving intervention on history and a measured behavioral change. |
| A task screen that predicts one TXC win explains the architecture. | Backtracking supplies a plausible positive example. | Selecting tasks after seeing TXC results is circular; benchmark papers show that task and implementation choices can reverse rankings. | Freeze an inventoried task panel, mechanism-family development split, untouched task holdout, thresholds, and architecture contrast before confirmatory training. |
| A positive raw-window screen predicts a TXC win. | A TXC can implement an arbitrary finite impulse-response filter before its nonlinearity. | Its dictionary is trained for reconstruction, not the target. High-variance nuisance modes can absorb capacity, while low-variance discriminative modes can be missed. | Call the screen an architecture opportunity estimate. Separately test whether learned codes cover the available target-discriminant subspace. |
| A Fourier TXC has a new spectral inductive bias. | A temporal filter has an exact Fourier representation. | A complete Fourier parameterization is only a basis change. | Claim a spectral bias only when band restrictions, hierarchical pooling, or frequency-dependent penalties change the hypothesis class or regularization. |

## Why the empirical power-law evidence is narrower than it looks

The public FutureLens experiment and our earlier corrective audit concern
unconditional GPT-2 residual trajectories on WikiText blocks, not task
rollouts. FutureLens compared a pure power regression with a pure exponential
on a de-persisted Frobenius-norm curve. The power model won that restricted
comparison. The corrective audit reproduced the restricted result, then added
floors, cutoffs, stretched exponentials, held-out persistent-subspace
estimation, signed directions, matrix spectra, article bootstraps, and three
centering choices. No layer-by-centering cell selected a pure power law.

The current defensible statement is:

> GPT-2 residual trajectories contain slow, heterogeneous, finite-range
> multiscale dependence with a persistent component over the measured range.
> A pure power law is not stable to broader model classes and centering
> choices.

This does not negate the Cagnetta et al. program. Their law is a corpus-level
coarse-grained description tied to a sampling floor and next-token
prediction. It says that a power-law approximation can organize an aggregate
learning horizon under declared assumptions. It does not require every layer,
task, sequence, or conditioned activation observable to have one exact
exponent.

## Rival screen families

| Family | What it estimates | Main advantage | Decisive failure | Decision |
|---|---|---|---|---|
| Scalar autocorrelation length or power-law exponent | Generic second-order memory | Cheap, interpretable, and close to the scaling-law motivation | Target-free, order-insensitive, rank-collapsing, and unstable under regimes and fit choices | Reject as a pass/fail screen |
| Unconditional multivariate spectrum | Frequency and rank of generic activation variation | Preserves multivariate modes and supplies a finite-data resolvability analysis | High-power modes may be task-irrelevant; phase or direction may still be omitted | Retain as the activation learnability axis |
| Conditional usable information | Held-out target risk reduction from ordered history beyond declared baselines | Directly target aligned and compatible with grouped cross-fitting | Probe-family relative and weak on mechanism | Primary eligibility gate |
| Target-residual lag operator and cross-spectrum | Signed, frequency-resolved linear relation between residual history and target | Connects correlation functions to a task and retains phase | Misses higher-order dependence and needs stationarity or event-local reinterpretation | Primary mechanistic module after or alongside the gate |
| Temporal interventions and relaxation | Behavior or activation change after controlled history perturbations | Strongest evidence for operational use and useful horizon | Expensive, task-specific, and often unable to preserve content exactly | Confirmatory mechanism test |
| Hybrid target-aligned multiscale screen | Eligibility, resolvability, mechanism, and architecture coverage | Separates “memory exists,” “target uses it,” and “dictionary can learn it” | More measurements and no universal scalar score | Recommended |

## Resolved design decisions

1. **The screen has two coequal axes.** The target axis measures whether ordered
   history predicts a local label beyond the anchor, the full multiset, and
   the best single offset. The activation axis measures whether the
   corresponding target-aligned mode is resolvable in the input covariance
   at the proposed dictionary capacity.
2. **Correlation functions remain central, but become conditional and
   matrix-valued.** The key object is a nuisance- and baseline-residualized
   target lag operator, not a scalar norm of unconditional autocovariance.
3. **Power laws become hypotheses, not assumptions.** Compare pure power,
   exponential, floor, cutoff, stretched, broken-scale, and regime-mixture
   explanations using grouped uncertainty and frozen fit ranges.
4. **Sparse events are analyzed in event time.** A pre-onset lag kernel or
   wavelet profile is primary; calling the Fourier transform of a few aligned
   events a stationary spectrum would be misleading.
5. **The screen can reject more strongly than it can certify.** A bag,
   best-offset, target-shift, shortcut, or held-out-family failure disqualifies
   a TXC-specific claim. Passing the frozen protocol means that no declared
   simpler explanation survived, not that the intended causal mechanism was
   proven.

## Required falsifiers

| Process | Why it fools a simpler screen | Required outcome |
|---|---|---|
| Label-independent \(1/f\) background plus anchor label | Strong generic power law | Reject ordered eligibility |
| Persistent topic or regime mixture | Long low-frequency tail without within-regime memory | Attribute to regime; reject unless within-regime target gain remains |
| Window count or mean | History and low-frequency gain without order | Ordered model ties full-multiset baseline |
| One informative named lag | Position matters but distributed history does not | Best-offset baseline closes the gap |
| Directed \(A\!\to\!B\) motif and its reversal | Same marginal content and power magnitude | Signed phase/order distinguishes them |
| Opposite lag kernels in two observable subtypes | Aggregate mean cancels | Heterogeneity is detected and subtype-gated predictor recovers signal |
| Opposite lag kernels in two hidden subtypes | Aggregate mean cancels but no deployable gate exists | Do not claim a common screen-positive mechanism |
| Ordered XOR | Linear lag operator is zero | Nonlinear usable-information sensitivity passes |
| Sparse pre-onset ramp | Global stationarity fails | Event-local ordered gain and lag profile pass |
| Matched processes differing only beyond \(T\) | Distant tail looks scientifically relevant | Every \(T\)-window method ties exactly |

## Novelty boundary

Conditional probing, generalized residual covariance, Geweke decomposition,
multitaper spectra, locally stationary spectra, wavelet coherence, kernel
cross-spectral operators, transfer entropy, and surrogate data are established
methods. The defensible contribution is their prospective combination:

> a grouped, task-local, target-residual correlation atlas that is frozen
> before dictionary training and tested for its ability to predict both the
> sign and the useful window of TXC-versus-baseline gaps on held-out task
> families.

Do not claim a new correlation estimator without an additional theorem. The
scientific novelty is the dictionary-architecture-independent screen, its
falsification protocol, and the prospective screen-to-architecture
validation. Every score remains conditional on the frozen LM, layer,
projection, target, and observer family.
