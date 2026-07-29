# Conditional information and spectral evidence

Full-text evidence cards for core-packet entries 10--12 and every paper in
Directions 2 and 3 of the annotated reading list.

**Audit status.** The assignment contains 20 unique primary sources. Entry 12
contains two distinct papers—the 2000 surrogate-data review and the 1996
IAAFT method—and both receive cards. Nineteen full texts were accessible and
were read in full. Robinson (1995) was not openly accessible through the
publisher, JSTOR, Project Euclid, arXiv, OpenAlex, or Unpaywall during this
audit, so its entry below is an access record rather than a full-text evidence
card and is not counted among the 19 audited sources. No abstract-only claim
is used to design the screen.

## Synthesis: what a nontrivial spectral screen would have to do

A scalar activation autocorrelation, fitted power-law exponent, or decay
length cannot be the Temporal Screen. It is label-free, collapses a
high-dimensional process to one number, generally preserves its magnitude
under time reversal, and can be produced by topic mixtures, rare regime
switches, smooth nonstationarity, or a mixture of finite timescales. The
papers below jointly imply a stricter division of labor:

1. **The gate must be target-aligned and conditional.** Estimate whether
   ordered pre-target history improves held-out prediction of the local target
   beyond the anchor, an expressive permutation-invariant representation of
   exactly the same history, and preregistered nuisance variables. Under log
   loss this is conditional usable information for a declared predictor
   family, not a distribution-free estimate of Shannon information. Watson
   and Wright, Berrett et al., Shah and Peters, and McAllester and Stratos
   explain why a learned conditional null, probe class, and finite-sample
   qualification are unavoidable.
2. **Spectra should resolve the passed signal, not replace the gate.** After a
   task passes, estimate a local cross-spectral or frequency-resolved
   predictive-dependence object between a frozen low-dimensional activation
   subspace and the target process. Thomson supplies a finite-sample spectral
   estimator; Geweke and Schreiber supply linear and nonlinear directional
   predictive quantities. These are observational prediction measures, not
   intervention-level causality.
3. **Preserve rank and heterogeneity.** A layer-average spectrum can hide a
   small temporally useful subspace. Estimate a regularized spectral matrix or
   its leading eigenvalue/rank curve after a label-blind projection, and
   report distributions over documents, rollouts, tasks, layers, and event
   phases. Sun et al. give one high-dimensional estimator, while Besserve et
   al. give a nonlinear cross-spectral alternative. Neither licenses choosing
   the projection or kernel after looking at the label.
4. **Treat nonstationarity as a competing explanation.** Dahlhaus's local
   spectrum and Grinsted et al.'s time-frequency coherence allow structure to
   vary around an onset. Diebold and Inoue show that rare switches can imitate
   long memory, while Veitch and Abry show how scale estimates depend on the
   selected octaves. A candidate “power law” should survive breakpoint,
   state-mixture, local-stationary, finite-mixture-of-exponentials, and
   within-state comparisons.
5. **Use a hierarchy of matched nulls.** Full shuffles, block shuffles,
   circular target shifts, reversal, and IAAFT surrogates remove different
   properties. Schreiber and Schmitz show that rejection applies only to the
   chosen null and that even an approximate surrogate can badly miscalibrate
   a nominal test. Residual marginal, spectrum, autocorrelation, group, and
   event-alignment mismatches must be reported for every control.

The resulting screen is deliberately two-stage. A task first passes a
grouped, nested, cross-fitted ordered-history log-loss test. Spectral analyses
then ask whether the passed gain is concentrated in a stable band, a local
pre-onset phase, a low-rank subspace, or a nonlinear phase relation. This is
more demanding than fitting a power law, but it also yields a falsifiable
architectural prediction: the TXC window should saturate near the
target-aligned predictive horizon, and a frequency-biased model should help
only when the corresponding band survives the conditional and surrogate
controls.

## Core spectral estimators and surrogate nulls

### Thomson (1982), *Spectrum Estimation and Harmonic Analysis*

**Question, object, and estimator.** Thomson asks how to estimate spectra,
lines, cross-spectra, and coherence from one short, finite record while
controlling the bias from leakage. For \(N\) contiguous, equally spaced
observations, the method forms \(K\) discrete-prolate-spheroidal-sequence
(DPSS) tapered Fourier transforms and averages their eigenspectra. With
time-bandwidth product \(NW\), approximately \(K=2NW\) tapers have their
spectral energy concentrated inside the chosen half-bandwidth \(W\). The
expected eigenspectrum is the true spectrum convolved with the taper's
spectral window; the leakage contribution is controlled by \(1-\lambda_k\),
where \(\lambda_k\) is the DPSS concentration eigenvalue.

**Assumptions, sampling, and uncertainty.** The derivations use a stationary
process and contiguous equal-rate sampling. Approximate Gaussianity gives
the chi-square and degrees-of-freedom calculations; the individual
eigenspectrum remains inconsistent, while their average has roughly \(4NW\)
degrees of freedom and becomes consistent under the stated asymptotics.
Cross-spectral examples assume that the two signals are meaningfully aligned
and warn that filtering, delay, and preprocessing affect coherence. The
paper's replicate/canonical-correlation construction is the relevant route
when independent trials exist. Language-model windows should never be
concatenated across rollout boundaries; spectra should be computed within
groups and uncertainty clustered over groups.

**Relevant result and limitations.** Multitapering reduces broadband leakage
without the resolution loss of one heavily smoothed periodogram. Thomson also
shows why highly colored spectra can correlate nominally distinct
eigencoefficients and why line and continuum components with different phase
can cancel in a coherence estimate. The paper repeatedly describes practical
extensions as approximate or heuristic rather than universally optimal.
Power and ordinary coherence remain second-order, stationary summaries and
cannot distinguish a trajectory from its reversal or establish causal
direction.

**Direct screen adaptation.** Inputs are one activation coordinate,
label-blind component, or paired activation/target series per document or
rollout; outputs are power, a spectral matrix, cross-spectrum, coherence, and
effective degrees of freedom. Preregister \(NW\), \(K\leq 2NW\), the frequency
grid, taper weighting, segment length, and event-relative pooling. The main
cost is \(K\) FFTs per sequence and component; DPSS and multitaper routines are
available in SciPy and MNE, although the primary paper predates a linked code
release. Bias is the local convolution plus residual leakage, and variance is
controlled by the number of tapers and independent groups. A required
falsification is a forward/reversed pair with identical power: the
unconditional estimate must agree while a genuine ordered target gate may
differ.

**Allowed decision and exact pointer.** This supports “second-order energy or
coherence is concentrated in these bands at this finite resolution.” It
cannot support “the task requires order,” “the spectrum is scale free,” or
“one process causes the other.” See Section III, especially Equations
(3.1)--(3.11), PDF pp. 1060--1062; Section IV on local and broadband bias,
pp. 1062--1065; and Section XIV on cross-spectra, coherence, colored spectra,
and replicated observations, pp. 1087--1090.

### Geweke (1982), *Measurement of Linear Dependence and Feedback Between Multiple Time Series*

**Question, object, and estimator.** Geweke decomposes linear dependence
between jointly stationary vector processes into instantaneous, forward, and
reverse feedback. The time-domain quantity
\(F_{Y\rightarrow X}=\log(|\Sigma_1|/|\Sigma_2|)\) compares the generalized
variance of one-step prediction errors for \(X\) using its own past with the
error after adding past \(Y\). A spectral factorization assigns this reduction
to frequencies; under the paper's invertibility conditions, the
nonnegative frequency-domain measure integrates to the time-domain feedback.
Unlike activation power, it is directional and aligned to a declared target
process.

**Assumptions, sampling, and uncertainty.** The processes are wide-sense
stationary, purely nondeterministic, and admit invertible autoregressive and
moving-average representations with spectral density bounded away from zero.
The proposed estimator truncates a finite VAR at lag \(p\), fits it by OLS,
and uses Gaussian independent innovations for likelihood-ratio and
chi-square inference. Lag order, omitted common drivers, contemporaneous
mixing, and a noninvertible representation can invalidate the decomposition;
the paper notes that without invertibility the frequency integral can be
strictly smaller than the time-domain measure. Finite-sample frequency-wise
inference is not solved by the asymptotic result and is identified as needing
Monte Carlo study.

**Relevant result and limitations.** In the empirical example, feedback is
concentrated in a narrow low-frequency region that a global time-domain test
can obscure. Coherence alone is insufficient because opposite directional
components can cancel or combine. The measure is linear predictive
dependence: an unobserved state, shared topic, or omitted lag can produce
apparent feedback, so “Granger feedback” here must not be renamed causal
influence.

**Direct screen adaptation.** Represent the activation history by a frozen
label-blind projection and the evolving target by a scalar or small vector;
fit nested VARs on training groups and score one-step predictive covariance
or held-out log loss on held-out groups. Outputs are total and
frequency-resolved activation-to-target predictive gain. Preregister lag
order selection, projection dimension, frequency smoothing, and nuisance
processes; report reverse-target and circular-shift controls. Compute is
ordinary VAR fitting plus spectral factorization, but parameter count grows
quadratically with the joint dimension and linearly with \(p\). A decisive
synthetic suite includes known unidirectional VAR coupling, bidirectional
coupling, an omitted common driver, and a nonlinear coupling invisible to a
linear VAR.

**Allowed decision and exact pointer.** This can say “past projected
activations add linear predictive information about an evolving target in
these frequency bands beyond the target's own past.” It cannot establish
intervention causality, nonlinear dependence, or order dependence for a
single rollout-level label. See the setup and assumptions on PDF pp. 1--2;
the feedback measures in Equations (2.3)--(2.6), pp. 3--4; the spectral
decomposition in Equations (3.1)--(3.6), pp. 4--6; estimation in Section 4,
pp. 6--7; and the narrow-band example and finite-sample caveat in Sections
5--6, pp. 8--10.

### Schreiber and Schmitz (2000), *Surrogate Time Series*

**Question, object, and null.** This review asks how to test a time-series
statistic against a clearly stated composite null when the null's nuisance
parameters are not known. It separates “typical-realization” surrogates,
which sample a fitted model, from constrained randomization, which preserves
selected properties of the observed record. Fourier, amplitude-adjusted
Fourier, iterative Fourier, multivariate, and annealed constrained surrogates
occupy different points in that space. A rank test with \(M=1/\alpha-1\)
surrogates is an exact one-sided level-\(\alpha\) test only when the data and
surrogates are exchangeable under the chosen null.

**Assumptions, sampling, and uncertainty.** The familiar Fourier family is
designed around a stationary Gaussian linear process, optionally observed
through a static invertible transform. The review shows that stationarity,
endpoint mismatch, finite-record spectral leakage, a non-Gaussian marginal,
cross-channel structure, and nonindependent phases can all invalidate an
apparently simple surrogate. Simulated annealing can constrain
autocorrelations, cross-correlations, endpoints, local mean/variance, or
other statistics, but the resulting ensemble and convergence need auditing.
For grouped LM data, surrogate generation must stay inside each document or
rollout and the final test statistic must be aggregated at that group level.

**Relevant result and limitations.** The review's main result is conceptual:
rejection only rejects the specified null. It does not prove deterministic
chaos, generic nonlinearity, direction, causality, or usefulness for a
downstream label. A phase-randomized record may also remove event alignment
or local nonstationarity in addition to nonlinear phase coupling. Conversely,
an overly constrained annealed surrogate may preserve the very structure the
test is meant to remove.

**Direct screen adaptation.** Inputs are a scalar or frozen low-dimensional
activation trajectory within each group; outputs are \(M\) matched surrogate
datasets, residual mismatch diagnostics, and the rank of a preregistered
target-aligned statistic. Material choices are the explicit null, \(M\),
Fourier versus block/annealed generator, convergence tolerance, number of
restarts, protected group/event boundaries, and one- versus two-sided test.
FFT surrogates cost \(O(MN\log N)\); annealing can be orders of magnitude
slower. The review points to the TISEAN implementation. Before use, synthetic
calibration must cover a Gaussian AR process, a static monotone transform of
one, a nonlinear phase-coupled process, a local-regime switch, and a
label-independent long-memory nuisance.

**Allowed decision and exact pointer.** This supports “the measured
target-aligned statistic is incompatible with this precisely described,
diagnostically matched surrogate null.” It cannot identify a unique
mechanism. See Section 3 on surrogate testing and rank logic, PDF pp. 4--6;
Section 4.3 on iterative Fourier methods and Section 4.6 on multivariate
surrogates, pp. 8--11; Section 5 on constrained randomization, pp. 12--16;
the nonstationary and multivariate examples in Section 6, pp. 16--20; and
the interpretation warnings in Sections 7--8, pp. 20--24.

### Schreiber and Schmitz (1996), *Improved Surrogate Data for Nonlinearity Tests*

**Question, object, and estimator.** This short paper introduces iterative
amplitude-adjusted Fourier-transform (IAAFT) surrogates for the null that an
observed scalar series is an invertible static transform of a stationary
Gaussian linear process. Starting from a random permutation of the observed
values, each iteration (i) replaces Fourier magnitudes with those of the
observed series while retaining current phases and then (ii) rank-orders the
result back onto the observed marginal values. The output approximately
matches both the observed periodogram and value distribution.

**Assumptions, finite-sample behavior, and result.** The method assumes one
stationary scalar record and a meaningful Fourier representation. Figure 1
shows that ordinary AAFT surrogates can be biased toward a whiter spectrum.
Figures 2--3 show iterative reduction of the spectral discrepancy. In the
paper's nonlinearity example, AAFT rejects 66% of nominally null realizations
at a nominal 5% level, while roughly seven IAAFT iterations restore
approximately calibrated behavior. This is direct evidence that an
unmeasured surrogate mismatch can dominate the test.

**Failure modes and direct adaptation.** Iterations introduce phase
correlations, convergence can stall, and exact preservation of both the
marginal and periodogram is not guaranteed. Windowing and endpoint artifacts
remain. For LM activations, generate surrogates per group and per
preregistered scalar projection, never by concatenating rollouts; protect
event boundaries only if that protection is part of the stated null. Inputs
are a length-\(N\) scalar series and iteration tolerance; outputs are an
ensemble plus marginal, periodogram, autocorrelation, and phase diagnostics.
Cost is repeated FFT plus sorting, approximately
\(O(MIN\log N)\) for \(M\) surrogates and \(I\) iterations. TISEAN contains a
reference implementation. A required falsification pairs an
amplitude-spectrum-only target, which should survive IAAFT, with a
phase/order-defined target, which should collapse if the null is matched.

**Allowed decision and exact pointer.** IAAFT can test whether a statistic
needs structure beyond an approximately matched marginal and linear
spectrum. It cannot show causality, direction, or target relevance on its own,
and reversal is a separate orientation test. See the null and AAFT failure on
PDF p. 1 and Figure 1; the two-step iteration and convergence in Figures
2--3, p. 2; the rejection-rate example in Figure 4, p. 3; and the phase,
windowing, and convergence caveats on pp. 3--4.

## Direction 2: target-aligned and conditional information

### Watson and Wright (2021), *Testing Conditional Independence in Supervised Learning Algorithms*

**Question, object, and estimator.** Conditional predictive impact (CPI)
tests whether features \(X_S\) improve a trained supervised predictor for
\(Y\) after controlling for remaining features \(X_R\). On held-out rows it
replaces \(X_S\) with a conditional knockoff
\(\tilde X_S\sim P(X_S\mid X_R)\), computes the paired loss difference
\(\Delta_i=L(\hat f,\tilde Z_i)-L(\hat f,Z_i)\), and tests whether its mean is
positive. The observable is therefore target-aligned predictive degradation,
not generic dependence in \(X\).

**Assumptions, sampling, and uncertainty.** Rows are treated as iid, the
predictor is fixed before testing, and the conditional sampler must preserve
the relationship between \(X_S\) and \(X_R\) while removing its extra
relationship with \(Y\). A paired \(t\)-test is proposed for large samples
and a Fisher/permutation test for small samples. For moment-limited losses the
tested null can be weaker than full conditional independence; log-likelihood
loss is needed to align the predictive distribution more closely with the
information-theoretic question. Overlapping activation windows violate iid
sampling, so folds and uncertainty must be grouped by rollout, document,
writer, or problem.

**Relevant result and failure modes.** The framework supports arbitrary
learners and losses and directly compares the same fitted predictor on real
and conditionally replaced inputs. Its validity is only as good as the
knockoff generator: a poor \(P(H\mid A,B,C)\) model can create apparent
importance by distribution shift. Because the predictor is not refit on the
knockoff distribution, CPI is also a reliance test for that trained model,
not automatically the information available to an optimally refit bag model.

**Direct screen adaptation.** Let \(X_S\) be ordered history, \(X_R\) the
anchor, bag, and nuisance variables, and \(Y\) a local label. Train both the
probe and conditional generator strictly within outer training folds, score
paired held-out log losses, and bootstrap groups. Inputs are grouped
activation windows and labels; outputs are a paired loss distribution and
confidence interval. Compute is probe training plus one or more conditional
resamples per example. Hyperparameters include the knockoff model, resample
count, loss, folds, and one-sided threshold. Use exact synthetic conditional
samplers before learned ones, and falsify with a target depending only on the
anchor while history remains strongly correlated through a nuisance state.
The paper supplies Algorithm 1; no primary-source software release was
required for the audit.

**Allowed decision and exact pointer.** A calibrated CPI can say that this
fixed predictor relies on history beyond the declared controls. It cannot,
without a trustworthy conditional sampler and grouped calibration, certify
conditional independence or causal necessity. See the knockoff construction
and CPI definition in Section 3.1, Equations (1)--(3), PDF pp. 5--7; the
large- and small-sample tests in Section 3.2, pp. 7--8; and Algorithm 1 and
the distinction between predictive and conditional-independence nulls,
pp. 8--10.

### Berrett, Wang, Barber, and Samworth (2020), *The Conditional Permutation Test for Independence While Controlling for Confounders*

**Question, object, and estimator.** The conditional permutation test (CPT)
tests \(X\perp Y\mid Z\) when the conditional law \(Q(\cdot\mid Z)\) of \(X\)
is known. Instead of a uniform row permutation, it samples permutation
\(\pi\) with probability proportional to
\(\prod_i q(X_{\pi(i)}\mid Z_i)\), which preserves compatibility between
permuted \(X\) and each row's \(Z\). Any preregistered statistic may then
compare the observed and conditional-permutation datasets.

**Assumptions, sampling, and uncertainty.** The theorem assumes iid triples
\((X_i,Y_i,Z_i)\), a correct known conditional density \(q(X\mid Z)\), and
exchangeable Monte Carlo draws from the nonuniform permutation law. Pair-swap
Markov chains provide a practical sampler. Theorem 1 gives finite-sample
type-I control under the exact law. Theorems 4--5 bound inflation under
misspecification using total-variation or KL discrepancy between the true and
fitted conditional laws, but those discrepancies are usually unknown in
high-dimensional activation space.

**Relevant result and failure modes.** CPT formalizes why a global shuffle is
the wrong exact null when history covaries with position, topic, anchor, or
length. It also makes the hard part explicit: natural-language
\(P(H\mid A,B,C)\) is not known, and an apparently plausible generator can
invalidate the nominal \(p\)-value. The Markov chain must mix, and overlapping
windows do not supply iid rows.

**Direct screen adaptation.** Use CPT only in synthetic tasks with a known
conditional generator, or after a separately calibrated low-dimensional
conditional model. Inputs are group-independent windows, controls, labels,
and \(q\); outputs are a Monte Carlo null distribution and rank \(p\)-value.
Material choices are the test statistic, number of permutations, swap-chain
length and thinning, conditional density, and group blocking. Compute is
roughly the test-statistic cost times the number of sampled permutations,
plus density evaluation and MCMC. In real LM data, report it as a
falsification control rather than an exact test unless conditional calibration
is demonstrated. Synthetic falsification should vary the KL error of \(q\)
and confirm the theoretical type-I inflation.

**Allowed decision and exact pointer.** Under its exact assumptions, CPT can
reject conditional independence while preserving declared confounder
relationships. It cannot survive an unvalidated conditional generator or
establish intervention causality. See Equation (4) and Section 3 for the
conditional permutation law, PDF pp. 6--8; Theorem 1 for exact validity,
pp. 8--9; Algorithms 1--2 for pairwise MCMC, pp. 10--12; and Theorems 4--5
on model misspecification, pp. 14--16.

### Runge (2018), *Conditional Independence Testing Based on a Nearest-Neighbor Estimator of Conditional Mutual Information*

**Question, object, and estimator.** Runge combines the Frenzel--Pompe
nearest-neighbor conditional-mutual-information estimator with a local
permutation null. A shared maximum-norm radius around each \((X,Y,Z)\) sample
defines neighbor counts in the required marginal spaces. Under
\(X\perp Y\mid Z\), \(X\) is reassigned among observations with nearby \(Z\),
approximately preserving \(X\)--\(Z\) dependence while removing the
conditional \(X\)--\(Y\) relation.

**Assumptions, sampling, and uncertainty.** The estimator expects iid samples
from a smooth continuous distribution and enough local neighbors in the
conditioning space. Its material hyperparameters are the CMI neighbor count
\(k_{\mathrm{CMI}}\), permutation-neighborhood size
\(k_{\mathrm{perm}}\), metric, and number \(B\) of surrogates. Very small
\(k_{\mathrm{perm}}\) can leave residual dependence or make assignments
degenerate; a large value approaches a global shuffle and breaks
\(X\)--\(Z\). Dimensionality rapidly depletes local samples. Time-series lag
vectors additionally need separation or group blocking so temporally adjacent
rows are not treated as independent.

**Relevant result and direct adaptation.** Simulations show better calibration
and power than global shuffling in nonlinear conditional-dependence examples,
but performance depends materially on neighborhood choices and sample size.
Raw \(dT\)-dimensional transformer windows are outside the credible regime.
For the screen, freeze a label-blind projection on training groups, estimate
CMI only in a small projected space, conduct the local permutations within
appropriate strata, and aggregate uncertainty over held-out groups. Inputs
are projected \(H\), \(Y\), and controls \(Z\); outputs are estimated CMI and
a surrogate rank. Complexity is dominated by nearest-neighbor search plus
\(B\) local assignment problems. The method is available in Tigramite's
CMIknn implementation.

**Allowed decision and exact pointer.** This is a nonlinear sensitivity check
for residual dependence after low-dimensional conditioning. It cannot be a
model-free high-dimensional gate, cannot establish causality, and can mistake
projection or local-null error for dependence. Calibrate it on a zero-CMI
nonlinear process with strong \(X\)--\(Z\) and \(Y\)--\(Z\) associations, then
on a matched conditional alternative. See the CMI estimator and local
permutation algorithm in Section 2, Equations (1)--(4) and Algorithm 1, PDF
pp. 3--5; sensitivity to \(k_{\mathrm{perm}}\) in Section 3 and Figures 1--2,
pp. 5--8; and the limitations and Tigramite reference in Section 5, p. 10.

### Goldfeld and Greenewald (2021), *Sliced Mutual Information: A Scalable Measure of Statistical Dependence*

**Question, object, and estimator.** Sliced mutual information (SMI) averages
ordinary scalar MI over independent random one-dimensional projections of
high-dimensional \(X\) and \(Y\). Conditional SMI additionally projects
\(Z\) and conditions each scalar-slice estimate on that projected control.
The Monte Carlo estimator averages \(m\) random projection pairs and applies
a one-dimensional MI estimator to each slice.

**Assumptions, sampling, and uncertainty.** The theory uses iid observations,
random directions independent of the data, bounded or regular sliced
densities, and an error-controlled scalar MI estimator. Theorem 1 separates a
Monte Carlo projection term of order \(m^{-1/2}\) from scalar-estimator error
\(\delta(n)\). Smoothness or log-concavity assumptions give the stated sample
rates. Group dependence and adaptive projection selection are absent from
the analysis and need outer-fold handling for LM activations.

**Relevant result and limitations.** Proposition 1 shows nonnegativity,
zero iff independence, and an upper bound by ordinary MI. The paper also
emphasizes that SMI is a distinct dependence measure: the MI--SMI gap can be
unbounded, and SMI can increase under deterministic processing, so it does
not obey the usual data-processing inequality. Conditional SMI with a
projected \(Z\) is not the same as full
\(I(X;Y\mid Z)\). Learned, label-selected projections would further
invalidate the clean random-slice interpretation.

**Direct screen adaptation.** Inputs are frozen, whitened activation windows,
local labels, and controls; outputs are SMI or conditional-SMI distributions
over random slices. Preregister \(m\), the scalar MI estimator and its
neighbors/bins, whitening, projection dimension, and random seed; average
within group and bootstrap groups. Compute is \(m\) projection passes and
scalar MI estimates, far cheaper than full high-dimensional density
estimation. Algorithm 1 supplies pseudocode, but no external software was
needed for this audit. Falsify with dependence confined to one weak random
direction, where finite \(m\) should reveal low power, and with a target
dependence removed only by conditioning on the full rather than projected
\(Z\).

**Allowed decision and exact pointer.** SMI can triangulate whether
high-dimensional dependence appears across fixed random projections. It
cannot quantify ordinary CMI in bits, select a temporal horizon alone, or
show direction or causality. See Definitions 1--2 and Equation (4), PDF
pp. 3--5; Proposition 1 and Remarks 2--4, pp. 5--7; estimator Equation (6)
and Theorem 1, pp. 8--10; the processing counterexample in Proposition 4,
pp. 11--12; and Algorithm 1 in Appendix B, p. 19.

### Schreiber (2000), *Measuring Information Transfer*

**Question, object, and estimator.** Transfer entropy asks whether the past of
a source process \(J\) improves prediction of the next state of target
process \(I\) beyond \(I\)'s own \(k\)-step past:
\[
T_{J\rightarrow I}
=\sum p(i_{n+1},i_n^{(k)},j_n^{(\ell)})
\log\frac{p(i_{n+1}\mid i_n^{(k)},j_n^{(\ell)})}
{p(i_{n+1}\mid i_n^{(k)})}.
\]
It is a directional conditional mutual information and vanishes under the
corresponding generalized Markov condition. A known common driver can be
included in the conditioning set.

**Assumptions, sampling, and uncertainty.** The paper uses stationary Markov
approximations of orders \(k\) and \(\ell\), with finite-state counts or
continuous-space coarse graining/kernel correlation integrals. Markov order,
bin width or kernel radius, and sample size determine bias. The paper often
sets \(\ell=1\) for tractability. A \(q=2\) correlation-integral approximation
is computationally useful but loses the exact KL form and therefore the
nonnegativity guarantee. Overlapping lag tuples and trials need grouped
resampling in an LM setting.

**Relevant result and direct adaptation.** Coupled tent-map simulations show
directional transfer where ordinary symmetric mutual information does not
identify direction; a heart/breathing example illustrates real data use.
The simulation uses ten realizations of 100,000 iterations, which also
exposes how sample-hungry direct density estimation is. For evolving labels
such as backtracking pressure or local misalignment state, replace plug-in
density estimation with nested cross-fitted predictive log loss:
compare \(q(Y_{t+1}\mid Y_{\leq t},C)\) with
\(q(Y_{t+1}\mid Y_{\leq t},H_t,C)\). Inputs are grouped trajectories; outputs
are directional held-out gain versus lag and history length. Preregister
Markov order, probe class, folds, and nuisance histories.

**Allowed decision and exact pointer.** Transfer entropy supports
“activation history predicts an evolving target beyond the target's own
persistence.” It does not prove intervention causality; hidden common causes,
state-estimation error, and conditioning-set choice remain. It is also
ill-posed for a single rollout-level label without a local target process.
See the generalized Markov condition and Equation (3), PDF p. 2; transfer
entropy Equation (4), p. 2; conditioning on a common driver and continuous
estimators, pp. 2--3; and the tent-map and physiological examples in Figures
1--2, pp. 3--4.

### Papapetrou and Kugiumtzis (2013), *Markov Chain Order Estimation with Conditional Mutual Information*

**Question, object, and estimator.** For a finite-alphabet Markov chain, the
paper tests successive lags using
\(I_c(m)=I(X_t;X_{t-m}\mid X_{t-1},\ldots,X_{t-m+1})\). The largest
significant \(m\) estimates the chain order. Plug-in entropy/count estimates
are compared with an analytic bias correction and with a surrogate test
based on globally randomized sequences.

**Assumptions, sampling, and uncertainty.** The process is stationary,
discrete, and finite-order Markov. The number of contingency-table cells
grows exponentially with alphabet size and \(m\), so sparse counts create a
positive CMI bias. The proposed test uses \(M=1000\) randomized sequences and
\(\alpha=.05\). Those global randomizations do not preserve a lower-order
Markov null; the discussion explicitly proposes order-adjusted surrogates as
future work. Simulations treat the chain realization as the sampling unit,
not overlapping LM windows.

**Relevant result and limitations.** For binary chains with \(N=500\) and
true order five, the method reports about 94% correct selection in the stated
setting, but performance deteriorates sharply as alphabet size and order
increase. The analytic bias approximation is not an exact significance
bound. DNA examples require records as long as \(10^5\), underscoring the
sample burden.

**Direct screen adaptation.** Retain the incremental-horizon idea but replace
discrete plug-in CMI with nested cross-fitted probes:
\(\delta_m\) is the held-out log-loss gain from adding lag \(m\) after all
more recent lags, anchor, bag, and nuisances. Inputs are grouped activation
windows and targets; outputs are \(\delta_m\), a cumulative saturation curve,
and group-bootstrap intervals. Preregister the lag order, nesting,
capacity-matched optional residual branch, and multiplicity correction.
Compute is one nested model per lag and fold. Synthetic falsification should
include a true finite-order process, a long-memory nuisance with an
anchor-only target, and redundant correlated lags that make attribution
order-dependent.

**Allowed decision and exact pointer.** This can estimate a probe-relative
minimal useful horizon. It cannot recover a representation-independent
Markov order or causal delay. See the CMI order statistic in Section 2,
Equations (3)--(5), PDF pp. 3--4; the analytic and surrogate tests in Section
3, pp. 4--6; simulation Tables 1--4, pp. 7--10; and the global-surrogate and
sample-size limitations in Section 5, pp. 11--12.

### Shah and Peters (2020), *The Hardness of Conditional Independence Testing and the Generalised Covariance Measure*

**Question and impossibility result.** The paper asks whether finite-sample
conditional independence can be tested without structural assumptions. Its
central result is negative: over unrestricted absolutely continuous
distributions, any test with valid level against the conditional-independence
null has no power above its level against any alternative. Bounded support
does not remove the obstruction. This rules out advertising any activation
screen as a universal, model-free CMI test.

**Constructive estimator and assumptions.** The generalized covariance
measure (GCM) regresses \(X\) and \(Y\) separately on \(Z\), then tests the
mean product of residuals. Under the null, asymptotic normality requires the
product of nuisance regression mean-squared errors to vanish fast enough
(\(A_fA_g=o_p(n^{-1})\)), plus control of weighted errors. A sample-split
version avoids some dependence between nuisance fits and test residuals.
Rows remain iid. The multivariate extension uses a maximum over residual
cross-products with a high-dimensional Gaussian approximation.

**Relevant limitation.** GCM is powerful for alternatives with nonzero
conditional covariance after nuisance regression. It can miss nonlinear
conditional dependence with zero residual covariance. Misspecified nuisance
regressions, overlapping windows, adaptive feature selection, and
non-grouped splits break the claimed calibration.

**Screen consequence and falsification.** The Temporal Screen must declare a
finite probe family, split groups before every nuisance fit, and publish
matched synthetic type-I and power curves. GCM can be one linear residual
sensitivity check, but the primary claim should be conditional usable
information, not universal independence. A necessary synthetic pair has
identical zero conditional covariance in both cases but nonlinear dependence
in one, demonstrating the GCM blind spot.

**Allowed decision and exact pointer.** This paper supports the methodological
decision that assumptions and task-specific power are part of the screen
specification. It does not support interpreting a GCM nonrejection as
conditional independence. See Theorem 2 and its bounded-support corollary in
Section 2, PDF pp. 4--6; the GCM statistic in Equations (2)--(3), p. 6;
Theorem 6 and nuisance-rate conditions, pp. 8--10; and the sample-split and
multivariate extensions in Theorems 8--9, pp. 11--13.

### McAllester and Stratos (2020), *Formal Limitations on the Measurement of Mutual Information*

**Question and theorem.** The paper asks how large a distribution-free,
high-confidence lower bound on mutual information can be when only \(N\) iid
samples are available. Theorem 1.1 shows that, for a broad class of
data-dependent bounds, the reported value is with high probability at most
\(2\ln N+5\), regardless of the true MI. A related KL result saturates at
\(\ln N\) even when one marginal density is known. These are worst-case
limits, but they directly constrain headline “bits of temporal information”
claims from a neural estimator.

**Alternative estimator and assumptions.** The difference-of-entropies
(DoE) estimator fits unconditional and conditional density models and
subtracts their held-out cross-entropies. This is neither a guaranteed upper
nor lower MI bound. For a fixed bounded log-loss model, ordinary held-out
concentration applies; selecting architectures or hyperparameters on the
same test set removes that guarantee.

**Relevant result and limitations.** In a 128-dimensional synthetic Gaussian
experiment, DoE tracks the known MI better than several variational lower
bounds, which saturate or become unstable. Training uses 3,000 optimization
steps with minibatches of 128, and hyperparameters are tuned with access to
the known answer. Shuffled pairs produce near-zero estimates, but this is a
sanity check rather than a universal guarantee.

**Direct screen adaptation.** Report the paired held-out log-loss difference
between a nested ordered predictor and its anchor-plus-bag submodel. Call it
conditional usable information for that probe class; do not call it an
absolute Shannon-information estimate. Split and bootstrap documents or
rollouts, freeze all model choices before the outer test fold, and publish
raw per-group loss differences. Inputs are windows, labels, and controls;
outputs are a loss-gap curve with confidence intervals. Compute is two
nested predictors per fold and window size. Falsify with shuffled labels,
anchor-only targets, and a high-MI construction whose lower-bound estimator
is known to saturate.

**Allowed decision and exact pointer.** This supports held-out predictive gain
as an operational, probe-relative quantity and cautions against absolute-bit
claims. It cannot establish conditional independence, order, or causality
without the corresponding baselines and interventions. See Theorem 1.1 on
PDF pp. 1--2; the KL limitation in Theorem 3.1, pp. 4--5; DoE Equations
(14)--(17) and Theorem 5.1, pp. 6--7; and Figure 2 and Tables 1--2,
pp. 8--9.

## Direction 3: spectra, phase, direction, and nonstationarity

### Robinson (1995), *Gaussian Semiparametric Estimation of Long Range Dependence* — inaccessible full text

The full paper could not be obtained from an open primary-source endpoint
during this audit. The verified bibliographic record is *The Annals of
Statistics* 23(5), 1630--1661, DOI
[10.1214/aos/1176324317](https://doi.org/10.1214/aos/1176324317). The
publisher abstract describes a local Gaussian semiparametric likelihood
based on low frequencies, with consistency, asymptotic normality, efficiency,
finite-sample Monte Carlo, and a Nile-data application; it also says
Gaussianity is not needed asymptotically.

Because the full assumptions, bandwidth conditions, theorem statements, and
finite-sample results were not audited, this source contributes no design
claim to the Temporal Screen ledger. The reading-list annotation that local
Whittle estimates should receive a bandwidth-stability curve remains a
proposal to verify after lawful full-text access, not an evidence-backed
conclusion of this card.

### Veitch and Abry (1999), *A Wavelet-Based Joint Estimator of the Parameters of Long-Range Dependence*

**Question, object, and estimator.** For a second-order stationary
long-range-dependent process, the paper estimates the memory parameter and
scale from discrete-wavelet-transform coefficients. Wavelets with \(M\)
vanishing moments suppress low-order polynomial trends; at octave \(j\), the
sample mean of squared coefficients estimates scale-dependent energy. A
bias-corrected log-scale diagram regresses \(\log_2\) energy against octave,
with slope determining the memory parameter and intercept determining scale.

**Assumptions, sampling, and uncertainty.** The asymptotic model is
second-order stationary with a power-law singularity near zero frequency and
a sufficiently regular short-memory factor. Correct octave selection is
essential: at least three scaling octaves are needed, fine scales contain
short-memory contamination, and coarse scales contain very few coefficients
plus boundary effects. Approximate coefficient decorrelation motivates the
weighted regression covariance. The paper treats one contiguous record; LM
adaptation must estimate within groups and combine group-level slopes rather
than concatenate them.

**Relevant result and limitations.** The DWT gives an \(O(N)\) estimator,
near-decorrelated multiscale coefficients, and robustness to polynomial
trends. The paper's conclusion identifies automatic scale-range selection,
time-varying long-range dependence, and stationarity testing as unresolved.
Level shifts and regime changes can still create a false scaling region.

**Direct screen adaptation.** Inputs are scalar frozen activation projections
per sufficiently long group; outputs are per-group octave-energy curves,
memory slopes, fit residuals, and cross-group heterogeneity. Preregister
wavelet family, \(M\), boundary rule, minimum coefficient count, and all
candidate octave ranges; compare with a frequency-domain estimator and a
breakpoint/mixture model. Compute is linear in sequence length. The paper
provides the estimator but no linked modern software; standard wavelet
libraries implement the DWT. Synthetic falsification should contrast true
fractional noise, a finite mixture of exponentials, a level shift, and rare
regime switching at matched apparent slope.

**Allowed decision and exact pointer.** This can support a scale-stability
statement for second-order memory in a scalar projection. It cannot show
target relevance, order, direction, or causality, and a layer-average slope
cannot exclude a sparse useful subspace. See the wavelet assumptions and
scale behavior in Sections II--III, PDF pp. 1--4; the energy estimator and
weighted log-scale regression in Equations (4)--(10), pp. 3--6; statistical
properties and octave selection in Sections IV--V, pp. 6--13; limitations in
Section VI, pp. 14--16; and the implementation recipe in Section VII,
pp. 16--17.

### Dahlhaus (2012), *Locally Stationary Processes*

**Question, object, and estimator.** Dahlhaus formalizes a nonstationary
triangular array \(X_{t,T}\) that is locally approximated around rescaled time
\(u=t/T\) by a stationary process \(\widetilde X_t(u)\). Its evolutionary
spectrum \(f(u,\lambda)\) and local covariance vary smoothly with sequence
position. Local periodograms, tapered segments, kernels, and local Whittle
likelihoods estimate this time-varying object.

**Assumptions, sampling, and uncertainty.** The core theory requires smooth
variation in \(u\), regular transfer functions, growing sequence length, and
a window bandwidth that shrinks in relative time while containing more
observations. For a local covariance estimate, Theorem 2.1 exhibits the
tradeoff: nonstationarity bias scales with the squared local bandwidth while
variance scales inversely with observations in the window. The framework is
not an automatic model for abrupt reasoning-state switches; those require
piecewise segmentation or a change-point model. Independent replicated
rollouts are still the uncertainty units.

**Relevant result and limitations.** The paper gives a unique local spectrum
under its representation and extends the framework to multivariate,
wavelet, stationarity-test, and local-long-memory settings. It explains why
one spectrum from concatenated reasoning traces is scientifically
ill-defined: early, pre-onset, and post-onset phases may have different
second-order structure. The price is reduced frequency resolution in short
local windows and smoothing across precisely the transition of interest.

**Direct screen adaptation.** Align groups at a preregistered local event,
freeze a low-dimensional activation projection, estimate multitaper or local
Whittle spectra in sliding/event bins, and aggregate estimates across groups.
Inputs are event-aligned grouped trajectories; outputs are
\(f(u,\omega)\), local spectral rank, and target cross-spectra with
group-bootstrap intervals. Hyperparameters are temporal bandwidth, taper,
frequency smoothing, projection dimension, and boundary handling. Compute is
one spectral estimate per local bin and group. Synthetic calibration must
include smoothly drifting spectra, an abrupt switch, a stationary long-memory
process, and a phase-aligned event with no spectral change.

**Allowed decision and exact pointer.** This can say that second-order
timescales change around a local phase or event under a local-stationarity
model. It cannot show that the change predicts the target, that the event
causes it, or that a power law generated it. See the triangular-array and
local approximation in Section 2, Equations (3)--(5), PDF pp. 3--6; local
covariance bias and variance in Theorem 2.1, pp. 7--9; the general linear
definition and unique spectrum in Section 4 and Theorem 4.3, pp. 15--21; and
multivariate, wavelet, stationarity, and local-long-memory extensions in
Sections 7.1--7.3 and 7.10, pp. 43--49 and 62--64.

### Grinsted, Moore, and Jevrejeva (2004), *Application of the Cross Wavelet Transform and Wavelet Coherence to Geophysical Time Series*

**Question, object, and estimator.** The paper localizes shared energy and
phase between two nonstationary series. A complex Morlet continuous wavelet
transform gives time-scale coefficients; the cross-wavelet transform
highlights common power, while wavelet coherence normalizes a smoothed
cross-spectrum by local auto-power. Phase arrows summarize local relative
phase. The paper uses \(\omega_0=6\), approximately ten scales per octave,
and Monte Carlo AR(1) red-noise surrogates.

**Assumptions, sampling, and uncertainty.** A lead-lag reading requires a
coherent narrowband relationship, stable phase, and an unambiguous mapping
from phase angle to delay. Smoothing in time and scale is part of the
estimator and materially changes significance. The cone of influence marks
edge-contaminated regions. Approximately 1,000 surrogate pairs calibrate
pointwise contours, but the many time-scale cells and phase arrows are not
independent; the paper warns against scattergun significance searches.

**Relevant result and limitations.** The method reveals transient shared
bands that a global correlation misses. Figure 4 also shows that smoothing
choices can create boundaries and change significant regions. The authors
explicitly state that a significant wavelet association is not reliable
evidence of causality and recommend a mechanistic model. An AR(1) null is too
simple for hierarchical language data.

**Direct screen adaptation.** Inputs are groupwise activation and evolving
target trajectories; outputs are event-relative coherence, phase, and
cluster-corrected significant regions. Replace the AR(1) null with
within-group circular target shifts, block surrogates, and matched
regime-switch controls; protect the cone of influence and control
multiplicity over time, scale, layer, and component. Compute is a CWT for each
group and surrogate. Reference Matlab code is linked from the paper's
companion site. Falsify with two series sharing a common amplitude envelope
but no lagged coupling and with a known delayed narrowband signal.

**Allowed decision and exact pointer.** This can localize a stable
second-order lead-lag pattern around an event. It cannot establish causal
flow, broad-band delay, or task usefulness. See Morlet CWT Equations (1)--(2)
and the cone of influence, PDF pp. 562--563; cross-wavelet Equation (5),
p. 564; coherence and smoothing Equations (8)--(10), pp. 564--565; the
surrogate procedure and scale choices, pp. 565--566; and Figure 4 plus the
causality and multiple-testing warnings, pp. 568--569.

### Diebold and Inoue (2001), *Long Memory and Regime Switching*

**Question, object, and estimators.** The paper asks when rare structural
breaks or persistent regime switching are observationally confounded with
fractional long memory. It studies a mean-plus-noise process with
sample-size-dependent break probability, a stochastic permanent-break
(STOPBREAK) model, and a Markov-switching model whose staying probabilities
approach one. Their partial-sum variances can scale like those of an
\(I(d)\) process even though the mechanism is regime change.

**Assumptions, sampling, and uncertainty.** The equivalence is asymptotic
under explicit sequences of shrinking break probabilities or increasingly
persistent transition probabilities. The Monte Carlo section uses 10,000
replications and the Geweke--Porter-Hudak low-frequency log-periodogram
estimator with the rule \(m=\sqrt T\). This is not a claim that every finite
regime process is literally fractionally integrated; it is a demonstration
of weak finite-sample identification from standard low-frequency diagnostics.

**Relevant result and limitations.** Propositions 1--3 derive fractional-like
partial-sum scaling for the three break/switch constructions. Simulated
log-periodograms appear linear at low frequencies and estimated \(d\) can
look like genuine long memory; with few breaks, the estimate may be bimodal
depending on whether a realization happened to contain a switch. Hence a
corpus-level power law can summarize event heterogeneity rather than a
scale-free within-state process.

**Direct screen adaptation.** Every long-memory candidate should be compared
with preregistered breakpoint, hidden-state, and finite-mixture-of-exponential
competitors. Fit within stable phases and within groups, report switch-count
heterogeneity, and test whether the target-aligned gain remains after
conditioning on inferred state. Inputs are grouped scalar/subspace
trajectories; outputs are competing predictive likelihoods, residual spectra,
and within-state decay estimates. Compute is modest for breakpoint and HMM
fits relative to dictionary training. Synthetic falsification should match
the observed low-frequency slope across true fractional noise, rare breaks,
and persistent switching, then ask whether the proposed screen distinguishes
them.

**Allowed decision and exact pointer.** This supports treating
regime-switching as a mandatory rival explanation for slow decay. It cannot
by itself choose between mechanisms in the observed LM data. See the
mixture-break construction and Proposition 1 in Section 3.1, working-paper
PDF pp. 6--9; STOPBREAK and Proposition 2, pp. 9--12; Markov switching and
Proposition 3, pp. 12--16; the 10,000-replication GPH experiments in Section
4, pp. 17--23; and the identification conclusion in Section 5, pp. 23--25.

### Nolte et al. (2008), *Robustly Estimating the Flow Direction of Information in Complex Physical Systems*

**Question, object, and estimator.** For two linearly coupled processes with a
delay, the cross-spectrum's phase is approximately linear in frequency. The
phase-slope index (PSI) sums the imaginary part of products of adjacent
complex coherencies over a selected band; its sign estimates which process
leads. A leave-one-epoch-out jackknife estimates its standard deviation, and
the normalized statistic is compared with \(|\mathrm{PSI}|>2\).

**Assumptions, sampling, and uncertainty.** PSI assumes a sufficiently
coherent delayed linear relation, consistent delay over the selected band,
adequate frequency resolution, and independent epochs for the jackknife. It
is insensitive to arbitrary instantaneous mixtures of independent sources
because those create real, zero-phase coherency under the model. The
simulation uses AR(5) sources, 60,000 samples, 1,000 random systems, and
several noise levels. The EEG application uses four-second epochs,
two-second Hanning segments, 50% overlap, and 0.5-Hz resolution.

**Relevant result and limitations.** PSI's simulated false-direction rate is
about 6% in the worst reported condition and 3.5% at finer resolution,
whereas a compared Granger method approaches 50% under instantaneous mixing.
The EEG analysis also exposes false peaks at filter/band edges. Nonlinear
coupling, frequency-dependent delays, hidden common dynamics, finite mixtures,
or weak coherence can defeat PSI; a lead is still not intervention causality.

**Direct screen adaptation.** Apply PSI only after a task passes the target
gate and only to an evolving target/proxy with meaningful spectral phase.
Inputs are multiple independent activation/target epochs; outputs are
bandwise direction and epoch-jackknife uncertainty. Preregister component,
band, resolution, epoch length, taper, and sign convention; use
circular-shift and instantaneous-mixture controls. Compute is cross-spectral
estimation plus an epoch jackknife. The paper provides equations rather than
a linked package. Falsify with a delayed linear VAR, an instantaneous mixture,
a nonlinear delay, and a band-edge artifact.

**Allowed decision and exact pointer.** PSI can support “a stable linear
lead-lag relation has this orientation under the stated band model.” It
cannot support generic information flow, causal onset prediction, or a
rollout-level label. See the phase model and Equations (1)--(2), PDF p. 1;
PSI and jackknife Equations (3)--(6), pp. 1--2; simulation design and false
rates, pp. 2--3; and EEG parameters and band-edge artifacts in Figures 3--4,
pp. 3--5.

### Sun, Li, Kuceyeski, and Basu (2018), *Large Spectral Density Matrix Estimation by Thresholding*

**Question, object, and estimator.** The paper estimates the \(p\times p\)
spectral density matrix of a high-dimensional weakly stationary process.
It averages raw periodograms across a frequency span \(m\), then applies an
entrywise hard, soft, or adaptive-lasso threshold \(\lambda\). A spectral
stability norm controls temporal dependence in the nonasymptotic error.

**Assumptions, sampling, and uncertainty.** The theory assumes weak
stationarity, tails and dependence controlled strongly enough for
concentration, approximate entrywise sparsity of the spectral matrix in the
observed coordinate basis, and \(\log p/n\rightarrow0\). Raw periodograms are
asymptotically unbiased but inconsistent, so \(m\) trades variance against
frequency smoothing bias. Entrywise thresholding need not preserve positive
semidefiniteness in finite samples. The proposed frequency-domain
sample-splitting selector treats separated periodograms as only
asymptotically independent.

**Relevant result and limitations.** Simulations over sparse VAR/VMA systems
with \(p=12\)--96 and \(n=100\)--600 show improved operator/Frobenius error
and edge recovery under the designed sparse truth, but recall can remain low.
Transformer neurons are not known to make an entrywise sparse spectral
matrix; selecting a rotation after seeing the target can manufacture a clean
low-rank or sparse story.

**Direct screen adaptation.** Freeze a label-blind PCA or random projection
on training groups, then estimate and threshold the projected spectral
matrix. Prefer reporting eigenvalue/rank and subspace-stability curves over
native-neuron edges. Inputs are groupwise \(r\)-dimensional trajectories;
outputs are regularized spectral matrices, leading eigenvalues, effective
rank, and uncertainty across groups. Hyperparameters are \(r,m,\lambda\),
threshold type, frequency grid, and PSD repair. Compute is FFT plus
\(O(p^2)\) matrices per frequency. Algorithms and simulation settings are in
the paper; no audited repository is required. Falsify with dense low-rank,
sparse full-rank, and rotated-sparse processes.

**Allowed decision and exact pointer.** This can support “the second-order
dependence occupies a stable multivariate subspace or sparse graph in this
frozen basis.” It cannot establish target alignment, order, direction,
causality, or a scale-free mechanism. See the spectral matrix and stability
definitions in Section 2.1, Equations (2.1)--(2.4), PDF pp. 3--5; averaged
periodogram and thresholds in Equations (2.5)--(2.8), pp. 5--7; consistency
theorems in Section 3, pp. 8--13; frequency-split tuning in Algorithm 1,
pp. 14--15; and VAR/VMA simulations in Section 4, pp. 16--19.

### Besserve, Logothetis, and Schölkopf (2013), *Statistical Analysis of Coupled Time Series with Kernel Cross-Spectral Density Operators*

**Question, object, and estimator.** The paper maps two stationary time
series into reproducing-kernel Hilbert spaces and Fourier-transforms their
lagged cross-covariance operators. The resulting kernel cross-spectral
density (KCSD) can detect frequency-specific nonlinear dependence invisible
to linear cross-correlation. With characteristic kernels, Theorem 1 relates
zero KCSD norm at every frequency to pairwise independence across all lags.

**Assumptions, sampling, and uncertainty.** The estimator assumes jointly
stationary series, bounded kernels, summable lag covariances/cumulants, and
appropriate tapering. Theorem 5 gives asymptotic unbiasedness of a tapered
kernel periodogram. Squaring its norm introduces an autospectral-trace bias
(Theorem 6); independent repeated trials allow the unbiased cross-trial
inner product in Theorem 7 and Corollary 8. Kernel bandwidth and lag window
govern power, and one long realization is weaker than multiple independent
trials.

**Relevant result and limitations.** RBF kernels detect simulated
phase--amplitude coupling, and a string kernel handles a symbolic Markov
example. Theorem 1 concerns pairwise lagged independence, not full process
independence. Proposition 2 upgrades it only under strong causal-graph
conditions—no confounding plus Markov and faithfulness assumptions. KCSD is
non-directional, stationary, kernel-sensitive, and expensive: forming Gram
matrices is \(O(N^2)\) before FFT/lag aggregation.

**Direct screen adaptation.** Treat independent rollouts as trials, freeze
activation projections and kernels without label access, and use a target
kernel appropriate to binary, ordinal, or continuous local labels. Outputs
are frequency-resolved KCSD norms with cross-trial unbiased estimates.
Preregister kernels, bandwidths, taper, lag truncation, projection, and trial
aggregation. The primary paper provides formulas and experiments but no
audited linked package. Falsify with independent series sharing marginals, a
linear lagged relation, pure nonlinear phase--amplitude coupling, an omitted
common driver, and a nonstationary switch.

**Allowed decision and exact pointer.** KCSD can support “these stationary
processes have nonlinear lagged dependence concentrated in these bands under
this kernel.” It cannot establish direction, intervention causality,
target-useful order, or a practical v0 screen at raw transformer dimension.
See the operator construction and Theorem 1 in Sections 2--3, PDF pp. 2--4;
the graphical-model qualification in Proposition 2, pp. 4--5; the tapered
estimator and Theorem 5, pp. 5--6; squared-norm bias and cross-trial correction
in Theorems 6--7 and Corollary 8, pp. 6--7; and nonlinear and symbolic
experiments in Section 5, pp. 7--9.

## Audit conclusion

The 19 accessible full texts converge on a narrow claim. Generic activation
memory is scientifically interesting but cannot select TXC tasks. The
screen's primary pass/fail statistic should be grouped, nested, cross-fitted
target-prediction gain for ordered history beyond anchor and bag. A spectral
module becomes informative only after that gate and only if it preserves
event locality, multivariate rank, group heterogeneity, and matched nulls.

The minimum defensible spectral follow-up is:

1. multitaper auto- and cross-spectra on a frozen label-blind subspace;
2. event-local spectra or coherence rather than concatenated global FFTs;
3. a spectral-matrix eigenvalue/effective-rank curve rather than one layer
   exponent;
4. frequency-resolved predictive dependence for an evolving local target;
5. IAAFT, circular-shift, reversal, block-shuffle, and regime-switching
   competitors, each named by the property it destroys;
6. group-bootstrap uncertainty and synthetic calibration for both type-I
   error and power.

Even that package establishes observational target-aligned temporal
dependence, not causality. Causal language requires an intervention on the
history that preserves anchor, content, position, and other admissible
controls while changing only the relevant temporal relation, followed by a
measured target or behavior change.
