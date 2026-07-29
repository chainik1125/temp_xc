# A target-aligned multiscale correlation screen for temporal dictionaries

**Recorded:** 2026-07-29

**Status:** research proposal after a 58-source full-text audit; no new
architecture result is claimed

**Working name:** Target-Aligned Multiscale Correlation screen (TAMC)

## Decision

The correlation-function idea should remain central, but the object must be
more informative than one decay exponent or one correlation length. The
proposed screen measures:

1. whether ordered history predicts a declared local target beyond the anchor,
   the complete unordered history, and the best single offset;
2. which signed lag modes carry that incremental target signal;
3. whether the modes are shared or heterogeneous across sequences;
4. whether those modes lie in stable raw-covariance directions under a linear
   proxy and are actually covered by a trained dictionary's held-out codes.

The first item is the eligibility gate. The next three form the correlation
atlas that explains the mechanism and makes prospective architecture
predictions. This preserves the Cagnetta--Ganguli scaling-law motivation while
closing the gap between “the activation process has memory” and “this task
requires an ordered temporal representation.”

The screen is independent of the dictionary architecture being evaluated
during task selection, but it is not representation-independent: every result
is conditional on the frozen LM, layer, activation projection, target, and
observer family. It predicts an opportunity for an order-sensitive window
representation; it does not uniquely select a TXC or guarantee that sparse
reconstruction training finds the target-relevant direction.

## Why one decay length failed

A scalar decay summary collapses four distinctions that matter:

- **shared versus sequence-specific structure:** persistent directions can
  rotate or change sign across sequences and cancel in an ensemble average;
- **target signal versus nuisance memory:** topic, style, position, and
  repeated content can dominate the activation spectrum without predicting
  the task label;
- **DC versus order:** a mean, count, or persistent state has low-frequency
  power but does not require slot order;
- **magnitude versus direction:** power and matrix norms discard the phase
  that distinguishes \(A\!\to\!B\) from \(B\!\to\!A\).

The earlier FutureLens observation illustrates the estimator problem. A
de-persisted Frobenius-norm curve preferred a pure power law to a pure
exponential. Our prior corrective audit, separate from the 58-paper
literature audit, was consistent with a narrower conclusion once floors,
cutoffs, stretched exponentials, signed directions, centering choices, and
grouped uncertainty were included: slow, heterogeneous, finite-range
multiscale dependence with a persistent component, rather than one stable
pure law. That is scientifically interesting, but it does not select a task.

The [contradiction matrix](evidence/contradiction-matrix.md) records the full
evidence and the failure each stage below is designed to remove.

## 1. Scientific object

Let \(g\) index the highest dependence unit: a document, rollout, problem,
writer, conversation, or story. At an operational prediction point \(t\), let:

- \(Y_{g,t}\) be a local state, transition, onset, or other declared target;
- \(A_{g,t}\) be the anchor activation available at prediction time;
- \(H_{g,t}^{(T)}=(A_{g,t-T},\ldots,A_{g,t-1})\) be the strictly pre-target,
  ordered history;
- \(M_{g,t}^{(T)}=\{A_{g,t-\tau}:\tau=1,\ldots,T\}\) be the mathematical
  multiset containing exactly the same vectors without their slot identities;
- \(C_{g,t}\) contain preregistered, deployment-available nuisance variables.

Group identifiers define splits and bootstrap units. They are not predictor
features. A nuisance belongs in \(C\) only when conditioning is justified by
the scientific question; an arbitrary adjustment can create a collider,
remove the phenomenon, or leak the target.

The exact multiset is a population object, not a promise that one finite
DeepSets network captures every invariant function. In experiments,
\(\phi_{\mathrm{bag}}(H)\) denotes the declared finite bag observer. Any
ordered-versus-bag gain is therefore observer-relative and may reflect an
underpowered invariant branch. A capacity-matched observer trained and tested
with a fresh hidden permutation for every example is a co-primary order
control.

### 1.1 Target contract

The target must be aligned to one prediction point and be allowed to switch
within a sequence. Onset inputs are strictly pre-onset. A rollout-level
“contains misalignment” label is unsuitable for a local-state screen when a
brief positive span relabels every neutral window in the rollout.

Before analysis, record:

- source and construction of the target;
- prediction point and allowed inputs;
- group identifier and sampling unit;
- event count, base rate, sequence lengths, and overlap among windows;
- possible lexical, positional, length, template, and source shortcuts;
- whether the scientific estimand weights groups or events equally.

For onset tasks, define the at-risk set and censoring rule before selecting
anchors. Match negative anchors on sequence phase, remaining eligible length,
and opportunity for a future event; otherwise the choice of prediction point
can itself reveal the onset label. Report both a minimal scientifically
necessary nuisance set and a prespecified expanded set so collider or
over-adjustment sensitivity remains visible.

## 2. Eligibility: conditional ordered usable information

Fit three nested or carefully matched observers on outer training groups:

\[
q_{\mathrm{anchor}}(Y\mid A,C),
\]

\[
q_{\mathrm{easy},T}
\left(
Y\mid A,C,M(H^{(T)}),A_{t-\tau^\star}
\right),
\]

and

\[
q_{\mathrm{ordered},T}
\left(
Y\mid A,C,M(H^{(T)}),A_{t-\tau^\star},H^{(T)}
\right).
\]

The best single offset \(\tau^\star\), every preprocessing transform, and all
regularization are selected only inside the outer training fold. The ordered
observer contains the easy observer as an exact submodel and can set its
ordered residual branch to zero.

The primary score is the group-averaged held-out proper-risk reduction

\[
S_{\mathrm{ord}}^{\mathcal Q}(T)
=
\mathbb E_g
\left[
L_g(q_{\mathrm{easy},T})
-
L_g(q_{\mathrm{ordered},T})
\right].
\]

With Bayes-optimal unrestricted observers under log loss, the corresponding
population quantity is the conditional mutual information

\[
I\!\left(Y;H^{(T)}
\mid A,M(H^{(T)}),C,A_{t-\tau^\star}\right).
\]

With a frozen finite family \(\mathcal Q\), the empirical quantity is
conditional usable information. This qualification is a strength: the
information relevant to a practical dictionary must be accessible to a
declared observer, while high-dimensional Shannon information is not
estimable at the available sample size.

For a continuous target, replace log loss with a frozen proper scoring rule
such as Gaussian negative log likelihood; do not call every risk reduction
mutual information.

Also report:

\[
S_{\mathrm{history}}(T)
=L(q_{\mathrm{anchor}})-L(q_{\mathrm{ordered},T}),
\]

the bag-only ordered increment, and the ordered-versus-best-offset risk gap.
The task is specifically TXC-motivated only if history, order, and
distributedness survive. The following outcomes have different meanings:

| Result | Interpretation | Architecture prediction |
|---|---|---|
| No history gain | The anchor is sufficient for the declared observer | No explicit-history architecture is motivated |
| History gain, no order gain | Denoising, occupancy, repetition, or a bag statistic helps | SAE plus pooling or T-SAE should be competitive |
| Order gain, no gain beyond best offset | One named position carries the signal | Positional/stacked SAE is the relevant baseline |
| Stable gain beyond declared bag and best-offset observers, also surviving fresh-permutation controls | Ordered windows contain usable information unexplained by the declared easier observers | An order-sensitive window representation is motivated; architecture comparisons must determine whether TXC is best |

The primary observer should be small: a ridge or affine-softmax baseline,
DeepSets bag branch, and rank-limited ordered branch with closely matched
capacity and optimization. The ordered model must also be trained on fresh
per-example random permutations, with the permutation hidden from the model;
this preserves the vector multiset and model capacity while removing stable
slot assignment. Exact nesting prevents a negative population gain but does
not make every positive finite-observer gain attributable to order. One
shallow nonlinear observer is a prespecified sensitivity analysis for
XOR-like dependence. Large task-specific architectures are inappropriate
screeners.

## 3. The nontrivial correlation object

### 3.1 Residual target lag operator

Fit a frozen, label-blind activation projection \(P\) on outer training groups
and let

\[
Z_{g,t}=P A_{g,t}\in\mathbb R^p.
\]

Let the easy baseline be

\[
B_{g,t}^{(T)}
=
\left(A_{g,t},C_{g,t},M(H_{g,t}^{(T)}),
A_{g,t-\tau^\star}\right).
\]

Cross-fit the nuisance regressions

\[
m_\tau(B)=\mathbb E[Z_{t-\tau}\mid B],
\qquad
\pi(B)=\mathbb E[U_t\mid B],
\]

where \(U_t\) is a scalar, one-hot, ordinal, or continuous representation of
the target. On held-out groups form

\[
R_{g,t,\tau}=Z_{g,t-\tau}-\widehat m_\tau(B_{g,t}^{(T)}),
\]

\[
V_{g,t}=U_{g,t}-\widehat\pi(B_{g,t}^{(T)}).
\]

The target-residual lag operator is

\[
K_B(\tau)
=
\mathbb E_g\mathbb E_{t\mid g}
\!\left[R_{g,t,\tau}V_{g,t}^{\top}\right],
\]

This is the main correlation-function extension. Unlike unconditional
activation autocovariance, it is aligned to the declared target. Unlike an
operator or Frobenius norm, the signed matrix retains lag orientation and
cross-channel structure. Under
\(Y_t\perp H_t^{(T)}\mid B_t^{(T)}\), it is zero. The converse does not hold,
because higher-order dependence and conditional effects that cancel across
values of \(B\) may have zero averaged cross-covariance. The displayed
expectation weights groups equally and positions equally within group; an
event-weighted estimand must be reported separately. Inference inherits the
generalized-covariance-measure requirement that the cross-fitted nuisance
errors converge quickly enough for their product to be negligible. This is
an averaged partial cross-moment, not a conditional-independence certificate.

### 3.2 Joint linear opportunity

Stack \(R_{t,1:T}\) into \(R_T\), define

\[
\Sigma_R=\operatorname{Cov}(R_T),
\quad
\Sigma_V=\operatorname{Cov}(V),
\quad
K_T=\operatorname{Cov}(R_T,V),
\]

and estimate the regularized linear opportunity

\[
Q_T
=
\operatorname{tr}\!\left[
\Sigma_V^{\dagger/2}
K_T^\top
(\Sigma_R+\lambda I)^{-1}
K_T
\Sigma_V^{\dagger/2}
\right].
\]

For multiclass targets, encode \(V\) with \(K-1\) full-rank contrasts; the
pseudoinverse notation above also handles the singular covariance of a
redundant one-hot encoding. For a standardized scalar Gaussian target this is
a ridge partial-\(R^2\)-like quantity. In binary equal-covariance Gaussian
classification, using total residual covariance gives a bounded monotone
transform of Mahalanobis discriminability; use the within-class residual
covariance when \(D^2\) itself is the intended quantity. Report

\[
Q_{\mathrm{dist}}(T)
=
Q_T-\max_{\tau\le T}Q_{\{\tau\}}
\]

as joint linear gain beyond one informative offset. It can also arise from
repeated noisy measurements or an underfit invariant observer, so it is not
by itself evidence of distributed order. With regularized finite-sample
estimates, report the untruncated difference and its grouped interval;
clipping it to zero would hide estimator failure.

### 3.3 Frequency and phase

For the finite pre-target lag path, transform the full operator:

\[
\widetilde K_B(\omega)
=
\sum_{\tau=1}^{T}h_\tau K_B(\tau)e^{-i\omega\tau}.
\]

This is a finite lag-kernel basis transform. Report its singular values and
complex phase, but do not call it a spectrum or multiple coherence.

A separate stationary analysis is permissible only when \(R_t\) and \(V_t\)
form a fixed, dense, jointly locally stationary residual time series. It then
requires full auto- and cross-spectral densities, including negative lags.
For scalar \(V\), residual multiple coherence is

\[
\kappa^2(\omega)
=
\frac{
S_{VZ}(\omega)
S_{ZZ}(\omega)^{-1}
S_{ZV}(\omega)
}{
S_{VV}(\omega)
}.
\]

For vector \(V\), report the eigenvalues of the canonically whitened
cross-spectral matrix rather than applying this scalar ratio. Estimate the
stationary objects with multitapers, shrinkage or ridge inversion selected
inside training folds, and whole-group resampling. Report:

- integrated non-DC target-aligned lag-kernel energy;
- complex phase of the finite kernel or a separately declared stationary
  directional statistic;
- the band containing most target-aligned energy;
- spectral-matrix eigenvalues and effective rank.

For sparse onsets, \(K_B(\tau)\) is an event-triggered pre-onset kernel.
Fourier or wavelet coefficients are then a finite basis decomposition of the
kernel, not a stationary process spectrum. The lag-domain path remains
primary.

Positive-versus-negative lag asymmetry is a separate descriptive analysis
that requires constructing post-target as well as pre-target kernels; it is
undefined for the strictly pre-target operator above and is never part of the
prediction-time input.

### 3.4 Higher-order sensitivity

A cross-covariance misses targets encoded in variance, covariance, parity, or
phase-amplitude coupling. Prespecify a low-dimensional quadratic feature
basis \(\psi(R_{t,\tau},R_{t,\tau'})\), cross-fit its conditional mean given
\(B\), and estimate the averaged generalized cross-moment

\[
D_B(\tau,\tau')
=
\mathbb E_g\mathbb E_{t\mid g}
\left[
\left\{
\psi(R_{t,\tau},R_{t,\tau'})
-
\mathbb E[\psi(R_{t,\tau},R_{t,\tau'})\mid B_t]
\right\}
V_t^\top
\right],
\]

or a kernel cross-spectral norm on the frozen projection. The nonlinear
ordered usable-information score remains the umbrella check. A result that
exists only under a selected high-capacity kernel is exploratory.

## 4. Shared law and sequence heterogeneity

Estimate per-group contributions

\[
\widehat K_g(\tau)
=
\frac{1}{n_g}
\sum_{i\in g}
\widehat R_{i,\tau}\widehat V_i^\top.
\]

The shared operator is

\[
\overline K(\tau)=\mathbb E_g[K_g(\tau)].
\]

For \(G\) groups, let
\(\widehat\Omega_g(\tau)\) estimate the within-group sampling covariance of
\(\operatorname{vec}\widehat K_g(\tau)\). A finite-\(G\) noise-corrected
scalar deviation summary is

\[
\widehat{\mathcal V}_K(\tau)
=
\frac{1}{G-1}
\sum_{g=1}^{G}
\left\|
\operatorname{vec}\widehat K_g(\tau)
-
\operatorname{vec}\overline{\widehat K}(\tau)
\right\|_2^2
-
\frac{1}{G}
\sum_{g=1}^{G}
\operatorname{tr}\widehat\Omega_g(\tau).
\]

Functional PCA of \(K_g(\tau)-\overline K(\tau)\), fit only on training
groups, can expose recurring sequence subtypes. Report both the common mode
and the noise-corrected deviation spectrum.

Per-group operators require enough replicated eligible targets within each
group. When rollouts have only one or a few events, use a hierarchical
shrinkage model or aggregate scientifically exchangeable groups; do not
interpret an unidentifiable per-rollout \(K_g\). A subtype used to gate a new
prediction must be assigned from deployment-available covariates using only
the outer training fold. A subtype discovered from label-dependent
\(\widehat K_g\) is descriptive and cannot serve as a deployment-time gate.

Interpretation requires care:

- large \(\overline K\), small \(\widehat{\mathcal V}_K\): one shared temporal
  relation;
- small \(\overline K\), large \(\widehat{\mathcal V}_K\), observable subtype:
  a gated or multi-feature representation may be appropriate;
- small \(\overline K\), large \(\widehat{\mathcal V}_K\), hidden subtype:
  the pattern is consistent with heterogeneous cancellation if the sampling
  model is adequate, but it does not supply a deployable screen;
- both small: no detectable target-aligned second-order mechanism.

The last two cases prevent “heterogeneity” from becoming an unfalsifiable
explanation for a failed average.

## 5. Input resolvability and unsupervised learnability

The target axis above can find a low-variance direction that an unsupervised
reconstruction dictionary will ignore. TAMC therefore keeps a separate
activation axis.

On a frozen label-blind subspace, estimate the raw input spectral covariance

\[
S_X(\omega)
=
U(\omega)\Lambda(\omega)U(\omega)^*.
\]

Use group bootstraps and matched surrogates to identify stable eigenmodes
above the finite-data floor. Let \(P_{\mathrm{res}}(\omega)\) project onto
those raw-input modes. Separately estimate the easy-baseline-residualized
class-mean path \(\widehat\Delta_B(\omega)\) and its conditional within-class
noise spectrum \(S_{R\mid Y,B}(\omega)\). Define the whitened target direction

\[
\delta_B(\omega)
=
S_{R\mid Y,B}(\omega)^{-1/2}
\widehat\Delta_B(\omega).
\]

Map the stable raw-input modes through the same whitening transform and let
\(\Pi_{\mathrm{res}}(\omega)\) project onto their span. The full residual
linear target opportunity is

\[
J_Y(\omega)=\|\delta_B(\omega)\|_2^2,
\]

and the resolvable portion

\[
J_{\mathrm{res}}(\omega)
=
\left\|
\Pi_{\mathrm{res}}(\omega)\delta_B(\omega)
\right\|_2^2.
\]

The integrated coverage ratio

\[
\rho_{\mathrm{res}}
=
\frac{\int J_{\mathrm{res}}(\omega)d\omega}
{\int J_Y(\omega)d\omega}
\]

asks whether target-relevant signal lies in stable raw-covariance input modes
under this linear proxy. A rank-\(H\) linear autoencoder
supplies a sharper capacity hypothesis
by replacing \(P_{\mathrm{res}}\) with the leading stable covariance
eigenspace \(P_H\). This is only a diagnostic approximation for sparse
ReLU--TopK training.

This two-axis view handles the two main false positives:

- abundant low-frequency nuisance gives high input power but near-zero
  target alignment;
- a tiny discriminative direction gives high conditional predictive value
  but low raw-covariance coverage.

Call \(\rho_{\mathrm{res}}\) a **linear covariance-resolvability proxy**, not
statistical learnability by an unsupervised sparse dictionary. After training,
measure actual coverage on held-out nonlinear codes with a frozen probe or a
prespecified code-space mean/covariance statistic.

## 6. Conjectural extension of the Cagnetta--Ganguli horizon

As a scaling ansatz, suppose the leading singular value of the
target-residual operator obeys

\[
\sigma_1(K_B(\tau))
\simeq
a\tau^{-\beta_Y},
\]

while the grouped estimation floor behaves as

\[
b_G(\tau)\simeq c(\tau)G^{-\nu}.
\]

If \(c(\tau)\) is approximately constant on the fitted range, their
intersection gives

\[
\tau_Y^\star(G)
\asymp
\left(\frac{a}{c}\right)^{1/\beta_Y}
G^{\nu/\beta_Y}.
\]

Under an independent fixed-dimensional group CLT, \(\nu=1/2\), recovering the
same \(G^{1/(2\beta_Y)}\) form as the original resolvability argument. For
overlapping activation windows, unequal rollout sizes, growing projection
dimension, or simultaneous lag selection, that exponent is not assumed.
Estimate \(\nu\) by group subsampling and estimate the floor with grouped
resampling. Replacing an independent pair count with \(G\) is justified only
when groups are independent and carry asymptotically fixed information.

The primary empirical horizon does not require a power law. Define
\(\tau_Y^\star\) from the significant lag support whose simultaneous lower
confidence band exceeds a matched target-shift floor, and separately report
the farthest supported lag. Do not require the support to begin at lag one:
delayed or gapped motifs are valid. Require the proposed horizon to agree
with held-out ordered-information saturation.

There are two distinct horizons:

1. **target resolvability:** \(K_B(\tau)\) or conditional usable information
   is above its grouped null;
2. **reconstruction-mode resolvability:** the corresponding activation mode
   is stable above the input-covariance floor.

Their conjunction is a stronger TXC opportunity hypothesis. It is still not
an architecture-learning theorem because the sparse nonlinear optimization
dynamics remain unspecified.

This grouped, nuisance-residualized, target-operator extension is the precise
open theory question to discuss with Surya and Francesco.

The heterogeneity correction, linear covariance-resolvability ratio, and
target-conditioned horizon are synthesized heuristics from the audited
literature, not established results in those papers. They require the
synthetic calibration and coverage checks below before use on natural tasks.

## 7. TXC theory connection

A TXC preactivation is

\[
a_j(t)
=
\sum_{\tau=0}^{T-1}
w_j(\tau)^\top x_{t+\tau}
+b_j.
\]

Its exact finite Fourier representation is

\[
a_j(t)
=
\frac{1}{T}
\sum_\omega
\widehat w_j(\omega)^*
\widehat x_t(\omega)
+b_j.
\]

An unconstrained TXC is therefore a finite impulse-response filter bank before
ReLU--TopK. A full Fourier parameterization has identical expressive power
and is only a basis change at the hypothesis-class level. Optimization,
initialization, and parameter geometry can still create an implicit bias, so
an empirical difference without an explicit constraint should be described
as an optimization effect rather than a new spectral hypothesis class.

In the linear rank-\(H\) relaxation, an MSE autoencoder spans the leading
eigenspace of the \(T\)-window covariance. Under local stationarity, that
covariance is block Toeplitz and approximately Fourier diagonal for large
\(T\). This yields testable, limited predictions:

- an order-invariant mean sees mainly the DC component;
- a T=1 SAE cannot implement a cross-position filter;
- a positional/stacked SAE can use one or several named offsets;
- a TXC can represent signed non-DC, phase-bearing filters;
- a spectral TXC gains an inductive bias only through band restrictions,
  hierarchical pooling, or frequency-dependent regularization.

For equal-covariance Gaussian classes,

\[
X_T\mid Y=c\sim\mathcal N(\mu_c,\Sigma_T),
\qquad
\Delta_T=\mu_1-\mu_0,
\]

and the optimal linear discriminability is

\[
D_T^2
=
\Delta_T^\top\Sigma_T^{-1}\Delta_T
\approx
\frac{1}{2\pi}
\int
\widehat\Delta(\omega)^*
S_Z(\omega)^{-1}
\widehat\Delta(\omega)
d\omega.
\]

TAMC estimates this target opportunity, its easy-baseline residual, and its
coverage by stable raw-covariance modes under a linear proxy. For a genuinely
linear encoder \(E\), post-training coverage is

\[
D_E^2
=
(E\Delta_T)^\top
(E\Sigma_T E^\top)^\dagger
(E\Delta_T),
\]

not inferred from raw power alone. For the actual nonlinear sparse code,
measure held-out target coverage with the same frozen probe family or a
prespecified code-space mean/covariance statistic; the linear formula does
not apply directly.

## 8. Estimation contract

1. **Split by the highest dependence unit.** Overlapping windows never cross
   folds. Default reporting weights groups equally; event-weighted results
   are a named sensitivity.
2. **Use nested cross-fitting.** Projection, nuisance residualization,
   best-offset search, probe selection, regularization, early stopping, and
   calibration occur inside each outer training fold.
3. **Use one common cohort across \(T\).** Report a maximal-data sensitivity
   separately when long windows remove examples.
4. **Freeze the primary probe.** Use one linear/rank-limited family for the
   confirmatory score and one shallow nonlinear sensitivity. Do not maximize
   over observers after seeing results.
5. **Report the full \(T\) curve.** Define the useful horizon by a rule frozen
   on synthetics, bootstrap the whole curve jointly, and report simultaneous
   uncertainty. The earliest \(T\) within a calibrated tolerance of the best
   point is a selected statistic and cannot use pointwise intervals.
6. **Bootstrap groups, not windows.** Refit nuisance models in a full
   bootstrap when feasible; a cluster multiplier bootstrap on cross-fitted
   group scores is the cheaper approximation.
7. **Use simultaneous uncertainty for lag and \(T\) curves.** A pointwise
   excursion at one selected lag or one selected window is not a horizon.
8. **Preserve natural prevalence.** The primary log-loss estimand uses the
   deployment prevalence; balanced analyses are sensitivities.
9. **Record estimator failures.** Non-PSD spectral matrices, unstable
   inverses, optimizer boundaries, centering sensitivity, and negative
   nested gains remain visible.

## 9. Null and intervention ladder

Each control answers one declared question:

| Control | Preserves | Destroys or changes | Interpretation |
|---|---|---|---|
| Fresh hidden within-window permutation | Vector multiset and anchor | Presented slot assignment | Primary order-information comparison when the observer is refit |
| Block shuffle | Local content and coarse regimes | Fine order across blocks | Scale-locality diagnostic |
| Target shift within stable phase, no wrap | Activation trajectory and target prevalence | Event alignment | Target-alignment floor |
| Inter-event-interval shuffle | Burst distribution | Order among intervals | Separates burstiness from interval correlations |
| IAAFT surrogate | Marginal distribution and approximate power spectrum | Higher-order phase coupling | Nonlinear spectral falsifier, not a unique mechanism test |
| Fixed reversal | Information and magnitude power | Orientation relative to a fixed model | Orientation stress test with distribution-shift caveat |
| Refitted reversal | All information under a bijection | Nothing for an expressive observer | Sanity check: performance should recover |
| Content-preserving history edit | Declared content/marginals | A specific temporal relation | Needed for causal language |

Contextual activations already encode earlier tokens and position. Shuffling
the vector sequence removes the arrangement presented to the observer, not
all temporal information inside each vector. Audit absolute/relative position
decodability and pair the activation screen with a text- or token-side control
when possible.

## 10. Synthetic calibration suite

The screen is frozen only after it separates known mechanisms:

| Generator | Expected history | Expected order | Correlation prediction | Required decision |
|---|---:|---:|---|---|
| IID target from anchor plus independent \(1/f\) background | zero beyond anchor | zero | strong generic tail, zero target operator | reject |
| Window mean or count | positive | zero beyond bag | DC target signal | reject TXC-specific claim |
| One named old offset | positive | may beat bag | lag spike, no distributed gain | positional baseline |
| Directed \(A\!\to\!B\) motif | positive | positive | signed phase-bearing kernel | pass |
| Motif versus reversal | positive | positive | equal magnitude power, opposite phase | distinguish |
| Opposite kernels in observable subtypes | positive after gating | positive | mean washout, high structured heterogeneity | subtype-gated pass |
| Opposite kernels in hidden subtypes | weak/zero aggregate | ambiguous | mean washout without deployable gate | no global claim |
| Ordered XOR | nonlinear positive | nonlinear positive | zero linear \(K_B\) | nonlinear sensitivity only |
| Persistent topic/regime mixture | apparent positive | usually zero | low-frequency tail removed within regime | reject or attribute |
| Sparse pre-onset ramp | positive | positive | event-local low-frequency/derivative path | pass |
| Burst process with shuffled intervals | process memory | target dependent | separates bursts from interval order | mechanism-specific |
| Identical through lag \(T-1\), different after \(T\) | identical at \(T\) | identical at \(T\) | distant spectra differ | every \(T\)-window model ties |

Calibrate the smallest effect of interest, observer capacity, ridge grid,
horizon tolerance, group-stability rule, and null thresholds on these
generators before scoring new tasks.

## 11. Prospective benchmark protocol

1. Inventory plausible task families without looking at new dictionary
   results. Include screen-negative tasks.
2. Mark every task whose TXC result has already been inspected as development
   evidence. This includes Backtracking, EM, RLHF, sparse probing, deletion,
   and the other July task pilots.
3. Split by mechanism family, data source, generator, writer/template, and
   label rule. Random task identifiers do not define independence.
4. Tune TAMC and validate type-I/type-II behavior on synthetics and
   development tasks.
5. Freeze the task score, useful-window prediction, architecture contrast,
   task weighting, missing-run policy, and all thresholds.
6. Score the untouched task panel and preregister the predicted sign and
   useful \(T\) for every task before training a dictionary.
7. Train matched SAE, T-SAE, positional/stacked SAE, ordered TXC, and
   order-destroyed controls on both screen-positive and screen-negative tasks.
8. Test whether the continuous TAMC score predicts the TXC-minus-baseline gap
   and whether its horizon predicts TXC saturation or peak.
9. Keep an append-only task and experiment registry so failed tasks cannot
   disappear.

The confirmatory unit is a task family, not another random seed or another
example from a task used to design the screen.

## 12. Implementation sequence

### Phase A: estimator validation

- implement grouped synthetic generators and nested risk scores;
- validate lag-operator recovery, phase, higher-order false negatives, and
  heterogeneity correction;
- verify matrix-spectrum Hermiticity/PSD and grouped bootstrap coverage;
- freeze the initial observer and \(T\) grids.

**Exit condition:** the screen rejects every designed false positive and
recovers the directed, distributed positives with calibrated uncertainty.

### Phase B: retrospective task atlas

- apply TAMC to Backtracking and every other already-inspected task;
- produce one target-usefulness curve, lag/operator panel, heterogeneity
  panel, and null panel per task;
- determine whether the prior TXC successes and failures match the screen.

This phase can invalidate the proposal but cannot confirm it because the task
results informed its design.

**Exit condition:** TAMC ranks Backtracking above the order-invariant and
global-label tasks without using any dictionary metric. If it does not, revise
or stop before creating new benchmarks.

### Phase C: prospective task holdout

- freeze an untouched, mechanism-stratified task panel;
- publish screen decisions and window predictions before dictionary runs;
- test screen-to-architecture association across all tasks.

**Exit condition:** a prespecified association with uncertainty that excludes
the synthetic-calibrated null and survives retention of screen-negative tasks.

### Phase D: spectral architecture

Only after Phase C supports the screen, compare:

- real-space TXC;
- full Fourier reparameterization, expected to tie absent optimization
  effects;
- band-limited or multiresolution spectral TXC;
- frequency-dependent sparsity/reconstruction penalties.

This prevents the screen from being tuned to justify an architecture that was
already selected.

## 13. Outputs

The first complete implementation should produce:

- `task_registry.csv`: immutable task provenance, mechanism family, and
  development/holdout status;
- `screen_protocol.json`: frozen observers, splits, grids, controls, and
  thresholds;
- one group-level prediction ledger per task;
- ordered-information and useful-horizon curves;
- shared and heterogeneous target-residual lag operators;
- local spectral/coherence and effective-rank panels when valid;
- surrogate/null distributions;
- preregistered architecture predictions;
- the eventual screen-score versus TXC-gap analysis including all failures.

## 14. Claims and stop conditions

TAMC can support:

- ordered local history contains target-usable information beyond declared
  easier baselines;
- the information is concentrated at particular lags, frequencies, phases,
  or sequence subtypes;
- the target-relevant mode is or is not covered by stable covariance modes
  under a declared linear reconstruction proxy;
- a frozen task score prospectively predicts architecture gaps and useful
  windows.

It cannot by itself support:

- a universal pure power law in activations;
- causal use of the history;
- guaranteed TXC feature recovery;
- a novel spectral estimator;
- superiority of a Fourier parameterization without an actual constraint or
  regularizer.

Stop or narrow the project if any of the following occurs:

1. the screen cannot reject a label-independent power-law background;
2. the full-multiset or best-offset baseline closes the Backtracking gap;
3. results are dominated by one group, fit range, centering choice, or probe;
4. useful horizons do not agree with held-out predictive saturation;
5. the frozen score fails to predict architecture behavior on the untouched
   task panel.

## 15. The collaboration question

The useful question for Surya and Francesco is not whether another aggregate
activation curve follows a power law. It is:

> Can the finite-data resolvability theory be extended from an unconditional
> token covariance norm to a grouped, nuisance-residualized target
> cross-covariance operator, with a principled decomposition into a shared law
> and sequence- or task-specific deviations?

A derivation of its sampling floor, resolvable horizon, and connection to
reconstruction-mode learnability would supply the missing theory. TAMC
provides the empirical protocol against which that theory can be falsified.

## Evidence

The complete audit is in:

- [language and scaling evidence](evidence/language-and-scaling.md);
- [information and spectral evidence](evidence/information-and-spectra.md);
- [activation and benchmark evidence](evidence/activations-and-benchmarks.md);
- [contradiction matrix](evidence/contradiction-matrix.md);
- [annotated reading list](reading-list.md);
- [exact GPT-2/WikiText correlation recheck](../../../results/temporal_screen/correlation_recheck_20260729/summary.md).
