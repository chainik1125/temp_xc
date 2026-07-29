## Theory and literature for a power-spectrum temporal screen

## Bottom line

A spectral screen is worthwhile, but only as a *second-order routing
instrument*. It can cleanly separate stable/DC structure, slow persistence,
line-like periodicity, and broadband fluctuations. It cannot by itself
distinguish dynamics that have the same power but different phase, direction,
localization, or higher-order dependencies. In particular, time reversal and
the sign of a real sinusoid are invisible to an ordinary power spectrum.

The result in Cagnetta et al. justifies Fourier analysis more narrowly than the
strongest reading suggests. They use a corpus-averaged two-point function that
depends only on lag. Conditional on that lag-stationary description,
translation invariance makes temporal Fourier modes the natural diagonal
coordinates of the second-order covariance operator. Their reported power-law
decay of a *lagwise matrix norm* does **not** prove that language activations
have a low-frequency-dominated power spectrum.

The first experiment should therefore use a statistically controlled,
multivariate screen with DC reported separately, multitaper AC spectra,
sequence-level bootstrap uncertainty, shuffle and colored-noise nulls, and an
explicit stationarity check. For the model, the best first intervention is not
to delete DC unconditionally. It is to give DC its own branch and budget, then
normalize the AC reconstruction loss by background band power. A
frequency-Matryoshka loss is a useful second ablation, but only with correction
for the repeated overweighting of low frequencies inherent in naive nested
cutoffs.

## What the Francesco result establishes

Cagnetta et al. define the lag-$n$ token covariance matrix

$$
C_{\mu\nu}(n)
=
\mathbb{P}(X_i=\mu,X_{i+n}=\nu)
-
\mathbb{P}(X_i=\mu)\mathbb{P}(X_{i+n}=\nu),
$$

and summarize its strength by $\lVert C(n)\rVert_{\mathrm{op}}$. In the local
source and current paper they report
$\lVert C(n)\rVert_{\mathrm{op}}\asymp n^{-\beta}$, with
$\beta=0.88\pm0.06$ on TinyStories and $0.94\pm0.16$ on WikiText. WikiText is
better described by a broken power law, and both corpora show a localized
feature near lag 10. Their purpose is a signal-to-noise argument:
$P_n^\ast\asymp\lVert C(n)\rVert_{\mathrm{op}}^{-2}\asymp n^{2\beta}$, not
spectral estimation. See [Cagnetta, Raventós, Ganguli, and Wyart,
2026](https://arxiv.org/abs/2602.07488), especially the token-correlation and
sampling-noise sections.

The definition suppresses the absolute position $i$, thereby using a
wide-sense-stationary, or second-order translation-invariant, summary. This is
not itself an empirical stationarity test: corpus averaging can conceal
document-position and domain changes. Conditional on this summary, define the
full two-sided matrix covariance sequence by

$$
C(-n)=C(n)^\top,\qquad
S(\omega)=\sum_{n=-\infty}^{\infty}C(n)e^{-i\omega n},
$$

where $S(\omega)$ is a Hermitian positive-semidefinite cross-spectral density
matrix. Equivalently, the covariance over a long finite window is block
Toeplitz and is approximately block-diagonalized in temporal Fourier
coordinates. This is the precise motivation for a frequency-domain
parameterization of a temporal crosscoder.

There is a useful conditional implication. If a *scalar, eventually
nonnegative* covariance obeys
$c(n)\sim a n^{-\beta}$ with $0<\beta<1$, then its low-frequency density has
the Abelian asymptotic form

$$
S(\omega)\propto |\omega|^{\beta-1}
\quad\text{as}\quad \omega\rightarrow0.
$$

Thus $\beta<1$ implies a low-frequency divergence. If the Francesco exponents
were exponents of such a scalar covariance, they would imply only weak red
spectra: approximately $|\omega|^{-0.12}$ and $|\omega|^{-0.06}$ at the
central estimates. This calculation is suggestive, not a valid transform of
their plotted norm.

## What it does not establish

- A norm is not a covariance function. Taking
  $\lVert C(n)\rVert_{\mathrm{op}}$ discards signs, phases, and changes in the
  leading singular directions. The Wiener-Khinchin transform applies to
  $C(n)$, not to the sequence of its norms.

- Identical norm decay can hide arbitrary carrier frequencies. For example,
  $C(n)=a n^{-\beta}\cos(\omega_0n)uu^\top$ has
  $\lVert C(n)\rVert_{\mathrm{op}}\asymp n^{-\beta}$ for every $\omega_0$,
  but its spectral singularities lie near $\pm\omega_0$, not necessarily at
  DC.

- Stationarity does not imply low-frequency dominance. White noise,
  an alternating process, and a narrow high-frequency tone can all be
  stationary.

- Second-order stationarity does not imply full distributional stationarity.
  Randomly chunking a concatenated corpus makes an averaged estimator
  convenient; it does not prove that topic, style, document position, or
  activation mean is constant through the stream.

- A power spectrum loses temporal location and ordinary univariate power
  loses phase. A changepoint and a stationary broadband process can have
  similar averaged power. A trajectory and its reversal have the same power.
  The two directions around the FrequencyBench circle also have the same
  scalar power unless cross-channel phase is retained.

- Pairwise spectra are blind to purely higher-order structure. Two processes
  can have the same $S(\omega)$ but different equality patterns, burst
  structure, bispectra, or predictive information.

These limitations are not reasons to reject the screen. They identify the
screen's correct output: a route to *DC*, *stationary spectral*,
*phase-sensitive*, *localized*, or *higher-order* analysis.

## What the language literature supports

Long-range organization in text is well established, but the measured object
matters.

- [Ebeling and Pöschel
  (1994)](https://doi.org/10.1209/0295-5075/26/4/001) found power-law decay of
  character-level mutual information over part of its range in two literary
  corpora. [Montemurro and Pury
  (2002)](https://doi.org/10.1142/S0218348X02001257) mapped word sequences to
  numerical series and found long-range fractal correlations beyond
  sentence-scale syntax. These are foundational evidence for long memory, not
  universal estimates of an activation PSD.

- [Altmann, Pierrehumbert, and Motter
  (2009)](https://doi.org/10.1371/journal.pone.0007678) showed that individual
  word recurrences are bursty and often fit stretched exponentials rather than
  a Poisson process. This warns that an averaged stationary spectrum can mix
  discourse-local bursts with genuine scale-free stationary dependence.

- [Lin and Tegmark
  (2017)](https://doi.org/10.3390/e19070299) connected power-law mutual
  information to hierarchical probabilistic grammars and proved that generic
  finite-state Markov models instead give exponential decay. This supports a
  multiscale interpretation, but it still does not select DC over an
  oscillatory carrier.

- [Takahashi and Tanaka-Ishii
  (2019)](https://doi.org/10.1162/coli_a_00355) compared several long-memory
  statistics on text and generated language. Their results show that
  conclusions depend on the symbol-to-number mapping and statistic; mutual
  information, fluctuation analysis, recurrence autocorrelation, and Taylor's
  law are not interchangeable.

- A very recent preprint, [Yang et al.
  (2026)](https://arxiv.org/abs/2604.05536), reports an approximately
  $f^{-5/3}$ spectrum for a scalar *contextual-embedding step* signal across
  corpora and languages, with the scaling disrupted by token shuffling and
  absent from static embeddings. This is the closest direct precedent for the
  proposed activation screen, but it is a new preprint, uses a derived scalar
  observable rather than the token covariance matrix, and its exponent should
  not be identified with Cagnetta et al.'s $\beta$.

The machine-learning result usually called spectral bias is adjacent but
different. [Rahaman et al.
(2019)](https://proceedings.mlr.press/v97/rahaman19a.html) show that ReLU
networks tend to learn low-frequency *target functions over input space*
earlier. That result does not say that low temporal frequencies contain most
language-activation power, nor that a sparse temporal autoencoder should
prefer them. It instead gives a confound to test: an apparent low-frequency
advantage could arise from optimization even when task-relevant temporal
power is balanced.

Sparse-coding work supports structured and shift-aware dictionaries.
[Olshausen and Field
(1996)](https://doi.org/10.1038/381607a0) learned localized, oriented,
bandpass atoms from sparse coding; [Grosse et al.
(2007)](https://ai.stanford.edu/~ang/papers/uai07-shiftinvariantsparsecoding.pdf)
learned shift-invariant sparse audio bases; and [Bristow, Eriksson, and Lucey
(2013)](https://openaccess.thecvf.com/content_cvpr_2013/html/Bristow_Fast_Convolutional_Sparse_2013_CVPR_paper.html)
made convolutional sparse coding efficient through frequency-domain
optimization. These precedents motivate temporal band structure, but none
establishes the particular per-band sparsity penalty needed here.

[Bussmann et al.
(2025)](https://openreview.net/forum?id=m25T5rAy43) train Matryoshka SAEs by
requiring nested dictionary prefixes to reconstruct independently. Nesting
*frequency supports* rather than dictionary widths is a new adaptation and
should be presented as such.

## A defensible multivariate spectral screen

Assume independent sequences or documents indexed by $s$, with activations
$x_{s,t}\in\mathbb{R}^d$. Never Fourier-transform across a document boundary.
All centering, channel scaling, and band choices must be fit on the training
split and then frozen.

### Keep DC observable

Estimate one global training mean $\bar x$, then define

$$
u_{s,t}=x_{s,t}-\bar x,\qquad
\mu_s=\frac{1}{T_s}\sum_t u_{s,t},\qquad
r_{s,t}=u_{s,t}-\mu_s.
$$

Record sequence-mean energy as a separate DC statistic and estimate the AC
spectrum only from $r_{s,t}$. With a unitary transform, use

$$
q_{\mathrm{DC}}
=
\frac{T_s\lVert\mu_s\rVert^2}
{T_s\lVert\mu_s\rVert^2+\sum_t\lVert r_{s,t}\rVert^2}.
$$

This separates persistent between-sequence semantics from within-sequence
dynamics. Demeaning each sequence without retaining $\mu_s$ would silently
delete exactly the stable/DC tasks. Conversely, leaving the large global
activation bias in the periodogram would make DC dominance nearly automatic
and scientifically uninteresting.

### Estimate AC power without bin chasing

Use DPSS multitaper estimates on fixed-length windows, with window length $W$
reported and frequencies in cycles per token. Thomson's multitaper method was
designed to control the finite-sample bias/variance tradeoff and provides
replicate taper estimates; see [Thomson
(1982)](https://doi.org/10.1109/PROC.1982.12433). Welch averaging is an
acceptable cheaper baseline; see [Welch
(1967)](https://doi.org/10.1109/TAU.1967.1161901). A bare rectangular-window
periodogram should not be the headline estimator.

For the matrix estimate $\widehat S_s(f)$, report both:

- raw total power $\operatorname{tr}\widehat S_s(f)$;
- channel-whitened total power, using a regularized covariance transform fit
  on training data.

The raw quantity preserves activation geometry; the whitened quantity prevents
a few high-variance channels from defining the answer. Their disagreement is
itself diagnostic. Zero padding may smooth a plot but must never be described
as improved frequency resolution: the Rayleigh scale remains $1/W$.

Use a small preregistered set of bands, aligned to the model's DCT partition,
rather than testing every bin independently. For example: DC, low AC, middle
AC, and high AC, with exact bin edges stored in the run manifest. Repeat the
analysis at multiple $W$ values so a result is not an artifact of one DCT
grid.

### Core sequence-level metrics

Compute these per independent sequence before aggregating:

- **DC fraction** $q_{\mathrm{DC}}$.

- **Normalized AC band-power vector**
  $p_b=\int_{f\in b}\operatorname{tr}\widehat S(f)\,df/
  \int_{f>0}\operatorname{tr}\widehat S(f)\,df$.

- **Spectral centroid** and **spectral entropy** of the normalized AC
  band-power vector. These distinguish slow concentration from broadband
  activity without asserting a power law.

- **Line excess**: peak power divided by a locally smoothed colored
  background, with significance evaluated against a maximum-over-frequency
  null. Peaks should not be tested against white noise when the background is
  red; [Vaughan
  (2005)](https://doi.org/10.1051/0004-6361:20041453) gives the relevant
  warning and test structure.

- **Spectral effective rank** of
  $\widehat S(f)/\operatorname{tr}\widehat S(f)$ within each band. Equal trace
  power can be carried by one coherent activation direction or diffusely by
  many directions; a spectral crosscoder should behave differently in those
  cases.

- **Red-slope diagnostic**, fit only over a declared range and only after
  comparison with flat, broken-power-law, and smooth curved alternatives.
  Do not infer scale-freeness from a straight-looking log-log plot. The
  model-comparison and bootstrap principles in [Clauset, Shalizi, and Newman
  (2009)](https://doi.org/10.1137/070710111) are the appropriate standard,
  although PSD ordinates require a spectral likelihood rather than their
  independent-tail likelihood.

### Measure task distinguishability directly

The scientific question is not merely whether average spectra differ. Freeze a
low-dimensional feature vector

$$
\phi_s=
[q_{\mathrm{DC}},\log E_{\mathrm{AC}},p_1,\ldots,p_B,
\text{centroid},\text{entropy},\text{line excess},
\text{band ranks}],
$$

then train one regularized linear classifier on $\phi_s$ to predict the
synthetic task identity or ground-truth sequence latent. Split by generating
sequence and seed. Report balanced accuracy, cross-entropy improvement over
the class-prior predictor, and a sequence-bootstrap confidence interval.
Calibrate significance by rerunning the entire classifier under permuted task
labels.

For per-token latents, supplement class-conditional spectra with
cross-spectral coherence between a one-hot encoding of the latent and the
activation sequence. Power alone asks where activations vary; coherence asks
where variation is task-relevant.

A task is **spectrally screenable** only if all of the following hold:

- held-out performance exceeds the 99th percentile of the label-permutation
  null;
- the independent-sequence 95% interval excludes chance;
- the result replicates across generator seeds and at least two compatible
  window lengths;
- an ordered-versus-shuffled effect exceeds both sampling uncertainty and the
  known labeler noise floor;
- no single preprocessing choice, especially DC retention, reverses the
  conclusion.

## Nulls and uncertainty

The unit of resampling is the independent generated sequence or source
document, never an overlapping window or token. Bootstrap sequences within
each task and seed; report medians and percentile intervals across the full
pipeline. For per-bin plots, construct a simultaneous maximum-statistic
envelope from null replicates rather than attaching uncorrected intervals to
every bin.

Use four null families with distinct interpretations:

- **Within-sequence token permutation** preserves the token multiset and exact
  DC coefficient while destroying all order. It is the primary
  temporal-versus-composition null.

- **Block permutation** with several block sizes preserves structure below the
  block scale and destroys longer-range order. It locates the scale at which a
  spectral effect appears.

- **Colored Gaussian or fitted linear-process surrogates** preserve a smooth
  background spectrum but contain no narrow line or nonlinear structure. They
  are the correct null for peak significance.

- **Fourier phase-randomized surrogates** preserve univariate power while
  destroying temporal phase and much higher-order structure. Surrogate-data
  testing was formalized by [Theiler et al.
  (1992)](https://doi.org/10.1016/0167-2789(92)90102-S). For multivariate
  signals, use a construction that explicitly states whether cross-channel
  phase/coherence is preserved. If task information survives only in the
  original and not the phase surrogate, ordinary power is not a sufficient
  screen.

Retain the existing FrequencyBench random-embedding symmetry null. The circle
embedding gives symbol space a frequency geometry; an exchangeable random
embedding does not. A spectral distinction that vanishes under the random
embedding is evidence for meaningful temporal geometry rather than symbol
identity.

## Stationarity and localization gate

Before assigning a single spectrum to a task, split every long sequence into
early/late and coarse time blocks. Compare:

- means and covariance traces;
- normalized band-power vectors using Jensen-Shannon distance;
- line frequency and line power;
- task-classifier accuracy when trained on one time block and tested on
  another.

Bootstrap a threshold for these discrepancies from exchanges of same-time
blocks across independent sequences. If the early-to-late discrepancy exceeds
that threshold, label the task nonstationary and route it to a short-time
Fourier, wavelet, or changepoint analysis. Priestley's
[evolutionary-spectrum framework
(1965)](https://doi.org/10.1111/j.2517-6161.1965.tb01488.x) is the classical
formalization of a time-varying spectrum.

This matters for the repository's tasks. Stable noisy classes are mostly DC;
AR persistence and smooth drift are red; cyclic tones have narrow lines;
changepoints and bursts are broadband but localized; signed direction and
phase-pair tasks can share power while differing in phase; equality-pattern
tasks can evade all second-order spectra. The screen should report that
routing, not force every task onto one low-to-high axis.

## Consequences for the spectral crosscoder

The existing FrequencyBench result already shows that DCT-band structure can
create strong random-initialization access to a tone. A fair new comparison
must therefore keep trained and untrained controls, matched total active-atom
budgets, and raw reconstruction NMSE alongside task recovery.

### Recommended first-order objective

Keep separate DC and AC branches and normalize reconstruction by training-set
band energy:

$$
\mathcal{L}_{\mathrm{bal}}
=
\lambda_{\mathrm{DC}}
\frac{\lVert P_{\mathrm{DC}}(x-\hat x)\rVert^2}
{\mathbb{E}_{\mathrm{train}}\lVert
P_{\mathrm{DC}}(x-\bar x)\rVert^2+\epsilon}
+
\sum_{b\in\mathrm{AC}}\lambda_b
\frac{\lVert P_b(x-\hat x)\rVert^2}
{\mathbb{E}_{\mathrm{train}}\lVert
P_b(x-\bar x)\rVert^2+\epsilon}
+
\mathcal{L}_{\mathrm{sparse}}.
$$

Use equal $\lambda_b$ for the first balanced ablation, clip inverse-power
weights so a nearly empty noisy band cannot dominate, and keep the total
BatchTopK budget matched to TXC. This is band whitening in the *loss*, not
necessarily whitening the final representation. Always report ordinary
unweighted NMSE as the capability check.

Run three DC ablations:

- ordinary full-band reconstruction;
- explicit DC branch with its own atom and sparsity budget;
- AC-only training with sequence mean passed through an unpenalized side
  channel.

Dropping DC without a side channel is acceptable as a diagnostic, not as the
default model: it makes stable temporal tasks impossible by construction.

### Frequency-Matryoshka ablation

Let $B_1\subset B_2\subset\cdots\subset B_m$ be cumulative frequency supports
and require nested model prefixes to reconstruct $P_{B_j}x$. This imports the
independent-prefix idea of Matryoshka SAEs into temporal scale. A naive sum
$\sum_j\lVert P_{B_j}(x-\hat x_j)\rVert^2$ counts the lowest bins in every
term and the highest bins only once, manufacturing the low-frequency bias the
experiment is meant to study.

Use one of two corrections:

- normalize every prefix loss by its band energy and choose per-frequency
  weights so the total coefficient accumulated by each Fourier bin is equal;
- sample one cutoff per batch and apply inverse-inclusion-probability
  weighting to the included bins.

Compare low-to-high nesting with high-to-low nesting and a random nested-band
control. A benefit shared by all three is a generic multi-loss regularization
effect; a benefit unique to low-to-high nesting is evidence for a temporal
scale hierarchy.

### Minimum decisive experiment

At matched parameter count, realized L0, training tokens, window, and seed,
compare:

- TXC;
- current hard-band Spectral-TXC;
- DC-separated Spectral-TXC;
- DC-separated plus band-balanced loss;
- the corrected frequency-Matryoshka variant.

Evaluate all models on at least a DC task, the circle-tone ladder, a
changepoint/localized task, a phase-sensitive task, and the random-embedding
null. The main score is a frontier across task recovery and unweighted NMSE,
not mean recovery pooled over tasks. The strongest useful outcome may be a
router: the screen predicts which architecture wins in each regime, even if
no single spectral architecture dominates TXC everywhere.

## Actionable recommendation

Implement the screen first and freeze its bands and nulls before training new
models. Use it to assign every synthetic task to one of five regimes: DC,
red/broadband stationary, narrow-line, phase-sensitive, or
localized/higher-order. Then make **DC-separated plus band-balanced
reconstruction** the primary spectral-crosscoder candidate. Treat
frequency-Matryoshka as a preregistered secondary ablation with corrected
frequency weights. Do not use “language is low-frequency dominated” as the
model prior: the literature supports long memory and multiscale structure,
while the low-frequency claim remains representation- and observable-specific.
