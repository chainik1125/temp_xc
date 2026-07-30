---
author: Dmitry
date: 2026-07-29
tags:
  - results
  - complete
---

## Controlled frequency and secret-sharing results

## Bottom line

The strongest new result is interpretability, not a universal accuracy win.
On controlled HMMs, the spectral crosscoders put the planted slow and fast
sources in the theoretically expected DCT bands. Spectral v1 and the global
frequency-Matryoshka model both localize 100% of the pure slow and alternating
sources, and 87.5% of the mixed-frequency sources across two seeds. The single
mixed-HMM miss is the borderline \(\lambda=+0.5\) source, whose two lowest
bands have nearly tied probe scores.

The global-selection Matryoshka variant additionally reallocates its sparse
support in the expected direction: its lowest-band selection share is 45.8%
on slow HMMs, while its highest-band share is 56.7% on alternating HMMs.
Spectral v1's fixed per-band quotas remain close to 25% in both cases. This is
the clearest evidence that a frequency-aware sparse dictionary can expose
temporal scale without a post-hoc feature search.

The performance result is more qualified. Frequency Matryoshka slightly
outperforms TXC-post on slow and mixed HMMs, but TXC-post remains best on the
pure alternating HMM. Spectral v1 alone does not beat TXC-post on these
factorial HMMs. The earlier periodic-velocity and multilane benchmarks remain
the large spectral wins; simple high frequency is not sufficient to predict a
win.

The exact polynomial-clock task is also not a spectral success case. TXC is
best on the fresh \(h=1\) diagnostic, and the corrected reviewer-response
\(h=2\) curves rise much faster than the new spectral points. This is
consistent with the secret living in finite-field higher differences rather
than in a dominant linear Fourier mode.

## Protocol

The controlled suite uses independent representation-training, probe-training,
and evaluation episode pools. The observation alphabet or HMM emission matrix
is shared across pools, preventing the alphabet mismatch and repeated-anchor
leakage found in earlier secret-sharing experiments.

All fresh controlled cells use two model seeds. Nominal window support is
matched at \(W\) active entries:

- BatchTopK SAE uses one active entry per token;
- TXC-post uses \(k_{\mathrm{pos}}=W\) shared entries per window;
- Spectral v1 and Spectral Matryoshka use \(k_{\mathrm{pos}}=1\), giving
  \(k_{\mathrm{win}}=W\).

Parameter counts are reported but not matched. Spectral v1 reserves equal
selection budgets in four DCT bands. Spectral Matryoshka uses the same
band-limited dictionary with one global selection pool and a nested
low-to-high frequency reconstruction penalty.

## Factorial frequency HMMs

Each source is a symmetric two-state Markov chain with autocovariance
\(\lambda^{|\ell|}\). Positive \(\lambda\) produces slow, low-frequency
dynamics; negative \(\lambda\) produces alternating, high-frequency dynamics.
The three tasks use \(W=8\):

- slow: \((+0.9,+0.9,+0.9,+0.9)\);
- alternating: \((-0.9,-0.9,-0.9,-0.9)\);
- mixed: \((+0.9,+0.5,-0.5,-0.9)\).

Held-out linear recovery of the full latent trajectories is:

| Model | Slow \(R^2\) | Alternating \(R^2\) | Mixed \(R^2\) |
|---|---:|---:|---:|
| BatchTopK SAE | 0.546 ± 0.006 | 0.541 ± 0.004 | 0.548 ± 0.001 |
| TXC-post | 0.843 ± 0.016 | **0.844 ± 0.006** | 0.742 ± 0.009 |
| Spectral v1 | 0.819 ± 0.021 | 0.745 ± 0.003 | 0.738 ± 0.009 |
| Spectral Matryoshka | **0.858 ± 0.013** | 0.826 ± 0.012 | **0.750 ± 0.007** |

The Matryoshka increments over TXC are small: +0.014 on slow and +0.008 on
mixed. On alternating, TXC is +0.018 above Matryoshka and +0.100 above
Spectral v1. Two seeds are enough to establish the qualitative localization
result, but not to treat these small performance gaps as precise population
effects.

### Frequency localization

The mixed-HMM band-only probes recover the intended ordering:

| Source | Expected band | Spectral v1 band-only \(R^2\) | Matryoshka band-only \(R^2\) |
|---|---|---:|---:|
| \(\lambda=+0.9\) | \(k=0\) | **0.74**, 0.12, -0.00, -0.01 | **0.73**, 0.12, -0.00, -0.02 |
| \(\lambda=+0.5\) | \(k=1\text{--}2\) | 0.27, **0.27**, 0.10, 0.01 | 0.27, **0.27**, 0.10, 0.03 |
| \(\lambda=-0.5\) | \(k=5\text{--}7\) | 0.01, 0.05, 0.11, **0.49** | 0.01, 0.05, 0.10, **0.53** |
| \(\lambda=-0.9\) | \(k=5\text{--}7\) | -0.00, 0.07, 0.16, **0.78** | -0.00, 0.05, 0.11, **0.81** |

The star in the figure marks the band with maximum theoretical DCT energy.
For \(\lambda=+0.5\), the seed-mean values round to a tie; one seed selects
\(k=0\) and the other selects \(k=1\text{--}2\). Every other source is
localized correctly in both seeds.

Time shuffling destroys the mixed and alternating spectral probes
(mean \(R^2\) from -0.333 to -0.046, depending on model and task). The slow
process retains substantial recovery after shuffling because a highly
persistent window contains state-composition information even after its order
is removed. This control therefore distinguishes order-sensitive fast modes
from slow occupancy rather than serving as a universal zero baseline.

## Polynomial-clock secret sharing

The exact construction is

\[
P(t)=B_0+B_1t+\cdots+B_{h-1}t^{h-1}+Yt^h \pmod q,
\]

where \(Y\) is the secret. Any \(W\le h\) distinct observations are independent
of \(Y\), while \(W=h+1\) observations determine it. The symbolic
finite-field interpolation oracle is perfect whenever the threshold is met.

For the fresh \(h=1,q=31\) diagnostic, all models are at the \(1/31\) chance
level at \(W=1\). Chance-normalized secret recovery is:

| Model | \(W=1\) | \(W=2\) | \(W=4\) |
|---|---:|---:|---:|
| BatchTopK SAE | -0.002 | **0.056** | 0.181 |
| TXC-post | -0.004 | 0.053 | **0.620** |
| Spectral v1 | -0.001 | 0.026 | 0.463 |
| Spectral Matryoshka | — | 0.031 | 0.424 |

The small SAE/TXC ordering at \(W=2\) should not be over-read; both recover
only about 5% of the available signal. By \(W=4\), TXC clearly exceeds both
spectral variants.

For \(h=2,q=11\), the new spectral balanced accuracies are:

| Model | \(W=2\) | \(W=3\) | \(W=6\) |
|---|---:|---:|---:|
| Spectral v1 | 0.094 | 0.114 | 0.265 |
| Spectral Matryoshka | 0.095 | 0.099 | 0.233 |

Both satisfy the \(W\le2\) chance ceiling. At the \(W=3\) threshold, Spectral
v1 reaches only 0.114 accuracy. The corrected reviewer-response TXC \(k=2\)
curve is 0.15 at \(W=3\), 0.32 at \(W=4\), 0.56 at \(W=5\), and 0.91 at
\(W=10\); TXC \(k=5\) reaches 0.96 at \(W=10\). Those curves use a broader
independent \(k\) sweep, so the comparison is selection-asymmetric, but they
rule out a spectral secret-sharing win under the tested settings.

This negative result is scientifically useful. Although the leading
coefficient is recovered by an \(h\)-th finite difference in scalar
coordinates, the observed symbols are modular and embedded in arbitrary
orthonormal directions. The task is higher-order temporal binding, not a
power-readable tone.

## Denoising DC versus AC usage

The three-seed replay uses the best Spectral-v1 Denoising cell from the paper
synthetic sweep, \(T=2,k_{\mathrm{pos}}=20\). It separates decoder energy from
task-relevant information.

- Decoded reconstruction energy excluding bias is 30.6% DC and 69.4% AC.
- Activation-weighted decoder-coefficient energy is 20.8% DC and 79.2% AC.
- The observed activation windows are 76.0% DC.
- The hidden support trajectories are 92.5% DC.

Despite the substantial AC reconstruction traffic, the linear hidden-state
probe is completely concentrated in the DC features:

| Code block | Hidden-state Ridge \(R^2\) |
|---|---:|
| Full code | 0.4117 ± 0.0190 |
| DC features only | **0.4118 ± 0.0190** |
| AC features only | -0.00038 ± 0.00001 |

Thus the answer to “are the useful Denoising features DC?” is yes. AC atoms
are active and carry most reconstruction energy, but they add no held-out
linear information about the hidden state. This supports the interpretation
that TXC's Denoising advantage is aligned with a static/DC latent, while the
spectral model spends extra capacity reconstructing AC fluctuations.

## Combined interpretation

The results suggest a routing story rather than a replacement story:

- TXC remains the safest default for generic temporal binding and the
  polynomial-clock task.
- Multiband spectral models are strong when the task is power-readable, as in
  periodic velocity and multilane tones.
- Frequency Matryoshka with global selection is the most promising spectral
  variant: it reallocates firing events to the planted band, slightly beats
  TXC on slow and mixed HMM recovery, and closes most of Spectral v1's
  alternating gap.
- Denoising uses its DC block for the task and its AC block for
  reconstruction. Removing DC globally would destroy the useful signal.
- A high-frequency task is not automatically a spectral win. TXC can learn
  an alternating HMM, while band structure is most valuable when it resolves
  superposed or class-varying frequencies.

The next model ablation should therefore make selection adaptive across bands
without forcing a universal low-frequency preference, and should report both
band occupancy and band-only task recovery. The next task screen should
distinguish linear spectral identity from finite-field or other higher-order
temporal binding.

## Limitations

- The controlled suite has two seeds; small architecture gaps need more seeds
  before headline use.
- The reviewer-response \(h=2\) baselines select the best \(k\) independently
  at each \(W\), while the spectral points use one fixed sparsity rule.
- Only \(h=1\) and \(h=2\) were run. The \(h=3,q=7\) diagnostic was left out
  because the reviewer curve already supplies the relevant \(h=2,W\le10\)
  baseline and the total run was kept under the compute cap.
- The Matryoshka arm bundles global cross-band selection with the nested
  frequency loss. Its adaptive occupancy cannot be attributed to the
  Matryoshka penalty independently without a global-selection/no-penalty
  control under this exact HMM recipe.
- Band localization is linear-probe localization of a feature block, not a
  proof that individual atoms are monosemantic.
- The HMMs are stationary and linear at second order. They do not test phase
  labels, localized changepoints, or higher-order temporal structure.

## Artifacts and cost

The controlled run completed 70/70 cells with no failures. Its conservative
ledger estimate is $4.55. The Denoising replay used 777.3 GPU-seconds, or
$1.08 at the same conservative effective rate of $5/hour. Adding these to the
prior $37.60 experiment estimate gives **$43.23**, below the $50 cap.

Canonical outputs:

- `figures/controlled_shamir_recovery.{png,pdf}`;
- `figures/controlled_hmm_frequency_localization.{png,pdf}`;
- `figures/denoising_frequency_usage.{png,pdf}`;
- `results/controlled_frequency_analysis.{json,csv}`;
- `results/controlled_frequency_provenance.json`;
- `results/controlled_frequency_suite_remote/`;
- `results/denoising_frequency_usage_remote/result.json`.
