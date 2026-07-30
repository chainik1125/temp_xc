---
author: Dmitry
date: 2026-07-30
tags:
  - results
  - complete
---

## Routed Fourier Spectral Matryoshka

## Bottom line

The spectral advantage exists, but in a narrower regime than “high
frequency.” A TXC can learn a single alternating process directly. The large
spectral win appears when each episode contains several simultaneous,
random-phase narrowband causes and the model must separate *which cause* is
present from *at which frequency and phase* it evolves.

Across two seeds, a real-Fourier spectral crosscoder reaches direct latent
reconstruction \(R^2=0.963\) on balanced-frequency mixtures and \(0.950\) on
high-frequency-crowded mixtures. The matched TXC reaches \(0.762\) and
\(0.760\); the matched token SAE reaches only \(0.127\) and \(0.125\). A DCT
spectral model is intermediate at \(0.835\) and \(0.837\). This is a large,
replicated architecture-level effect under the tested construction.

The new learned Spectral Matryoshka is more qualified. Learning relative
frequency-loss weights without imposing a low-to-high ordering does not
improve on the unweighted Fourier model. Letting those weights route global
BatchTopK competition gives a small improvement only in the high-frequency-
crowded task: \(+0.0039\) mean \(R^2\) over Fourier global. The gain is
positive in both seeds but varies from \(+0.0071\) to \(+0.0007\), so it is a
promising curriculum effect rather than a headline result.

## Regime designed to expose a spectral advantage

Each sparse narrowband task has:

- \(T=8\), input width 48, and 24 ground-truth causes;
- exactly four active causes per episode;
- a random phase and amplitude for every active cause;
- orthogonal sine/cosine emission planes, making direct latent recovery
  identifiable without a learned probe;
- exact Fourier-bin frequencies \(1/8\), \(2/8\), and \(3/8\);
- either 8/8/8 causes across the three AC frequencies, or a high-frequency-
  crowded 3/6/15 allocation.

The last contrast tests whether learned routing reacts to dictionary demand,
not merely to signal power. The observed data spectrum differs sharply
between the two tasks even though both contain four active causes per episode.

The evaluation matches nominal support at \(L_0=8\) per window and trainable
parameters near 49,600:

- TXC: 49,600 parameters and 64 unconstrained temporal atoms;
- spectral arms: 49,503--49,507 parameters and 255 band-limited atoms;
- BatchTopK SAE: 49,712 parameters and 512 token atoms.

The SAE trains with one selection per token. Its inference threshold is
calibrated to exactly \(L_0=8\) on the episode-disjoint probe split; held-out
evaluation \(L_0\) is 8.34 on balanced and 8.03 on crowded. Crosscoder
evaluation \(L_0\) ranges from 7.69 to 8.16.

## Reconstruction results

The primary metric projects reconstructed observations onto the known
orthonormal emission planes. An all-zero reconstruction scores exactly zero,
so inactive causes cannot inflate the result.

| Task | SAE | TXC | DCT global | Fourier global | Learned loss | Learned routing |
|---|---:|---:|---:|---:|---:|---:|
| Sparse balanced | 0.1268 ± 0.0032 | 0.7619 ± 0.0145 | 0.8347 ± 0.0009 | **0.9628 ± 0.0004** | 0.9627 ± 0.0005 | 0.9626 ± 0.0005 |
| Sparse high-crowded | 0.1252 ± 0.0059 | 0.7600 ± 0.0006 | 0.8369 ± 0.0044 | 0.9502 ± 0.0050 | 0.9496 ± 0.0044 | **0.9541 ± 0.0005** |

The Fourier gain over TXC is \(+0.2010\) on balanced and \(+0.1902\) on
high-crowded. DCT improves over TXC by only \(+0.0728\) and \(+0.0769\).
This Fourier--DCT gap matters: the source phase is random, so sine/cosine
pairs form a phase-equivariant two-dimensional subspace. A fixed DCT
coordinate must approximate phase shifts less naturally.

The active-only diagnostic gives the same story. Fourier global obtains
\(R^2=0.9775\) and \(0.9674\) on active causes, versus \(0.8397\) and
\(0.8386\) for TXC. Fourier also leaks less reconstruction into inactive
emission planes: approximately 0.3% of active target energy, versus 1.6% for
TXC.

The token SAE is intentionally a standard matched-budget baseline. Its low
score is expected in this regime: one token-level selection must explain a
token containing four superposed causes, while the window models can assign
their eight selections across sources and temporal modes.

## Why alternating HMM is not the advantage regime

The same two-seed run includes slow and alternating factorial-HMM controls:

| Task | TXC | Fourier global | Learned loss | Learned routing |
|---|---:|---:|---:|---:|
| Slow HMM | **0.8194 ± 0.0106** | 0.8025 ± 0.0037 | 0.8023 ± 0.0036 | 0.8025 ± 0.0036 |
| Alternating HMM | **0.8149 ± 0.0060** | 0.7655 ± 0.0061 | 0.7659 ± 0.0061 | 0.7656 ± 0.0065 |

A single alternating source has a distinctive high-frequency spectrum, but
it does not require a spectral dictionary. An unconstrained TXC atom can
learn the \((-1)^t\)-like template directly. Fourier structure pays off when
frequency is a reusable factor shared across many simultaneous identities
and phases, not when “the task is fast” in isolation.

## Interpretability from frequency-constrained features

The frequency blocks light up in the expected proportions. The table compares
expected latent Fourier power with routed-model selection events, ordered as
DC / Fourier 1 / Fourier 2 / Fourier 3--4:

| Task | Expected power | Routed selection share |
|---|---|---|
| Slow HMM | 78.10 / 14.10 / 4.25 / 3.55% | 43.22 / 23.72 / 16.06 / 16.99% |
| Alternating HMM | 1.06 / 2.54 / 4.35 / 92.05% | 5.94 / 12.48 / 15.25 / 66.34% |
| Sparse balanced | 0 / 33.33 / 33.33 / 33.33% | 4.20 / 32.01 / 31.97 / 31.82% |
| Sparse high-crowded | 0 / 12.50 / 25.00 / 62.50% | 2.12 / 12.93 / 23.84 / 61.12% |

Thus a simple count of feature firings recovers the planted spectral mixture
almost exactly in the sparse tasks. Every spectral task/model/seed cell also
assigns all 24 causes to the expected frequency band. This is useful
interpretability “for free” at the *band* level.

That claim has an important boundary: band identity is imposed by the decoder
parameterization, and the frequencies are exactly on-grid. The result shows
that the model uses the correct constrained blocks; it does not show that
unconstrained atoms would discover frequency, nor that individual atoms are
monosemantic.

Three quantities must remain distinct:

- expected band power is a property of the data;
- selection share is observed model use;
- learned difficulty weight is loss emphasis, not estimated power.

## The new order-free Matryoshka

The previous low-to-high nesting encoded a preference the task did not
justify. Version 2.2 instead gives every frequency band a learned relative
weight. The weights are initialized from band width, normalized jointly, and
updated adversarially from band-specific residual difficulty with entropy and
floor regularization. There is no fixed “low frequencies are cheapest”
ordering.

The routed variant turns these learned weights into detached
\(\sqrt{w_b/p_b}\) multipliers on the global BatchTopK *scores*. The selected
code amplitudes themselves are unchanged. This lets a difficult band win more
support without allowing the routing scale to manufacture reconstruction
amplitude.

On the high-crowded task, the high-frequency routing multiplier rises to
1.106 at step 1,500 while the middle bands fall to about 0.92. This is the
correct direction and coincides with the small routed improvement. On the
balanced task, the largest excursion is only about 5.8%.

All routing multipliers return to within 0.1% of one by the end of training.
The learned loss weights similarly return near their band-width prior
\([0.125,0.25,0.25,0.375]\). The current evidence therefore supports a
*transient routing curriculum*, not a stable learned frequency allocation.
The main performance win comes from Fourier factorization itself.

## What the power-spectrum screen predicts

The screen now supports the following working prediction:

- expect a spectral advantage for simultaneous, phase-varying causes whose
  identities reuse a small set of frequencies;
- expect a larger advantage as more cause identities compete within the same
  window and temporal mode becomes a reusable factor;
- do not predict an advantage from high frequency alone;
- do not expect linear power to solve finite-field secret sharing or other
  higher-order temporal binding;
- use DC removal only as a targeted ablation, because earlier Denoising
  experiments place essentially all task-relevant hidden-state information in
  DC features.

The next decisive tests are off-grid frequencies, drifting frequencies or
chirps, partially overlapping emission directions, and phase-labelled tasks.
Those would test whether the Fourier advantage survives beyond its current
best-case construction.

## Limitations

- There are two model seeds, enough for replication but not inferential
  uncertainty.
- Frequencies are stationary and exactly aligned to Fourier bins.
- Emission planes are orthogonal and the ground-truth frequency labels are
  known.
- Band localization is partly guaranteed by architectural constraints.
- Parameter count and nominal support are matched, but atom count is not.
- The adaptive residual uses a known-noise subtraction heuristic. Because the
  reconstruction depends on the noisy input, this is not an unbiased residual
  correction.
- The routed improvement is small and its magnitude varies across the two
  seeds.

## Artifacts, source, and cost

The two routed panels completed 36/36 non-smoke cells with no failures. The
matched SAE completed 4/4 cells, and the support-calibrated evaluation reused
the frozen checkpoints rather than retraining.

Canonical artifacts:

- `figures/spectral_matryoshka_routed/spectral_advantage_direct_latent_r2.{png,pdf}`;
- `figures/spectral_matryoshka_routed/spectral_advantage_frequency_diagnostics.{png,pdf}`;
- `figures/spectral_matryoshka_routed/spectral_matryoshka_routing_trajectories.{png,pdf}`;
- `results/spectral_matryoshka_routed_analysis.{json,csv}`;
- `results/spectral_matryoshka_provenance.json`;
- the four frozen remote-result directories named in the provenance file.

The exact launch commits were `4b415cf21` for routed seed 1,
`c3d84d01e` for seed 2, `6ce05fe08` for the SAE training panel, and
`e9c2af750` for support-calibrated evaluation. The final conservative running
total is **$47.245971**, leaving **$2.754029** under the $50 cap.
