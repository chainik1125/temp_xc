---
author: Dmitry
date: 2026-07-30
tags:
  - results
  - complete
---

## Plain Fourier XC on backtracking

## Bottom line

The parameter-matched, non-Matryoshka Fourier XC does not reproduce the
long-window improvement of Aniket's TXC on backtracking. On the recovered
evaluation cohort it is indistinguishable from TXC at \(T=1\), has a
seed-variable advantage at \(T=2\), is slightly behind at \(T=4\) and \(T=6\),
and is clearly behind at \(T=10\). The Fourier curve rises from \(T=1\) to
\(T=2\), then stays almost flat before declining at \(T=10\); Aniket's TXC
curve continues to improve with context.

The frequency decomposition is nevertheless interpretable. DC activation
mass falls monotonically from 100% at \(T=1\) to 56% at \(T=10\), while the
combined AC share rises to 44%. The shares are exceptionally stable across
the three training seeds. Backtracking features are therefore not simply DC
features: DC remains the largest block, but longer windows recruit a large
and reproducible AC component.

## Matched experiment

The Fourier model changes only the temporal parameterisation of Aniket's
reviewer-response recipe:

- 20,000 training steps for \(T=1,2,4,6,10\);
- training seeds 1, 2, and 42;
- batch size 1,024, learning rate \(3\times10^{-4}\), 1,000 warmup steps;
- the same reconstruction, AuxK, dead-feature, and decoder-projection rules;
- global per-example TopK\((20T)\) followed by ReLU;
- real orthonormal Fourier bands with sine/cosine partners kept together;
- no Matryoshka loss, adaptive frequency loss, or learned routing;
- widths chosen separately at each \(T\) to match the TXC trainable parameter
  count.

The Fourier widths are 32,768, 65,532, 98,298, 131,064, and 131,067 atoms.
The parameter mismatch is zero at \(T=1\), at most eight parameters for
\(T=2,4,6\), and 8,197 out of 2.684 billion parameters at \(T=10\).

The primary metric is Aniket's fixed 32-feature sparse logistic probe,
evaluated in five question-grouped folds. Values below are means and sample
standard deviations over the three representation-training seeds.

| T | Fourier XC sensitivity | TXC pinned reference | Difference |
|---:|---:|---:|---:|
| 1 | 0.2184 ± 0.0016 | 0.2178 ± 0.0049 | +0.0006 |
| 2 | 0.2428 ± 0.0189 | 0.2289 ± 0.0064 | +0.0139 |
| 4 | 0.2429 ± 0.0125 | 0.2466 ± 0.0072 | -0.0037 |
| 6 | 0.2446 ± 0.0059 | 0.2512 ± 0.0058 | -0.0066 |
| 10 | 0.2398 ± 0.0067 | 0.2548 ± 0.0085 | -0.0150 |

These differences are descriptive, not paired estimates: the Fourier and TXC
curves do not share the same ordered evaluation cohort.

## Frequency use

The table reports activation-mass shares, not dictionary sizes. All
frequency blocks at a given \(T\) have matched atom counts apart from integer
rounding, and selection competes globally across them.

| T | DC share | Combined AC share |
|---:|---:|---:|
| 1 | 1.0000 ± 0.0000 | 0.0000 |
| 2 | 0.7932 ± 0.0086 | 0.2068 |
| 4 | 0.6566 ± 0.0092 | 0.3434 |
| 6 | 0.5982 ± 0.0127 | 0.4018 |
| 10 | 0.5592 ± 0.0018 | 0.4408 |

At \(T=10\), the remaining activation mass is split consistently across the
three AC blocks: 15.84%, 17.44%, and 10.81%. This is useful interpretability
"for free": the fixed Fourier support makes it possible to say which temporal
subspaces the selected features occupy without fitting a post-hoc spectral
probe.

It is not, however, evidence that the AC features improve the task. The
longest window has the largest AC share and the largest deficit relative to
TXC. A direct causal test should clamp individual bands or retrain with a
matched DC-only/AC-only support budget.

The fixed-probe ordering controls provide a separate diagnostic. The
three-seed mean PR-AUC gap between ordered codes and shuffled codes is 0 at
\(T=1\), 0.0005 at \(T=2\), 0.0083 at \(T=4\), 0.0123 at \(T=6\), and 0.0048
at \(T=10\). The model does encode temporal order, but the order signal peaks
at \(T=6\) rather than explaining the whole long-window curve.

## Artifact boundary

Aniket's exact 20,335-row T16 activation artifact is not present in Git, the
available RunPod volumes, or the experiment handoff. Replaying the frozen
extractor under the current CUDA stack gives small numerical differences and
a different eligible cohort. The recovered artifact therefore preserves the
published row count and class balance but not the ordered cohort hash:

- recovered artifact SHA-256:
  `1681b7e6ef68ccc207a5d9af2c4ba3d4646056ccdca0bc3d5c09bc3b43c2125f`;
- recovered cohort SHA-256:
  `9137bda110780afc1965f453669d434100e1e495bd8f4dfaf8713c4bb6516c0d`;
- pinned reference cohort SHA-256:
  `f397f4caf6212825bd98b1b82be932ae634f01a716fd7e3642fd3d7640b27c0b`.

The first ten offsets are replayed activations; the last six offsets are
copied bit-for-bit from the official artifact. Consequently this result is a
recovered-artifact sensitivity analysis, not a bit-exact replication or a
clean causal comparison to the pinned TXC/SAE curves. The plot carries this
warning prominently.

## Artifacts and compute

The publication bundle is under
`results/backtracking_fourier_matched/reviewer-five-point-v1/publication/`:

- `backtracking_fourier_summary.png` and `.pdf`: comparison and band-use plot;
- `backtracking_fourier_summary.json`: exact aggregates and provenance;
- `backtracking_fourier_summary.csv`: tidy plotting table.

All 15 cell results and training summaries are stored alongside the
publication bundle. The sweep ran on three RunPod H100s for the endpoint
cells and three RunPod RTX 5090s for the middle windows. The recovery job used
a short B200 allocation. Wall time multiplied by the listed instance rates
gives a conservative total of approximately $37, below the $50 cap. All six
training/evaluation pods are in `EXITED` state.
