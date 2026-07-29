---
author: Dmitry
date: 2026-07-29
tags:
  - results
  - complete
---

## Synthetic task-spectrum screen

## Answer

Frequency information can screen temporal tasks, but a scalar power spectrum
is not a sufficient universal screen.

The useful object is a *routing vector* with at least four components:

- signed DC coefficients for stable or slowly changing state;
- AC power for magnitude and periodicity;
- complex cross-spectral terms for phase and direction;
- sensitivity to window length for localization and nonstationarity.

Across the synthetic panel, ordinary AC power is nearly perfect for clean
periodic magnitude tasks and progressively recovers one tone inside a
three-tone superposition. It remains exactly at chance for two direction/sign
tasks whose information lives in cross-channel phase. Global spectral
summaries overlap substantially across otherwise different tasks. This argues
for using a spectral routing vector to choose an architecture or loss, not for
assigning a universal scalar "temporality" score.

## Protocol

For every datasource and seed, the generator materializes 640 independent
sequences from one synthetic world:

- 128 sequences fit an eight-dimensional PCA basis without using probe
  examples or labels;
- 512 disjoint sequences form the probe set;
- only the leading window from each sequence is used, so a hidden
  sequence-level phase or state cannot cross validation folds;
- window lengths are 4, 8, 16, and 32 tokens;
- results average three generator seeds and five stratified or ordinary
  cross-validation folds;
- linear probes are compared with shuffled-label nulls.

The three representations are:

- **Power (AC):** normalized diagonal spectral power after per-window
  demeaning, plus log total AC energy;
- **Cross-spectrum:** Power (AC) plus real and imaginary off-diagonal
  cross-spectral terms;
- **Signed DC:** the unsquared mean coefficient vector in the calibrated PCA
  basis.

Classification numbers are balanced accuracy; regression numbers are
out-of-fold $R^2$.

## Main results

Selected three-seed means are:

| Task target | Window | Chance | AC power | Cross-spectrum | Signed DC |
|---|---:|---:|---:|---:|---:|
| periodic velocity | 4 | 0.100 | 0.625 | 0.494 | 0.127 |
| periodic velocity | 32 | 0.100 | 0.991 | 0.864 | 0.156 |
| periodic magnitude | 4 | 0.333 | 0.990 | 0.951 | 0.356 |
| multilane lane-0 velocity | 4 | 0.100 | 0.242 | 0.286 | 0.105 |
| multilane lane-0 velocity | 8 | 0.100 | 0.485 | 0.546 | 0.122 |
| multilane lane-0 velocity | 16 | 0.100 | 0.708 | 0.725 | 0.141 |
| multilane lane-0 velocity | 32 | 0.100 | 0.790 | 0.827 | 0.134 |
| phase-only sign | 4 | 0.500 | 0.483 | 0.826 | 0.507 |
| phase-only sign | 32 | 0.500 | 0.517 | 0.984 | 0.496 |
| motion direction | 4 | 0.500 | 0.493 | 0.933 | 0.496 |
| motion direction | 32 | 0.500 | 0.501 | 1.000 | 0.511 |
| permuted schedule | 4 | 0.100 | 0.128 | 0.114 | 0.109 |
| permuted schedule | 32 | 0.100 | 0.527 | 0.415 | 0.102 |
| recipe equality | 4 | 0.500 | 0.686 | 0.655 | 0.597 |
| recipe equality | 32 | 0.500 | 0.480 | 0.486 | 0.562 |

These results give five clean conclusions.

### Power detects periodic magnitude

For FrequencyBench velocity, AC-power accuracy rises from 0.625 at four tokens
to 0.991 at 32 tokens. The phase-pair magnitude class is already 0.990 at four
tokens. These are the tasks for which a band-partitioned spectral crosscoder
has a strong mechanistic prior.

### Multilane power gives a benchmark prediction

The multilane target selects lane 0 from three simultaneous circle tones and
predicts its ten-way velocity class. AC-power accuracy rises monotonically
from 0.242 at four tokens to 0.485, 0.708, and 0.790 at 8, 16, and 32 tokens.
The corresponding shuffled-label means stay near 0.100. Cross-spectral
accuracy is 0.286, 0.546, 0.725, and 0.827, while signed DC remains weak at
0.105--0.141.

This gives a specific screen-to-benchmark prediction. Lane velocity is
primarily a power-aligned target embedded in spectral superposition, not a
stable-mean target. A band-partitioned or band-balanced crosscoder should
therefore have a relevant inductive bias for multilane recovery. The four-token
cell has both detectable spectral signal and substantial headroom, so it is a
better architecture discriminator than the longer windows, where even the
fixed screen approaches high accuracy. The small cross-minus-power increment
of 0.017--0.061 leaves room for phase-aware structure but does not make phase
the primary prediction.

This is a directional prediction, not a promised ranking or effect size. The
screen uses fixed spectral summaries rather than learned sparse codes, and it
targets one lane while the benchmark averages recovery over all lanes. Its
role is to identify the mechanism and informative window regime; matched
training results must decide whether any spectral variant actually beats TXC.

### Power erases direction

The phase-pair sign and signed-motion direction tasks were designed so the
opposing labels have the same marginal power. AC-power accuracy remains at
chance for every window. The imaginary cross-spectrum raises accuracy to
0.826--0.984 and 0.933--1.000 respectively. A real power-only screen would
misclassify these as non-temporal even though their time dependence is
extremely strong.

This is also why a DCT representation is not the exact finite-window
diagonalization of a translation-invariant process. A complex DFT or an
explicit sine/cosine pair is needed to preserve general phase.

### DC is task signal, not merely nuisance

Removing DC helps isolate periodic AC structure, but it is unsafe as a
universal model change. At a four-token window:

- signed DC predicts changepoint mode at 0.478 balanced accuracy versus 0.125
  chance;
- it predicts the backtracking control parameter at $R^2=0.651$;
- it predicts hedging confidence at $R^2=0.307$, while AC power is below zero.

For backtracking, the full power representation reaches $R^2=0.868$, compared
with 0.445 after DC removal. The correct ablation is therefore a separate DC
branch and budget, alongside a remove-DC negative control.

### Localization is visible as window instability

Backtracking strength is recoverable from a four-token window but collapses as
the window grows: AC-power $R^2$ moves from 0.445 at four tokens to -0.477 at
32 tokens. Changepoint time-since-switch behaves similarly. An averaged long
window washes out local state.

The permuted schedule shows the opposite resolution effect. It is essentially
invisible at four or eight tokens, then reaches 0.527 power accuracy at 32
tokens. This is broadband, task-specific structure rather than evidence that
low frequencies are generically dominant.

## Global spectra do not order the tasks

Across raw task activation distributions:

- DC fractions range from roughly 0.1% to 12%;
- low-frequency shares of AC power range from roughly 22% to 58%;
- normalized AC spectral entropy ranges from 0.74 to nearly 1.00.

Those summaries identify the deliberately narrow phase-pair process and the
nearly broadband permuted process, but many qualitatively different tasks
cluster together. Filler channels and task-independent variance can dominate a
trace spectrum. A task-conditioned probe is much more informative than ranking
tasks by low-frequency mass alone.

These are global, time-averaged summaries. They cannot prove stationarity:
a localized or drifting process can share the same global power with a
stationary process. Stationarity requires blockwise or position-conditioned
spectra and uncertainty over independent sequences or documents.

## Proposed routing screen

For a candidate temporal task:

1. Estimate DC separately and estimate AC spectra without crossing sequence or
   document boundaries.
2. Compare signed-DC, power, and complex cross-spectral probes against
   sequence-level shuffled-label nulls.
3. Repeat over several window lengths.
4. Route high power-excess tasks to a banded spectral model.
5. Route high cross-minus-power tasks to a phase-aware complex or paired
   sine/cosine model.
6. Preserve a DC branch when signed DC carries task signal.
7. Route strongly window-sensitive tasks to localized or multiresolution
   models.
8. Treat near-null second-order screens as a reason to test higher-order and
   explicitly positional structure, not as proof that the task is
   non-temporal.

For language activations, the confirmatory version should replace the simple
Hann periodogram with multitaper or Welch estimates, bootstrap over documents,
and test stationarity across document position and domain. Those upgrades are
specified in [[theory_and_literature]].

## Consequences for the crosscoder benchmark

The overnight benchmark therefore compares:

- TXC-pre as the equal-window-support baseline;
- TXC-post as the position-mixing, lower-code-support comparator;
- the existing multiband spectral TXC;
- explicit DC removal;
- a bandwidth-normalized band-dominance penalty;
- a frequency-Matryoshka reconstruction loss;
- their combination;
- global top-k selection, which removes forced per-band occupancy.

The screen predicts that no single spectral variant should win every task.
Multiband models should be strongest on periodic magnitude tasks. The
multilane result sharpens that prediction: at the four-token benchmark window,
lane power is above chance but far from saturated, so band partitioning has a
plausible superposition advantage with measurable headroom. Phase-only and
permuted controls test whether gains are merely DCT alignment, and the colored
task tests recovery from temporal covariance. The benchmark has no
DC-positive stable-state task, so it cannot establish that deleting DC is
generally safe. The signed-DC and cross-spectral screening arms remain
essential because power alone cannot recover stable state, direction, or sign.

## Artifacts

- `results/task_screen.json`: full nested results and configuration.
- `results/task_screen.csv`: one row per target, seed, and window.
- `results/task_screen_aggregate.csv`: three-seed aggregates.
- `figures/task_screen_separability.png`: representation-by-window comparison.
- `figures/task_spectrum_summary.png`: global spectrum overlap.
- `code/run_task_screen.py`: reproducible runner.
- `code/plot_task_screen.py`: aggregation and plotting.
