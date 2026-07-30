---
author: Dmitry
date: 2026-07-29
tags:
  - results
  - complete
---

## Spectral-crosscoder benchmark

## Answer

The spectrum is useful for *routing* temporal tasks, but ordinary power is not
a universal measure of temporality. The screen correctly predicts that
band-partitioned models work well when task information is encoded in periodic
magnitude, and fail when the same marginal power hides a phase or direction
label.

The learned-model result has two parts:

- allowing post-nonlinearity temporal position mixing with approximately
  \(kT\) support explains most of the improvement over the original TXCs;
- explicit multiband structure then provides a smaller, task-dependent
  inductive-bias benefit on power-readable periodic tasks.

None of the proposed v2 penalties is a universal improvement. Removing DC is a
useful high-pass ablation but trades recovery against reconstruction and would
delete real signal on DC-positive tasks. Band-power flattening gives only a
small frequency-task increment. The tested frequency-Matryoshka objective is
effectively neutral. A phase-aware complex or paired sine/cosine route is the
most important next architecture, because the real banded DCT models fail the
phase-only control despite reconstructing it well.

## Question

Can a frequency-domain task screen predict when spectral structure is useful,
and do DC removal, band-power flattening, frequency-Matryoshka loss, or global
support selection improve the existing spectral temporal crosscoder?

The primary metrics are normalized task-recovery scores. Frequency,
multilane, phasepair, and permuted tasks use linear probes of the learned
codes; colored sources use chance-adjusted decoder-direction recovery.
Absolute scores are therefore comparable between models *within* a task, not
between tasks.

## Benchmark design

The main panel contains eight models, five tasks, and three seeds, for 120
full-training cells. Every cell uses the same datasource, seed, optimizer
steps, learning rate, dictionary width, and approximately 1,024 reconstructed
token positions per step. Realized L0 per window, NMSE, and parameter count
are reported alongside recovery.

The two repository TXCs are not interchangeable baselines:

- TXC-pre has approximately the same realized active-code count per window as
  the spectral models, but applies its nonlinearity before temporal mixing.
- TXC-post mixes positions before its nonlinearity, but has only about one
  active code per window rather than approximately \(T\).

The additional `v2_full_global` control is therefore essential. A full DCT
basis spans every temporal kernel, so this model is an orthonormal
reparameterization of a monolithic post-squash temporal model with global
\(kT\)-scale support. It matches the multiband model's window support and
realized L0, though not its parameter count.

## Model ablations

- `spectral_v1`: the existing multiband model with reserved support per band.
- `v2_remove_dc`: centers each window, hard-masks DCT-0 decoder coefficients,
  and still reconstructs the original uncentered target.
- `v2_dominance`: a one-sided width-normalized band-power-flattening penalty.
  The name is retained in machine-readable outputs, but this is not a learned
  dominance detector.
- `v2_freq_matryoshka`: nested low-to-high reconstruction losses at frequency
  boundaries. Under disjoint per-band support it is mathematically a mild
  low-frequency loss reweighting, not nested feature dictionaries.
- `v2_combined`: DC removal plus band-power flattening.
- `v2_global`: retains multiband atoms but replaces reserved per-band support
  with one global BatchTopK pool.
- `v2_full_global`: the matched-support monolithic control.

## Integrity and seed sensitivity

The canonical analyzer requires the exact expected task/model/seed panel,
successful status, exact target training horizon, finite primary/NMSE/L0
metrics, consistent parameter counts, unique training identities, and matched
batch-token accounting before it permits plotting.

Seeds 1 and 2 start full training from step 0. In the main panel, seed 42
continues the smoke-to-gate checkpoint and therefore does not restore the
synthetic window buffer's data-stream state. All main models share this
continuation recipe, but seed 42 is not exchangeable with seeds 1 and 2. The
analysis reports fresh seeds 1--2 as the primary sensitivity check and the
three-seed panel as supporting evidence. The matched full-band control uses
fresh full-training seeds 1, 2, and 42.

## Complete benchmark results

The strict gate accepts all 120 main cells and all 15 matched-control cells.
There are no missing, failed, unexpected, retried, or fairness-invalid cells.
The table reports three-seed mean normalized recovery after the colored-source
chance correction for hard-DC masks:

| Model | Periodic | Multilane | Phase sign | Permuted | Colored |
|---|---:|---:|---:|---:|---:|
| TXC-pre | 0.143 | 0.013 | -0.013 | 0.014 | 0.085 |
| TXC-post | 0.336 | 0.129 | 0.722 | 0.063 | -0.102 |
| spectral v1 | 0.960 | 0.469 | -0.014 | 0.122 | 0.107 |
| remove DC | 0.954 | 0.503 | -0.007 | 0.091 | 0.103 |
| band flattening | **0.965** | 0.468 | -0.008 | 0.119 | **0.110** |
| frequency-Matryoshka | 0.960 | 0.468 | -0.012 | 0.122 | 0.106 |
| combined | 0.958 | **0.503** | -0.010 | 0.091 | 0.103 |
| global top-k | 0.959 | 0.421 | -0.011 | 0.111 | 0.109 |
| full-band control | 0.928 | 0.394 | **0.999** | **0.136** | -0.067 |

The decimals should not be over-interpreted as independent discoveries. Band
flattening's 0.005 improvement over spectral v1 on periodic velocity is
consistent in all three paired seeds but small. Its 0.003 colored-source
increment and global top-k's 0.002 increment are similarly modest. The
frequency-Matryoshka result is numerically indistinguishable from spectral v1.

The task screen's qualitative predictions are borne out:

- periodic magnitude is power-readable, and every multiband model beats both
  TXCs by a large margin;
- simultaneous multilane tones favor band structure and reserved per-band
  occupancy;
- phase-only sign is invisible to power and fails under every banded model,
  while the full-band control reaches 0.999;
- the permuted schedule is weakly screenable at \(T=8\), and every model
  remains weak;
- colored covariance is modestly recovered by multiband models, while the
  support-matched full-band control is below chance.

The phasepair result is the strongest falsification of a scalar-power story.
The keep-DC banded models reach NMSE 0.197--0.214 without exposing the label
linearly, whereas the full-band control reaches 0.999 recovery at NMSE 0.177.
Good reconstruction therefore does not imply preservation of phase-sensitive
task information.

For colored sources, the repository evaluator originally counted zeroed DCT-0
decoder slices when estimating chance for hard-DC models. The canonical
analyzer recomputes chance using 192 nonzero candidates rather than 256 and
records the correction. The corrected remove-DC and combined scores are 0.103,
not the raw summary's 0.085.

### Fresh-seed sensitivity

Seeds 1 and 2 are the clean fresh-training comparison. On periodic velocity,
spectral v1 exceeds the full-band control by 0.029 on average, with paired
differences of 0.033 and 0.025. Their NMSE is effectively identical and their
realized L0 differs by about 1%, while spectral v1 uses 25% as many parameters.

On multilane, spectral v1 exceeds the full-band control by 0.079 in the fresh
seeds, with paired differences of 0.085 and 0.073. This is an accessibility
win rather than a reconstruction win: spectral v1's NMSE is about 0.503 versus
0.403 for the full-band model. The three-seed differences, 0.031 and 0.076 on
periodic and multilane respectively, have the same sign as the fresh-only
comparison.

## Architecture interpretation

The matched control completed 15/15 full cells with no failures, missing
cells, retries, identity collisions, or fairness errors.

On the two completed main tasks, multiband spectral v1 improves task recovery
over the full-band control at closely matched realized L0:

| Task | Full-band | Multiband v1 | Multiband minus full | NMSE full | NMSE multiband |
|---|---:|---:|---:|---:|---:|
| periodic velocity | 0.928 | 0.960 | +0.031 | 0.555 | 0.556 |
| multilane | 0.394 | 0.469 | +0.076 | 0.400 | 0.504 |

The fresh-seed paired differences are +0.025 and +0.032 for periodic
velocity, and +0.073 and +0.085 for multilane. The frequency result is a
cleaner architecture win because reconstruction is unchanged. Multilane
instead shows improved linear access to simultaneous frequency latents at the
cost of worse aggregate reconstruction.

The multiband model uses roughly one quarter of the full-band model's
parameters. This is a favorable efficiency result, but it also means the
control isolates a *bundled band prior*--band-limited atoms, support
allocation, and reduced parameterization--rather than any one component.

On periodic velocity, global support selection within the multiband model is
essentially tied with reserved per-band support. The remaining gain over the
full-band model is therefore mostly attributable to the band-limited
parameterization. On multilane, the decomposition

```text
full-band/global  ->  multiband/global  ->  multiband/per-band
      0.394                0.421                  0.469
```

shows contributions from both band limitation and reserved per-band
occupancy. This is the benchmark's clearest superposition result.

The phase-only control supplies the corresponding failure mode. The full-band
control reaches 0.999 recovery, whereas every banded spectral variant is near
zero. Binning frequencies independently therefore exposes simultaneous
power-aligned tones but prevents the cross-band combinations required for the
phase/sign label. TXC-post reaches 0.722, supporting the same distinction:
position mixing helps, but the band partition can be actively harmful when
phase is the task variable.

The v2 regularizers do not rescue that failure:

- band-power flattening adds about 0.005 on periodic velocity and is neutral
  or worse elsewhere;
- frequency-Matryoshka is effectively tied with spectral v1;
- DC removal raises multilane recovery from 0.469 to 0.503 but raises NMSE
  from 0.504 to 0.688;
- combining DC removal with flattening behaves like DC removal alone;
- no single variant wins across task types.

## Interpretation contract

The final claims must distinguish:

- task-relevant recovery from aggregate reconstruction;
- band limitation from forced per-band occupancy;
- a hard high-pass DC ablation from a generally useful regularizer;
- low-frequency loss reweighting from a feature-level Matryoshka dictionary;
- synthetic mechanism validation from evidence about natural language.

In particular, the benchmark contains no DC-positive stable-state task. The
screening results on changepoints, backtracking, and hedging therefore remain
the reason not to generalize a remove-DC win from this panel.

## Cost and provenance

The main run used a conservative estimated $27.48 and the matched control
$1.67, for a combined conservative estimate of **$29.15**. This is a
deliberately inflated ledger estimate rather than a Modal invoice and remains
well below the $50 cap.

The run used one A10G sequentially. Its completed GPU session ran for about
5.46 hours; the short smoke and interrupted-launch sessions are included in
the cost. Both final session ledgers report `complete`, and the main Modal app
reports `stopped` with zero active tasks.

Machine-readable provenance in `results/provenance.json` records:

- the exact main and control source commits;
- Modal app and function-call IDs;
- equality of each tracked and frozen configuration;
- SHA-256 hashes for raw results, summaries, spend ledgers, gate reports,
  frozen configs, and plans;
- the combined cost and integrity receipt.

Canonical outputs are in `results/overnight_remote/`; the four final benchmark
figures are in `figures/`.
