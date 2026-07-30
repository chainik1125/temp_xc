## Backtracking replication: plain Fourier XC

This isolated experiment compares a parameter-matched Fourier crosscoder to
Aniket's reviewer five-point TXC backtracking curve.

The frozen reference is `origin/neurips-aniket` commit `d9c7fc7b2`, protocol
`2026-07-26.t16.1`:

- windows: `1, 2, 4, 6, 10`
- training seeds: `1, 2, 42`
- 20,000 steps, batch size 1,024, learning rate `3e-4`
- warmup 1,000 steps and schedule seed `907000 + 100 * train_seed`
- fixed 20,335-row T16-valid evaluation cohort
- five question-grouped folds and fixed sparse probes at 8, 16, and 32 features

The new model changes only the temporal parameterisation. It uses the same
per-example `TopK(20*T)` followed by ReLU, reconstruction loss, AuxK,
dead-feature bookkeeping, and decoder projection as `TXCBase`. Each atom lives
in DC or one of three real-Fourier AC bands, with sine/cosine quadratures kept
together. Selection is global across all bands. There is no BatchTopK,
Matryoshka loss, adaptive frequency loss, or learned routing.

Dictionary width is chosen separately at each T to minimize the total trainable
parameter-count difference from Aniket's 32,768-atom TXC:

| T | Fourier atoms | TXC parameters | Fourier parameters | Difference |
|---:|---:|---:|---:|---:|
| 1 | 32,768 | 268,472,320 | 268,472,320 | 0 |
| 2 | 65,532 | 536,911,872 | 536,911,868 | -4 |
| 4 | 98,298 | 1,073,790,976 | 1,073,790,970 | -6 |
| 6 | 131,064 | 1,610,670,080 | 1,610,670,072 | -8 |
| 10 | 131,067 | 2,684,428,288 | 2,684,420,091 | -8,197 |

On the RunPod reference checkout, stage the two Python modules under
`purified/experiments/power_spectrum/code/`, then run:

```bash
BACKTRACKING_FOURIER_PHASE=memory-smoke \
  bash purified/experiments/power_spectrum/code/run_backtracking_fourier_runpod.sh
```

Only after the exact-width, full-batch smoke succeeds:

```bash
BACKTRACKING_FOURIER_PHASE=all \
  bash purified/experiments/power_spectrum/code/run_backtracking_fourier_runpod.sh
```

Completed cells delete only their new optimizer-state file after a successful
20,000-step training return (or after evaluation when running eval-only). The
model weights, training summary, sparse codes, fold predictions, band-only
probes, and result JSON remain. Reruns recognize this completed-and-cleaned
state.

## Evaluation-artifact recovery

Aniket's bit-exact T16 evaluation artifact was not present in the Git history,
the attached RunPod volumes, or the public experiment handoff. Replaying the
frozen extractor in the current CUDA environment introduces small numerical
drift, so the strict artifact gate correctly rejects the replay.

The fallback is therefore marked as a *sensitivity analysis*, not an exact
replication. It:

- retains the published 20,335-row count and 2,498-row class balance;
- selects the lowest replay-RMSE candidates within each class, preserving
  source order;
- copies the six offsets available in the official artifact bit-for-bit;
- records the different cohort hash and replay-error quantiles in a manifest;
- requires the explicit `--allow-recovered-artifact` flag at evaluation.

Fourier results on this artifact must not be presented as directly
bit-exact-comparable to Aniket's published TXC values. The published TXC curve
can be shown as a dashed reference, with the artifact mismatch stated in the
figure and results text.

## Completed result

The 15-cell, three-seed sweep is complete. The recovered-cohort Fourier
PR-AUC curve at the fixed 32-feature probe is 0.2184, 0.2428, 0.2429, 0.2446,
and 0.2398 for \(T=1,2,4,6,10\). Aniket's pinned TXC curve is 0.2178, 0.2289,
0.2466, 0.2512, and 0.2548. Thus the plain Fourier model has a
recovered-cohort advantage at \(T=2\), is close at \(T=4\) and \(T=6\), and
loses by 0.0150 PR-AUC at \(T=10\).

DC activation-mass share declines monotonically from 1.000 at \(T=1\) to
0.559 at \(T=10\). The remaining 0.441 is carried by AC blocks, so the
representation is not merely using DC features even though DC remains the
largest component.

See `analysis/backtracking_fourier_results.md` for the complete table,
frequency-use interpretation, provenance boundary, and compute accounting.
The comparison plot and machine-readable summary are in
`results/backtracking_fourier_matched/reviewer-five-point-v1/publication/`.
