## Power-spectrum temporal screen

This directory isolates the power-spectrum screening and spectral-crosscoder
experiments requested on 2026-07-29.

The work has two linked goals:

- test whether frequency-domain summaries distinguish temporal synthetic tasks;
- improve a spectral crosscoder relative to a matched temporal crosscoder (TXC).

Both goals are complete. The task screen shows that AC power routes periodic
magnitude, cross-spectra are required for phase and direction, signed DC is
task signal, and window sensitivity diagnoses localization. The complete
120-cell main benchmark plus 15-cell control shows that temporal position
mixing explains most of the gain over TXC, with an additional task-dependent
multiband benefit on power-readable tasks. No v2 regularizer wins universally.
The combined conservative compute estimate is $29.15 against a $50 cap.

Read-only source material:

- Francesco et al. source and figures:
  `../temporal_screen_1/papers/francesco/` in the primary checkout;
- existing FrequencyBench and spectral-TXC history in this repository;
- existing synthetic benchmark generators and matched TXC training harness.

All new implementations, frozen configurations, run manifests, cost logs,
results, figures, and conclusions for this run belong in this directory.

## Directory contract

- `analysis/`: theory notes and frequency-screen analyses.
- `code/`: self-contained experimental implementation and runners.
- `configs/`: frozen experiment configurations.
- `results/`: machine-readable outputs.
- `figures/`: rendered plots.
- `tests/`: focused correctness tests.

The original checkout is not modified by this experiment. Development occurs
on the isolated branch `codex/spectral-screen-overnight-20260729`; the
checkout is at the neutral ignored path `.worktrees/power-spectrum`. That
host-side location is operational metadata rather than part of the experiment.

## Reading order

- `analysis/theory_and_literature.md`: what the Francesco result does and does
  not imply, plus the proposed statistically controlled screen.
- `analysis/task_screen_results.md`: three-seed synthetic screening results.
- `analysis/benchmark_results.md`: matched learned-model results, ablations,
  failure modes, and cost/provenance.
- `results/provenance.json`: run IDs, source commits, integrity receipt, cost,
  frozen-config checks, and raw-artifact hashes.
- `code/spectral_txc_v2.py`: experiment-local spectral-crosscoder ablations.
- `configs/overnight.json`: frozen matched benchmark and cost envelope.
- `configs/matched_control.json`: bounded full-band, matched-window-support
  control that separates the multiband prior from total support.

## Reproduction

Run the task screen, its plots, and the focused tests with:

```bash
uv run python -m experiments.power_spectrum.code.run_task_screen
uv run python -m experiments.power_spectrum.code.plot_task_screen
uv run python -m pytest experiments/power_spectrum/tests -q
uv run python -m ruff check experiments/power_spectrum
```

Inspect the exact paid-compute plan without allocating a GPU:

```bash
uv run --with modal modal run \
  experiments/power_spectrum/code/modal_benchmark.py --stage plan
uv run --with modal modal run \
  experiments/power_spectrum/code/modal_matched_control.py \
  --stage plan
```

The main frozen plan is 2,448,000 optimizer steps, 7.15 estimated A10G hours,
and $35.75 under a deliberately conservative effective rate of $5/hour. The
matched full-band control adds 312,000 steps, 0.95 estimated hours, and $4.74,
for a combined planned estimate of $40.49. The independent jobs' 7.5-hour and
1.1-hour inner deadlines bound new GPU time to $43.00 at the same inflated
rate; combined spend must still be checked explicitly because their ledgers
are separate. Each job uses one GPU sequentially, checkpoints atomically, and
resumes completed cells.

Launch the durable overnight call only after the paid smoke stage succeeds:

```bash
uv run --with modal modal run \
  experiments/power_spectrum/code/modal_benchmark.py \
  --stage smoke
uv run --with modal modal run --detach \
  experiments/power_spectrum/code/modal_benchmark.py --stage overnight
```

After retrieving the completed JSONL, the canonical analyzer rejects missing
or failed cells, verifies the exact training horizon and required metrics, and
computes paired deltas versus both TXC baselines and spectral v1:

```bash
uv run python -m experiments.power_spectrum.code.analyze_benchmark
uv run python -m experiments.power_spectrum.code.plot_benchmark
```
