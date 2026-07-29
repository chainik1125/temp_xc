## Power-spectrum temporal screen

This directory isolates the power-spectrum screening and spectral-crosscoder
experiments requested on 2026-07-29.

The work has two linked goals:

- test whether frequency-domain summaries distinguish temporal synthetic tasks;
- improve a spectral crosscoder relative to a matched temporal crosscoder (TXC).

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
in the dedicated worktree `/private/tmp/temp_xc_spectral_screen` on branch
`codex/spectral-screen-overnight-20260729`.

## Reading order

- `analysis/theory_and_literature.md`: what the Francesco result does and does
  not imply, plus the proposed statistically controlled screen.
- `analysis/task_screen_results.md`: three-seed synthetic screening results.
- `code/spectral_txc_v2.py`: experiment-local spectral-crosscoder ablations.
- `configs/overnight.json`: frozen matched benchmark and cost envelope.

## Reproduction

Run the task screen, its plots, and the focused tests with:

```bash
uv run python -m experiments.power_spectrum.code.run_task_screen
uv run python -m experiments.power_spectrum.code.plot_task_screen
uv run pytest experiments/power_spectrum/tests -q
uv run ruff check experiments/power_spectrum
```

Inspect the exact paid-compute plan without allocating a GPU:

```bash
uv run modal run experiments/power_spectrum/code/modal_benchmark.py --stage plan
```

The frozen worst-case plan is 2,448,000 optimizer steps, 7.15 estimated A10G
hours, and $35.75 under a deliberately conservative effective rate of $5/hour.
The runner has a 7.5-hour inner deadline, a 7h45 remote hard timeout, and a
$45 usable ledger plus a $5 reserve. It uses one GPU sequentially, checkpoints
atomically, and resumes completed cells.

Launch the durable overnight call only after the paid smoke stage succeeds:

```bash
uv run modal run experiments/power_spectrum/code/modal_benchmark.py \
  --stage smoke
uv run modal run --detach \
  experiments/power_spectrum/code/modal_benchmark.py --stage overnight
```
