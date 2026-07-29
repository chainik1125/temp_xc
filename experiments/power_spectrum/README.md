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
