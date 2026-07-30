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
multiband benefit on power-readable tasks. A subsequent 70-cell controlled
suite shows that planted slow and fast HMM sources activate the expected DCT
feature blocks, while exact polynomial-clock secret sharing remains a TXC
strength rather than a spectral win. The final conservative compute estimate
is $43.23 against a $50 cap.

## Paper synthetic extension

`configs/paper_synthetic_v1.json` freezes a new Spectral-v1 evaluation on the
two synthetic tasks used in Figure 2 of the paper:

- Denoising: best full-code hidden-state linear-probe
  `R²_global` on `toy_markov_n20_d40_noisy`;
- Coupling: best static decoder `gAUC` on the maximum-overlap
  `toy_coupled_noisy_K10_M20_d256_pB05_np10` regime.

The runner ports the historical generators, data seeds, training recipe, and
evaluation definitions into this isolated experiment. Its focused grid
contains the paper-best TXC cell plus dense-code alternatives on Denoising and
the paper-best neighborhood on Coupling. Published TopK SAE, T-SAE, and
TXC-base numbers are extracted byte-for-byte from the pinned Figure 2 data
rather than re-trained under the newer repository protocol:

```bash
uv run python -m experiments.power_spectrum.code.extract_paper_baselines
uv run python -m experiments.power_spectrum.code.run_paper_synthetic_v1 \
  --mode plan
uv run --with modal modal run \
  experiments/power_spectrum/code/modal_paper_synthetic_v1.py --stage smoke
uv run --with modal modal run --detach \
  experiments/power_spectrum/code/modal_paper_synthetic_v1.py --stage full
```

After the durable run completes:

```bash
uv run --with modal modal run \
  experiments/power_spectrum/code/modal_paper_synthetic_v1.py --stage fetch
uv run python -m experiments.power_spectrum.code.analyze_paper_synthetic_v1
uv run python -m experiments.power_spectrum.code.plot_paper_synthetic_v1
```

The plan contains 30 three-seed cells and 768,000 optimizer steps. Its
conservative estimate uses the measured paid-smoke throughput and stays below
the $14 hard ledger cap and 2.75-hour inner deadline. The Coupling evaluator
filters near-zero time-mean decoder atoms before cosine normalization: non-DC
DCT atoms have mathematically zero time mean, and normalizing float32
cancellation residue would create arbitrary directions. The raw historical
calculation is retained as `gauc_paper_raw` for sensitivity. This
maximum-overlap Coupling target is rank one, so Denoising is the more
discriminating result.

The completed paper-synthetic result is:

| Task | TopK SAE | T-SAE | TXC-base | Spectral v1 |
|---|---:|---:|---:|---:|
| Denoising \(R^2\) | 0.363 | 0.382 | **0.483** | 0.412 |
| Coupling gAUC | 0.842 | **0.990** | **0.990** | 0.969 |

Spectral v1 beats both token baselines on Denoising but not TXC-base. The
Coupling target is rank one and saturating, so it is less discriminating.
The comparison figure is
`figures/paper_synthetic_v1_comparison.{png,pdf}`.

## Controlled HMM and polynomial-clock extension

`configs/controlled_frequency_suite.json` defines the exact
polynomial-clock/Shamir generator for \(h=1,2\), plus slow, alternating, and
mixed-frequency factorial HMMs. Representation training, probe training, and
evaluation use episode-disjoint pools with a shared observation alphabet or
emission matrix.

The \(h=2,W\le10\) SAE/TXC curves come from the corrected
`dmitry-txcwins-10h:docs/dmitry/reviewer_responses/reviewer_responses_1.md`
table. The current reduced plan therefore reserves new \(h=2\) compute for
Spectral v1 and Spectral Matryoshka. The fetched full artifact preserves the
initial 70-cell plan, which had already completed its small local baseline
cells before the reviewer-response reuse was applied.

Inspect or run the reduced plan with:

```bash
uv run python -m experiments.power_spectrum.code.run_controlled_frequency_suite \
  --mode plan
uv run --with modal modal run --detach \
  experiments/power_spectrum/code/modal_controlled_frequency_suite.py \
  --stage full
```

After fetching a complete result, render all controlled figures and aggregate
tables with:

```bash
uv run python -m experiments.power_spectrum.code.plot_controlled_frequency_results
```

The main findings are:

- Spectral Matryoshka has held-out latent \(R^2\) 0.858 / 0.826 / 0.750 on
  slow / alternating / mixed HMMs, versus TXC-post 0.843 / 0.844 / 0.742.
- Both spectral models recover the expected band in 100% of pure slow and
  alternating source-seed cases and 87.5% of mixed cases.
- Global Matryoshka selection moves 45.8% of firing events into the lowest
  band on slow HMMs and 56.7% into the highest band on alternating HMMs.
- On the paper Denoising replay, DC-only features retain the full hidden-state
  probe \(R^2\) (0.4118 versus 0.4117), while AC-only features have
  \(R^2\approx0\), even though AC atoms carry most reconstruction energy.
- Polynomial-clock secret sharing is not power-readable: TXC is strongest on
  fresh \(h=1\), and the reviewer-response \(h=2\) TXC curve rises faster than
  the new spectral points.

See `analysis/controlled_frequency_results.md` for the protocol, tables,
interpretation, limitations, and cost accounting.

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
- `analysis/controlled_frequency_results.md`: exact polynomial-clock results,
  controlled frequency-HMM localization, and Denoising DC/AC usage.
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
