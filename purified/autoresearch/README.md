# Autoresearch — synthetic temporal benchmarks

This directory is the home of the **temporal-property autoresearch program**:
find a *measurable* temporal property of real LM behaviour, fit a faithful
synthetic *mirror* of it, then benchmark whether a window/temporal dictionary
exploits that structure better than a per-token SAE. Each benchmark is a
**self-contained subdir** holding its docs, scripts, figures, and results.

> **Prime directive.** Success = a sound, reproducible **verdict** — positive or
> negative. We never tune the labeler, statistic, capacity regime, probe, or
> metric to manufacture a "win". See [`autoresearch_spec.md`](autoresearch_spec.md) § 0.

## The loop (6 stages)

`propose & preregister → operationalize on real data (the labeler) → measure the
temporal signature (gate) → fit a synthetic mirror → validate the mirror →
benchmark architectures`. Stages and validity gates: [`autoresearch_spec.md`](autoresearch_spec.md).
How any synthetic bench is built/scored (ground truth `F`, capacity matched +
anchored on `F` + swept into the scarce regime, power-of-two tiled eval,
memorization-free linear probes, frontier reporting):
[`synthetic_benchmark_guidance.md`](synthetic_benchmark_guidance.md). The DC/AC
frequency lens: [`frequency_lens.md`](frequency_lens.md).

## Benchmarks (status)

| benchmark | dynamics class | stage | verdict | headline |
|---|---|---|---|---|
| [`backtracking/`](backtracking/) | self-exciting / recurrent (**AC**) | bench run | **POSITIVE** | window λ-recovery **0.95** (T≥4) vs per-token **DPI floor 0.41**, robust at `d_sae<F` |
| [`signed_motion/`](signed_motion/) | order-sensitive step (**AC**) | bench run | **NEGATIVE** | no arch recovers the sign in the scarce regime (`#windows=2F` memorization confound) |
| [`topic_switching/`](topic_switching/) | change-point / sticky (DC+AC) | measured | **ABORT** | autocorrelation is 82% per-doc *composition*, not order; labeler inadequate |
| [`changepoint/`](changepoint/) | change-point / absorbing | spec only | *gated* | shared topic+EM generator; gated on a valid real anchor |

Each benchmark subdir contains (where applicable): `prereg.md` (frozen
preregistration), `measurement.md` (the measure→mirror record), `bench_spec.md`
(frozen architecture-test spec), `bench_record.md` (architecture results),
the `*.py` scripts that produce them, `figs/`, and `results/` (derived stats
JSON). `signed_motion/` uses `bench.md` (single combined writeup).

## Running a benchmark's scripts

Scripts are a package; run from `purified/` as
`.venv/bin/python -m autoresearch.<bench>.<script>`. The canonical leaderboard
(shared) stays at `../results/leaderboard.jsonl`; real-label inputs (e.g. the
Ward backtracking labels) stay at `../results/`. Examples:

```bash
cd purified/
# backtracking (the positive result)
.venv/bin/python -m autoresearch.backtracking.gating         # § 8 per-token vs window ceilings
.venv/bin/python -m autoresearch.backtracking.kernel_order   # held-out kernel-length (K) selection
.venv/bin/python -m autoresearch.backtracking.measure        # measure real backtracking (stages 2-3)
.venv/bin/python -m autoresearch.backtracking.mirror         # fit + validate the synthetic mirror
.venv/bin/python -m autoresearch.backtracking.run_grid 6     # the 120-cell architecture grid
.venv/bin/python -m autoresearch.backtracking.render_figs    # frontier figures + stats
# topic-switching (the abort)
.venv/bin/python -m autoresearch.topic_switching.measure
```

All results route through the canonical runner (code-version stamped); no edits
to `temp_bench/core/`.
