---
status: active
created: 2026-07-21
for: runpod
venue: runpod
---

# Stage 6 — build + evaluate the first grounded benchmarks (`assumption_consequence`, `hedging_drift`)

**Goal.** Turn the two solid grounded SPECs into `✓`-registered benchmarks, run
them through the **fair-backbone B×A grid**, and extend
[`experiments/explorations/synthetic/REPORT.md`](../experiments/explorations/synthetic/REPORT.md)
with their rows — the first time a benchmark *discovered by the expansion loop*
faces the architectures. This closes the measure→mirror→**bench** loop.

Scope is exactly these two (both AC-distinct + DC — the axes worth testing):
- **`assumption_consequence`** — AC / directed-transition. Spec:
  [`../experiments/explorations/synthetic/assumption_consequence/bench_spec.md`](../experiments/explorations/synthetic/assumption_consequence/bench_spec.md);
  canonical mirror `mirror_params_g7.json`.
- **`hedging_drift`** — DC / slow-drift. Spec:
  [`../experiments/explorations/synthetic/hedging_drift/bench_spec.md`](../experiments/explorations/synthetic/hedging_drift/bench_spec.md);
  canonical mirror `mirror_params_hier.json`.
`list_item_parallelism` (redundant class, weak mirror) and `self_reference_echo`
(SPEC*) are **out of scope**.

## ⚠️ The prime discipline — BLIND evaluation

The per-architecture predictions are **already frozen** in each `bench_spec.md`
(§ 5–6): assumption_consequence → windows > per-token, additive weaker; hedging →
per-token does fine (persistent DC state). **You must run and report without
tuning anything to make them come true.**

- Build the generator, evaluator, and capacity/window/probe design **to the spec's
  frozen conventions** (Part II: `d_sae` anchored on `F`, `L=32`, tiled `T`,
  memorization-free per-tile **linear** probe, `[chance, oracle]` normalization).
  Do NOT adjust any of these after seeing a metric.
- After the grid, compare actual results to the frozen predictions and **report the
  verdict honestly** (POSITIVE / NEGATIVE / SPLIT). A **failed** prediction (e.g.
  per-token unexpectedly recovers the directed dependency) is a real, citable
  finding — report it, never retune. Prime directive: a sound verdict, never a win.

## Build (follow the changepoint / backtracking template exactly — no new patterns)

Each frozen spec is a two-layer generative process (a fitted mirror + the standard
emission). Read the spec for `F`, the latents, and the chance/oracle for each.

1. **Generators** — add `assumption_consequence()` and `hedging_drift()` to
   `src/temp_bench/data/synthetic.py`, mirroring `semi_markov_modes()` /
   `self_exciting()`:
   - assumption_consequence: the 3-state {N,A,C} Markov chain from
     `mirror_params_g7.json` → Layer-2 emission over `F = 3 + K_c = 20` orthonormal
     dirs (state + content), backtracking's emission pattern.
   - hedging_drift: the `hier_ar1` process from `mirror_params_hier.json` (per-doc
     slow level + AR(1)) → emission per its spec's `F`.
   - Expose ground truth in `extra` (as backtracking exposes `lambda_labels`,
     changepoint `mode_labels`): the state/level labels **and** the directed
     next-state / drift targets the evaluator will probe.
2. **Datasources** — register `toy_assumption_consequence_*` and
   `toy_hedging_drift_*` in `configs/data.yaml` (generator + params + a `notes`).
3. **Evaluators** — add two probe add-ons dispatched from
   `src/temp_bench/evals/synthetic_recovery.py` on the new `extra` keys, exactly
   like `lambda_recovery` / `changepoint_recovery` (no-op for other benches ⇒
   **protocol stays 1.3.0**, additive). Per the specs: for assumption_consequence
   a categorical **state** probe (DC) + a **directed next-state dependency** probe
   (AC, chance = marginal transition freq, oracle = the Markov one-step
   conditional); for hedging a **confidence-level** probe (DC) normalized to its
   spec's chance/oracle. All linear/logistic, per-tile at the leading edge,
   split-by-example, over the common `L`-tiling.
4. **Tests** — a `tests/test_*_bench.py` per bench (generator shapes + ground-truth
   labels + the evaluator returns the right keys), like `test_changepoint_bench.py`.

## Gate first (§ 8 sanity — do NOT skip)

Before the full grid, confirm on each new generator: (i) the latent's **oracle is
reachable** by a probe on the noiseless emission, and (ii) the **chance floor sits
where the spec says** (e.g. assumption_consequence AC chance = marginal transition
freq). If a latent isn't separable oracle-vs-chance, stop and report — a
degenerate benchmark is not worth a grid. (Analogue of `backtracking/gating.py`.)

## The grid

Run **both** datasources through the **uniform fair-backbone design** the full
rerun used (`src/explorations/synthetic/design.py` + the shared
`grid.run_pool`): the 6 fair-backbone archs (`batchtopk_sae`, `tsae`,
`stacked_batchtopk`, `txc_batchtopk_pre/post`, `spectral_txc`) × `d_sae ∈
{F//2, F, 2F}` × `T ∈ {1,2,4,8}` × `k_pos ∈ {1,2,4,8,16}` (respect `d_sae ≥
k_pos·T`) × seeds `{1,2,42}` + the untrained-encoder control. Everything through
the **canonical runner** (`run_experiment`) → `results/leaderboard.jsonl`. (Runs
on CPU; the A40 is pathologically slow for these tiny `d_in` models — see STATUS.)

## Render + record

1. Add the two benches to
   `experiments/explorations/synthetic/registry.py` as `Bench()` entries with
   their latent-axes (the new metric keys), `F`, and `F_note`; `render_report`
   regenerates `REPORT.md` (matrix rows + NMSE/eauc panels + the three figures)
   from the fresh leaderboard rows.
2. Write a `synthetic/<bench>/bench_record.md` per bench (arch frontier + the
   **frozen-prediction-vs-actual** check + a one-line verdict), following
   `backtracking/bench_record.md` / the single-source record pipeline.
3. **Update [`BENCHMARKS.md`](../experiments/explorations/synthetic/BENCHMARKS.md)**:
   flip both rows `reg. ✗ → ✓` and fill the **arch-verdict** column with the honest
   result.

## Acceptance gate

- Both generators + datasources + evaluators built to the template; per-bench
  tests pass; the § 8 gate passed on both.
- Grid complete through the canonical runner, **0 failures**, code-version stamped.
- `REPORT.md` shows the two new benches (rows + panels + figures) regenerated from
  raw JSON; per-bench `bench_record.md` written with the blind
  prediction-vs-actual verdict; `BENCHMARKS.md` updated.
- **No prediction was tuned for.** Committed + pushed to `origin/arxiv` with
  **scoped commits** (no stray logs). Then STOP for review.

## Hard rules

`TEMP_BENCH_ALLOW_DIRTY=1`; `.venv/bin/python`; **never edit `temp_bench/core/`**;
new arch/eval/generator = a file drop + a `configs/` entry (plugin-only);
everything through the canonical runner (code-version stamped); paper-section
names; the evaluator protocol stays **1.3.0** (the add-ons are no-ops for existing
benches). When done + reviewed, delete this briefing.
