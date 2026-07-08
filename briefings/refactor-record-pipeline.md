# Briefing — extract the shared record pipeline; then legibility cleanups

```
status:  active
created: 2026-07-08
for:     RunPod or local CC agent (CPU-only — NO GPU, NO tokens, NO grid re-runs)
venue:   any
```

## Why (the debt)

Every synthetic benchmark **re-implements** its `run_grid.py` + `render_figs.py`
by copying the "changepoint template." Current cost:

- `render_figs.py`: 397 (backtracking) / 478 (changepoint) / 589 (frequency) /
  139 (signed_motion) lines.
- `run_grid.py`: 129 / 131 / 157 lines; frequency also spawned a **second**
  driver (`run_grid_bands.py`, 118).
- ≈ 2,100 lines of near-parallel scaffolding, growing ~600–700 per new bench,
  and the copies will drift.
- **`src/explorations/` — the reserved home for exactly this reusable library
  code — is absent.** The intended structure exists but has never been used.

The *single-source record pipeline* (leaderboard → `render_figs` → `figs/*` +
`results/*_stats.json` + auto-filled `<!-- AUTO:* -->` blocks in
`bench_record.md`) is a good pattern. The problem is it was copy-pasted per
bench instead of factored into a library.

## Goal

Lift the **common scaffolding** into a shared library at
`src/explorations/synthetic/`, reducing each bench's `run_grid.py` /
`render_figs.py` to a **thin, config-only driver** (its arch list, `d_sae`/`T`
grid, datasources, and its bench-specific table/figure *specs*) that calls the
shared functions. This is a **refactor behind an unchanged output contract** —
the safest kind, with a mechanical acceptance test (below).

## THE ACCEPTANCE GATE (read this first — it is the pass condition)

This refactor touches the code that produces **published numbers**. The
leaderboard (`results/leaderboard.jsonl`) is the source of truth and is **not**
modified. After refactoring, regenerate every existing bench from the unchanged
leaderboard:

```
for b in backtracking signed_motion changepoint frequency; do
  .venv/bin/python -m experiments.explorations.synthetic.$b.render_figs
done
git diff --stat   # inspect
```

**PASS = the `<!-- AUTO:* -->` blocks in every `bench_record.md` and every
`results/*_stats.json` are numerically identical to before the refactor**
(figure binaries may differ trivially in bytes; the *numbers* must not move). If
any number changes, the refactor changed behaviour — stop and diagnose; do not
"accept" a drift. If you discover a pre-existing bug in a renderer while
extracting it, **flag it in the PR, do not silently fix it** (silently changing
a published number violates the prime directive).

Recommended: before you start, snapshot the current outputs
(`git stash`-clean tree → the committed `bench_record.md`/`*_stats.json` ARE the
baseline; diff against them).

## Steps

0. **Package the lib.** Create `src/explorations/__init__.py` and
   `src/explorations/synthetic/__init__.py`. Confirm `explorations` is importable
   (it shares the `src/` root with `temp_bench`); if the packaging config
   (`pyproject.toml`) doesn't pick it up, add it. Verify `import
   explorations.synthetic` works from the repo root venv.

1. **Extract the record/renderer scaffolding** into (suggested shape — the
   module split is yours to choose; the output contract is the hard requirement):
   - `record.py` — read the leaderboard, filter to a bench's rows (by datasource
     + arch set), fill the named `<!-- AUTO:* -->` blocks in a `bench_record.md`,
     write the `*_stats.json`. The bench passes in its table specs
     (which columns, which rows, how to aggregate over seeds).
   - `figs.py` — shared figure styling, `.pdf`/`.png`/`.thumb.png` save helpers,
     the frontier/curve plot primitives the benches share.
   - `grid.py` — enumerate cells (archs × `d_sae` × `T` × seeds × datasources +
     the untrained / `k_pos` controls) and invoke the **canonical runner**
     (`temp_bench.core.runner.run_experiment`). Bench-specific knobs come from a
     config object the driver passes in.
   Extract the **common 60–70%**; leave genuinely bench-specific plots as
   per-bench specs. Do NOT force four different figure sets into one function.

2. **Reduce each bench** (`backtracking`, `changepoint`, `frequency`,
   `signed_motion`) `run_grid.py` / `render_figs.py` to thin drivers that
   declare config + specs and call the shared lib. Fold `frequency`'s
   `run_grid_bands.py` into the shared grid driver (it's the symptom).

3. **Run the acceptance gate** (above). Zero numeric drift is required.

4. **Tests:** add `tests/test_record_pipeline.py` (or similar) for the shared
   lib; keep the full suite green (`bash scripts/agent_smoke_test.sh` — currently
   78 passed / 1 skipped).

## Bundle these minor legibility cleanups (cheap, same PR)

- **Stale docstrings:** `src/temp_bench/data/synthetic.py` header still says
  "*Two* generators map to the paper's two benchmarks" (there are **6**:
  markov, coupled, signed_motion, self_exciting, semi_markov_modes,
  cyclic_tones); `src/temp_bench/data/__init__.py` lists only markov/coupled_hmm.
  Refresh both.
- **Reverse index:** add a comment block at `_GENERATORS` mapping generator →
  its datasource(s) → bench.
- **Arch grouping:** in `configs/archs.yaml`, add comment headers marking the
  **current fair-backbone suite** (batchtopk_sae, tsae, stacked_batchtopk,
  txc_batchtopk_pre/post, spectral_txc{,_dcac,_full}) vs **TopK-legacy**
  (topk_sae, stacked_sae, txc_base) vs **off-axis** (mlc = layer crosscoder,
  sae_arditi).

## Hard rules / constraints

- **Never edit `src/temp_bench/core/`.** The shared lib goes in
  `src/explorations/` (library code — allowed, not core).
- **`results/leaderboard.jsonl` is read-only** here. No grid re-runs, no new
  cells, no verdict changes, no touching frozen `bench_spec.md` bodies or the
  hand-written `bench_record.md` prose (only the AUTO blocks regenerate).
- **Do not move `temp_bench/evals/` helpers** (the tiled-probe / window-sampling
  code in `synthetic_recovery.py`) — those are correctly shared framework code
  and imported by evaluators. The duplication in scope is the *experiment-side*
  grid/render scaffolding only.
- `TEMP_BENCH_ALLOW_DIRTY=1`, `.venv/bin/python`, small described commits, push
  `arxiv`, co-author line `Co-Authored-By: Claude Opus 4.8
  <noreply@anthropic.com>`.

## Out of scope (do NOT do)

New benchmarks, new archs, re-running any grid, changing any verdict or number,
`core/` edits, touching real-LM datasources or leaderboard contents.

## When done

Delete this briefing file (per `briefings/README.md` — briefings don't
accumulate). The record of what happened is the git history + the shrunken
drivers + `src/explorations/synthetic/`.
