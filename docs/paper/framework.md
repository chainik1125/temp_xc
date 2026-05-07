---
title: "Framework design: principles and contract"
author: agent_paper
date: 2026-05-03
status: locked
---

This document codifies the framework that all paper experiments run on.
**Every agent must read this before writing experiment code.** The design
is constrained by ten principles that exist because the paper has 72
hours and seven components running in parallel — friction kills throughput
exactly when the deadline is tightest.

## The ten principles

### 1. Configs are the source of truth, not code

Architecture hyperparameters, data sources, and component cell sweeps
live in version-controlled YAML under `configs/`. Code reads
configs. **Hyperparameters never live in `.py` files** except as defaults
the YAML overrides.

> *Test: if you grep `src/temp_bench/architectures/*.py` for a
> hard-coded `d_sae=18432` or `T=5`, you've found a bug.*

### 2. A datasource is a config, not a constant

`SUBJECT_MODEL = "gemma-2-2b-it"` does NOT appear in any component
script. Instead, the component references a named datasource
(`gemma_2_2b_it_l13_fineweb_24k128`) defined in
`configs/datasources.yaml`. Switching IT → BASE = edit one yaml
field; framework handles the rest.

### 3. Two-tier deterministic cache

Three keys, all `sha256(canonical_json(inputs))[:16]`:

```
act_cache_key  = hash(subject_model, layer, hookpoint, dataset, n_seqs, seq_len, tokenizer_revision)
train_key      = hash(arch_class, arch_version, hparams, seed, training_cfg, act_cache_key)
eval_key       = hash(train_key, eval_protocol_version, eval_cfg)
```

Same inputs → same key → cache hit. No randomness. Replaces the earlier
`secrets.token_hex(4)` design.

### 4. Single runner, single pathway to leaderboard

`temp_bench.runner.run_cell` is the **only** function that may append a
row to `results/leaderboard.jsonl`. If you bypass it, your row never
appears in the paper. This is enforced by:

- The `LeaderboardRow` Pydantic schema rejects rows missing
  `eval_key`, `train_key`, or `schema_version`.
- `append_leaderboard` validates against the schema before writing.
- A pre-flight check warns if any row in the leaderboard lacks a
  matching `train_key` checkpoint or a matching `eval_key` metrics file.

Bypassing the runner is a documentation-policy violation (PROTOCOL.md § 11).

### 5. Per-component eval protocol versioning

Each `experiments/cN/run.py` declares:

```python
EVAL_PROTOCOL_VERSION = "1.0.0"
```

A bug fix in C3's probing AUC formula is one bumped string + a re-run.
Trained checkpoints survive (their cache keys do not include eval_protocol_version).

### 6. Per-cell idempotency

`run_cell` is idempotent. Running it twice does nothing the second time
(cache hit). Therefore:

- Re-running `experiments/run_all.sh` is always safe.
- An interrupted run resumes by just re-running.
- "Force regenerate" is a knob, not a workflow change.

### 7. Schema-checked leaderboard

Every `leaderboard.jsonl` row is validated against `LeaderboardRow`
(see `src/temp_bench/schemas.py`). Malformed rows are rejected at
append time. A `schema_version` field enables future migrations.

### 8. Architecture-specific complexity is encapsulated in the arch class

`configs/locked_archs.yaml` resolves a `class_path` to a class
implementing `TempBenchArch`. All optional behaviour
(Bricken-resample hooks, matryoshka loss, multi-distance contrastive)
lives behind that interface. The runner does not branch on arch type.

> *Test: if `runner.py` imports a specific architecture class, you've
> found a bug.*

### 9. Component-specific complexity is encapsulated in the eval_fn

A component's bespoke logic (Wang procedure, passage-discrimination
probe) lives in `src/temp_bench/eval/<name>.py` as a callable
`eval_fn(model, eval_cfg) -> dict[str, float]`. Components don't add
flags to the runner; they pass a different `eval_fn`.

### 10. Everything that can be wrong is testable

`tests/` under `` covers cache-key determinism, runner
idempotency, schema validation, and a per-component smoke run with a
1-step dummy arch. CI doesn't exist on `final` — but
`scripts/smoke_test.sh` runs the whole test suite, and is invoked
on every agent session.

## What this means for each scenario

### Scenario A — "miracle TXC found 1 day before deadline"

1. Drop `src/temp_bench/architectures/txc_miracle.py` (one new class
   subclassing `TempBenchArch`).
2. Add four lines to `configs/locked_archs.yaml`.
3. `bash experiments/run_all.sh`.

Only `txc_miracle` cells compute. C3, C4, C5, C7 share the new
checkpoint (same act-cache key). C6 trains with brickenauxk (different
training_cfg) — different train_key, separate run. C1, C2 use synthetic
data — also separate.

**New code: ~150 lines (one class). Time: bottlenecked by training
the new arch.**

### Scenario B — "bug found in TXC-pro encode"

1. Fix the code.
2. Bump `arch_version` from `"1.0.0"` → `"1.1.0"` in yaml.
3. `bash experiments/run_all.sh`.

Only `txc_pro` cells re-train. Other archs cached.

**Diff: 1 yaml line + the fix. Time: bottlenecked by re-training
txc_pro across all components.**

### Scenario C — "metric formula wrong in C3"

1. Fix in `src/temp_bench/eval/probing.py`.
2. Bump `EVAL_PROTOCOL_VERSION` in `experiments/c3_probing/run.py`.
3. `python -m experiments.c3_probing.run`.

All checkpoints cached; only evaluations re-run (~1 hour).

**Diff: 1 const bump + the fix. Time: eval-bottlenecked, not
train-bottlenecked.**

### Scenario D — "switch C3, C4, C5 from IT to BASE"

1. Edit `experiments/c{3,4,5}/run.py` — change the `DATASOURCE`
   constant from `gemma_2_2b_it_l13_fineweb_24k128` to
   `gemma_2_2b_base_l13_fineweb_24k128`.
2. Add the new datasource to `configs/datasources.yaml` if it doesn't
   exist yet.
3. `bash experiments/run_all.sh`.

Activation cache rebuilds once (shared across C3/C4/C5). Training and
evaluation cache invalidate per arch. Eval rolls.

**Diff: 1–2 lines per script. Time: bottlenecked by act-cache rebuild
+ all trainings.**

### Scenario E — "add Bricken to all of C3"

1. In `experiments/c3_probing/run.py`, override
   `runner.default_training_cfg(arch)` with a `BrickenConfig` injection.
2. Run.

Different training_cfg → different train_key → C3 trains a separate
"with-Bricken" checkpoint per arch. Other components' checkpoints are
untouched. Both with and without Bricken stay in the cache.

The leaderboard distinguishes them by `train_key`; the per-component
writeup explicitly reports which `train_key` family was used.

## Files

- `configs/locked_archs.yaml` — arch registry
- `configs/datasources.yaml` — data-source registry
- `src/temp_bench/config.py` — yaml loaders, schemas
- `src/temp_bench/schemas.py` — Pydantic models
- `src/temp_bench/cache.py` — checkpoint + leaderboard ops
- `src/temp_bench/runner.py` — `run_cell`, `preflight`
- `tests/` — cache-key + idempotency + smoke tests
- `scripts/smoke_test.sh` — runs preflight + tests

## Hard rules (codified in PROTOCOL.md § 11)

1. Hyperparameters never live in `.py` files except as defaults.
2. Datasources are named, not hard-coded.
3. Use `runner.run_cell` for every cell that produces a
   leaderboard row.
4. Bumping `arch_version` is the canonical way to invalidate trained
   checkpoints. Don't delete files manually.
5. Bumping `EVAL_PROTOCOL_VERSION` is the canonical way to invalidate
   eval results.
6. New architectures: yaml entry + class file. Nothing else.
7. New components: copy the cN template (~30 lines), set
   `EVAL_PROTOCOL_VERSION`, name the datasource.
