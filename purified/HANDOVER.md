# HANDOVER — arxiv branch post-refactor (2026-05-27)

**Branch**: `arxiv`. HEAD at commit time of this writeup: see the most
recent commit in `git log arxiv -3`.

**Where things stand**: the **framework v2 skeleton + engine** is built,
smoke-tested, and **all 5 paper-section experiments route end-to-end**.
**§ 4 synthetic** is fully wired — `python run.py synthetic --arch
<any> --seed 0 --smoke` trains and evaluates, writes a real
leaderboard row with code-version stamp, and re-runs cache-hit. The
4 real-LM evaluators (`probing`, `backtracking`, `em`, `rlhf`) have
their entry points wired but their `Evaluator.eval()` bodies raise
`NotImplementedError` with a clear pointer to the load-bearing v1 code
on `origin/final` that needs porting.

**Tests**: `45/45` passing under `pytest tests/ -q`.

## TL;DR — what to do on your next session

```bash
cd $(git rev-parse --show-toplevel)/purified
.venv/bin/python -m pytest tests/ -q            # should pass 45/45
TEMP_BENCH_ALLOW_DIRTY=1 python run.py validate # registry self-check
TEMP_BENCH_ALLOW_DIRTY=1 python run.py synthetic --arch txc_base --seed 0 --smoke
```

Then read `purified/docs/framework_v2.md`. Then this file.

## What was built (vs what was promised in CLEANUP_PLAN.md)

### Step 1 — Cleanup (DONE)

Committed at `4583d880`. Removed 278 files (~45K lines) of wasteland
+ intermediate writeups. Repo root is now `README.md + purified/`.

### Step 2 — Framework v2 spec (DONE)

`purified/docs/framework_v2.md` is the spec — single source of truth
for the framework, written for agents-first. Codifies:

- Paper-section names (synthetic / probing / backtracking / em / rlhf)
  replacing the cN convention.
- Plugin pattern: arch / eval / datasource / experiment all extend
  via single-file drop + YAML entry. Never edit core.
- Cache contract: `train_key`, `eval_key`, `data_key` SHA-16-hex.
- Code-version stamp on every result row.
- Token shuffle buffer as default training data path (replacing v1's
  whole-sequence sampling).

### Step 3 — Forced migration (PARTIAL — engine done, real-LM evals stubbed)

Done end-to-end:

- `purified/run.py` — single CLI dispatcher with subcommands:
  `<experiment>`, `sweep`, `reproduce`, `render-figures`, `validate`.
- `temp_bench/interfaces/` — three ABCs: `TempBenchArch` (with
  `arch_version` + `consumes ∈ {token, window, sequence}`),
  `BatchIter` protocol, `Evaluator` ABC.
- `temp_bench/core/` — schemas (with `CodeVersion`), config (registry
  loaders + deterministic cache keys), cache (flock-protected
  JSONL append), code_version (capture with dirty-tree gate),
  runner (`run_experiment` + `run_sweep`), trainer (unified SAE
  loop; dispatches token/window/sequence BatchIter).
- `temp_bench/data/` — `activation_buffer` (token shuffle, literature
  standard), `window_buffer` (T-window shuffle), `sequence_buffer`
  (legacy mode for v1 ports that do internal window sampling),
  `synthetic` (markov + coupled_hmm), `real_lm` (activation cache
  build + refill source).
- `temp_bench/archs/` — all 9 architectures:
  - **Ours**: `txc_base`, `txc_pro`, `topk_sae`, `stacked_sae`,
    `mlc`, `sae_arditi` — ports from `origin/final` with v2 attrs.
  - **Adapter-wrapped**: `tsae` (AI4LIFE-GROUP/temporal-saes),
    `tfa` (TFA paper reference impl), `tfa_pos`. Labeled
    `arch_version="2.0.0-port"` to flag they're our adaptations,
    not verbatim upstream.
- `temp_bench/evals/` — 5 evaluator modules:
  - `synthetic_recovery.py` — **fully implemented**: eAUC, gAUC, NMSE.
  - `probing.py`, `backtracking.py`, `em.py`, `rlhf.py` — stubs
    with `NotImplementedError` + pointers to the v1 code that needs
    porting from `origin/final`.
  - `legacy/` — preserved utility modules (`case_study.py`,
    `detection.py`, `steering_hooks.py`, `steering_protocols.py`)
    that the real-LM evaluators will delegate to when ported.
- `experiments/` — one entry-point dir per paper section + `TEMPLATE/`.
  Each `run.py` is a thin wrapper around `run_experiment`.
- `experiments/render_paper_figures.py` — skeleton for figure rendering
  (Figs 2-6); concrete render functions raise NotImplementedError with
  port pointers to `origin/final` analysis.py files.
- `configs/archs.yaml`, `configs/data.yaml`, `configs/experiments.yaml` —
  three new YAML registries replacing v1's `locked_archs.yaml` +
  `datasources.yaml` (preserved as `.v1.bak`).
- `tests/test_v2_*.py` — 5 test modules covering interface contracts,
  cache-key determinism, code-version capture, buffer shapes, and
  end-to-end synthetic smoke. **45/45 passing.**

## Smoke validation results

Ran on local 5090, WSL, fresh venv (uv-managed). All "smoke" runs use
tiny dims (d_in=16, d_sae=16, n_steps=10) — no real training,
~hundreds of milliseconds each.

```
python run.py validate                                                   ✅
python run.py synthetic --arch txc_base   --seed 0 --smoke               ✅
python run.py synthetic --arch topk_sae   --seed 0 --smoke               ✅
python run.py synthetic --arch stacked_sae --seed 0 --smoke              ✅
python run.py synthetic --arch tsae       --seed 0 --smoke               ✅
python run.py probing      --arch txc_base --seed 0 --smoke              → NotImplementedError (expected; routing works)
python run.py backtracking --arch txc_base --seed 0 --smoke              → NotImplementedError (expected; routing works)
python run.py em           --arch txc_base --seed 0 --smoke              → NotImplementedError (expected; routing works)
python run.py rlhf         --arch txc_base --seed 0 --smoke              → NotImplementedError (expected; routing works)
```

For the synthetic runs, every result row landed in
`results/leaderboard.jsonl` with:

- `schema_version: "2.0.0"`
- `code_version.commit_sha`: full 40-char SHA at run time
- `code_version.dirty: true` (arxiv branch is dirty during dev)
- `code_version.diff_sha256`: hex hash of the working diff
- Metrics dict containing `eauc`, `gauc` (where applicable), `nmse`.

Re-running the same cell produces a cache-hit (`train_cached=True,
eval_cached=True`), confirming the deterministic key contract.

## What's left to do (ordered by priority)

### Priority 1 — Port the real-LM evaluators (~1-2 day each)

Each one is a focused single-file migration. The Evaluator class is
already in place; the `eval()` method body needs porting from
`origin/final`. Pointers are written into each stub's
`NotImplementedError` message.

**`temp_bench/evals/probing.py`** — § 5.1
- Port `data/nlp/probe_cache.py` + `probe_tasks.py` from origin/final.
- Port the canonical probing protocol from
  `origin/final:purified/experiments/c3_probing/run.py:my_eval_fn`.
- Headline metric: `mean_auc` over 36 SAEBench tasks.

**`temp_bench/evals/backtracking.py`** — § 5.2
- Port `data/nlp/ward.py` (Ward stage B rollout labels) from origin/final.
- Detection skeleton is already in
  `temp_bench/evals/legacy/detection.py:detect_case_study`.
- Inducement protocol: port from
  `origin/final:purified/experiments/det_steer/run_c7_locked.py`.
- Headline metric: `detection_pr_auc` + an inducement gap-recovery
  number.

**`temp_bench/evals/em.py`** — § 5.3 (highest cost)
- Port the Wang 4-stage screening procedure from
  `origin/final:purified/experiments/c6_em/run.py`.
- Requires a judge client (Anthropic Haiku) + LoRA-organism loader
  for Qwen. Costs ~$0.50/cell in API spend.
- Headline metric: `peak_align` at `coh ≥ 30`.

**`temp_bench/evals/rlhf.py`** — § 5.4
- Port preference loader + rank-based decomposition from
  `origin/final:purified/experiments/c5_steering/run.py`.
- Steering primitives are in `temp_bench/evals/legacy/steering_*.py`.
- Headline metric: `preference_auc`.

### Priority 2 — Port the figure renderers (~half day each)

`experiments/render_paper_figures.py` has skeleton render functions
that all raise NotImplementedError with port pointers to
`origin/final` analysis.py files. Each renders one paper figure (Fig 2
through Fig 6) by querying the leaderboard + matplotlib.

### Priority 3 — Backfill activation caches for full reproduction

Real-LM reproduction needs `acts.npy` activation caches built. The
build function exists at
`temp_bench.data.real_lm.build_activation_cache`. To rebuild a paper
section's cache:

```python
from temp_bench.core.config import load_datasource
from temp_bench.data.real_lm import build_activation_cache
ds = load_datasource("gemma_2_2b_it_l13_fineweb_24k128")
build_activation_cache(ds)   # ~3 H100-hours
```

Alternatively, port the v1 `sync_from_hf.sh` script (deleted in
step 1 cleanup) to pull pre-built caches from
`han1823123123/temp-bench-data` on HF.

### Priority 4 — Production hardening (low-stakes polish)

- Replace TFA's port with a true upstream wrapper. The current
  `temp_bench/archs/tfa.py` is our adaptation of the TFA paper's
  reference impl (with Han's `scaling_factor` heuristic for real-LM
  activations). If a clean upstream pip package becomes available
  for TFA, replace with an adapter that wraps it.
- Same for T-SAE: `temp_bench/archs/tsae.py` is our port of
  `AI4LIFE-GROUP/temporal-saes`. Could be wrapped as a true
  dependency if their package is pip-installable.
- Update `pyproject.toml` to optionally depend on the upstream
  packages so adapters can `import temporal_saes` directly.

## Notable design choices worth knowing

### Tolerating both v1 (tuple) and v2 (dict) train_step contracts

The trainer (`temp_bench/core/trainer.py`) accepts BOTH old (tuple
`(loss, info)`) and new (dict `{"loss": ..., ...}`) return values from
`arch.train_step`. This lets us port v1 archs in-place without
rewriting their loss code. TXC-base was rewritten to the v2 contract;
the other archs (txc_pro, stacked_sae, mlc, sae_arditi, tsae, tfa)
still return tuples — and that's fine.

### Three `consumes` modes

- `"token"` — arch sees `(B, d_in)`. Trainer uses ActivationBuffer
  (i.i.d. tokens from a shuffle buffer; literature standard).
- `"window"` — arch sees `(B, T, d_in)`. Trainer uses WindowBuffer
  (i.i.d. T-windows from buffered sequences).
- `"sequence"` — arch sees `(B, seq_len, d_in)`. Trainer uses
  SequenceBuffer (legacy mode for v1 ports that do internal window
  sampling). Effectively the v1 behavior, opt-in by the arch.

New archs should prefer `"token"` or `"window"`. Legacy ports use
`"sequence"`.

### Code-version stamp ≠ cache key

The `code_version` field is recorded on every row but does NOT feed
the cache hash. This is deliberate: cosmetic code edits shouldn't
invalidate trained checkpoints. The mechanism for "this code change
matters" remains the explicit `arch_version` bump in
`configs/archs.yaml`.

### k_win clipping in TXC-base

`txc_base.py` clips `k_win = min(k_pos * T, d_sae)` so toy configs
where `k_pos * T > d_sae` don't crash (warns instead). This was
needed for the smoke test (synth_smoke has d_sae=16).

## Files / dirs to know

```
purified/
├── run.py                          ← dispatcher (single CLI entry)
├── CLAUDE.md                       ← agent quick-start
├── CLEANUP_PLAN.md                 ← phase plan
├── HANDOVER.md                     ← this file
├── docs/framework_v2.md            ← framework spec (read FIRST)
├── docs/figs/                      ← rendered paper figures (fig2_*.{pdf,png})
├── configs/
│   ├── archs.yaml                  ← 9 archs registered
│   ├── data.yaml                   ← 8 datasources
│   ├── experiments.yaml            ← canonical paper sweep configs
│   └── sweeps/                     ← agent-defined custom sweeps go here
├── src/temp_bench/
│   ├── core/                       ← runner, cache, schemas, trainer, code_version
│   ├── data/                       ← buffers + synthetic generators + real_lm cache
│   ├── archs/                      ← all 9 architectures
│   ├── evals/                      ← 5 evaluators (1 implemented, 4 stubs)
│   │   └── legacy/                 ← preserved v1 eval primitives
│   ├── interfaces/                 ← 3 ABCs (architecture, batch_iter, evaluator)
│   ├── training/bricken.py         ← Bricken plug-in (preserved from v1)
│   └── utils/                      ← seed, gpu_locks, plotting, etc.
├── experiments/
│   ├── synthetic/run.py            ← § 4 entry point (works end-to-end)
│   ├── probing/run.py              ← § 5.1 entry point (routes OK, eval stub)
│   ├── backtracking/run.py         ← § 5.2 entry point
│   ├── em/run.py                   ← § 5.3 entry point
│   ├── rlhf/run.py                 ← § 5.4 entry point
│   ├── TEMPLATE/run.py             ← copy-to-extend
│   └── render_paper_figures.py     ← Figs 2-6 (stubs)
├── tests/                          ← 5 v2 test modules, 45/45 passing
├── checkpoints/                    ← 23 v1 trained models (preserved from final-aniket)
└── results/leaderboard.jsonl       ← currently has smoke-test rows from this session
```

## Things to clean up later

- `purified/results/leaderboard.jsonl` has ~4 smoke-test rows from the
  validation runs. They're tagged `eval_cfg.smoke=True`. Either prune
  them or let the analysis filter them out (analysis code should
  filter `smoke=True` for paper headlines).
- `purified/checkpoints/` has 23 pre-existing v1 checkpoints. They use
  v1 train_keys (different hash space than v2) — they won't cache-hit
  for v2 runs. Decide whether to keep them as historical artifacts or
  prune.
- `configs/locked_archs.v1.bak`, `configs/datasources.v1.bak` —
  preserved for reference. Delete once you're confident in v2.
