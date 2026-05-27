# arxiv branch cleanup plan

**Branch**: `arxiv` (off `origin/final-aniket` @ `3bec1ac0`).
**Goal**: minimal infra to reproduce the paper's **main findings** (excluding
appendix). Aggressive nuking allowed; full code lives on `final` and earlier
branches if we need to refer back.

## Current state assessment

The arxiv branch is a **thin shell** inherited from `final-aniket`. It is
already much leaner than `final`:

- `src/temp_bench/architectures/` has **only `base.py`** (the abstract
  interface). `txc_base.py`, `txc_pro.py`, baselines — **all missing**.
- `experiments/cN_*/` contains **only `README.md` stubs** (no `run.py`,
  no `analysis.py`).
- `src/temp_bench/data/`, `training/` — only `__init__.py` (no toy/nlp
  generators, no shared trainer).
- `results/leaderboard.jsonl` is **0 bytes**. `checkpoints/` has 23
  pre-existing dirs.
- `agents/` already cleaned to README + template.

What IS present:
- The framework primitives: `cache.py`, `config.py`, `runner.py`,
  `schemas.py`, `eval/` (case_study, detection, steering_hooks,
  steering_protocols), `training/bricken.py`, `utils/` (gpu_locks,
  seed, save_figure, tokens, shuffles), `plotting/`.
- Configs: `locked_archs.yaml`, `datasources.yaml`.
- The paper: `docs/aniket/{main,appendix,checklist}.tex` + `figs/` + `refs.bib`.
- Component writeups: `docs/components/c{1..7}.md`.
- `experiments/det_steer/` has ad-hoc runners (`run_c7_locked.py`,
  `run_steering_ab.py`, `validate_protocols.py`).

## What the paper actually needs (main body, NOT appendix)

Main-body experiments per `purified/docs/aniket/main.tex`:

- **C1** — Synthetic TopK sweep (NMSE/AUC) — synthetic
- **C2** — Synthetic coupled features (gAUC) — synthetic
- **C3** — Sparse probing on Gemma-2-2B-IT L13
- **C5** — RLHF steering (negative case, real)
- **C6** — Emergent misalignment (negative case, real)
- **C7** — Backtracking on DeepSeek-R1-Distill (primary real-world win)

**C4 — qualitative latents — APPENDIX ONLY.** Excluded from this cleanup.

Architectures locked at two: `txc_base`, `txc_pro`. Baselines: `topk_sae`,
`tsae_paper`, `stacked_sae`, `tfa`, `mlc`, `sae_arditi`.

## Three-step cleanup

### Step 1 — Delete obvious cruft (THIS PASS)

Nuke without ceremony:

- **All `__pycache__/`** directories (build artifacts).
- **Repo-root wasteland** (outside `purified/`):
  - `/home/elysium/temp_xc/papers/` (5 wasteland reference docs)
  - `/home/elysium/temp_xc/docs/` (top-level wasteland)
  - `/home/elysium/temp_xc/RUNPOD_INSTRUCTIONS.md`
  - Root-level `CLAUDE.md` (purified/ has its own).
- **Component _paper_assets/ dirs**: `c1_paper_assets/`, `c2_paper_assets/`,
  `c7_paper_assets/` — superseded by `docs/aniket/figs/`.
- **C4 docs**: `docs/components/c4.md` (appendix-only, not main).
- **Exploratory writeups**: `c7_optimal_analysis.md`.
- **Duplicate paper_results md**: `c1_paper_results.md`, `c2_paper_results.md`,
  `c7_paper_results.md` — keep `cN.md` only; the formal results are in `main.tex`.
- **Pod-specific scripts**: `scripts/{bootstrap_runpod,bootstrap_local,sync_from_hf,wasteland_refresh}.sh`.
- **det_steer/results/** if present.

Commit message: `arxiv cleanup step 1 — nuke wasteland + duplicate paper writeups`.

### Step 2 — Design the centralized framework (DESIGN ONLY THIS PASS)

A single doc proposing the unified training+evaluation framework with
experiment+code-version tracking. **No code changes yet.**

Required properties (from Han):
1. **Centralized training + evaluation** shared between synthetic and
   real-world experiments. Per-experiment divergence requires explicit
   justification.
2. **Centralized result tracking** — every result row records (a) the
   training/eval recipe (already in `train_key`/`eval_key`) AND (b) the
   **git commit SHA** of the codebase at run time.

Design sketch (to refine):

- **Unified runner**: existing `temp_bench.runner.run_cell` extended to
  accept a uniform `(arch, data_spec, train_cfg, eval_cfg)` regardless
  of synthetic vs real. A `data_spec` resolves to either a toy generator
  or an activation cache loader.
- **Unified dataloader interface**: `temp_bench.data.load_batches(spec) → BatchIter`
  with two implementations:
  - `toy_batches(generator_name, params)` — synthetic
  - `cached_act_batches(datasource_name)` — real-LM
- **Unified trainer**: `temp_bench.training.train_sae(arch, batch_iter, cfg)`
  with optional Bricken plug-in. Currently scattered; consolidate.
- **Code-version tracking**: extend `LeaderboardRow` and `manifest.jsonl`
  schemas with `code_commit_sha: str` field. Populated by
  `subprocess.check_output(["git", "rev-parse", "HEAD"])` in `run_cell`.
  Reject runs with a dirty working tree unless an explicit override flag is set.
- **Per-experiment driver template**: each `experiments/cN_*/run.py` is a
  thin wrapper that builds the `(arch, data_spec, train_cfg, eval_cfg)`
  tuple and calls `run_cell`. ≤ 100 lines each.

Output: `docs/framework_v2.md` — design doc. Han reviews before Step 3.

### Step 3 — Force-migrate code from `final` (after design approval)

Port the MINIMUM code needed for the 6 main-body components from
`final` HEAD into the new framework structure. Breakage allowed; fix
as we go.

Scope:
- Architectures from `final` → `src/temp_bench/architectures/`:
  `txc_base.py`, `txc_pro.py`, `topk_sae.py`, `tsae_paper.py`,
  `stacked_sae.py`, `tfa.py` (+ `tfa_pos.py` if used in main body),
  `mlc.py`, `sae_arditi.py`.
- Toy generators from `final` → `src/temp_bench/data/toy/`: only the
  ones used in C1/C2 main figures (Markov for C1, coupled HMM for C2).
  Drop the 30+ exploratory generators (aurora, chord, dewdrop, harbor,
  …) — they belong to the appendix at most.
- Real-LM activation cache from `final` → `src/temp_bench/data/nlp/`.
- Shared trainer from `final` → `src/temp_bench/training/sae_trainer.py`
  (already exists somewhere on `final`).
- Per-component runners → `experiments/c{1,2,3,5,6,7}_*/run.py` +
  `analysis.py`, thin wrappers around `run_cell`.

Out of scope for arxiv:
- C4 (qualitative latents — appendix only).
- All exploratory C2 setups beyond the headline Setup D + Setup E.
- Detection-axis ablations for C5/C6/C7 (unless cited in main.tex).
- All per-agent briefings (the orchestration story belongs in old branches).

Each migrated file gets a header comment `# Ported from origin/final:<path>@<commit>`.

## Order of operations

1. **Now**: write this plan + execute Step 1 (delete cruft) + commit + push.
2. **Then**: write `docs/framework_v2.md` (Step 2 design) — request Han review.
3. **After review**: execute Step 3 (forced migration) — separate PR-sized commits
   per architecture / per component for easy revert.

## Reversion strategy

- arxiv branch is brand new. Every commit is recoverable by `git reset --hard`.
- The full code lives on `origin/final` (HEAD `457888a3`). Any deletion can be
  recovered by `git checkout origin/final -- <path>`.
- Trained checkpoints are pinned in `checkpoints/manifest.jsonl` on `final`
  with HF URLs — never lost.
