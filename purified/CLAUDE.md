# CLAUDE.md — purified/ paper-ready framework

You are an AI agent working on the `final` branch. **All work happens
inside `purified/`.** Wasteland code (the soup of 50+ TXC variants from
Phase 2-7 hill-climbing) was deleted from this branch — it lives only on
`origin/han-phase7-unification`. Wasteland *docs* are still here in
`docs/`, `papers/`, and `RUNPOD_INSTRUCTIONS.md` because component
writeups cite them heavily.

## Quick start

**Always work from `purified/` as your shell's cwd.** The framework's
`.venv`, configs, results, and checkpoints all resolve from here. Do NOT
operate from the repo root — `set_agent_env.sh` will refuse, the
`.venv/bin/python` paths get longer, and `git add -A` would risk
staging root-level cruft.

```bash
# 1. cd to purified/ (works on local + RunPod)
cd $(git rev-parse --show-toplevel)/purified

# 2. Build the venv if it doesn't exist (one-time per pod / fresh clone)
[ -d .venv ] || uv sync

# 3. Pin GPU + set AGENT_NAME + TEMP_BENCH_POD_MODE
source scripts/set_agent_env.sh <agent_name>

# 4. Verify env (CRITICAL preflight failures are fatal)
bash scripts/agent_smoke_test.sh

# 5. (Ephemeral pods only) pull cached checkpoints + activations from HF
[ "$TEMP_BENCH_POD_MODE" = "ephemeral" ] && bash scripts/sync_from_hf.sh
```

First-time pod provision (tokens + uv install + git checkout) is run
**by the user, not by an agent**. `scripts/bootstrap_runpod.sh` is
interactive (`read -rs` for token input) and an agent session cannot
enter input. By the time you start, Han has already run it; tokens are
in `/workspace/.tokens/` (or `~/.tokens/` locally) and the venv exists.
If `agent_smoke_test.sh` flags missing tokens, **stop and ping Han** —
do not try to populate the tokens yourself.

## Layout

```
configs/                         # locked_archs.yaml, datasources.yaml
src/temp_bench/                  # the library (paper-ready, single source of truth)
  ├ architectures/               # locked: 8 archs, registered in locked_archs.yaml
  ├ data/{toy,nlp}/              # generators + activation cache
  ├ training/                    # shared trainer + Bricken (opt-in)
  ├ eval/                        # synthetic, probing, qualitative, case_study
  ├ case_studies/                # C5 steering, C6 EM, C7 backtracking (temp-bench)
  ├ plotting/, utils/            # helpers (gpu_locks, set_seed, save_figure)
  ├ config.py                    # yaml loaders + cache-key computation
  ├ cache.py                     # checkpoint + leaderboard ops (only writers)
  ├ runner.py                    # run_cell — single canonical pathway
  └ schemas.py                   # Pydantic models (LeaderboardRow, …)
experiments/c{1..7}_*/           # one dir per paper component
results/
  ├ act_cache/<act_cache_key>/   # activation caches keyed deterministically
  ├ runs/<eval_key>/             # per-cell metrics + plots
  └ leaderboard.jsonl            # append-only, schema-validated, all agents append
checkpoints/<train_key>/         # trained models keyed deterministically
  └ manifest.jsonl               # append-only HF + local registry
docs/
  ├ components/c{1..7}.md        # paper-section writeups (the source of truth)
  └ paper/                       # framework.md, architecture.md, hardware.md, outline.md
agents/<name>/                   # per-agent briefing + log
tests/                           # pytest — cache-keys, schemas, runner, gpu_locks
scripts/                         # bootstrap, smoke, set_agent_env, sync_from_hf
```

## The seven paper components

| C | Subject | Lead arch | Lead agent | Hardware |
|---|---|---|---|---|
| C1 | Synthetic TopK sweep (NMSE/AUC) | TXC-base + TXC-pro vs TopK-SAE / TFA / Stacked | agent_paper | local 5090 |
| C2 | Synthetic coupled features (gAUC) | TXC-base + TXC-pro at multiple T | agent_paper | local 5090 |
| C3 | Sparse probing | TXC-base + TXC-pro vs T-SAE / TopK-SAE / MLC | agent_nlp | 1× H100 |
| C4 | Qualitative latents (Pareto) | TXC-pro vs T-SAE | agent_nlp | shares C3 cache |
| C5 | RLHF steering | TXC-base + TXC-pro vs T-SAE | agent_steer | 1× A40 |
| C6 | Emergent misalignment | SAE-arditi vs TXC-base+brickenauxk | agent_em | 1× H100 |
| C7 | Backtracking (Ward Stage B) on Llama-3.1-8B BASE L10 | TXC-base + TXC-pro + Stacked + TopK-SAE + TFA + T-SAE + MLC | agent_back | 1× A40 |

Full delegation table + GPU pinning: `agents/README.md`.

## The two TXC architectures

Locked across the paper — no per-component hill-climbing. Spec details
in `docs/paper/architecture.md`.

- **TXC-base** = `txc_bare_antidead_t5` — vanilla TopK temporal crosscoder
  (T=5) + tsae_paper anti-dead stack. No matryoshka, no contrastive,
  no Bricken (Bricken is opt-in per component).
- **TXC-pro** = `phase5b_subseq_h8` — subseq encoder (T_max=10, t_sample=5)
  + matryoshka H8 + multi-distance InfoNCE.

Free per-component knobs: ``k_pos`` (sparsity), ``d_sae`` (dict size).
Other arch hparams are fixed in `configs/locked_archs.yaml`.

## First actions on every session

1. `source scripts/set_agent_env.sh <agent_name>` (pins GPU, sets
   `AGENT_NAME`, sets `TEMP_BENCH_POD_MODE`)
2. `bash scripts/agent_smoke_test.sh` (env + tests + preflight; CRITICAL
   warnings are fatal)
3. Read **`agents/<your_name>/briefing.md`** top-to-bottom. The top
   section is Han's mandate (read-only). The bottom sections are
   *your previous self's* state-of-the-world — current state, next
   action, pitfalls. PROTOCOL.md § 14.
4. Read `decisions.md` for locked decisions you must respect.
5. Read `docs/components/c{N}.md` for any component you'll touch.
6. Skim the last 10 entries of `results/leaderboard.jsonl`.

**Before you exit or anticipate context compact**, overwrite the
agent-owned sections at the bottom of your briefing — see
`agents/_briefing_template.md` and PROTOCOL.md § 14. There is no
separate handover file and no `log.md` — git history of your briefing
is the audit trail.

## Hard rules

0. **Always cd into `purified/` first.** All paper paths and the venv
   are relative to this dir. `set_agent_env.sh` enforces this.
1. **Wasteland code is on `origin/han-phase7-unification`, not here.**
   Read it via `git show origin/han-phase7-unification:<path>`. Never
   `import` from anywhere outside `temp_bench`. To port code, copy once
   with attribution + the source commit hash in a header comment.
2. **Hyperparameters in YAML, not code.** Edit
   `configs/locked_archs.yaml` and `configs/datasources.yaml`. Hardcoded
   `d_sae=18432` or `subject_model="..."` in a `.py` is a framework bug.
3. **Use `runner.run_cell` for every cell.** It is the only path that
   appends to `leaderboard.jsonl`. Schema validation is mandatory.
4. **Bump `arch_version` to invalidate trained checkpoints.** Don't
   delete `.safetensors` files manually.
5. **Bump `EVAL_PROTOCOL_VERSION` to invalidate eval results.**
6. **Two TXCs only.** Don't introduce a third TXC variant. If you
   genuinely need to (e.g. paper revision), raise it in
   `docs/components/cN.md` first. Per-experiment training augmentations
   (Bricken etc.) are allowed if disclosed; see PROTOCOL.md § 5.
7. **Never edit another agent's directory or paper-territory files.**
   `agents/<other_agent>/` is theirs. `docs/paper/`, `decisions.md`,
   `agents/README.md`, `configs/locked_archs.yaml`, `pyproject.toml`,
   and `uv.lock` are agent_paper's (the last two because dependency
   changes affect every agent's venv — atomic pyproject + lockfile
   commits are the only safe form, and only agent_paper coordinates
   them). If you have a change that would touch any of those, **STOP**:
   add to "Open questions for Han" in your own briefing and surface
   it. Han or agent_paper will land the cross-territory edit. Even
   if Han verbally approves in chat, do not commit cross-territory
   edits yourself — surface as a written request, get explicit
   go-ahead, then let agent_paper integrate. Rationale: with 5
   agents touching the repo concurrently, "Han said it's fine" loses
   provenance the next time the briefing is read by a post-compact
   instance, and a pyproject change without the matching uv.lock
   bumps surprises every other agent's `uv sync`.
8. **Always set `TQDM_DISABLE=1`** before any Python invocation.
9. **GPU pinning on shared pods is mandatory.** Each agent's
   `set_agent_env.sh` entry pins one primary GPU. To use spare pool
   GPUs, claim via `temp_bench.utils.gpu_locks.claim_gpu(idx)`.
   See PROTOCOL.md § 12 (pinning) + § 13 (multi-GPU Primary + Pool).
10. **Results live in state, not prose.** Numbers in the `## Results`
    section of any `docs/components/cN.md` are NEVER hand-typed. The
    block between `<!-- BEGIN AUTO-RESULTS -->` and `<!-- END AUTO-RESULTS -->`
    is owned by `experiments/cN_*/analysis.py` + rewritten by
    `temp_bench.report.render(component="cN")`. Edit `analysis.py`,
    not the .md. See PROTOCOL.md § 7 *Results live in state*.
11. **T-SAE = paper-faithful Ye et al. 2025 only.** When the paper says
    "T-SAE" or "TSAE", it means the registered `tsae_paper` arch,
    sourced from `origin/han-phase7-unification:src/architectures/tsae_paper.py`
    — Matryoshka BatchTopK + AuxK + temporal contrastive + threshold
    inference. The wasteland's `tsae_ours.py` (TopK + 50/50 split + no
    AuxK + only 2 recon terms) is **deprecated and must NEVER be ported
    or imported**. If you find a TSAE comparison or baseline that traces
    back to `tsae_ours.py`, treat it as wasteland reference only.

## How to record results

**Use `temp_bench.runner.run_cell`. Don't write your own caching or
leaderboard logic.** See `docs/paper/framework.md` for the design.

```python
from temp_bench import runner
from temp_bench.schemas import TrainingConfig

result = runner.run_cell(
    component="c3",
    arch_name="txc_base",
    seed=42,
    datasource_name="gemma_2_2b_it_l13_fineweb_24k128",
    training_cfg=TrainingConfig(),
    eval_cfg={"k_feat": 20, "S": 32},
    eval_protocol_version="1.0.0",
    train_fn=my_train_fn,
    eval_fn=my_eval_fn,
)
# result.eval_key, result.train_key, result.cached
```

The runner:
- Computes deterministic `train_key` and `eval_key` from inputs.
- Skips training if a checkpoint with that `train_key` exists.
- Skips evaluation if that `eval_key` is in `leaderboard.jsonl`.
- Saves the trained checkpoint, run-dir metrics, and one validated row
  in the leaderboard. Schema-rejected rows abort the cell.
- On ephemeral pods (`TEMP_BENCH_POD_MODE=ephemeral`), auto-pushes the
  checkpoint to `han1823123123/temp-bench-models`.

A run-dir is created at `results/runs/<eval_key>/` containing
`metrics.json` and any plots (use `temp_bench.plotting.save_figure`,
which writes both `.png` and `.thumb.png`).

### Aggregate results: results live in state

Per-component summary numbers + paper-bound plots flow through
`experiments/cN_*/analysis.py` and `temp_bench.report.render(...)`.
The script queries `leaderboard.jsonl`, computes summary stats, saves
aggregate plots to `experiments/cN_*/plots/`, and rewrites the
AUTO-RESULTS block of `docs/components/cN.md`. Hand-typing numbers
into the .md is forbidden — see PROTOCOL.md § 7 and Hard Rule #10.

```python
from temp_bench import report

# Render one component (writes results.json + plots + rewrites cN.md):
report.render(component="c1")

# Render every component (idempotent):
report.render_all()
```

`experiments/_analysis_template.py` is the starting point — copy into
your component's `experiments/cN_*/analysis.py` and implement
`run_analysis() -> AnalysisResult`.

## How to record checkpoints

The runner already saves the trained checkpoint to
`checkpoints/<train_key>/` and appends a validated row to
`manifest.jsonl`. On persistent pods (H100, H200, local 5090), HF
backup is recommended at session end:

```python
from huggingface_hub import HfApi
from temp_bench.config import checkpoint_dir

train_key = result.train_key
HfApi().upload_folder(
    folder_path=str(checkpoint_dir(train_key)),
    path_in_repo=train_key,
    repo_id="han1823123123/temp-bench-models",
    repo_type="model",
)
```

Two private HF repos:
- `han1823123123/temp-bench-models` — checkpoints, keyed by `<train_key>`
- `han1823123123/temp-bench-data` — activation caches, judge transcripts

See `checkpoints/README.md` for full upload recipe.

## Pod modes — persistent vs ephemeral

`TEMP_BENCH_POD_MODE` is set by `set_agent_env.sh`:

- **persistent** (H100, H200, local): `/workspace` survives stop/start.
  HF backup is optional but recommended at session end.
- **ephemeral** (4× A40): `/workspace` is wiped on pod stop. Bootstrap
  pulls from HF (`scripts/sync_from_hf.sh`); `cache.save_checkpoint`
  auto-pushes to HF on save. Push failure is fatal.

## Adding an architecture

1. Drop a class in `src/temp_bench/architectures/<name>.py` subclassing
   `TempBenchArch`.
2. Add an entry to `configs/locked_archs.yaml` (class_path, arch_version,
   hparams).
3. Re-run any component's `experiments/cN_*/run.py` — only the new
   arch's cells compute. Everything else is cached.

## Markdown style

ATX headings (no H1 — Obsidian renders the filename as the title);
dash `-` bullets; fenced code blocks with a language tag; YAML
frontmatter (`author`, `date`, `tags`) on every `docs/` file; tags
must be `kebab-case`.

## Where to look for more

- **PROTOCOL.md** — full operating contract.
  - § 1 branch model, § 2 wasteland boundary, § 3 filesystem ownership,
    § 4 cache-key contract, § 5 two-TXC discipline, § 6 baselines,
    § 7 component writeup template, § 8 anti-conflict workflow,
    § 9 stop conditions, § 10 paper agent (orchestrator),
    § 11 framework discipline, § 12 GPU pinning, § 13 multi-GPU access.
- **docs/paper/framework.md** — the modularity design (10 principles,
  cache contract, version bumping)
- **docs/paper/architecture.md** — locked TXC spec, per-experiment
  training knobs (Bricken)
- **docs/paper/hardware.md** — pod specs, parallelism strategy,
  multi-GPU access example, storage layout
- **docs/paper/outline.md** — paper structure, headline figures
- **docs/components/c{1..7}.md** — per-component setup, hypothesis,
  results, caveats, reproduction
- **agents/README.md** — agent-to-component-to-pod mapping (roster only;
  protocol details live in PROTOCOL.md)
- **tests/** — `pytest -q` to verify framework contract holds

## Quick reference

```bash
# Han runs once per fresh pod (interactive — agent CANNOT run this):
#     cd /workspace/temp_xc/purified && bash scripts/bootstrap_runpod.sh

# session start (every restart — this IS what the agent runs):
cd /workspace/temp_xc/purified
source scripts/set_agent_env.sh <agent_name>
bash scripts/agent_smoke_test.sh

# pull from HF (ephemeral pods only)
bash scripts/sync_from_hf.sh

# run a component (idempotent — safe to re-run)
TQDM_DISABLE=1 .venv/bin/python -m experiments.c1_synthetic_topk.run

# read a wasteland file (code is on origin/han-phase7-unification)
git show origin/han-phase7-unification:src/architectures/txc_bare_antidead.py

# port a wasteland file into temp_bench (then add header comment)
git show origin/han-phase7-unification:src/architectures/txc_bare_antidead.py \
  > src/temp_bench/architectures/txc_base.py
```
