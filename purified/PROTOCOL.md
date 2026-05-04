# PROTOCOL.md — agent operating rules

Concrete rules for multi-agent coordination on this paper. Read fully before
your first action.

## 1. Branch model

- **`final`** is the only paper branch. All work commits here.
- Never push to `main`, `han-phase7-unification`, `em-nanda`,
  `aniket-ward-stage-b`, or any other branch. Those are wasteland.
- Pull `final` before each session: `git pull --rebase origin final`.
- Push frequently — at least once per substantive change. If a push
  conflicts, rebase (don't merge).

## 2. The wasteland boundary

Wasteland **code** (`src/`, `experiments/`, `references/`, `tests/`,
`scripts/`, root `pyproject.toml`, `Dockerfile`, etc.) is **deleted from
the `final` branch** as of 2026-05-03. It lives only on
`origin/han-phase7-unification`. Wasteland **docs** (`docs/`, `papers/`,
root `CLAUDE.md`, `RUNPOD_INSTRUCTIONS.md`) are kept on `final` because
component writeups cite them heavily.

The deletion is git-level enforcement of the quarantine: an accidental
`from src.architectures.tfa import ...` now raises `ModuleNotFoundError`
immediately rather than silently picking up wasteland code.

### Reading wasteland code from origin

```bash
# refresh once per session (purified/scripts/wasteland_refresh.sh)
git fetch --all --prune

# read a wasteland code file
git show origin/han-phase7-unification:src/architectures/txc_bare_antidead.py

# list wasteland code paths
git ls-tree -r origin/han-phase7-unification --name-only | grep '^src/architectures'

# port a wasteland file into temp_bench (then add a header comment with
# the source path + commit hash)
git show origin/han-phase7-unification:src/architectures/txc_bare_antidead.py \
    > purified/src/temp_bench/architectures/txc_base.py
```

After porting, add a header comment to the new file:

```python
# Ported from origin/han-phase7-unification:src/architectures/txc_bare_antidead.py
# at commit <sha> on YYYY-MM-DD by <agent>.
```

Live imports from origin are forbidden; one-time copies with
attribution are fine.

### 2a. Cross-branch reads (em-nanda, aniket-ward-stage-b)

Dmitry's emergent-misalignment work and Aniket's backtracking work also
live on other branches and are still being updated. Same pattern:

```bash
git show origin/em-nanda:docs/dmitry/results/em_features/em_nanda_results_paper.md
git show origin/aniket-ward-stage-b:docs/aniket/experiments/ward_backtracking/handoff_neurips_push.md
```

We never merge any sibling branch into `final` — that would freeze
stale snapshots and create conflict surface.

## 3. Filesystem ownership

| Path | Owner | Mutability |
|---|---|---|
| `purified/agents/<name>/` | Agent `<name>` only | Owner write, others read |
| `purified/docs/components/c{N}.md` | Component lead (see briefing) | One agent at a time; coordinate via header comment |
| `purified/docs/paper/` | Agent PAPER | PAPER write, others read |
| `purified/src/temp_bench/` | First mover, then negotiate | Treat as shared; small additive PRs |
| `purified/results/runs/<run_id>/` | Run owner | Append-only after run completes |
| `purified/results/leaderboard.jsonl` | All | **Append-only**; use `flock` |
| `purified/checkpoints/manifest.jsonl` | All | **Append-only**; use `flock` |

## 4. Cache-key contract (replaces the old run_id pattern)

The framework computes deterministic cache keys for you. **Do not
allocate run-ids manually.** Three keys, all 16-hex-char SHA-256 prefixes:

- `act_cache_key` — `(subject_model, layer, hookpoint, dataset, n_seqs, seq_len, tokenizer_revision)`
- `train_key` — `(arch_class, arch_version, hparams, seed, training_cfg, act_cache_key)`
- `eval_key` — `(train_key, eval_protocol_version, eval_cfg)`

The framework guarantees:
- Same inputs → same key → cache hit (re-run is safe + idempotent).
- Bumping `arch_version` invalidates `train_key` (forces retrain).
- Bumping `EVAL_PROTOCOL_VERSION` invalidates `eval_key` (forces re-eval, retains training).

See `docs/paper/framework.md` for the full design.

## 5. Two-TXC discipline

- **TXC-base** = `txc_bare_antidead_t5`. Implementation lives in
  `src/temp_bench/architectures/txc_base.py`. Registered in
  `configs/locked_archs.yaml` as `txc_base`.
- **TXC-pro** = `phase5b_subseq_h8`. Implementation in
  `src/temp_bench/architectures/txc_pro.py`. Registered as `txc_pro`.

Hyperparameters live in `configs/locked_archs.yaml` only — never as
constants in `.py` files. A component may override a hyperparameter via
the `per_component_hparams` map in the yaml entry (e.g., a different
`d_sae` for C6's 14B-organism cells).

If you find yourself wanting to change the architecture's structure
(not just hparams), **stop and post to `docs/components/cN.md` first**.
Justify it. The paper makes a "two architectures everywhere" claim
that breaks if any component silently drifts.

**Per-experiment training knobs** (e.g., Bricken resample, mixed
precision) are allowed if you disclose them in `docs/components/cN.md`
and either (a) cite prior evidence supporting them on a comparable
setup, or (b) run an A/B at small scale and report the verdict. These
are training-time augmentations that do not change the architecture's
mathematical identity. They flow through `TrainingConfig` (different
config → different `train_key` → separate cache entries, both kept).
See `docs/paper/architecture.md` § *Per-experiment training knobs* for
the full opt-in table.

## 6. Baselines (also locked)

| Slug | Description |
|---|---|
| `topk_sae` | Per-token TopK SAE, k=k_pos, d_sae=d_sae. The simple baseline. |
| `tsae_paper` | T-SAE (Bhalla et al. 2025). Use the paper's released config. |
| `tfa` | Temporal Feature Analysis (priors_in_time). Used in C1/C2/C7 only. |
| `mlc` | Multi-layer crosscoder (Lieberum et al. 2024, paper config). C3 only. |
| `sae_arditi` | EM-only baseline. The C6 winner. |

## 7. Component writeup template

Each `docs/components/cN.md` follows this structure:

```markdown
---
component: cN
status: planning|running|complete
lead: <agent name>
last_update: YYYY-MM-DD
---

## Hypothesis
(what this component proves for the paper, in 1-2 sentences)

## Setup
(data, models, hardware, hyperparameters, seeds)

## Results

<!-- BEGIN AUTO-RESULTS -->
(auto-generated by experiments/cN_*/analysis.py — do not hand-edit)
<!-- END AUTO-RESULTS -->

## Caveats
(seed variance, brittleness, things we tried that didn't work)

## Reproduction
(exact commands)
```

### Results live in state, not prose

**Hard rule**: numbers in the `## Results` section are NEVER
hand-typed. The block between `<!-- BEGIN AUTO-RESULTS -->` and
`<!-- END AUTO-RESULTS -->` is owned by `experiments/cN_*/analysis.py`
and rewritten atomically by `temp_bench.report.render(component=...)`.

Why: hand-typed numbers drift from the leaderboard. They also can't be
re-rendered when a new seed lands or a metric is fixed. Sourcing every
paper-relevant number from `results/leaderboard.jsonl` via a deter-
ministic analysis script keeps the writeup coherent with state.

Workflow when adding a new result:

1. Run cells through `temp_bench.runner.run_cell` — appends to the
   leaderboard, writes per-cell `results/runs/<eval_key>/`.
2. Edit `experiments/cN_*/analysis.py:run_analysis()` (copy from
   `experiments/_analysis_template.py` if it doesn't exist). Query
   the leaderboard via `temp_bench.report.query_leaderboard(...)`,
   compute summary stats, save aggregate plots to
   `experiments/cN_*/plots/`, return an `AnalysisResult`.
3. Run `temp_bench.report.render(component="cN")` (or
   `render_all()`). It writes `experiments/cN_*/results.json` and
   rewrites the AUTO-RESULTS block in `docs/components/cN.md`.
4. Commit `analysis.py`, `results.json`, the regenerated plots, and
   the regenerated `cN.md` together.

The framework guarantees idempotency: running `render` twice with no
new cells produces an identical .md. CI runs `check_markers()` to
verify the markers are present.

What CAN go in `cN.md` outside the AUTO-RESULTS block:
- Hypothesis, Setup, Caveats, Reproduction — human prose.
- Reference numbers from prior work (origin/em-nanda, papers/) — those
  are external state, not paper claims; clearly delimit with a header
  like "Reference numbers (wasteland — for context only)".

What MUST go through `analysis.py`:
- Any number that appears in the paper's figures, tables, or claims.
- Any plot that the paper renders.

### Component docs vs agent briefings

- **`docs/components/cN.md`** is **component-centric**: hypothesis,
  setup, results, caveats. It outlives the agent. When agent_nlp gets
  reassigned, `c3.md` stays.
- **`agents/<name>/briefing.md`** is **agent-centric**: identity +
  current state + next action. It's about the agent, not the
  component.

Agent briefings **point at component docs** for technical detail; they
do NOT duplicate hypothesis / setup / results. If you're tempted to
duplicate, you have the abstraction wrong — push the technical content
into the component doc, leave a one-line pointer in the briefing.

## 8. Anti-conflict workflow

For any file under shared ownership:

1. `git pull --rebase origin final`
2. Open the file. If header comment names another agent and is <2 hr old,
   ping in `docs/components/cN.md` "Status" line and back off.
3. Add a header comment with your name + start time before editing:
   `<!-- editing: agent_nlp 2026-05-03T14:30Z -->`
4. Edit. Commit with the same agent name in the message.
5. Remove the header comment in the same commit.
6. `git push`. If push fails: pull-rebase, resolve, push again. Never
   force-push `final`.

## 9. Stop conditions

Stop and write to your agent log if:

- A component number diverges from another agent's run on the same arch+seed
  by more than 2× σ_seeds. Investigate before adding more rows.
- An architecture's training crashes silently (NaN, dead-feature collapse).
- A baseline number contradicts a published paper by more than 0.05 AUC.
- You're tempted to introduce a third TXC variant. (Don't.)

## 10. Paper agent (orchestrator)

Agent PAPER is the only agent allowed to:

- Edit `docs/paper/`.
- Update `docs/components/cN.md` "Hypothesis" or "Caveats" sections without
  notifying the component lead.
- Decide cross-component questions (notation, figure style, story arc).

PAPER does not own training compute beyond the local 5090. PAPER's
day-to-day is: read leaderboard, draft sections, raise issues to component
leads via their agent dirs, integrate component writeups into the paper.

## 11. Framework discipline (load-bearing — do not deviate)

These rules exist because a 72-hour, 7-component, 5-agent paper cannot
absorb framework friction. Read `docs/paper/framework.md` for the full
rationale.

1. **Hyperparameters in YAML, not code.** Edit
   `configs/locked_archs.yaml` and `configs/datasources.yaml`. Never
   hard-code `d_sae=18432` or `subject_model="gemma-2-2b-it"` in a `.py`.
2. **Datasources are named, not constants.** Each component declares
   `DATASOURCE = "<name>"` referencing `configs/datasources.yaml`.
3. **Use `runner.run_cell` for every cell.** It is the only path that
   appends to `leaderboard.jsonl`. Schema validation is mandatory.
4. **Bump `arch_version` to invalidate trained checkpoints.** Don't
   delete `.safetensors` files manually; the cache contract relies on
   keys, not file lifetimes.
5. **Bump `EVAL_PROTOCOL_VERSION` to invalidate eval results.**
6. **Add an arch = yaml entry + class file.** Nothing else. If you
   need to touch a component's runner, you've found a framework bug.
7. **Add a component = copy `experiments/_runner_template.py` + set
   `EVAL_PROTOCOL_VERSION` and `DATASOURCE`**. The eval logic goes in
   `src/temp_bench/eval/<name>.py`, not in the runner script.

### Code reuse contract (load-bearing — non-negotiable)

The single most important framework property: **the same architecture
or trainer or eval used in two components is the SAME code, not a
fork**. With 5 agents touching the repo in 72 hours, parallel ports
are the dominant failure mode. Enforced by structure + tests:

**One arch = one class file = one yaml entry.**
- `configs/locked_archs.yaml` lists all archs. One class_path each.
- `temp_bench/architectures/<name>.py` contains the class. One file.
- `tests/test_arch_registry.py` enforces both directions: every yaml
  entry has an importable class; every .py is referenced by some yaml
  entry (no orphans, no duplicates).
- Per-component customisation goes in `per_component_hparams.cN`.
  C7's d_sae=32768 (Llama d_in=4096) is set there, NOT in a forked
  `txc_base_for_c7.py`. Class file stays canonical.
- If you need to extend an existing arch, edit the file in place and
  bump `arch_version`. Forking with a new name is a last resort —
  raise it in `docs/components/cN.md` first.

**Arch porting: first-needs-it ports it.**
- Class `.py` files in `temp_bench/architectures/` are open ownership.
  Any worker who needs an arch ports it from the wasteland (with
  header attribution per § 2) and removes the entry from
  `tests/test_arch_registry.py::KNOWN_UNPORTED`. agent_paper is NOT
  the gatekeeper.
- YAML registration (`configs/locked_archs.yaml`) is agent_paper's
  territory (cross-cutting, must stay canonical) — but all 9 entries
  are already in. Workers only write the class file matching the
  existing `class_path`.
- This avoids agent_paper bottlenecking on a slow local 5090 while
  faster pod agents wait. Concretely: agent_nlp ported `tsae_paper`
  + `txc_base` (their C3 needs); agent_em ported `sae_arditi` (C6);
  any worker can port stacked_sae / tfa / mlc / txc_pro when they
  need them.

**One trainer for all SAE-family archs.**
- `temp_bench.training.train_sae(model, batch_iter, training_cfg)` is
  the canonical training entry. Components pass an instantiated model
  (built via `config.instantiate_arch(spec, d_in=…)`) and call it.
- Per-arch behaviour (auxK, contrastive, matryoshka, decoder
  projection) lives in the arch class via `train_step(x)` and
  `post_step()` overrides on `TempBenchArch`. The trainer does not
  branch on arch type.
- Components do **not** write training loops. If you find yourself
  writing `for step in range(...): forward(); loss.backward(); ...`
  in `experiments/cN_*/run.py`, stop — that logic belongs in the
  arch's `train_step` (per-arch loss) or the shared trainer
  (everything else: optimizer, warmup, grad clip, snapshots, Bricken).

**Shared eval modules per metric class.**
- `temp_bench.eval.synthetic` — C1, C2 toy NMSE / AUC / gAUC.
- `temp_bench.eval.probing` — C3 sparse probing (mean-pool + S-tail).
- `temp_bench.eval.qualitative` — C4 Top-256 cumulative SEMANTIC.
- `temp_bench.eval.steering` — C5 V7 + PP coh-vs-success curves.
- `temp_bench.eval.case_study` — C6 + C7 case-study harness.
- Each component's `my_eval_fn` composes these primitives. Components
  do NOT roll their own probing pipeline / judge dispatch / Pareto
  computation. If a function is missing, add it to the shared module
  and PR it; do not inline.

**Component runners are thin.**
- `experiments/cN_*/run.py` is ~30 lines: imports, `COMPONENT`
  + `DATASOURCE` + `EVAL_PROTOCOL_VERSION` constants, a `my_train_fn`
  that calls `train_sae`, a `my_eval_fn` that calls
  `temp_bench.eval.<module>.<fn>`, and `runner.run_cell` calls in a
  loop over `(arch, seed, eval_cfg)`. `experiments/_runner_template.py`
  is the canonical scaffold.

**Why this is non-negotiable.**
- Two ports of TopK-SAE = two probing tasks debugging two slightly
  different MSE definitions. Cache keys diverge. Leaderboard becomes
  uncomparable. We cannot afford this.
- One trainer = one place to land Bricken, mixed-precision, snapshot
  cadence, gradient-accumulation. Parallel reimplementations diverge
  silently.
- Shared eval = one definition of "PR-AUC at 12% positive" across
  C7's PR-AUC and any other component that wants imbalance-aware
  classification. Same denominators, same plots, comparable numbers.

## 12. GPU pinning on shared pods

When two or three agents share a pod, each agent **must** pin
``CUDA_VISIBLE_DEVICES`` before any CUDA code runs. Otherwise PyTorch
defaults every process to ``cuda:0`` and agents collide on the same GPU.

**Session start (every agent, every pod, every restart):**

```bash
cd /workspace/temp_xc/purified
source scripts/set_agent_env.sh <agent_name>     # pins GPU + sets AGENT_NAME
bash scripts/agent_smoke_test.sh                 # verifies pinning
```

The smoke test calls ``runner.preflight()`` which warns if:
- ``CUDA_VISIBLE_DEVICES`` is unset on a multi-GPU pod
- ``torch.cuda.device_count() > 1`` after pinning (env var didn't take)
- ``AGENT_NAME`` is unset (leaderboard rows would be tagged "unknown")

The agent → GPU mapping is in `purified/agents/README.md` *Active
roster*; the executable copy is `scripts/set_agent_env.sh`. They must
match.

**Why pinning matters more than usual here:**
Agents share the same network volume (one `purified/checkpoints/`
tree). If two agents accidentally trained the same `(arch, seed,
training_cfg, act_cache_key)` cell on different physical GPUs, both
would write the same `train_key`. cuBLAS heuristics differ across
H100 SKUs so the saved weights would be near-identical but not
bit-identical — undefined cache state.

The pinning makes every shared-pod cell run on a single, stable GPU.
Re-running yields the same outputs (modulo cuDNN nondeterminism, which
is suppressed by `set_seed(deterministic=True)`).

## 13. Multi-GPU access (Primary + Pool protocol)

When N named agents share a pod with M ≥ N GPUs, each agent has one
**primary** GPU (always owned, pinned by `set_agent_env.sh`). Remaining
GPUs form a **pool** that any agent may claim via the
`temp_bench.utils.gpu_locks` lockfile manager.

For the 4× A40 pod (2 named agents):
- **Primary (reserved):** agent_steer = GPU 0; agent_back = GPU 1.
- **Pool (shared):** GPUs 2 and 3.

For the 2× H100 pod (2 named agents): no pool — each agent's primary
IS its only GPU. Multi-GPU not applicable here.

### Claiming a pool GPU

```python
from temp_bench.utils.gpu_locks import claim_gpu, claim_gpus
import os, subprocess

# Single spare claim
with claim_gpu(2, note="C5 seed-1 parallel"):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "2", "AGENT_NAME": "agent_steer"}
    subprocess.run(["python", "-m", "experiments.c5_steering.run", "--seeds", "1"], env=env)

# Multi-GPU claim (atomic — all or nothing)
with claim_gpus([2, 3], note="C5 seeds 1+2 parallel"):
    procs = []
    for gpu, seed in [(2, 1), (3, 2)]:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "AGENT_NAME": "agent_steer"}
        procs.append(subprocess.Popen(["python", "-m", "experiments.c5_steering.run", "--seeds", str(seed)], env=env))
    for p in procs:
        p.wait()
```

Each subprocess passes preflight because each sees exactly one GPU.
The parent agent's role is coordination (claim → launch → wait → release).

### Why Primary + Pool, not strict modulo

Pure modulo partitioning (steer = {0, 2}, back = {1, 3}) is too rigid:
if agent_back finishes early, GPUs 1+3 sit idle while agent_steer
crawls. Primary + Pool keeps each agent's primary inviolable while
letting either claim spare capacity opportunistically.

### Hard rules

1. **Never use another agent's primary.** Pool only.
2. **Always claim before pinning.** Launching with
   `CUDA_VISIBLE_DEVICES=2` without `claim_gpu(2)` first is a
   PROTOCOL.md § 13 violation — silent contention.
3. **Use `claim_gpus(...)` for multi-claim**, not nested `claim_gpu`s.
   The atomic version sorts indices and releases everything if any
   claim fails — eliminates deadlocks where two agents hold one of
   each other's targets.
4. **Lock files are best-effort coordination.** A misbehaving agent
   that ignores claims will collide. The framework gives you visible,
   debuggable failure when locks are honored.
5. **Stale locks auto-reclaim.** If a previous pod crashed mid-claim,
   `cleanup_stale()` (run by the smoke test) GCs locks whose PID is
   no longer alive.

### Multi-GPU is multi-process, not multi-CUDA-device

Our "no DDP" decision (`docs/paper/hardware.md` § Single-GPU vs
multi-GPU) means: to use N GPUs, an agent launches N subprocesses,
each with `CUDA_VISIBLE_DEVICES=<single_idx>`. The parent process
never sees more than one GPU — `runner.preflight()` would warn if it
did. The lock manager prevents two agents from racing for the same
spare; the per-process pinning prevents subtler cross-agent CUDA
contention.

## 14. Briefing maintenance (across context-compact boundaries)

LLM-agent context windows fill up; eventually context gets compressed
and an agent's working memory of the current session is gone. The
agent's **identity** survives (their name, mandate), but their
**state at compact time** must be written down explicitly or it's lost.

We use a **single rolling briefing** per agent at
`purified/agents/<name>/briefing.md`, with explicit section ownership.
There is no separate handover file, no dated archive, no log.md —
git history + this briefing + `decisions.md` carry everything.

### Section ownership (load-bearing)

The briefing is one file with two ownership zones:

| Section | Owner | Mutable by |
|---|---|---|
| `## Identity + mandate (Han owns — agents do not edit)` | Han | Han only |
| `## Current state (agent owns — overwrite)` | the agent | self (overwritten each compact) |
| `## What I just did (agent owns — overwrite)` | the agent | self |
| `## Next action (agent owns — overwrite)` | the agent | self |
| `## Don't repeat (agent owns — overwrite)` | the agent | self |
| `## Open questions for Han (agent owns — overwrite)` | the agent | self |

Han's section is read-only to agents. Han may rewrite it at session
start to redirect priorities. The agent's sections are overwritten
freely by the agent itself (and only by the agent that owns the
briefing — no cross-agent edits, see § 3).

### When to update the briefing

- **Before any anticipated compact** — when context fills up. Update
  state, push, keep working.
- **At session end** — leave the briefing in a fresh state so the
  next-life instance resumes cleanly.
- **After a substantive milestone** — e.g. "C3 caching done, training
  starts next" deserves an immediate state refresh.

You can update more often than strictly necessary. The cost is
trivial (overwrite ~30-50 lines).

### Successor reads (post-compact)

The next-life instance of an agent, after compact, reads:

1. Auto-loaded `purified/CLAUDE.md` (the operating manual).
2. Their own `briefing.md` — top section (Han's mandate) + bottom
   sections (current state, next action, etc.).
3. `decisions.md` — locked decisions for context.
4. (For chronological detail, optional)
   `git log -p purified/agents/<name>/briefing.md` — every state
   transition is captured automatically.
5. `docs/components/cN.md` for any component about to be touched.

### Hard rules

1. **Identity continuity over instance freshness.** A successor
   instance of `agent_paper` is still `agent_paper`. Don't fork
   identity (no "agent_paper_v2"). Update the briefing, keep the name.
2. **Update before compact, not after.** Once compact has happened,
   the working state is gone. Pre-empt.
3. **Reference, don't duplicate.** The briefing points at
   `decisions.md` and `docs/components/cN.md`. It doesn't restate them.
4. **Concrete next action.** "Continue working on C3" is not a next
   action. "Run `python -m experiments.c3_probing.run --seeds 1
   --archs txc_base`, expect ~2 H100-hours" is.
5. **Han's section is read-only.** If you think Han's mandate has
   shifted, raise it in `## Open questions for Han` instead of
   editing his prose.
6. **No separate handover file. No log.md.** Each agent has exactly
   one rolling doc: `briefing.md`. Git history is the audit trail.
7. **Keep it ≤ 200 lines.** Briefings are loaded on every
   session-start; brevity matters.

### Template

Copy `purified/agents/_briefing_template.md` when provisioning a new
agent. Han fills in the top section (identity + mandate); the agent
fills in the rest at first compact.
