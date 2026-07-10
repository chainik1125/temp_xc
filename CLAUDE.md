# CLAUDE.md — temp_bench v2 (arxiv branch)

You are an AI agent working on the `arxiv` branch — the post-submission
cleanup branch off Aniket's camera-ready (`origin/final-aniket`). All
work happens at the repo root.

**Session start — who am I, and the three kinds of state:**

> **You are one of a few agents (`agents/README.md`).** Infer your id from your
> environment: a macOS/darwin session at `~/research/projects/temp_xc` is
> **`mac-local`** (prototyping, MPS); the Linux RunPod box at `/workspace/temp_xc`
> is **`runpod`** (heavy grids, CUDA). If ambiguous, ask. Read your
> **`agents/<id>/STATUS.md`** — your own working-state briefing (what you were
> mid-doing, git position, next action). **Rewrite it before any compact** so
> your next context window resumes cleanly.
>
> Then two *shared* stores: **`briefings/`** holds task/idea handoffs any agent
> can pick up (see `briefings/README.md`) — if one is `status: active` and
> matches your task, execute it and **delete it when done**. And
> **`experiments/explorations/synthetic/STATUS.md`** is the living
> research-program state (verdicts, roadmap, benchmark index) — update it when
> you advance the science.

**Read order on every session:**

> **Resuming synthetic-benchmark work?** The research-program state — what's
> active, verdicts, locked designs, the benchmark index — lives in
> `experiments/explorations/synthetic/STATUS.md`. Read it after your workspace
> `agents/<id>/STATUS.md` (which holds your *own* mid-task thread + git state).
> Trust these two over scattered notes.

1. `docs/framework.md` — the framework spec. Read top-to-bottom.
2. `experiments/explorations/synthetic/README.md` — the synthetic-benchmark program: the prime
   directive, the measure→mirror→bench loop + validity gates, the conventions
   every benchmark follows, and the benchmark index (one self-contained subdir
   per benchmark: backtracking, signed_motion, topic_switching, changepoint, …).

Before proposing, implementing, or running **any synthetic benchmark** (or the
measure→mirror→bench loop on real language), read `experiments/explorations/synthetic/README.md`
in full — the single governing doc: prime directive ("a sound verdict, never a
win"), the validity gates that keep agents on rails, and the conventions every
benchmark follows (ground truth, capacity, windowing, metrics).

**Layout.** `src/` is **importable library code only**: `src/temp_bench/` (the
framework + its registered arch/eval/generator plugins) and `src/explorations/`
(reusable library code an exploration develops that isn't ready for `temp_bench`
— empty today). **Experiments** live under `experiments/`: the official
paper-section runners (`synthetic`, `probing`, `em`, `rlhf`, `backtracking`;
dispatched by `run.py <section>`) and `experiments/explorations/<name>/` for
exploratory ones. Today the one exploration is the **synthetic-benchmark
program** (`experiments/explorations/synthetic/`) — its single governing doc
`README.md` + the `STATUS.md` scratchpad at the root, then one subdir per
benchmark with scripts + docs + `figs/` + `results/` co-located. Run a
benchmark's scripts as `.venv/bin/python -m
experiments.explorations.synthetic.<bench>.<script>` (e.g.
`…synthetic.backtracking.run_grid`). Core-framework docs live under `docs/`
(idea writeups under `docs/ideas/`, e.g. `frequency_lens.md`, the DC/AC lens).
The canonical leaderboard stays at `results/leaderboard.jsonl`. (Historical
migration docs were retired — recover from git history if needed.)

## Quick reference

```bash
# cd to the repo root
cd $(git rev-parse --show-toplevel)

# the dispatcher
python run.py validate                        # self-check the registries
python run.py synthetic --arch txc_base --seed 0 --smoke
python run.py sweep configs/sweeps/<sweep>.yaml
python run.py reproduce all                   # canonical paper sweeps

# tests
.venv/bin/python -m pytest tests/ -q
```

## Hard rules (v2)

1. **One canonical pathway**: every result goes through
   `temp_bench.core.runner.run_experiment`. Never write your own
   leaderboard append.
2. **Code-version stamped**: every row carries
   `code_version.{commit_sha, dirty, diff_sha256}`. Runner refuses
   dirty trees unless `--allow-dirty` (or `TEMP_BENCH_ALLOW_DIRTY=1`).
3. **Plugin extension only**: adding an arch / eval / experiment is a
   single file drop + YAML entry. Never edit `temp_bench/core/`.
4. **Token shuffle buffer is the default**: not whole-sequence sampling.
5. **Paper-section names, not cN**: synthetic / probing / backtracking /
   em / rlhf.
