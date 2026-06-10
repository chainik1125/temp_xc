# CLAUDE.md — temp_bench v2 (arxiv branch)

You are an AI agent working on the `arxiv` branch — the post-submission
cleanup branch off Aniket's camera-ready (`origin/final-aniket`). All
work happens at the repo root.

**Read order on every session:**

> **Picking up in-progress synthetic-benchmark work (e.g. after a compact)?**
> Read `synthetic/STATUS.md` **first** — the single living briefing
> of the current redo initiative (what's active, the locked design, the next
> actions, git state). It is the *one* file kept current before a compact; trust
> it over scattered notes.

1. `docs/framework.md` — the framework spec. Read top-to-bottom.
2. `synthetic/README.md` — the synthetic-benchmark program: the prime
   directive, the measure→mirror→bench loop + validity gates, the conventions
   every benchmark follows, and the benchmark index (one self-contained subdir
   per benchmark: backtracking, signed_motion, topic_switching, changepoint, …).

Before proposing, implementing, or running **any synthetic benchmark** (or the
measure→mirror→bench loop on real language), read `synthetic/README.md`
in full — the single governing doc: prime directive ("a sound verdict, never a
win"), the validity gates that keep agents on rails, and the conventions every
benchmark follows (ground truth, capacity, windowing, metrics).

**Layout.** Core-framework docs live under `docs/` (ideas /
explorations under `docs/ideas/`, e.g. `frequency_lens.md`, the DC/AC
lens). The synthetic-benchmark program lives under `synthetic/` — its
single governing doc `README.md` + the `STATUS.md` scratchpad at the root, then
one subdir per benchmark with docs + scripts + `figs/` + `results/` co-located.
Run a benchmark's scripts as `.venv/bin/python -m synthetic.<bench>.<script>`
(e.g. `synthetic.backtracking.run_grid`). The canonical leaderboard stays at
`results/leaderboard.jsonl`. (Historical migration docs were retired —
recover from git history if needed.)

## Quick reference

```bash
# cd to 
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
