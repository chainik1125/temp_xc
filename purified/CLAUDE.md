# CLAUDE.md — temp_bench v2 (arxiv branch)

You are an AI agent working on the `arxiv` branch — the post-submission
cleanup branch off Aniket's camera-ready (`origin/final-aniket`). All
work happens inside `purified/`.

**Read order on every session:**

1. `purified/docs/framework_v2.md` — the framework spec. Read top-to-bottom.
2. `purified/docs/reproduction_report.md` — current § 4 synthetic reproduction.
3. `purified/docs/ac_signed_motion_bench.md` — the AC / order-sensitive bench.

Before proposing or implementing **any synthetic benchmark**, read
`purified/docs/synthetic_benchmark_guidance.md` — the conventions every
synthetic task follows (ground truth, capacity, windowing, metrics).

Before running the **temporal-property autoresearch loop** (measuring temporal
structure in real language and mirroring it synthetically), read
`purified/docs/autoresearch_spec.md` — its prime directive and §3 validity
gates keep agents on rails (the goal is a sound verdict, never a "win").

All project docs live under `purified/docs/`. (The historical migration
docs `CLEANUP_PLAN.md` / `HANDOVER.md` were retired once the v2 migration
landed — recover from git history if needed.)

## Quick reference

```bash
# cd to purified/
cd $(git rev-parse --show-toplevel)/purified

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
