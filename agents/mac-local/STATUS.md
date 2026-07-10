# Working state — agent `mac-local`

**Last rewrite:** 2026-07-10 (Opus 4.8, local CC session).

## Who / where
Local CC on the Mac (Apple M5 Pro, MPS, no CUDA) at `~/research/projects/temp_xc`.
Role: prototyping, review, orchestration. Heavy grids go to `runpod`.

## Git
Branch `arxiv`, level with `origin/arxiv`, clean tree.

## Done this stretch
- Setup/cleanup: pushed the pending torch-index fix + removed dead `.bak`
  configs; switched `origin` to the SSH URL.
- Walked the user through the synthetic-benchmark program (backtracking,
  signed_motion, topic_switching, changepoint) + the DC/AC lens.
- Scoped + wrote the **frequency (cyclic-tone) bench** plan; `runpod` built it
  (verdict **POSITIVE** — see research STATUS). Reviewed it: acceptance clean;
  flagged that the "DCT-band decisive" headline conflates band-partition (a tie
  at matched budget) with the access prior — a calibration note, not a blocker.
- Wrote the **record-pipeline refactor** brief; `runpod` executed it (shared lib
  `src/explorations/synthetic/` + thin drivers + legibility cleanups).
  Independently verified **zero numeric drift** on all four records.
- Fixed the signed_motion CWD-relative figs-path bug.
- Established the `agents/` workspaces + `briefings/` conventions (this change).

## In flight
Nothing active.

## Next / open
- Optional low-priority: wire signed_motion's `bench.md` `<!-- AUTO:* -->` block
  into the shared `record.py` (currently hand-maintained; flagged in its
  render_figs docstring).
- Next exploration undecided — the base is clean for a new synthetic benchmark
  (the periodic axis is now covered; DC/AC/self-exciting/change-point/frequency
  all have benches).
