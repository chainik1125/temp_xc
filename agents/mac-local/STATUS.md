# Working state — agent `mac-local`

**Last rewrite:** 2026-07-10 (Opus 4.8, local CC session — pre-compact).

## Who / where
Local CC on the Mac (Apple M5 Pro, MPS, no CUDA) at `~/research/projects/temp_xc`.
Role: prototyping, review, orchestration. Heavy grids go to `runpod`.
I am the `mac-local` agent (inferred from darwin + this path — no need to be told).

## Git
Branch `arxiv`, **level with `origin/arxiv` @ `5f586be9`**, clean tree.
`origin` is the SSH URL (`git@github.com:chainik1125/temp_xc.git`).

## Done this stretch (all committed + pushed)
- Setup/cleanup: pushed the pending torch-index fix + removed dead `.bak`
  configs; switched `origin` to SSH.
- Walked the user through the synthetic-benchmark program (backtracking,
  signed_motion, topic_switching, changepoint) + the DC/AC lens, and through
  Dmitry's `dmitry-spectral-sprint2` FrequencyBench sprint.
- Scoped + wrote the **frequency (cyclic-tone) bench** plan; `runpod` built it
  (verdict **POSITIVE**). Reviewed: acceptance clean. Calibration note raised —
  the "DCT-band decisive" headline conflates the band *partition* (a tie at
  matched budget) with the *access prior*; the honest core is "position-mixing
  before the nonlinearity + budget + circle geometry," multiband ≈ vanilla at
  matched `k_win`. (Not yet edited into the record — see Open.)
- Wrote + dispatched the **record-pipeline refactor** brief; `runpod` executed
  it (shared lib `src/explorations/synthetic/` {grid,record,figs}.py + thin
  drivers + legibility cleanups). Independently verified **zero numeric drift**
  on all four records.
- Fixed the signed_motion CWD-relative figs-path bug (`37019b6d`).
- Built the **agents/ workspaces + briefings/** coordination model (`5f586be9`):
  three-state split (per-agent working STATUS / shared `briefings/` / shared
  research STATUS); agent id inferable from env; CLAUDE.md session-start updated.

## In flight
Nothing active. Clean stopping point.

## Next / open (nothing queued — user's call)
- **Optional calibration:** soften the frequency `bench_record.md` banner +
  research STATUS §0 headline to lead with the matched-budget tie / access-prior
  framing (the review critique). Low-risk prose-only edit; not yet done.
- **Optional low-priority:** wire signed_motion's `bench.md` `<!-- AUTO:* -->`
  block into the shared `record.py` (currently hand-maintained; flagged in its
  render_figs docstring).
- **Next exploration undecided.** The base is clean; DC / self-exciting /
  order-sensitive / change-point / frequency axes all have benches now.
