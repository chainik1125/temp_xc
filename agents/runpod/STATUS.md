# Working state — agent `runpod`

**Last rewrite:** 2026-07-11 (full clean-room rerun + purge COMPLETE).

## Who / where
Remote CC on RunPod (Linux) at `/workspace/temp_xc`. Git creds at `/workspace/.tokens/`.

## Last task: `briefings/full-rerun-and-purge.md` — DONE (briefing deleted)
Rebuilt the entire synthetic result set at protocol **1.3.0** under the uniform
design, regenerated all records + the program report, purged stale synthetic rows.
See `experiments/explorations/synthetic/STATUS.md` §0 for the science.

Key outcomes:
- ~2239-cell grid, **0 failures**. Ran on **CPU** (A40 is kernel-launch-latency-
  bound for these tiny d_in≤128 models → ~14% util; CPU ~7h). 12 workers, OMP=1
  (higher counts hit a memory-bandwidth/contention cliff). CUDA hidden via env.
- Anchor commit **C = `5b526c4d`** (protocol bump + drivers). Every fresh row
  stamps it; the purge kept synth iff `1.3.0 AND commit_sha==C AND seed∈{1,2,42}`
  (the seed clause dropped 12 stray timing-probe rows). Non-synth (356) preserved.
- Drift-sanity: overlapping cells reproduce **max|Δ|=0.000** on primary metrics.
- Program matrix filled 36/36 + both panels. All per-bench records regenerated;
  verdicts preserved; spectral_txc added as a fair column across all 4 benches.
- Committed as **C2** (see git log), pushed to origin/arxiv.

## Gotchas learned (for next heavy CPU grid on this box)
- 40+ concurrent `git diff HEAD` (runner code_version) SIGBUS on the growing
  tracked leaderboard/manifest → `git update-index --assume-unchanged` those
  churning files during the grid, `--no-assume-unchanged` before committing.
- Tiny models: GPU useless (latency-bound); CPU sweet spot ~12 workers, OMP=1.
- `pkill -f "<pattern>"` self-matches the launching shell → exit 144. Launch
  grids as harness background tasks; kill via TaskStop, not pkill.

## Next / open
Check `briefings/` for the next `status: active` brief. Nothing queued now.
The `.prepurge` leaderboard backup can be deleted once the push is confirmed good.
