# Working state — agent `runpod`

**Last rewrite:** 2026-07-14 (expansion Cycle 1 COMPLETE — stopped for review).

## Who / where
Remote CC on RunPod (Linux) at `/workspace/temp_xc`. Git creds at `/workspace/.tokens/`.
Claude API key at `/workspace/.tokens/anthropic_key` (validated; all 3 judge models OK).

## Last task: `briefings/grounded-benchmark-expansion.md` — Cycle 1 DONE
All four stages executed autonomously; **STOPPED for human review** (per the
briefing — Cycle 2 must not start before review; the briefing file stays until
the user reviews, then it is superseded by `expansion/README.md` and deleted).

Outcome (full detail: synthetic STATUS §0 + `expansion/LEDGER.md` cycle log):
- 10 cards frozen (commit `9fe8a29e`) → blind selection (4; 2/domain) →
  all 4 calibrated → **2 PROCEED** graduated as frozen specs
  (`synthetic/assumption_consequence/`, `synthetic/hedging_drift/`) and
  **2 ABORT** (both text-corpus, both skeptic kills after numeric-gate passes:
  leakage / mirror circularity — records at `expansion/records/`).
- Spend **$9.55 / $25** (meter `expansion/results/spend.json`). ~123k Haiku
  sentence labels; no architecture touched.
- Harness (reusable for Cycle 2): `src/explorations/synthetic/expansion/`
  (client+meter, signature+nulls, labeler, corpus, mirrors), 11 tests.

## Next / open
- **Blocked on user review of Cycle 1.** After review: delete the briefing;
  Cycle-2 targets are in the LEDGER cycle log (interaction/equality both
  domains, a text-corpus PROCEED, periodic/long-memory; preregister a
  non-fitted-moment mirror tolerance).
- The stage-6 blind B×A eval of the two new specs is a separate task the user
  must green-light (needs datasource plugins; nothing run this cycle).
- `results/leaderboard.jsonl.prepurge` backup can be deleted (push confirmed).

## Gotchas (this box)
- Claude 5-family models reject `temperature` (client handles).
- Tiny models: GPU useless here; CPU ~12 workers OMP=1 for temp_bench grids.
- `pkill -f` self-matches the launching shell → use TaskStop on harness tasks.
- `datasets` streaming can core-dump at interpreter exit AFTER writing cache —
  cosmetic (exit 134), check the cache file.
