# Working state — agent `runpod`

**Last rewrite:** 2026-07-14 (expansion Cycle 2 COMPLETE — stopped for review).

## Who / where
Remote CC on RunPod (Linux) at `/workspace/temp_xc`. Git creds at `/workspace/.tokens/`.
Claude API key at `/workspace/.tokens/anthropic_key` (all 3 judge models OK).

## Last task: `briefings/grounded-benchmark-expansion-cycle2.md` — DONE
All stages + both riders executed autonomously; **STOPPED for human review**
(no Cycle 3 before review; briefing stays until reviewed, then delete — the
expansion README is the standing doc).

Outcome (full detail: synthetic STATUS §0 + `expansion/LEDGER.md` cycle log):
- 4 new interaction/equality cards frozen under gates 7–8 (commit `927b1bc3`)
  → deterministic selection (6, 3/domain) → all calibrated + g7 re-exam.
- **self-reference-echo PROCEED → SPEC** (`synthetic/self_reference_echo/`).
- **g7 re-exam upgraded assumption-consequence SPEC*→SPEC** (asym 0.297 with
  the strict ctx=0 labeler — 2.2× stronger; canonical mirror = g7 fit).
- **gate-8 recheck downgraded hedging-drift SPEC→SPEC*** (long-memory ACF
  plateau; ar1 + semi-Markov both fail; hierarchical-AR(1) proposed for C3).
- 5 ABORTs, each on a distinct preregistered gate (sign-falsification ×1,
  gate-8 mirror ×3 — one a 4%-relative near-miss / prime re-freeze candidate —
  noise floor ×1). Text-corpus PROCEED target NOT met (reported honestly).
- Spend **$14.06 / $25** (meter `expansion/results/spend.json`; C1 archived
  as `spend_cycle1.*`). No architecture touched.

## Next / open
- **Blocked on user review of Cycle 2.** C3 targets are queued in the LEDGER
  cycle log: re-freeze list-item-parallelism (magnitude-relative gate-8
  tolerance) + computation-verification (periodic+self-exciting hybrid);
  Appendix-B menu extensions (hierarchical AR(1); periodic-Hawkes); the two
  still-frozen C1 cards (goal-restatement, enumeration-cadence).
- Stage-6 blind B×A eval of the two full SPECs needs a user green-light
  (datasource plugins to write; nothing run).
- `results/leaderboard.jsonl.prepurge` backup can be deleted (push confirmed).

## Gotchas (this box)
- Claude 5-family models reject `temperature` AND think by default — tight
  max_tokens ⇒ empty text (client handles both).
- Tiny models: GPU useless here; CPU ~12 workers OMP=1 for temp_bench grids.
- `pkill -f` self-matches the launching shell → use TaskStop on harness tasks.
- Background python: launch with `-u` or prints sit in the block buffer.
