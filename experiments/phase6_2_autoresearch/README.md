# Phase 6.2: autoresearch loop toward TXC-family parity with `tsae_paper`

**Status**: scaffold drafted, queued for launch after Phase 6.1
pipeline completes. See
[`docs/han/research_logs/phase6_2_autoresearch/brief.md`](../../docs/han/research_logs/phase6_2_autoresearch/brief.md)
for the motivating finding + candidate plan.

## Files

- `candidates.py` — 6 candidate arch recipes as Python dataclasses
  (name, base-arch, toggles, training-cost estimate).
- `propose.py` — deterministic + Claude-Sonnet hybrid selector that
  picks the next unexplored candidate given prior cycle results.
- `run_phase62_cycle.sh` — runs one (arch, seed) cycle end-to-end:
  train → encode → autointerp → append to results jsonl.
- `run_phase62_loop.sh` — top-level orchestrator; runs up to 6 cycles.
- `launch_after_phase61.sh` — wait-for-Phase-6.1-and-launch wrapper.

## Prerequisites

Phase 6.2 depends on Phase 6.1 having produced:
- All seed=42 concat_A/B/random z caches for the 9 Phase 6 archs.
- Triangle seed variance ckpts for Cycle F + tsae_paper + 2x2 cell.
- Probing regression numbers for Cycle F + 2x2 cell.
- §9.5 summary rewritten with the rigorous-metric baselines.

Verify before launching:

```bash
# 1. Phase 6.1 chain must have completed
grep -q "FULL PIPELINE DONE" logs/phase61_full_chain.log

# 2. Baselines table must be in summary.md §9.5
grep -q "Phase 6.1 update.*metric upgrade" \
  docs/han/research_logs/phase6_qualitative_latents/summary.md
```

## Launch (sequential, foreground)

```bash
source /workspace/temp_xc/.envrc
bash experiments/phase6_2_autoresearch/run_phase62_loop.sh
```

Runs up to 6 cycles sequentially. Each cycle trains a candidate arch,
evaluates, appends a row to `results/phase62_results.jsonl`, and
emits a summary line to stdout. Can be safely interrupted between
cycles.

To run a specific candidate without the selector:

```bash
bash experiments/phase6_2_autoresearch/run_phase62_cycle.sh C1
```

Valid IDs: C1, C2, C3, C4, C5, C6. See `candidates.py` for what each
does.

## Fitness function

Primary: `concat_random x/32` at seed=42 (temp=0, multi-judge).
Tie-breaker: `concat_A+B combined x/64`.
Constraint: probing Δ AUC ≤ 0.02 vs `agentic_txc_02` (rejects a
winner that degrades probing utility).

## Budget

- GPU: ~4 hr (6 cycles × ~40 min average)
- Haiku API: ~$3
- Sonnet API (proposer): ~$0.5

## Stopping criterion

Stop at 6 cycles, OR when a candidate hits `concat_random x/32 ≥ 10`
(matching `tsae_paper`'s 12/32 within 2 labels), OR when fitness
hasn't improved in 3 consecutive cycles.

## Phase 6.3 hand-off

Whichever candidate wins Phase 6.2 becomes the TXC champion. Phase
6.3 then retrains it at seeds {1, 2} and does a full Phase 6 apples-
to-apples comparison against `tsae_paper` / `tfa_big` on the rigorous
metric.
