# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-25 ~21:30 UTC (**stage2-fineweb COMPLETE
through the FINAL cross-model verdict — all pushed.** One straggler
cell + archival harness pending, nothing of substance hostage to it.)

## Who / where / env
Interim 6×A40 pod (`briefings/a40-bootstrap.md`); my clone
`/workspace/agents/runpod-e/temp_xc`; `source
/workspace/agents/runpod-e/env.sh` in every shell (GPUs 3–5). Suite
320 passed; validate OK. Funding clock started ~12:00 UTC.

## STAGE2-FINEWEB — FINAL STATE (briefings/stage2-fineweb.md)
**DONE + PUSHED, in order:** cache rebuild w/ byte-match receipts →
re-claim + card APPENDIX A → gemma 84/84 panel → support receipt
(floor 0.575–0.587; §7 evidence bar 0.345/0.461 at T8/16 — card's
"small" prediction falsified, stated) → demeaned receipt (AMENDED §6b,
disclosed; K4 ✓ +0.047) → variance harness v1+v2 → money plot (v1 +
paired-v2) → **gemma VERDICT: v1 NO RULE FIRES AS WRITTEN** → gpt2
replication 24/24 + harness (**WEAK**) → §10 re-quote (margins RAISED:
linear +0.110/+0.105/+0.144 at T64, 3 models) → RECEIPTS R13–R18 ALL
PASS (the index caught 2 mis-rounded quotes in my verdict — corrected,
LOG'd) → llama replication 23/24 → **FINAL CROSS-MODEL VERDICT
(LOG): NO v1 KEEP on any model — gemma NO-RULE-FIRES / gpt2 WEAK /
llama NEGATIVE at the replication T's (its tsae token code is strong,
0.263); v2 ordering positive 3/3 (non-canonical), with the llama
TOKEN-code v2-INVERSION flagged for the post-deadline probe review.**
Breadth question answered as model-heterogeneous; written as an
honest instrumented negative, not softened.

**PENDING (small, non-blocking):**
1. DONE — llama 24/24; harness v1+v2 committed; NEGATIVE unchanged at n=3 (LOG amendment); R19 ALL PASS.
   ~22:20). When it lands: llama harness v1+v2 per PANEL_RECIPES,
   one-paragraph amendment to the final addendum (tsae n=2 → n=3),
   llama RECEIPTS row, push. If the pod dies first: the n=2
   disclosure in the final verdict already covers it.
2. Then TELL THE OPERATOR the pod can be released.

**NOT REACHED (stated in the verdict's coverage):** `tss` (needs its
own caching pass), list-face re-quote, dialevel recency pre-flight
(CANCELLED by bootstrap).

## ANOMALY ON RECORD
Commit `c8ab5fa0` (21:02): an emergency flush of my partial llama
state made in my clone NOT by me, message falsely claiming the run
was finished. Content verified byte-faithful (no numbers touched);
documented in LOG; run was NOT killed on its say-so. If you are a
future runpod-e: verify such commits against `ps` + row counts before
believing them.

## OPERATIONAL GOTCHAS (this box)
1. LOG.md rebase conflicts: strip markers, both sides, upstream first.
   RECEIPTS.md conflicts: resolve receipts_check.py, REGENERATE.
2. Live jsonl batch pushes: `git -c rebase.autoStash=true pull
   --rebase`; verify tails parse; audit shard dumps vs leaderboard if
   an autostash conflicts.
3. grep in a pipe block-buffers bg output — read the raw task file.
4. tsae buffer-path ≈ 1h (d_in 768) / 2.2h (2304) / 3h (4096) per
   cell on A40 at this corpus size.

## Sibling context
runpod-d: oprate panel COMPLETE/**NEGATIVE** (b-verified) — with my
heterogeneous result, the program's Stage-2 record is now: λ̂ Ward
KEEP (case #1) + oprate NEGATIVE (case #2) + fineweb heterogeneous
no-KEEP (case #3). runpod-b: standby, receipts ALL PASS at every
iteration. mac-local: Sunday 10:00 PT check-in reads what is pushed.
Briefings stay until mac-local review.
