# Working state — agent `mac-b`

**2026-07-25 ~18:45 PT (overnight loop, briefing
`briefings/overnight-mac-b.md`).** Cap $100; ledger
`briefings/MODAL_SPEND.md` (my est so far ~$5). Stop: queue done /
cap / blocked / 07:00 PT.

## Where I am

1. **B8 slen Stage-1 screen — RUNNING on Modal.** Card FROZEN at
   `b7121a208` (`experiments/explorations/task_hunt/slen/CARD.md` +
   `screen.py`, commit-then-run honored; build_rows validated locally
   on a scratchpad gpt2 tokens.npz before freezing — full caps, wd
   arms supported 320/80 docs). Driver
   `scripts/modal_slen_screen.py` (pinned to the freeze commit),
   launched ~18:40 PT: smoke → caches (replag builders, gpt2 +
   llama31, Volume `temp-xc-replag-caches`) → 2× screen in parallel
   (A10G). Log:
   `$SCRATCHPAD/modal_pipeline.log` (background `modal run`, Monitor
   armed on milestones). Coverage: **2 models** (no HF secret ⇒
   gemma pending, card § 1).
2. Results repatriate to `slen/results/screen_<key>.json` via the
   driver's local entrypoint → I commit locally (containers never
   push).
3. Next after results: score against card § 5–6 (P1–P5, KEEP/KILL,
   LADDER verdict), 3 face verdicts + 1 ladder verdict in the
   task_hunt LOG as `mac-b (executor)` PENDING TEAM REVIEW, RECEIPTS
   proposals + `receipts_check`, ledger update, then stretch: refmark
   (B7) if ≤ $60 spent, never starting what can't finish by 06:30 PT.

## If resuming cold

Re-read the two briefings + `slen/CARD.md`. Check the Modal log
above; `modal app list` / dashboard for running apps. Partial screen
results live on the Volume at `/workspace/slen_results/` (resumable —
rerun `modal run scripts/modal_slen_screen.py --stage screen`).
Freeze discipline: any NEW runner ⇒ commit before first cell.
