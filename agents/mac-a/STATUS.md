# Working state — agent `mac-a`

**2026-07-25 ~22:40 PT — DELIVERABLE DONE + RATIFIED.** mac-local
wired R22 into receipts_check (26 claims ALL PASS, verified
reproducing locally; R5 amended superseded-pending-team-ratification,
clause operative until check-in) and approved the verdict
(`c5568fb72`). HF-mirror follow-up for the 3 Volume checkpoints is
recorded on their side. Now idle-looping (origin watcher armed) until
07:00 PT unless asks arrive.

Original verdict summary: tsae/T1
seed top-up {3,4,5} on Ward, n = 6 complete, b's frozen criterion MET
(paired one-sided 95% LB +0.0200 > 0, all 6 seeds positive; Welch 6v6
LB +0.0272 p = 0.0030; caveat-free new-only Welch LB +0.0357). Full
verdict + PROPOSED R5 update (→R22) + two named caveats (cross-cache
pooling; s3/s4 realized-l0 under band, POST-HOC exclusion variants
reported): LOG 2026-07-26 `mac-a (executor)` entry. PENDING TEAM
REVIEW; mac-local ratifies RECEIPTS.

## Where everything is
- Leaderboard: +3 rows (eval_keys 721bd3c6…, bfe6ea32…, 149a036a…),
  PIN `c93473ad3`, clean stamps, 0 dups. Panel file 87 cells.
  Bounds artifact: `lambda_intensity/results/topup_bounds_tsae.json`
  (+ committed `cache_fingerprint_topup.json`).
- receipts_check ALL PASS post-merge; 13 fixture tests pass. R4/R5
  as-written still PASS (pinned to round-1 seeds) until ratification.
- Checkpoints: Modal Volume `temp-xc-ward-caches`
  `checkpoints_topup/{a49569223227158e,2e8cf4b77839253e,a258f49f272d7a0a}`
  + `payloads/seed_{3,4,5}.json`. NO HF token here — mirror upload =
  Han/mac-local follow-up (HF_MIRROR.md rule).
- Modal: app `temp-xc-tsae-seedtopup`; caches on the same Volume
  (stream + labels + base/hs13, receipts PASSED in-container).
  Ops lesson: long runs DETACHED (non-detached client disconnect
  cancelled attempt 1); payloads persist server-side.
- Spend: mac-a actuals ≈ $19 of $150 cap (ledger corrected, total
  ≈ $39 est across agents).

## Remaining / next
- Stretch (§ 3, gate ≤$100 spent: CLEAR at $19): mac-b took refmark
  itself (running detached); B8 panel gate CLOSED by mac-local. Check
  LOG on wake for assist requests; otherwise idle-loop pull-rebase.
- At 07:00 PT stop; briefing retires at Sunday check-in.
- If resuming mid-anything: the loop is work → push → pull-rebase;
  ledger before/after every launch.
