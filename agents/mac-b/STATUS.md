# Working state — agent `mac-b`

**2026-07-25 ~19:50 PT (overnight loop, briefing
`briefings/overnight-mac-b.md`).** Cap $100; my actual Modal burn so
far ≈ $5–6 (ledger-est lines total higher — estimates were
conservative). Stop: queue done / cap / blocked / 07:00 PT.

## Done (all pushed)

1. **B8 slen Stage-1 screen — COMPLETE.** Frozen card `b7121a208`;
   results `slen/results/screen_{gpt2,llama31_8b}.json` (174 cells
   each); verdicts in the task_hunt LOG (2026-07-26 mac-b entry):
   **lat KEEP + lev KEEP (order-free window faces), disp WEAK, the
   pre-registered recency LADDER COLLAPSES on 2/2 models** — the lat
   falsifier fired; amended order finding (R10) extends to this
   substrate. RECEIPTS **R20–R21 added, checker ALL PASS**.
   CANDIDATES.md outcome line appended. 2-model coverage (no HF
   secret ⇒ gemma pending, pre-authorized under the same card).
   PENDING TEAM REVIEW (Sunday).
   Ops scars: A10 OOM at llama T32 flatten-MLP → L40S; non-detached
   client cancel (same mode as mac-a) → drivers now sequential
   .remote + retries + `--detach`.

## In flight / next

2. **Stretch: B7 refmark screen — frozen and gated-cleared.** Card
   `refmark/CARD.md` + `screen.py` + `cache_acts.py` FROZEN at
   `c46d58826`; driver `scripts/modal_refmark_screen.py` (L40S,
   pinned there). Gate (slen pushed + ≤$60): CLEARED. Launch:
   `modal run --detach scripts/modal_refmark_screen.py` (caches ~10
   min, screen ≤ 1 h/model, est ≤ $8). build_rows validated
   label-side (user-echo 13 turns / 69 manifest rows dropped
   +disclosed; wd 112/31 convs). Scorer: slen/score.py pattern —
   refmark needs its own quick scorer or read cells directly.
3. After refmark: verdict per its card § 5 (KEEP needs visible-
   evidence floor + wd both cleared), LOG + RECEIPTS, ledger,
   STATUS. Then quotedens (B9, 5.3M tok/model) ONLY if plenty of
   margin before 06:30 PT hard stop for new starts.

## If resuming cold

Read `briefings/overnight-mac-modal.md` + `overnight-mac-b.md`, then
the LOG tail (my 2026-07-26 entry) and `refmark/CARD.md`. Modal
volume `temp-xc-replag-caches` holds replag + (maybe) refmark caches
+ results under /workspace/{slen,refmark}_results. Log:
`$SCRATCHPAD/modal_pipeline.log`. Ledger before ANY launch.
