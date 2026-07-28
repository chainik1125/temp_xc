# Working state — agent `runpod-1`

**2026-07-28 ~07:30 UTC (morning rewrite) — ALL DELIVERABLES
SHIPPED. Night+sprint arc complete; nothing of mine in flight on
either GPU. Awaiting only hub ratifications + any new directives.
GPU 2 = runpod-2 (pf_pilot running there — never mine).**

## Delivered this arc (receipts = LOG shas)

1. **Paper-faithful sprint (task #12)**: plugin `paper_txc_base_v1t`
   (vendored 94119bc08, 8/8 contract tests, bitwise adapter parity)
   + CARD_PAPER_FAITHFUL.md (PIN d9235755b, ratified 16d26642c) +
   my shards A+B 8/8 cells ($14-15, under est) + grid COMPLETE
   21/21 fleet-wide (pod-B strays repatriated by hub 6e928e2bb).
   **Card §9 SCORED (5075d098e): E1 CONFIRMED (zero-picks monotone,
   onset T6-T8, ≤0.14% of budget) · E2 NOT CONFIRMED (null — pf ≡
   btk at matched budgets; T8 k5 3/3 ABOVE) · E3 PASS (archived T5
   anchors interpolate retrained T4-T6 both k = vendoring validated
   end-to-end).** pf renders live (RESULTS_paper-faithful.md +
   figs), regenerate via `analysis.py --arm paper-faithful`.
2. **RM_CERTIFICATE v1.0** (aa6067152 + review fixes fcd744e1f):
   census-first lemma pair; identity set; T2-T16 divergence map w/
   caveats; §3a METRIC-NULL; trace bounds (v2 arms 0/1120 sampled
   contacts; floors decline with T); controls; §7 paper-faithful
   corroboration; RLHF cross-venue identity now tensor-grade
   through T10/s42 (rmx_b checks 1-4; 5-6 append as extensions).
3. **11:00 PROTECTED btk renders SHIPPED EARLY** (6391ced2c,
   deliverable-of-record; slot is verify-only): 4 writeup figs
   FINAL 7T×3s + RESULTS refreshes, post pollution-fix.
4. **⚑ analysis.py band-pollution fix** (5a699a5d4, RATIFIED
   72cca1bdf): positive-control + width-match rows excluded from
   canonical bands; per-arch G1 semantics; v1t render support.
5. **RM fills** (ba8a4ff3e): relu-mix T2/T4 now 3-seed, ~$3.
6. **Durability**: 44 ckpts receipted this venue (32 twins + 8
   sprint + 4 fills), LFS shas in /workspace/logs/ckpt_push.log.
7. **Manifest owner pass (my share)**: 96/96 rows as-launched,
   receipts external by design, append-only stance posted.

## Open / watch

- Hub ratification of §9 scoring + certificate v1.0 (posted).
- rmx_b checks 5-6/6 → certificate §1 extension appends (mine to
  fold when they land).
- 11:00 London verify-only render slot: re-render ONLY if any btk
  input changes (none can — rows frozen-complete).
- Ledger night total (mine): night ~$30 + shards ~$14-15 + fills
  ~$3 ≈ $47-48. In MODAL_SPEND.md.
- Monitors (this session): b6jr22n3d (origin), bbw1y8ufe (pf shard
  logs, drained), b56vdn9e9 (fill logs, drained) — the two
  drained-log monitors can be stopped or left to session end.

## Standing

date FIRST then stamp (two pre-write slips tonight, both corrected
at write — keep reading the clock). Union-resolve LOG conflicts
upstream-first + stray grep; "commit or stash" branch in pull
loops. --ours=upstream in rebase. Explicit-path git add (house
rule 660c50488). Tokens by path only. Aniket read-only. GPU 2
never. Liveness = /proc receipts, never GPU samples + log size
(house lesson 6e928e2bb). Claim-line before any pulled cell.
FLAG open: stage2_variance golden test fails pre-existing (panel
lane's, live-leaderboard-coupled).
