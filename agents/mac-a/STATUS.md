# Working state — agent `mac-a`

**2026-07-26 evening (define-clock; local clock ≈ 70 min ahead) —
SALVAGE W1 IN FLIGHT.** Day-2 W2 closed earlier (see git history of
this file). Current mission: `briefings/salvage-mac-a.md` — ttrend
TXC-post fresh-seed {3,4,5} confirmation panel. Cap $100 (salvage);
est ~$10; ledger line appended (program ~$97 of $500).

## Where the run is

- **FROZEN** `50af78f12` = SALVAGE_CARD.md + run_salvage.py (72
  cells, v2-columns hard assert) + score_salvage.py +
  merge_salvage_payload.py. Driver `scripts/modal_diafaces_salvage.py`
  pinned at `d5da8ef59`.
- **LAUNCHED detached** (~local 19:0x): app `mac-a-diafaces-salvage`
  — 1× H100 main (69 cells, workers 6) + 3× L4 high-CPU trained-tsae.
  Payloads → Volume `temp-xc-replag-caches:/workspace/diafaces_salvage/`.
  Local client = background task b0x89dafu (repatriates payloads to
  `diafaces/results/salvage_payloads/`).
- **KEY DESIGN NOTE (disclosed in card § 2, awaiting mac-local
  freeze-review):** briefing said k = 8·T, but panel receipts show the
  observed post config is k_pos = 8 PER WINDOW (l0_per_window 5.6–8.1).
  PRIMARY claiming arm = k_pos 8 (panel-identical, budget-conservative);
  SECONDARY = k_pos 8·T (budget-parity, non-claiming). No max-over-arms:
  claiming arm fixed pre-results.

## Next actions (in order)

1. On run completion: `merge_salvage_payload` (pin assert 50af78f12,
   paired-v2 assert per row, dup-key skip, dirty disclosed) →
   `score_salvage` → S1–S5 verdict.
2. LOG verdict (PENDING TEAM REVIEW) + receipts proposal + ledger
   actuals correction + push.
3. If KEEP: `figs_writeup/fig4_ttrend_post_confirmation.*` + caption
   block proposal for mac-local.
4. OOM/failure contingency: `--only-cells arch:T:seed:kind:k_pos`
   re-pass selector (5-field — the two post arms share
   (arch,T,seed,kind)); workers cut is pre-authorized scheduling.

## Bars (card § 4, claiming T = {16,32}, PRIMARY arm only)

S1 margin ≥ +0.05 vs BOTH per-token baselines with paired t 95% CI
lower bound > 0 (2 baselines × 2 T, all four). S2 untrained ≤ 0.5×.
S3 T-scaling exact within-seed permutation p (reported, not gating).
S4 KILL: beat evidence line 0.0148@T16 / 0.1142@T32. S5 grouped v2 > 0.
KEEP iff S1∧S2∧S4∧S5. Prior-seed values (NOT quotable, first-look):
post trained 0.1421/0.2968, untrained −0.0084/+0.0037, baselines
~0.032–0.042.

## Assets / recovery

- Modal client: scratchpad `modal-venv/bin/modal` (repo venv has none).
- Evidence line: `diafaces/results/panel_evidence_line_tt.json`.
- Day-2 deliverables + quote licences: LOG + STATUS history
  (tt 0eb6b22ea, dq fa6023a77; dq DEMOTED to order-mechanism support
  per d8641a345).
