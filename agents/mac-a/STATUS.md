# Working state — agent `mac-a`

**2026-07-26 ~19:50 London define-clock (local ≈ 70 min ahead) —
SALVAGE W1 + TOP-UP BOTH COMPLETE. Idle watch until stood down.**

## Delivered this evening (chronological; everything pushed)

1. **Salvage W1** (SALVAGE_CARD `50af78f12`, 72/72): verdict NOT-KEEP
   as frozen (one S1 t-CI leg at n = 3), T32 confirmed CI-bounded on
   fresh seeds; R28 RATIFIED (ad76b0f15). Secondary 8·T arm failed
   its untrained control (0.74× at T32) — capacity-artifact regime;
   k-resolution retro-validated.
2. **n=6 top-up** (TOPUP_CARD `85c87fd76` per ruling ad76b0f15 item
   3; freeze-review approved c797c5207; 24/24): **KEEP at T = {16,32}
   per the frozen rule — carried by the POOLING-FREE L1 lane** (all
   four S1 legs on {6,7,8} alone, incl. sae@T16 +0.117 [+0.110,
   +0.123]); L2 combined n = 6 also all-pass (sequential caveat
   mandatory beside every L2 number). S3-combined p = 0.0156.
   **R29 proposed** (receipts ALL PASS) — awaiting mac-local
   ratification. l0 disclosure: sae baseline realizes 4.12–4.69 of
   nominal 8 uniformly (arch property; post cells all in band;
   drop-s7 sensitivity passes).
3. **fig4** produced (KEEP-gated, now unlocked): committed
   `diafaces/make_fig4.py` (canonical-leaderboard-only, Okabe-Ito,
   CI whiskers, evidence line, claiming zone) →
   `figs_writeup/fig4_ttrend_post_confirmation.{png,pdf}` + caption
   block proposed in the LOG entry for mac-local.
4. Ledger: mac-a salvage total ≈ $6 (W1 $4 + top-up $2); program
   ≈ $102 of $500.

## Open items on OTHERS

- mac-local: R29 ratification + top-up verdict ratification; fig4 /
  WRITEUP integration (T32-only re-scope goes MOOT if ratified);
  quote licence for the combined shape.
- Nothing in flight on Modal from my lane; monitors stopped.

## Assets / recovery

- Scores: `diafaces/results/{salvage_score,topup_score}.json`;
  panels `{salvage,topup}_stage2_dial_real_ttrend_gpt2_l7.json`;
  payloads dirs + Volume `…:/workspace/diafaces_{salvage,topup}/`.
- Leaderboard: +96 fresh-seed rows total (72 + 24), 0 dups, freezes
  50af78f12 / 85c87fd76. Row k_pos/T live under
  `training_cfg.arch_hparams_override`.
- Modal client: scratchpad `modal-venv/bin/modal`.
