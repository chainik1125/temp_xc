# Working state — agent `mac-a`

**2026-07-26 ~18:40 London define-clock (local ≈ 70 min ahead) —
SALVAGE W1 COMPLETE, verdict pushed (2eb4927c9). Idle watch.**

## Salvage W1 delivered (all PENDING TEAM REVIEW)

- **Panel 72/72** at freeze `50af78f12` (card + k-resolution ratified
  pre-results, 56654864d). +72 leaderboard rows, 0 dups, pins
  verified, l0 all in band. ~1 h queue starvation behind the
  txc-neurips app cost $0; actuals ≈ $4 (ledger corrected, −$6).
- **VERDICT: NOT-KEEP as frozen** — S1 conjunction fails at exactly
  batchtopk_sae@T16 (mean +0.084 ≥ bar, t-CI [−0.027, +0.196]
  straddles 0, all 3 seed margins positive, n = 3 power). Everything
  else passes; **T32 confirmed CI-bounded on fresh seeds over both
  baselines** (+0.246/+0.248, CIs clear), S3 exact p = 0.0093,
  untrained flat, v2 +0.260.
- Secondary 8·T arm (non-claiming): FAILS untrained control at T32
  (0.74×) — sparse per-window code carries the separation;
  retro-validates the k-resolution.
- **R28 proposed** (receipts_check ALL PASS; direct-add, ratification
  = mac-local).
- **Two paths offered to the team (their call, not mine):**
  (a) ratify a T32-only re-scope (passes every frozen bar per-T;
  post-hoc narrowing), or (b) 3-seed top-up {6,7,8} at post/T16 +
  baselines for n = 6 power, est ≤ $3. fig4 was KEEP-gated — not
  produced; ready on request if either path is ratified.

## Open items on OTHERS

- mac-local/Han: R28 ratification; T32-re-scope vs top-up decision;
  neurips-queue priority ruling (moot for W1 — it completed).
- mac-b: W2 GAP-B rawgate numbers still queue-blocked (their lane).

## Assets / recovery

- Score: `diafaces/results/salvage_score.json`; panel
  `salvage_stage2_dial_real_ttrend_gpt2_l7.json`; payloads
  `salvage_payloads/` + Volume `…:/workspace/diafaces_salvage/`.
- Executor/scorer/merge: `diafaces/{run_salvage,score_salvage,
  merge_salvage_payload}.py`; driver `scripts/modal_diafaces_salvage.py`
  (pin 50af78f12). Note: leaderboard row k_pos/T live under
  `training_cfg.arch_hparams_override`.
- Modal client: scratchpad `modal-venv/bin/modal`. Spend: mac-a
  salvage ≈ $4 of $100; program ≈ $95 of $500.
