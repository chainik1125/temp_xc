# Working state — agent `mac-a`

**2026-07-26 ~21:10 London — ACTMIX W1 (briefings/actmix-mac-a.md,
read-first actmix-shared.md). Stage 1 (btk-only implementation)
DONE + convention note posted to task_hunt LOG (single-source rule —
pods consume it verbatim). Now: Stage 2 CALIB card (decisive input to
Dmitry's paper-re-run gate, wanted before 9am PT 2026-07-27). Cap $40
W1 total / $150 day. NOTE: this lane SUPERSEDES mac-local's ~20:40
subagent dispatch — that dispatch was recalled cleanly at Han's
direction; nothing was frozen, launched, or pushed by the recalled
agents. mac-local freeze-reviews my cards in parallel — pull before
scoring.**

## ACTMIX W1 progress

1. **Stage 1 DONE** — `src/temp_bench/archs/btk_only.py` + 5 registry
   entries (`{batchtopk_sae,tsae,stacked_batchtopk,txc_batchtopk_pre,
   txc_batchtopk_post}_btkonly`, arch_versions 1.1.0 / 2.1.0-port);
   relu_mode hparam hashed into train_key; threshold_set flag replaces
   the -1.0 sentinel (legit negative thresholds); selection by signed
   value over raw pre-acts; AuxK unchanged (ReLU'd); fired ⇔ z != 0;
   neg_frac diagnostic. Tests: tests/test_btk_only.py, full suite 369
   passed, validate OK. Convention note = LOG entry "~21:05 London"
   (CANONICAL; pods follow, never fork).
2. **Stage 2 IN FLIGHT** — CALIB_CARD frozen `97fae183a` (pin re-taken
   from ORIGIN history after a pull-rebase rewrite — watch for this),
   **APPROVED expedited by mac-local 269b7d86c** with one post-run
   ADVISORY: surface per-cell `neg_frac` in the verdict. Launched
   detached ~21:30 BST, app `ap-NANQj1zSfcIiBELX4ydG9w`
   (mac-a-diafaces-calib): H100 main 18 cells + 2× L4 tsae; relu-mix
   arm = 20 REUSED rows (card § 3 eval_keys); est ~$3 ledgered.
   **RESULT (19/20 in, tsae_s4 in flight): IDENTITY — btk-only ≡
   relu-mix at hunt widths.** 18 main cells: |Δrec| ≤ 2.2e-08, Δl0 =
   0.0 EXACTLY; tsae s3 identical to all printed digits
   (0.022458/7.0037). Preliminary LOG entry PUSHED (00309362f) with
   pods heads-up (their k20/d18432 arms will likely no-op too — smoke
   + neg_frac first). Mechanism re-attribution: l0 shortfall = eval
   JumpReLU threshold pruning, NOT selection zero-picks; neg_frac ≡ 0
   proven BY the identity (advisory answered exactly). On s4 land
   (watch buqn611s8): repatriate all 3 payloads from Volume (local
   .get() prints not flowing — Volume finally-writes are the designed
   fallback) → merge (pin-asserted) → score → fig → FINAL verdict
   (draft at scratchpad verdict_draft.md: E1/E3 FAIL-by-no-movement,
   E4 vacuous-pass, l0 band out-of-band disclosed as designed) +
   **Stage-3 RE-SCOPE PROPOSAL** (~$5–8 identity-confirm probes +
   thin-pool divergence demo instead of $23–34 blind re-runs; needs
   mac-local ruling) → ledger actuals (≈$3).
3. **Stage 3 GATED on Stage 2** — KEEP-set survives-the-fix
   (R29 ttrend lane / R22 λ̂ cells / R27 dq cells vs SAME bars;
   verdicts SURVIVES / MOVED-MARGINS / DOES-NOT-SURVIVE).

## Salvage sprint (CLOSED df8043d6d — prior phase, all ratified)

W1 NOT-KEEP as frozen + R28; top-up KEEP {16,32} + R29; fig4 in
WRITEUP §4; mac-a salvage actuals ≈ $6. Quote licences per mac-local.

## Assets / recovery

- btk-only convention: `src/temp_bench/archs/btk_only.py` docstring +
  LOG note (~21:05). Registry: configs/archs.yaml ACTMIX block.
- Salvage-phase assets: diafaces/results/{salvage,topup}_score.json,
  panels + payloads, Volume /workspace/diafaces_{salvage,topup}.
  Leaderboard k_pos/T under `training_cfg.arch_hparams_override`.
- Modal client: scratchpad `modal-venv/bin/modal`. Ledger
  briefings/MODAL_SPEND.md (program ≈ $102 of $500 at last line).
