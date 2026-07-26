# Working state — agent `mac-a`

**2026-07-26 ~23:50 London — ACTMIX W1 LANE CLOSED from my side: all
deliverables landed and pushed. Stage 1 shipped (btk-only canonical);
Stage 2 IDENTITY verdict + R30 (ALL PASS; preliminary ratified
af2247d43, final rode the 20/20 push); Stage 3 cancelled by ruling
(KEEPs certified composition-robust BY IDENTITY); thin-pool DIAG
landed with DIVERGENCE as pre-registered (l0 0.696→1.007, rec
0.2471→0.1805 — identity is substrate, not implementation). mac-a
ACTMIX spend ≈ $4 of $40. Idle watch — poll origin per listening
topology (my binding gates: LOG rulings + actmix-* amendments); act
only on things addressed to mac-a. This lane SUPERSEDED mac-local's
~20:40 recalled subagent dispatch (nothing was frozen/launched/pushed
by it).**

## Delivered (ACTMIX W1)

1. **Stage 1** (92db86c41): btk-only variants — 5 registry entries
   (`*_btkonly`, 1.1.0 / 2.1.0-port), plugin file
   `src/temp_bench/archs/btk_only.py`, CANONICAL convention LOG note
   (~21:05; single-source — pods consumed it), threshold_set flag,
   neg_frac diagnostic, tests/test_btk_only.py, suite 369 green.
2. **Stage 2 CALIB** (card freeze `97fae183a`, approved 269b7d86c;
   verdict FINAL ~23:00 entry): **IDENTITY — btk-only ≡ relu-mix at
   hunt widths (d2048, k8).** 20/20 cells; max |Δrec| 0.0000@4dp (raw
   ≤ 2.2e-08); l0 EXACTLY equal ×20; slopes +0.0701 both arms; tsae
   twin predictions landed digit-exact. Mechanism re-attribution:
   realized-l0 shortfall = eval JumpReLU threshold pruning, NOT
   selection zero-picks; neg_frac ≡ 0 proven by identity (advisory
   answered). **R30 direct-added, ALL PASS.** Preliminary RATIFIED
   (af2247d43); final ratification rides the 20/20 push. Deliverables:
   calib_score.json + figs/calib_relu_vs_btk.{png,pdf} + R30.
   Leaderboard +20 rows @97fae183a, 0 dups.
3. **Stage 3**: CANCELLED as designed by af2247d43 § 2 — hunt KEEPs
   (R22/R27/R28/R29/tt-P1) composition-robust BY IDENTITY; ~$30
   released; mac-b appends the forensics corrigendum (their lane).

## In flight / next

- Nothing in flight. DIAG landed (freeze 63ac1b208, eval_key
  3a6f0bbe0f9a0b07, +1 canonical row): rec 0.1805 / l0 257.8 vs twin
  0.2471 / 178.16 — divergence as pre-registered; close-out LOG note
  ~23:45. Idle watch only.

## Assets / recovery

- Calib: CALIB_CARD.md, run/score/merge/make_fig scripts,
  results/calib_{score,stage2_*}.json, calib_payloads/, Volume
  `…:/workspace/diafaces_calib/`, figs/calib_relu_vs_btk.*.
- btk-only convention: arch file docstring + LOG ~21:05 note +
  configs/archs.yaml ACTMIX block. Evaluator l0 is sign-agnostic
  (`z != 0`) — verified.
- Identity condition (for anyone extending): btk-only ≡ relu-mix iff
  train selection never exhausts positives AND tracked threshold ≥ 0;
  divergence regime on this substrate = 8·T arm at T32 (k256/d2048).
- Modal client: scratchpad `modal-venv/bin/modal`. Ledger last line
  ~$105 program. Local modal runner may still be attached
  (harness task b5j39gav2) — payloads already repatriated manually;
  safe to TaskStop it.
