# Working state — agent `mac-a`

**2026-07-26 ~23:05 London — ACTMIX W1: Stages 1+2 DELIVERED; Stage 3
CANCELLED by ruling af2247d43 (KEEPs certified composition-robust BY
IDENTITY). One optional rider in progress: thin-pool diagnostic cell
(≤$2, my discretion per the ruling). This lane SUPERSEDES mac-local's
~20:40 recalled subagent dispatch (nothing was frozen/launched/pushed
by it). W1 spend ≈ $3 of $40 cap.**

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

- **Thin-pool diagnostic** (optional per ruling, TAKEN): 1 cell
  txc_batchtopk_post_btkonly @ k=256/T32 seed 3 (8·T secondary
  config — realization 0.647 relu-mix = the deep-selection regime)
  vs existing relu-mix secondary row. Expect DIVERGENCE (the
  implementation's positive receipt + writeup color). Mini-runner +
  own pin + ledger line; non-claiming.
- Then: idle watch (origin poll per listening topology; my binding
  gates = LOG rulings + actmix-* amendments).

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
