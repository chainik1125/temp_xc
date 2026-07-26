# Working state — agent `mac-b`

**2026-07-26 ~19:05 London (SALVAGE sprint).** **SALVAGE W2 COMPLETE**
— `txcwin/CROSSRATIFY.md` FINAL, all artifacts committed, everything
PENDING TEAM REVIEW **and pending Andrii's review**. mac-b salvage
actuals ≈ $3 of $60; program ≈ $103 of $500. Holding: respond to
review pings; report-state milestone ~22:00 London met early.

## Salvage W2 record (all pushed)

- **Verdicts (`txcwin/CROSSRATIFY.md`):** gpt2 r1/c1/c2/c3/c4 all
  SUPPORTED (c1–c3 at 11.3–21.9σ, strict worst-vs-best-seed); 8B
  c1/c2 SUPPORTED-WITH-GAPS (2.6/2.7σ), **c3@T8 NOT-REPRODUCED**
  (their own W3/W8; one collapsed seed) while T=16 is strict 12.4σ →
  claims re-pin (name model+T, 8B at T=16) or ~$5 seed top-up
  proposed to Andrii. Gaps G-1..G-6 (incl. G-6: report.py embedded
  audit pools gpt2+8B into 6-seed pseudo-cells — masks the c3
  contradiction). Receipts R-X1..X4 proposed — enter receipts_check
  ONLY after mac-local ratification + Andrii ack.
- **GAP-A (visible-cue, $0):** T=8 window-computable floor V-win
  +0.054/+0.097; both band readings per ruling 56654864d (gpt2 band 3;
  8B band 2 via oracle-position V-all); 8B quoting guard 46e0021a7
  (floor-vs-best-dict ~4×, NOT floor-vs-per-token). V-pos instrument
  caveat: nov_resid keeps position residual r≈+0.21/+0.17.
- **GAP-B (raw gate, ~$3):** ALL cells CANDIDATE. Claims' T=8:
  gpt2 gap_mean +0.101; 8B +0.071 (window gap +0.320; raw_last
  +0.216 vs win +0.537 — biggest asymmetry in-thread). 8B T=16 lean
  cell (BLAS int32 overflow; one-sided gate per c797c5207) passes
  via gap_mean.
- **Freeze chain:** card fedf75aa9 → V-win e844cce52 (ruling-ordered)
  → lean-arms ea7a50ea1 (ratified). Driver pins ONLY post-push SHAs
  (pre-push local SHA got rebase-rewritten once — lesson in LOG).
- Caches persisted on Volume `temp-xc-replag-caches`:
  txcwin_caches/{gpt2_L6, 8B_L12} + txcwin_crossratify_results/.

## Standing state

- Day-2 record all ratified (R25/R27, fig3, negatives table); day-2
  actuals ≈ $2.
- `uvx modal …` (plain `modal` not on PATH). `source
  scripts/set_agent_env.sh mac-b` each shell.
- Tokens rotate after the weekend (Han). Containers never push.
  Andrii = human collaborator on this branch; never modify their
  txcwin files (crossratify/ + CROSSRATIFY.md are mac-b's additions).
- Modal: NOTHING in flight. Post-deadline queue (mac-local gate):
  gemma overnight-card fills from Volume partials (~$4).
