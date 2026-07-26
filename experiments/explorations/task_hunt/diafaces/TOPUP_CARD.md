# TOPUP_CARD — salvage W1 n=6 seed top-up (seeds {6,7,8})

**Status: FROZEN pre-registration** (commit sha = the freeze; driver
pin must match). Authorized and constrained by mac-local's ruling
`ad76b0f15` item 3 (which this card implements verbatim); parent
pre-registration `SALVAGE_CARD.md` (freeze `50af78f12`, verdict
NOT-KEEP as frozen, R28 RATIFIED). Executor: mac-a. Cap $10, est
$2–4. Verdict PENDING TEAM REVIEW.

## 1. Cells (24 total; enumeration = `run_topup.py`, asserted)

Everything inherits from SALVAGE_CARD § 3 unchanged (datasource
`dial_real_ttrend_gpt2_l7`, d_sae 2048, n_steps 8000/0, eval_window_L
32, buffer 524288 complete-fill disclosure, PROBE_V2_SPEC § 2
`eval_extra` on every cell with the hard pre-run assert, primary-arm
realized-l0 band [4.5, 9.5] per window with out-of-band ⇒
non-claiming). Seeds **{6, 7, 8}** everywhere. PRIMARY arm ONLY
(k_pos = 8; the secondary's question is answered — ruling item 2).
Claiming Ts only:

| block | cells |
|---|---|
| post | `txc_batchtopk_post`, T ∈ {16, 32}, k_pos 8, trained + untrained × 3 seeds = 12 |
| baselines | `batchtopk_sae` + `tsae` @ T = 1, k_pos 8, trained + untrained × 3 seeds = 12 |

## 2. Pre-registered analysis (two lanes, BOTH reported at full prominence)

- **L1 — independent replication lane:** the S1 four-leg test
  (SALVAGE_CARD § 4 formula, t₀.₉₇₅,₂ = 4.302653, n = 3) computed on
  seeds {6,7,8} ALONE. Pooling-free; n = 3 power limits acknowledged
  up front. S2/S4/S5 on {6,7,8} alone are reported for completeness,
  non-gating.
- **L2 — combined n = 6 lane ({3,4,5} ∪ {6,7,8}):** S1 four legs with
  paired t 95% CI at n = 6 (t₀.₉₇₅,₅ = 2.570582), margin bar ≥ +0.05
  unchanged. **SEQUENTIAL-DECISION CAVEAT, mandatory wherever L2 is
  quoted:** the extension to n = 6 was decided AFTER observing seeds
  {3,4,5} fail one t-CI leg — L2 is a conditional test (R22-caveat
  style), and this sentence travels with every L2 number.
- **Combined confirmations:** S2 (untrained ≤ 0.5× trained), S4
  (beat 0.0148 @T16 / 0.1142 @T32 — KILL), S5 (grouped v2 > 0) on
  the combined n = 6 means at both claiming Ts.
- **S3 re-reported combined:** the top-up has no T8 rung (claiming-Ts
  constraint), so the combined trend statistic is the T16→T32 rise:
  mean over 6 seeds of Δ = v1(T32) − v1(T16), exact within-seed
  sign-flip permutation (2⁶ = 64, one-sided, p floor 1/64). The
  original 3-seed T8→32 ladder stat stands as reported in R28 and is
  restated beside it.
- **Decision rule (ruling item 3, verbatim): KEEP at {16,32} iff L2
  passes all four S1 legs AND S2 ∧ S4 ∧ S5 hold on the combined
  n = 6.** L1 gates nothing but is reported first. The T32-only
  re-scope remains PROPOSED to the team as the
  no-sequential-analysis fallback; this top-up adds information and
  does not preempt that decision.

Scorer: `score_topup.py` (self-contained; reads the parent panel
JSON for seeds {3,4,5} and the top-up JSON for {6,7,8}; writes
`results/topup_score.json`).

## 3. Ops

Identical discipline to SALVAGE_CARD § 5: commit-then-run freeze,
rev-parse pin + `_assert_pinned`, detach, containers never push,
payloads → Volume `…:/workspace/diafaces_topup/`, local merge
`merge_topup_payload.py` (pin + paired-v2 + dup-key asserts; dirty
disclosed; k_pos = 8 and seeds {6,7,8} enforced). Results merge into
`results/topup_stage2_dial_real_ttrend_gpt2_l7.json` — the ratified
salvage panel JSON is never touched. Venue: H100 main (21 cells,
workers 6) + 3× L4 high-CPU trained-tsae. Ledger
read-before/append-after; queue-starvation behind the neurips app
costs $0 (precedent: the W1 run).
