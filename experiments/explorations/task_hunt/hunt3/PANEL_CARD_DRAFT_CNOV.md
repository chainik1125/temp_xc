# CNOV_PANEL_CARD — FREEZE-READY DRAFT (PICK-PENDING; nothing runs before the pick + freeze)

**Status: launch-prep complete (ruling 1348a661a) — the 17:00 pick
chooses candidate A/B (one `DS` line in `run_cnov_panel.py`), then
card + runner + scorer freeze in ONE commit, driver pins from
origin-history rev-parse, and only then cells run. Screen basis:
KEEP 3/3 + panel-gate routing, ratified 36d3175ac.**

## 1. The task

`cnov` — kernel trailing rate (support 64 tok, HL 16) of
FIRST-IN-CONVERSATION token types on DailyDialog
(`labels/hunt3_dailydialog_<tok>.npz`, committed). Out-of-window
structural guarantee: pre-window occurrences are uncomputable from
any T-window. Honestly-stated weakening: the guarantee DECAYS with
T — mac-b's evidence line quantifies it (§ 3 S4).

## 2. The pick (A vs B), then constants

| | A `dial_real_cnov_gpt2_l7` | B `dial_real_cnov_gemma2_2b_l14` |
|---|---|---|
| screen order receipt (wd win−shuf @T32) | +0.026 (below bar) | **+0.039 (clears)** |
| screen KEEP legs | all pass | all pass |
| infra | warmest (ttrend panel twin; complete-fill buffer) | caches on Volume; buffer note verified in-container at freeze |
| d_in / d_sae ratio | 768 / 2048 (2.7× over) | 2304 / 2048 (0.89× under — DISCLOSED) |
| est cost | ~$3–4 | ~$4–6 |

Launch-prep recommendation: **B if the panel's purpose is the TXC
order case** (the only model whose order receipt clears the gate
bar); A if the purpose is cheapest confirmation on proven infra.
Both datasources are registered and validate. If B: mac-b re-runs
`panel_evidence_line_cnov.py` on the gemma labels pre-freeze
(minutes; same committed machinery) and § 3's S4 numbers update.

Constants (both candidates): d_sae 2048, k_pos 8, n_steps 8000,
buffer 524288, eval_L 32, V2 eval_extra verbatim (hard assert),
batch 1024//T. Arms: `txc_batchtopk_post` @ T ∈ {8,16,32} +
`batchtopk_sae` + `tsae` @ T1; seeds {9,10,11} (fresh); trained +
untrained; **30 cells** (15/15), enumeration hard-asserted.

## 3. Bars (S1–S5; scorer `score_cnov_panel.py` staged)

- **Claiming zone (RULED, f9319e59a): T = 16 ONLY.** T32 runs
  RUN-BUT-NOT-CLAIM (floor 0.402 = dq territory per the ruling; its
  S-rows are reported beside, never gating).
- **S1** (claiming T16, BOTH baselines): paired-by-seed margin mean
  ≥ +0.05 AND t 95% CI LB > 0 (n=3, df=2, t=4.302653).
- **S2**: untrained ≤ 0.5× trained at each claiming T.
- **S3**: T8→16→32 trend, exact 216-perm, reported not gating.
- **S4 (KILL, mac-b's instrument
  `results/panel_evidence_line_cnov.json`)**: trained post mean must
  EXCEED the visible-cue evidence line at its T — gpt2 labels:
  **r 0.2692 @T16, 0.4017 @T32** (floor tracks in-window kernel mass
  53%/80%). mac-b's flag adopted verbatim: the T32 bar is ~3.5×
  ttrend's; the ruling makes this binding: **T16 claims, T32
  reports**. (T64 excluded entirely — floor r 0.63.)
- **S5**: grouped v2 > 0 at claiming Ts.
- **KEEP iff S1(both) ∧ S2 ∧ S4 ∧ S5 at T16** (the sole
  claiming T; T32 rows land beside as run-but-not-claim).

## 4. l0 bands + pre-disclosures

Post arms ONLY: realized l0_per_window ∈ [4.5, 9.5] (salvage
convention; out-of-band ⇒ disclosed, cell non-claiming). Baselines
are QUOTED never band-gated (the R29 lesson), with the R30
pre-disclosure: expect sae ≈ 4.1–4.7/8 and tsae ≈ 7.0/8 realized —
eval-threshold pruning, an arch property, not a defect. Pre-named
traps from the screen (position inverted-0.86, doc-mean 0.86)
carried; the panel's untrained control + fresh seeds are the
instruments here (the screen's wd arms already showed the
within-dialogue gain survives).

## 5. Ops

Commit-then-run; pin from ORIGIN-history rev-parse post-push;
`_assert_pinned`; detached; containers never push; payloads to
Volume dir `/workspace/hunt3_cnov_panel` in `finally`;
repatriate-merge locally with pin/dup/V2 asserts (merge script to be
committed with the freeze); ledger before/after; H100 main (24
cells) + 3× L4 trained-tsae. Est § 2 per candidate; envelope: hunt
budget (released Stage-3 funds), cap headroom ≈ $45 of $60.
