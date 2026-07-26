# CALIB_CARD — ACTMIX Stage 2: relu-mix vs btk-only calibration mini-grid (ttrend)

**Pre-registration. Frozen BEFORE any btk-only cell runs; the freeze
commit SHA is pinned in `scripts/modal_diafaces_calib.py` and asserted
in-container.** Briefing: `briefings/actmix-mac-a.md` Stage 2 (read-first
`briefings/actmix-shared.md`). Arch convention: the Stage-1 LOG note
("~21:05 London", CANONICAL) + `src/temp_bench/archs/btk_only.py`.
Owner mac-a. mac-local freeze-reviews in parallel (hub watcher). ALL
readings PENDING TEAM REVIEW.

## § 1 Purpose and epistemic status (NON-CLAIMING)

This card is **descriptive calibration, not a claiming panel**: no
S-bars, no KEEP/KILL. A different sparsity composition is a different
pre-registration — nothing measured here mints or upgrades a hunt claim
(that is Stage 3's job, own card). What it IS: the decisive measured
input to **Dmitry's paper-re-run gate** (does the composition fix move
recovery-vs-T on the warmest hunt substrate, and in which direction),
and the sizing prior for the pods' Phase-B and my Stage-3 re-runs.

**Arm labels (mandatory everywhere):** `relu-mix` = ReLU→BatchTopK (the
unsuffixed archs; current v2 backbone). `btk-only` = BatchTopK on raw
pre-acts, no ReLU in the sparsity path (`*_btkonly` archs, Stage-1
convention). No `paper-match` arm here (that is mac-c's audit + pods).

## § 2 Design (40 logical cells; 20 computed, 20 reused)

Substrate `dial_real_ttrend_gpt2_l7` (warmest infra; 19-min turnaround
proven on the salvage panel). Constants IDENTICAL to the salvage/topup
panels — that identity is what makes the reuse legal:

- d_sae = 2048, k_pos = 8 (primary panel-identical arm ONLY; no 8·T
  secondary here — it failed its untrained control on the salvage panel
  and is out of scope), n_steps = 8000 trained / 0 untrained,
  buffer_tokens = 524288, eval_window_L = 32, token-shuffle buffer
  default, batch = 1024//T.
- V2 eval_extra on EVERY cell, verbatim PROBE_V2_SPEC § 2 (hard pre-run
  assert): `{"lambda_probe_v2": True, "lambda_v2_probe": "ridge",
  "lambda_v2_alphas": logspace(-2,4,13), "lambda_v2_n_windows": 8192,
  "lambda_v2_split": "trace"}`.
- Grid: {batchtopk_sae, tsae} @ T=1 and txc_batchtopk_post @
  T ∈ {4,16,32}, each × {relu-mix, btk-only} × seeds {3,4} ×
  {trained, untrained}. Untrained kept at BOTH seeds (my executor call:
  they are ~eval-cost-only and anchor the untrained-normalized
  realized-l0 fingerprint, mac-b's ACTMIX_FORENSICS convention).
- Metrics read per cell: `lambda_recovery` (v1 CANONICAL),
  `lambda_recovery_v2` (paired report), `l0_per_window`,
  `l0_per_token`, `nmse`.

## § 3 relu-mix arm = REUSED rows (zero compute; dup-key discipline)

The relu-mix side is **cited, not re-run**: the salvage W1 panel
(freeze `50af78f121d4c4cbe5024c93aeaa5a4753daed11`) already holds every
relu-mix cell of this grid at seeds {3,4} under the exact constants of
§ 2. The 20 reused rows, by canonical-leaderboard `eval_key`:

| arch | T | seed | kind | eval_key |
|---|---|---|---|---|
| batchtopk_sae | 1 | 3 | trained | `3e7472feb278e922` |
| batchtopk_sae | 1 | 3 | untrained | `c8af7733b91e88f1` |
| batchtopk_sae | 1 | 4 | trained | `ea801af31aa09eb9` |
| batchtopk_sae | 1 | 4 | untrained | `c799183552489fa5` |
| tsae | 1 | 3 | trained | `c6441f5d9a65180d` |
| tsae | 1 | 3 | untrained | `bfe04bdb3695d6f5` |
| tsae | 1 | 4 | trained | `d02c894a8c76a7e5` |
| tsae | 1 | 4 | untrained | `4cda4f0078fed728` |
| txc_batchtopk_post | 4 | 3 | trained | `f05faa4f38cd9966` |
| txc_batchtopk_post | 4 | 3 | untrained | `063d8160ff0cef41` |
| txc_batchtopk_post | 4 | 4 | trained | `2100877acb00c139` |
| txc_batchtopk_post | 4 | 4 | untrained | `f07cf092c7506ed4` |
| txc_batchtopk_post | 16 | 3 | trained | `f8ef0d74a9056bee` |
| txc_batchtopk_post | 16 | 3 | untrained | `4a8706f47a85025f` |
| txc_batchtopk_post | 16 | 4 | trained | `2f0a19c6b6701d81` |
| txc_batchtopk_post | 16 | 4 | untrained | `b45636613df76f37` |
| txc_batchtopk_post | 32 | 3 | trained | `a79ee7cbf6c36012` |
| txc_batchtopk_post | 32 | 3 | untrained | `c63bed8ea1226f5d` |
| txc_batchtopk_post | 32 | 4 | trained | `e03386fbd4efdf15` |
| txc_batchtopk_post | 32 | 4 | untrained | `061c4465a13e2181` |

All 20 carry paired v2 columns. `score_calib.py` selects the relu-mix
arm by THESE eval_keys only (no re-query drift); if any key is missing
from the canonical leaderboard the scorer hard-fails. No relu-mix cell
is re-run (dup-key discipline; the btk-only arm cannot collide — its
arch names and train_keys differ by construction).

## § 4 btk-only arm = the 20 computed cells + realized-l0 bands

20 cells: {batchtopk_sae_btkonly, tsae_btkonly} @ T=1 and
txc_batchtopk_post_btkonly @ T ∈ {4,16,32}, seeds {3,4}, trained +
untrained, k_pos = 8, constants § 2. Enumeration hard-asserted in the
frozen executor `run_calib.py` (count == 20, split (10 trained, 10
untrained), all k_pos == 8, all seeds ∈ {3,4}, per-cell V2 — the day-2
defect lesson).

**Pre-registered realized-l0 bands (the mixing fingerprint — doubly
load-bearing per the shared briefing):**

- btk-only cells (all archs, trained AND untrained): realized
  l0/selection-row ∈ **[6.5, 9.6]** of nominal 8 (post: `l0_per_window`;
  token archs: `l0_per_token` ≡ per-window at T=1). Stage-1 convention
  predicts ≈ 8.0 exactly at train-time selection; eval-time JumpReLU
  threshold adds jitter both directions. Cells outside the band are
  DISCLOSED and flagged in the score JSON; being descriptive, the cell
  still reports (nothing here claims), but an out-of-band btk-only cell
  weakens the "fix removes the l0 shortfall" reading and must be said
  so in the verdict.
- relu-mix cited cells: their observed l0 is already on record
  (sae ~4.1–4.7/8, tsae ~6.7/8, post 5.6–8.0/window rising with T) —
  quoted as-is; that shortfall IS the pathology being calibrated.

## § 5 Pre-registered directional expectations (restated from
`briefings/actmix-shared.md` BEFORE any btk-only cell; the verdict MUST
quote these four and say which held)

Let Δ(cell) = recovery(btk-only) − recovery(relu-mix), paired per seed.

- **E1 (sae improves most):** mean Δ_trained(batchtopk_sae@T1) is the
  largest of {sae, tsae, post@4, post@16, post@32}.
- **E2 (tsae moves least):** |mean Δ_trained(tsae@T1)| is the smallest
  of that set (tsae already realizes 6.7/8 — our licensed lead
  comparator moves least).
- **E3 (post low-T recovers):** mean Δ_trained(post@T4) >
  mean Δ_trained(post@T32) (harm was worst at small pools).
- **E4 (slope may soften):** slope_btk ≤ slope_relu, where slope = OLS
  of trained recovery on log2 T over T ∈ {4,16,32} per seed, then
  seed-mean. Reported as a direction check, NOT gated (n = 2 seeds —
  descriptive; no CI claims at n = 2).

Untrained sanity (reported): untrained Δ expected ≈ 0 (composition
change alone shouldn't create recovery in an untrained dict); a large
untrained Δ flags a selection-artifact regime and is said loudly.

## § 6 Outputs

`score_calib.py` (frozen) writes
`diafaces/results/calib_score.json`: per-cell table (both arms, both
recoveries, realized l0, band flags), paired per-seed Δ per (arch, T,
kind), seed-mean Δ, post slopes per arm + Δslope, E1–E4 booleans with
numbers, untrained sanity. Then (post-run ops, not frozen): one figure
(relu-mix vs btk-only per arm, canonical-leaderboard-sourced, fig1–4
conventions) + LOG verdict PTR quoting E1–E4 + ledger actuals. Nothing
quotable before mac-local review.

## § 7 Ops (all standing discipline)

Commit-then-run: this card + `run_calib.py` + `score_calib.py` +
`merge_calib_payload.py` frozen in ONE commit; driver pins that SHA
from `git rev-parse` (never hand-typed) + `_assert_pinned()`
in-container. Modal detached (`--detach`), H100 main block (18 cells,
queue-permitting; else L40S) + 2× high-CPU L4 for trained tsae_btkonly
(tsae-first scheduling). Containers NEVER push; payloads persist to
Volume `temp-xc-replag-caches:/workspace/diafaces_calib` in `finally`;
local repatriate → `merge_calib_payload.py` (pin-asserted rows, dup-key
skip, dirty disclosed as pool convention). Ledger
`briefings/MODAL_SPEND.md` read-before/append-after; **est ~$2–4, stage
cap ≤$8** (W1 cap $40, day cap $150). London define-clock timestamps.
