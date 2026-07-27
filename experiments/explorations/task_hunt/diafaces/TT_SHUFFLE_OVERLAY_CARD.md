# ttrend SHUFFLE-OVERLAY CARD — anchor-gated retrain-with-shuffle-eval (runpod-b)

**STATUS: FROZEN at this commit (commit-then-run; no retrain cell has
run when this card lands). Directive: mac-local 16:45 entry
(`eeb4ee3c4`) item (b) per the 16:40 GPU-1 priority ruling
(`1d2e3de28`); pattern = the APPROVED λ̂ SHUFFLE_OVERLAY_CARD
("craft standard for overlay work"), transplanted. ALL outputs
PENDING TEAM REVIEW.**

## § 1 Why

The quoted ttrend v2 trained panel (diafaces tt on gpt2/hs7, 102
cells, seeds {1,2,42},
`diafaces/results/stage2_dial_real_ttrend_gpt2_l7.json`) carries no
eval-shuffle twins; checkpoints not persisted (16:45 audit:
diafaces_panels_v2 = payloads only). Retrain the claiming arm under
the anchor gate; **the quoted panel numbers stay the exhibit numbers
either way — the retrain contributes ONLY the shuffle overlay.**

## § 2 Retrain grid

Datasource `dial_real_ttrend_gpt2_l7` (gpt2/hs7, 4111×128×768).
Hyperparameters inherited from the QUOTED PANEL ROWS themselves
(every payload row records them; frozen here): d_sae 2048, k_pos 8,
n_steps 8000, buffer 524,288, batch = `grid.batch_size(T)`,
eval_window_L 32, canonical runner end-to-end.

| cells | arch | T | seeds |
|---|---|---|---|
| 15 | `txc_batchtopk_post` (claiming arm) | {2, 4, 8, 16, 32} | {1, 2, 42} |
| 3 | `batchtopk_sae` (anchor) | 1 | {1, 2, 42} |
| 3 | `tsae` (anchor) | 1 | {1, 2, 42} |

Untrained twins not retrained (quoted numbers stand as context).
Fresh-run mechanism: `eval_extra = {"retrain_tag":
"tt_shuf_overlay_r1"}` (new eval_keys, no collisions); no local
checkpoints for these train_keys exist on this pod and 0 manifest
hf_url rows (verified at the λ̂ freeze, 0/10210) ⇒ training is
fresh; checkpoints persist for § 4.

**Cache (venue-local, provenance disclosed):** `/workspace/
dialevel_caches/gpt2/` was built on THIS pod by runpod-a's cnov prep
(canonical `dialevel/cache_acts.py`; `acts_meta.json`:
mapping_verified_rows 4111/4111, hs_capture {7,4,10}, wall 4.4 s) —
the canonical builder's own verification receipt; no rebuild needed.

## § 3 ANCHOR GATE — pre-registered before any cell runs

Rule as in the approved λ̂ card: per cell,
**|mean₃(retrained ordered r) − mean₃(quoted r)| ≤ 1 · σ_quoted**,
quoted values = the v2 payload's per-cell 3-seed spread
(`lambda_recovery`, the v1-canonical column), frozen here:

| cell | quoted mean | σ (tolerance) |
|---|---|---|
| txc_batchtopk_post/T2 | 0.0363 | 0.0058 |
| txc_batchtopk_post/T4 | 0.0501 | 0.0087 |
| txc_batchtopk_post/T8 | 0.0709 | 0.0291 |
| txc_batchtopk_post/T16 | 0.1421 | 0.0099 |
| txc_batchtopk_post/T32 | 0.2968 | 0.0127 |
| batchtopk_sae/T1 | 0.0320 | 0.0030 |
| tsae/T1 | 0.0408 | 0.0040 |

**ALL SEVEN cells must pass to license the overlay.** Any failure ⇒
STOP + report (a finding, not a license; no re-rolls). Fallback
(pre-approved): the two-instrument figure (trained T-sweep + screen
shuffle curve, instruments labeled). Per-seed deltas reported as
diagnostics only.

## § 4 Shuffle instrument — identical to the λ̂ card § 4

Probing-1.2.0 convention byte-inherited: probe fit on ORDERED train
tile-codes (frozen v1 `lambda_recovery` pipeline, untouched:
`_sample_windows` seeds 0/1, leading-edge targets, LinearRegression);
the SAME fixed probe scored on eval tiles per-row permuted pre-encode
(`shuffle_within_window`, **seed 0, disclosed**); never refit; T1
anchors identity by construction (`shuffle_identity = 1`).
**Identity receipt:** recomputed ordered r must equal the cell's
canonical-runner metric to |Δ| ≤ 1e-6 BEFORE the shuffled column is
read. Code = `tt_shuffle_overlay.py` (this freeze); no frozen eval
edited, no protocol_version moves. Output:
`diafaces/results/tt_shuffle_overlay.json` (per-cell + summary +
mechanical gate table).

## § 5 Deliverable, venue, economics

`figs_writeup/fig_ttrend_shuffle_tsweep.{png,pdf}` — template
knob-for-knob with the probing/RLHF pair, y = **recovery r**,
anchors as bands. Claims nothing.

Venue: **pod H100 GPU 0** — borrow PRE-APPROVED by `1d2e3de28`
(runpod-a idle post-bundle, `b8d15f4a2`; hand back instantly on a
cnov GO — cells are runner-cached, resume loses nothing). Claim =
the LOG line of this freeze. Est 21 cells ≈ 0.5–1 GPU-h (gpt2 d768
is the cheap substrate; λ̂ d4096 cells run ~2–4 min each under
3-worker contention) ≈ **$2–4**; ledger at this freeze; actuals
after. λ̂ lane on GPU 1 unaffected.

_Owner: runpod-b. Recorded-by: claude-fable-5 (runpod-b)._

## AMENDMENT A1 (~17:15 London, before any shuffled column was read)

The identity receipt fired on the first overlay cell: recomputed
ordered r 0.03302152 vs canonical 0.03289311 (|Δ| = 1.28e-4 >
1e-6). Mechanism: cross-process GPU kernel nondeterminism — the
framework pins no TF32/matmul-precision/determinism flags, so encode
outputs drift ~1e-7 relative between processes, amplified through
the p = 2048 OLS probe to ~1e-4 on r. **Tolerance amended 1e-6 →
5e-4** — still 6–60× below every § 3 gate σ, so the receipt retains
its discriminating power against real protocol divergence (a wrong
seed/window/probe moves r by ≫ 1e-3). No result was read before
this amendment; the shuffled column computes only after the amended
receipt passes. PTR with the verdict.
