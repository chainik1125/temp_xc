# DQ_T_FILL_CARD — dq panel T{6,10} fill (Han grid item 5), post arch × 3 seeds

**Agent:** runpod-b · **GPU:** pod A GPU 1 · **Frozen:** 2026-07-28 00:01 BST
(freeze commit = this card + `dq_t_fill.py`; pin asserted clean at launch).
**Directive:** Han matrix `1065b26cf` item (5) — dq T-grid verify + fill.
Coverage verified from the board (my 23:42 entry): trained
T{2,4,8,16,32}×3 complete on the window archs + T1 anchors ⇒ **missing
exactly T{6,10}**. Step-0 unblocked by runpod-a's pod-level dialevel
rebuild (their 00:05 entry, phase 1): `/workspace/dialevel_caches/
llama31_8b/hs14.npy` verified this beat (3.83 GB; acts_meta 3653×128×4096
fp16, mapping_verified_rows 3653, screen_hs 14).

## Cells (12 = 6 trained + 6 untrained twins)

Exactly the λ̂ T-fill shape (T_FILL_CARD.md precedent, same beat), on the
dq panel's venue:

- ds `dial_real_dqgap_llama31_8b_l14`; arch post only
  (`txc_batchtopk_post` — the panel/exhibit line arch); window_ts (6, 10);
  seeds {1, 2, 42}; d_sae 2048, k_pos 8, n_steps 8000, buffer_tokens
  524288; untrained twins per (T, seed) — all `run_panel.py` constants
  (which are the stage-2 constants + the panel's own WINDOW_TS/BUFFER).
- **V2 paired columns:** the PROBE_V2_SPEC § 2 block attached verbatim to
  every cell via `run_panel.V2` (`eval_extra` → hashes into eval_key) —
  the dq panel's v2-DEFECT AMENDMENT term; the existing dq panel rows
  carry these keys (verified on the board this beat), so the fill stays
  venue-consistent. (The λ̂ panel is v1-only — my λ̂ fill matched it; each
  fill matches ITS panel.)
- **eval_window_L = 30** for all 12 cells — same venue line as
  T_FILL_CARD (T∈{6,10} ∤ 32 tiling reshape; L=30 minimal shared; one
  exhibit caption line), applied to the same evaluator machinery (the dq
  face panels through the λ̂ evaluator per the datasource's design note).
- batch_size = `1024 // T` (grid convention). Output:
  `results/dq_t6t10_fill.json` under diafaces (panel results untouched).

## Open item (mac-local): dq shuffle-overlay columns

NOT in this card. The dq exhibit's shuffle treatment (tt-style fallback
vs λ̂-style overlay columns) is mac-local's call; if overlay columns are
wanted I pre-register them the same way as the λ̂ fill overlay (identity
receipt ≤ 2e-3, no anchor gate, L=30) in a follow-up line — the fill
checkpoints persist either way.

## Estimate + ledger

λ̂ fill actuals were 12 cells in 9 min ≈ $1; same shapes, same corpus
scale (3653×128 vs 4044×128) ⇒ **≈ $1–2**. Ledger line at launch;
actuals on landing.

## Mechanics

Canonical pathway via `grid.run_pool` → `run_experiment` (section
synthetic); AGENT_NAME=runpod-b inline; CUDA_VISIBLE_DEVICES=1;
TEMP_BENCH_ALLOW_DIRTY=1 under the launch-pin convention. Runner:
`dq_t_fill.py` (this commit). Rows checkpoint on landing. No re-rolls;
seed anomalies (λ̂ fill's T10/s2 precedent) reported as measured with
receipts.
