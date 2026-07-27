# T_FILL_CARD — λ̂ Stage-2 T{6,10} fill (Han grid), post arch × 3 seeds

**Agent:** runpod-b · **GPU:** pod A GPU 1 · **Frozen:** 2026-07-27 23:47 BST
(freeze commit = this card + `run_t_fill.py`; pin asserted clean at launch).
**Directive:** Han deliverables matrix `1065b26cf` item (4) — λ̂ T-grid
{1,2,4,6,8,10,16}; existing coverage T{2,4,8,16}×3 (stage-2 grid + overlay)
with T1 = sae/tsae anchors on-figure; **missing exactly T{6,10}**. Claimed
runpod-b in my 23:42 verdict entry (56d53c157) with the L venue flag below.

## Cells (12 = 6 trained + 6 untrained twins)

`design.uniform_cells` — the stage-2 pathway verbatim (`run_stage2.py`
constants), narrowed to the fill:

- ds `ward_real_lambda_base_l12` (substrate verified on this pod:
  `/workspace/conv_depth_caches/base/hs13.npy`, 4.0 GB — the same cache my
  overlay retrains served from today).
- archs = post only (`txc_batchtopk_post` — the exhibit's T-sweep line arch,
  matching the overlay grid); **window_ts = (6, 10)**; seeds {1, 2, 42};
  d_sae 2048, k_pos 8, n_steps 8000, buffer_tokens 524288 — all stage-2
  values. Untrained twins per (T, seed) at k_pos 8, n_steps 0 (uniform_cells
  default, matches the stage-2 grid's own design; negligible cost).
- batch_size = grid convention `1024 // T` (T6 → 170, T10 → 102 windows;
  equal B·T token-positions/step) — formulaic, no new knob.
- Output: `results/stage2_t6t10_ward_real_lambda_base_l12.json` (the main
  `stage2_…json` is NOT touched); rows land in the canonical leaderboard as
  always.

## Venue line (flagged in 56d53c157, unobjected one beat)

**eval_window_L = 30 for these four T-points** (trained + untrained at T6,
T10). The λ̂ evaluator tiles `n_tiles = L // T` with a hard reshape — T ∈
{6,10} do not divide the stage-2 default L=32 (would crash; λ̂-side cousin
of the probing phantom-T10 issue). L=30 is the minimal L divisible by both
fill Ts; per-cell `eval_window_L` is an existing supported knob (hashes into
eval_key); the quoted-panel L=32 points are untouched. One caption line on
the exhibit: "T∈{6,10} points evaluated at L=30 (tiling divisibility);
all other points L=32."

## Pre-registered overlay extension (post-landing, same beat or next)

Ordered/shuffled probe columns for the two new T-points via the frozen
overlay machinery (`shuffle_overlay._fit_ordered_and_shuffled`) pointed at
the fill checkpoints, with L=30: probe fit on ORDERED train tiles, same
fixed probe on per-row `shuffle_within_window(…, T, seed=0)` eval tiles;
identity receipt |ordered r − canonical row metric| ≤ 2e-3 (amended
tolerance, A2 precedent) must pass before the shuffled column is read.
**No anchor gate** — these are fresh primary cells (first rows at these
T-points); the canonical rows ARE the quoted numbers. No re-rolls; if a
receipt fails: STOP + report.

## Estimate + ledger

6 trained cells ≈ 10–18 min each (overlay-grid history at d2048, mid-T)
run 2 workers ≈ 40–60 min wall; untrained + evals ≈ minutes.
**≈ 1.3–1.8 GPU-h ≈ $4–6** (matrix envelope for (4)+(5) was $5–10; dq's
half is separately blocked — see LOG). Ledger line at launch; actuals on
landing.

## Mechanics

Canonical pathway via `grid.run_pool` → `run_experiment` (section
synthetic); AGENT_NAME=runpod-b set explicitly inline (agent-stamp
discipline per runpod-2's fix-forward 64083c940); CUDA_VISIBLE_DEVICES=1;
TEMP_BENCH_ALLOW_DIRTY=1 under the launch-pin convention. Runner:
`run_t_fill.py` (this commit). Rows checkpoint on landing.
