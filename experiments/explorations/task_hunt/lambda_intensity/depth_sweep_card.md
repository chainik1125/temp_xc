# λ̂ depth-sweep card — L{6,9,12,15,18} screen (P2(b), directive 059a66239)

**Class: SCREEN** (hunt3 discipline: scorer frozen before results; no
panel/claiming cells). Owner runpod-2, slot ~20:50–21:30 London
2026-07-27 (post-P1 GPU-2 drain). Approvals: split 5aa351a4e; prereq
staging + "Ward stays yours" + L9/L15 capture extension 121807fb0.

## Inherits frozen (card.md, candidate 1 — unchanged)

Row recipe (eligibility, by-trace 80/20 rng(7), per-trace cap 120,
class-stratified), targets (λ̂_hist tercile PRIMARY; full-λ̂ tercile
secondary always read against its 0.82 position floor; regression r),
T grid {2,4,8,16,32}, probe stack (problib, per-token / flatten /
window-mean / within-window-shuffle seed 23, MLP presence at T16,
permutation null seed 99), σ_null = 0.0031 (3σ = 0.0094) from the
original 17-cell null.

## This sweep (deviations, all frozen here)

1. **Layers**: hs {7, 10, 13, 16, 19} = resid_post L {6, 9, 12, 15, 18}
   (extends the frozen L10/L12 pair; L9/L15 are odd blocks — capture
   list extended per approval).
2. **Caches rebuilt locally** on GPU 2 via argv-driven
   `depth_sweep_build.py` (wraps `conversion_depth.cache_depth.main`
   with LAYERS = [6, 9, 12, 15, 18]; batch/seq/dtype/stream unchanged;
   stream = the sha-verified mirror restore, 4044×128). ~4.24 GB/layer
   fp16, ~21 GB/tag.
3. **Separate results store**: `results/lambda_depth_sweep.json` via
   `depth_sweep_screen.py` (screen.py with LAYERS + OUT overridden,
   protocol otherwise byte-identical). ALL five layers recompute on
   the rebuilt cache — the depth profile is single-cache-generation
   by construction; the frozen `lambda_screen.json` is never touched.
4. **D-K1 rebuild-consistency gate (must pass before any depth claim):**
   |per-token AUC(base/hs13, rebuilt) − frozen base/hs13 tok AUC|
   ≤ 3σ_null = 0.0094, primary target. Pass ⇒ rebuilt profile quotable
   and the frozen L10/L12 cells may be cited beside it. Fail ⇒ STOP,
   report the delta, no depth verdict (forward nondeterminism vs the
   original A40/driver build would be the suspect — that is a finding,
   not a license to pick the friendlier cache).

## Pre-registered readouts (directional, scored as written)

- **D-P1 (presence):** per-token primary AUC clears the position
  floor + 3σ at every layer ≥ L9, both tags run.
- **D-P2 (shape):** the per-token depth profile is unimodal with an
  interior peak in L9–L15 (candidate-1 ordering L12 ≥ L10 in all four
  frozen cells motivates this). A flat profile (max−min ≤ 3σ_null)
  reads "distributed, no depth localization" — reported as such.
- **D-P3 (T-story stability):** the window-ceiling − per-token gain at
  T32 is ≥ 0 at every layer; the order story stays negative (g_order
  ≤ 0 / shuffle cost ≤ +0.022-scale) — any layer where ORDER turns
  positive is flagged loudly (it would be the first order-carried λ̂
  cell) but claims nothing without a confirm run outside this card.

## Schedule, cost, descope

Reader weights (Llama-3.1-8B; distill twin if reached) pre-download
~20:00 (network-only, overlaps P1 tail). At GPU drain: base cache
build (~6 min) → base screens (5 layers, est 15–30 min) → D-K1 → LOG
verdict lines PTR. **Distill = stretch**: start only if wall < 21:20;
else base-only profile is the deliverable and distill parity carries
over as an open line (P4 held at L10/L12 in the frozen cells). GPU
cost ≈ 0.3–0.7 GPU-h ≈ $1–2; ledger line at launch. Screens only —
NOTHING here feeds the pick-gated cnov panel.
