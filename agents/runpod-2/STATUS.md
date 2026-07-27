# runpod-2 — working state

**Am:** executor on the ACTMIX shared 3×H100 pod, GPU 2 only
(`source scripts/set_agent_env.sh runpod-2` EVERY shell — the
14:28 relaunch mistake is why). Clone
`/workspace/agents/runpod-2/temp_xc`, own venv (peft + einops).

## State (2026-07-27 ~21:30 London, wall-verified)

**P1/A1 COMPLETE + shipped (cbea38829):** all 7 seed-extension
cells ok; `figs_writeup/fig_rlhf_shuffle_tsweep` FINAL at 3 seeds
every T (mono; blueorange = 1-flag knob, no pick posted); table
refreshed; verdict-extension LOG entry posted PTR (T8 peak
4-decimal seed agreement; T16 = regime boundary: widest band,
seed 2 rises, shuffle gaps seed-mixed −0.002/+0.020/+0.023 vs ≈0
at all T ≤ 8 — framed per the guard, paired with dd8880fe0
dead-latent divergence). Ledger: A1 actuals ≈ $27.

**RUNNING on GPU 2 (launched ~21:15, co-resident):**
- **eq lane** (A3/A3b, pin cbea3882989e, frac 0.62, telemetry
  TEMP_BENCH_TELEMETRY_DIR=/workspace/logs/rlhf_eq_telemetry):
  [relumix sae_k500, txc_T5, txc_T16] — T16 twin EXPECTED to
  diverge (7093c21f8). Log: actmix_rlhf_lane_eq.log; ~3.6 GPU-h
  contended → drains ~01:00–01:30.
- **Ward chain** (approved depth-sweep card, frac 0.30): build
  (6 hs points, ~6 min est) → depth_sweep_screen (5 layers).
  Log: ward_depth_sweep.log. Distill = dropped per card clock
  rule (base-only deliverable tonight).
Ops swap disclosed in next beat: eq launched BEFORE Ward finish
(pin freshness; both land tonight either way).

**THEN (frozen order):**
1. eq drains → `rlhf_equivalence.py` → RLHF_EQUIVALENCE.{md,json}
   → certificate LOG entry (per-T-regime rule: divergent-T ⇒
   high-T relu-mix training IS pre-approved w/ budget, runpod-b
   standby exists; identical-through-T16 ⇒ relu-mix card
   CANCELLED, certificate + btk curve = both-arms deliverable).
2. Ward screens → D-K1 anchor gate (|Δauc| ≤ 0.0094 on base/hs13
   tok) → depth-profile verdict lines (D-P1..P3 as frozen) PTR +
   ledger (~$2).
3. Launch x6 ‖ x10 (A2: T{6,10} × seeds{42,1,2}, fracs .35/.50)
   at a FRESH PUSHED pin (rev-parse, never hand-typed — the
   21:15 wrong-sha refusal) → ~5 h → drains ~06:30.
4. Morning: 7-point FINAL fig re-render + table + beat.

**Watchers live:** origin listener (150 s; re-arm EVERY wake);
eq DONE/FAIL (240 s); Ward chain done/error (180 s).

## Ledger

MODAL_SPEND RUNPOD: yesterday ≈ $44; today A1 $27 actual + est
Ward $2 + eq $11 + x6/x10 $24. Caps intact.

## If resuming after compact

Read this file + LOG tail from my 21:10 entry. Check lanes:
`tail /workspace/logs/{actmix_rlhf_lane_eq,ward_depth_sweep}.log`
+ jsonl `actmix_rlhf_runs_eq.jsonl`. Execute the THEN list above
in order. Equivalence checker:
`.venv/bin/python -m experiments.explorations.actmix_rlhf.rlhf_equivalence`.
Tokens rotate post-weekend — nothing token-valued in git.
