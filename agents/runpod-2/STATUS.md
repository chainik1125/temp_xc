# runpod-2 — working state

**Am:** executor on the ACTMIX shared 3×H100 pod, GPU 2 only
(`source scripts/set_agent_env.sh runpod-2` EVERY shell). Clone
`/workspace/agents/runpod-2/temp_xc`, own venv (peft + einops).

## State (2026-07-28 ~00:05 UTC / ~01:05 London, wall-verified)

**CERTIFICATE SHIPPED (this push):** RLHF equivalence 3/3
TENSOR-IDENTICAL through T16 (k500/T5/T16, s42; Δauc exactly 0
each; receipts + mechanism: boundary_min_pre ≥ 2.21 every logged
step — relu inert by margin; dead_frac 0.654 at T16 present but
non-contesting). Pre-registered T16 divergence (7093c21f8)
REFUTED — disclosed. **rmx_b16 DEAD. runpod-b launches rmx_b
only.** RLHF_EQUIVALENCE.{md,json} committed. Eq actuals $11.

**Queue (frozen order, GPU 2):**
1. **NOW: launch x6 ‖ x10** (A2 frozen: T{6,10}×3 btk, fracs
   0.35/0.50) at fresh pushed pin via rev-parse. ~8 GPU-h
   co-resident → drains ~08:00-08:30 UTC. Arm DONE watchers.
2. **rmx_a behind them** (A5: relumix T{1,2,4,6}×3, cheap-first;
   T4-btk is runpod-a's per ratified swap be3d3fddc — landing on
   their pod, s42 already in at 0.6185). ≈ 6.5 GPU-h.
3. **Morning: 7-point FINAL fig re-render** {1,2,5,6,8,10,16} +
   table + beat = HARD POINT (af7d0869b). 8-point exhibit render
   {1,2,4,5,6,8,10,16} when T4×3 + relumix grid complete.
4. Ledger actuals per lane; STATUS + listener re-arm every wake.

**Fleet refs:** runpod-b = rmx_b T{8,10}×3 (their pod, unblocked
by the certificate; by-T split per A5 — NOT by-seed). runpod-a =
x4 lane (T4 btk ×3, s42 landed) + R30 twins. Morning render
consumes: my A1 rows + x6/x10 + runpod-a's x4 (T4 excluded from
the 7-point hard render; enters at 8-point).

## Watchers

Origin listener 150s — re-arm EVERY wake. Per-lane DONE watchers
armed at each launch (grep '\[lane <name>\] DONE' or FAIL).

## Ledger

Yesterday ≈ $44; today (07-27) A1 $27 + Ward $2 + eq $11 actuals
≈ $40; queued est: x6/x10 ~$24 + rmx_a ~$20. Caps intact.

## If resuming after compact

Read this + LOG tail from the 00:02 certificate entry + CARD § 7
A5. Lanes: `tail /workspace/logs/actmix_rlhf_lane_*.log`, jsonl
`actmix_rlhf_runs_*.jsonl`, `nvidia-smi` GPU 2. Execute queue in
order. Renderer: `.venv/bin/python -m
experiments.explorations.actmix_rlhf.render_writeup_fig --tag
final` (mono). Checker: `…actmix_rlhf.rlhf_equivalence`.
