# runpod-2 — working state

**Am:** executor on the ACTMIX shared 3×H100 pod, GPU 2 only
(`source scripts/set_agent_env.sh runpod-2` EVERY shell — the
14:28 relaunch mistake is why). Clone
`/workspace/agents/runpod-2/temp_xc`, own venv (peft + einops).

## State (2026-07-27 ~22:30 UTC / ~23:30 London, wall-verified)

**Shipped today:** A1 FINAL RATIFIED (af7d0869b) — 3-seed fig +
table + verdict extension. Ward depth-sweep verdict posted PTR
(0811644b1). Eq lane 2/3 CERTIFIED IDENTICAL (sae_k500 + txc_T5:
torch.equal all shared tensors, Δauc exactly 0, provenance
receipts staged — train_cached=False, walls 13.4/55.7 min,
distinct train_keys a67f63b5e0e15d6e/eff51d4fb0ec4088).

**RUNNING on GPU 2:** eq lane cell 3/3 — relumix txc_T16 s42
(EXPECTED divergent, 7093c21f8), solo since ~20:24 UTC, lands
~23:05 UTC. PID 76437, 36.9 GB. Log quiet = nohup block buffer
(flushes on exit; verified alive via /proc + nvidia-smi).

**⚑⚑ NEW GOVERNING DIRECTIVE: 1065b26cf (Han's deliverables
matrix).** Relu-mix RLHF arm REQUIRED at every grid T except
certified-identical points — A3/A3b cancel branch SUPERSEDED (eq
lane = certification+telemetry, unchanged value, different
consequence). btk T4×3 added (grid floor; T5 stays bonus).
**CARD § 7 A5 frozen this commit** — lanes x4 / rmx_a / rmx_b /
rmx_b16 in cells.py.

## SPLIT PROTOCOL (runpod-b: read this at width-match drain)

Pre-auth UNCONDITIONAL per 1065b26cf. Pin = the commit adding
CARD A5 (contains lanes; resolve via rev-parse on YOUR clone
after pull). Run from YOUR pod/GPU with YOUR env sourcing:

- **rmx_b = yours:** relumix txc T{8,10} × seeds {42,1,2} —
  6 cells ≈ 8.9 GPU-h solo ≈ $27. Launch any time from your
  drain; no dependency on my lanes.
  `nohup .venv/bin/python -m experiments.explorations.actmix_rlhf.run_cells --lane rmx_b --pin $(git rev-parse HEAD) ...`
- **rmx_b16 = yours, CONDITIONAL:** relumix txc T16 s1/s2 —
  launch ONLY after my T16 gate posts DIVERGENT (LOG certificate
  entry, ~00:00 London). Identical ⇒ never runs (certificate line
  covers T16). ≈ 5.3 GPU-h ≈ $16.
- **rmx_a = mine:** T{1,2,4,6} × 3 on GPU 2 behind btk lanes.
- Contention math is per-GPU — your pod, your rates. Ledger your
  own est/actuals lines.

## THEN (frozen order, GPU 2)

1. **T16 lands (~23:05 UTC)** → commit leaderboard/manifest rows
   → `.venv/bin/python -m experiments.explorations.actmix_rlhf.rlhf_equivalence`
   → certificate LOG entry w/ receipts (train_keys, wall_s,
   train_cached=False, telemetry trace files, distinct-ckpt
   no-aliasing). Consequence per A5: DIVERGENT T16 ⇒ rmx_b16
   armed for runpod-b (report IMMEDIATELY per 361de3cb2);
   IDENTICAL ⇒ certificate line, rmx_b16 dead. Push FAST —
   runpod-b reads the gate at their ~01:00 drain.
2. **Launch x6 ‖ x10** (A2 frozen: fracs 0.35/0.50) at fresh
   pushed pin via rev-parse. ~8 GPU-h co-resident → drains
   ~07:00-07:30 UTC.
3. **x4 at first drain slot** (x6 drains first ~05:30 UTC → x4
   co-resides with x10 tail, 2-way max — never 3-way, untested
   contention). ~2 GPU-h.
4. **rmx_a behind the btk lanes** (T{1,2,4,6}×3, cheap-first
   order per cells.py). ≈ 6.5 GPU-h.
5. **Morning: 7-point FINAL fig re-render** {1,2,5,6,8,10,16} +
   table + beat = the HARD POINT (af7d0869b — unchanged by A5).
   8-point exhibit re-render later when T4×3 + relumix grid land.
6. Ledger actuals per lane; STATUS + listener re-arm every wake.

## Watchers live

Origin listener (150 s; re-arm EVERY wake); eq DONE/FAIL
b296ty8ip (240 s). Arm per-lane DONE watchers at each launch.

## Ledger

MODAL_SPEND RUNPOD: yesterday ≈ $44; today A1 $27 + Ward $2
actuals; est queued: eq ~$11 + x6/x10 ~$24 + x4 ~$6 + rmx_a ~$20.
Day caps intact ($150/day).

## If resuming after compact

Read this file + LOG tail from my 21:45 entry + CARD § 7 A5.
Check lanes: `tail /workspace/logs/actmix_rlhf_lane_*.log` + jsonl
`actmix_rlhf_runs_*.jsonl`; `nvidia-smi` for GPU 2 occupancy.
Execute the THEN list in order. Checker:
`.venv/bin/python -m experiments.explorations.actmix_rlhf.rlhf_equivalence`.
Renderer: `... .render_writeup_fig --tag final` (mono default).
