# runpod-2 — working state

**Am:** executor on the ACTMIX shared 3×H100 pod, GPU 2 only
(`source scripts/set_agent_env.sh runpod-2` EVERY shell). Clone
`/workspace/agents/runpod-2/temp_xc`, own venv (peft + einops).

## SPRINT ACK (03f533cc3; posted 01:30 UTC / 02:30 London, wall-verified)

**agentic_txc_02 port STARTED CPU-side (recon in progress —
upstream chain extracted from 94119bc08, audit §6 facts pinned:
MatryoshkaTXCDRContrastiveMultiscale, k_win=100·T, scales 3,
γ=0.5).** ETA: plugin + contract tests + CARD (w/ port-cost flag)
frozen by **~04:00-04:30 UTC**. GPU untouched before the card —
x6/x10 keep GPU 2 until drain (~08:00-08:30 UTC, protected btk
exhibits; 11:00 renders unaffected). Honest grid-cost note for
the card: 21 cells (T{1,2,4,6,8,10,16}×3, T5 = archived anchor)
at v2-walls × contrastive-heaviness ≈ 30-37 GPU-h ≈ $90-110 —
ABOVE the $60-90 est; card will include a PILOT CELL (T2/s42,
~30 min) to measure the heaviness factor before committing the
full grid, plus shardable lane definitions so free pod-A GPUs can
take T-shards after probing (shard split = card author's, per the
map).

## State (2026-07-28 ~00:12 UTC / ~01:12 London, wall-verified)

**CERTIFICATE SHIPPED + RATIFIED (89370c68a; refutation praised as pre-registration working):** RLHF equivalence 3/3
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
2. **rmx_a CANCELLED-WITH-CERTIFICATE (A5b, my card-owner
   ruling per 89370c68a)** — matrix fulfillment = certificate
   line; **AUTO-RE-OPEN binding: any divergent rmx_b per-cell
   check re-cards it as eq-extension** (watch runpod-b's rows,
   run per-cell checker on each rmx_b landing). T4-btk is
   runpod-a's (s42 in at 0.6185).
3. **Morning: 7-point FINAL fig re-render** {1,2,5,6,8,10,16} +
   table + beat = HARD POINT (af7d0869b). 8-point exhibit render
   {1,2,4,5,6,8,10,16} when T4×3 + relumix grid complete.
4. Ledger actuals per lane; STATUS + listener re-arm every wake.

**Fleet refs:** runpod-b = rmx_b T{8,10}×3 (their pod, unblocked
by the certificate; by-T split per A5 — NOT by-seed). runpod-a =
x4 lane (T4 btk ×3, s42 landed) + R30 twins. Morning render
consumes: my A1 rows + x6/x10 + runpod-a's x4 (T4 excluded from
the 7-point hard render; enters at 8-point).

## Durability (b4ec84b04 item 2 — COMPLIANT)

**26/26 trained RLHF ckpts mirrored** to the ratified path
(`temp-bench-data/ckpts/<train_key>/`), certificate-evidence 6
first. **Spot-check receipt: T16 twin 5774f6c8b6d28938 HF-LFS
sha256 == local sha256 (2d6a3289810f144a…) — MATCH.** Receipts:
`experiments/explorations/actmix_rlhf/results/hf_durability_receipts.jsonl`
(train_key + sha256 + hf_path × 26). Auto-push finding: hf_url is
schema-only on this branch (confirmed independently by runpod-1/a).
x6/x10 + any future lane ckpts push at lane completion (cadence
rule). 4 pre-ratification strays in the model repo
(temp_xc_a40_checkpoints/actmix_rlhf_checkpoints/) — bonus
copies, hub may clean.

## Watchers

Origin listener 150s — re-arm EVERY wake. Per-lane DONE watchers
armed at each launch (grep '\[lane <name>\] DONE' or FAIL).

## Ledger

Yesterday ≈ $44; today (07-27) A1 $27 + Ward $2 + eq $11 actuals
≈ $40; queued est: x6/x10 ~$24 (rmx_a $20 CANCELLED per A5b). Caps intact.

## If resuming after compact

Read this + LOG tail from the 00:02 certificate entry + CARD § 7
A5. Lanes: `tail /workspace/logs/actmix_rlhf_lane_*.log`, jsonl
`actmix_rlhf_runs_*.jsonl`, `nvidia-smi` GPU 2. Execute queue in
order. Renderer: `.venv/bin/python -m
experiments.explorations.actmix_rlhf.render_writeup_fig --tag
final` (mono). Checker: `…actmix_rlhf.rlhf_equivalence`.
