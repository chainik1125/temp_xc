# Working state — agent `runpod-1`

**2026-07-27 ~16:25 London — P1 DELIVERED: formal verdict 88a955623
(PTR) + FINAL 3-seed fig inside the extended 16:45 gate. Directive
059a66239 (saturation to 22:00, report ~21:30). Remaining: post-42
addendum, P2 layer sweep, post-1/2 relaunch, 21:30 report.**

## Delivered today (all pushed)

- **FINAL fig** figs_writeup/fig_probing_shuffle_tsweep (3 seeds all
  T; pair-style mono; blueorange = 1-flag re-render if 17:00 meeting
  picks it).
- **Formal verdict** (LOG ~16:10 entry): level story = flat at SAE
  anchor T1–T4 (0.8985/0.8975/0.8988 vs 0.8993), decline to T16
  (0.8794, T16−T1 −0.019); margins VANISH (best-T −0.0005); order-gap
  0→+0.030@T8→+0.023@T16 quotable only cross-task/decline-mitigation
  (guard 5aa351a4e); tsae 0.8718±0.0008 NOT-met w/ 7d caveats;
  E-scoring 1/2/5/7/8 MET, 3/4/6 NOT MET; G5 PASS both k; G1
  over-admission disclosed (+3–5%, +19%@T16, tsae +13–21%).
- **Ledger**: day actuals ≈ $92 (+$17 corr), proj ≈ $110 vs $150 cap.
- Earlier: INTERIM fig (approved), FREEZES lineage fix (ratified),
  CARD §7e (ratified), tsae column complete, P2 freeze af95601dd.

## Evening schedule (decide live, ~17:15)

1. **GPU1 post-42 drains ~17:15** ({T16,T4} shard): KILL GPU1 chain
   before its post-1/2 pass starts (kill bash wrapper + python child;
   NEVER touch GPU 2 pids) → launch P2 extraction on GPU1
   (`layer_sweep.extract` both models sequential, ~25 min; needs
   HF_HOME=/workspace/hf_cache env for llama31 from_pretrained).
2. **GPU0 post-42 drains ~18:00** ({T1,T2,T8} shard): kill chain,
   then probes sharded: llama sweep on GPU1, gemma on GPU0
   (`layer_sweep.sweep <model> <hs...>`, resid-L default set:
   llama 8 15 22 29, gemma 7 14 21 — check LOG first for a
   mac-local overrule to hs-index; capture union covers both).
3. ~19:00 screens done → `score_sweep` → commit results + LOG line
   (PIN af95601dd… actually PIN = HEAD at launch, record it) +
   ledger actuals.
4. ~19:00 **relaunch post-1/2** as separate sweep both GPUs
   (`experiments.probing.actmix.sweep --arm btk-only --txc-archs
   txc_batchtopk_post_btkonly --seeds 1 2 --shard-index N
   --shard-count 2`, TEMP_BENCH_ALLOW_DIRTY=1, nohup) — 10 cells ≈
   5 GPU-h fits the 19:00–21:30 window.
5. **Post-42 addendum** to the verdict when its column lands (~18:00)
   + post-1/2 status at the 21:30 report.
6. **21:30 report** (LOG): P1 close + P2 depth profiles + ledger
   actuals + STATUS rewrite + push.

## Standing

- Monitors: origin bf369am3s (THE listener), grid biki784bc.
- λ̂-Ward = runpod-2's (no reassignment; my 19:45 Ward-update
  deadline passes silently unless my (a) finished early — it won't).
- Tokens by path only; rotations post-weekend. GPU 2 never mine.
- Card clause for post-1/2 cut+relaunch: §3 sequencing allows post
  seeds 1/2 to trail; relaunch is the SAME cells at the same PIN
  lineage (cache-hit-safe if any partial trained).
