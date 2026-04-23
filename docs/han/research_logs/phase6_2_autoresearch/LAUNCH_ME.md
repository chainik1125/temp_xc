---
author: Han
date: 2026-04-23
tags:
  - todo
---

## REMINDER: Launch Phase 6.2 autoresearch when Phase 6.1 pipeline completes

Phase 6.1's `run_phase61_full_chain.sh` is currently running (triangle
seed-variance + encoding + autointerp + probing + assemble). Phase 6.2
is queued and ready.

### How to launch

**Option A — fire-and-forget:** start a background launcher that
watches for the Phase 6.1 DONE marker and kicks off Phase 6.2.

```bash
cd /workspace/temp_xc
bash experiments/phase6_2_autoresearch/launch_after_phase61.sh \
  > logs/phase62_launcher.log 2>&1 &
```

**Option B — manual, once Phase 6.1 logs show DONE:**

```bash
grep -q "FULL PIPELINE DONE" logs/phase61_full_chain.log \
  && bash experiments/phase6_2_autoresearch/run_phase62_loop.sh
```

### Current implementation status

Ready to run (seeded by Track 2's 5/32 random baseline):

- **C5 `phase62_c5_track2_longer`** — Track 2 with `min_steps=10000`.
  Reuses existing `agentic_txc_10_bare` dispatch + `--min-steps`
  override. ~60 min GPU.
- **C6 `phase62_c6_bare_batchtopk_longer`** — 2×2 cell (Phase 6.1 #4)
  with `min_steps=10000`. Reuses `agentic_txc_12_bare_batchtopk`.
  ~60 min GPU.

Needs new arch class before C3 (the highest-prior candidate!) can run:

- **C1** — Track 2 + matryoshka H/L
- **C2** — Track 2 + InfoNCE contrastive
- **C3** — Track 2 + matryoshka + contrastive (≈ tsae_paper on TXC base).
  **This is the candidate most likely to close the gap to
  `tsae_paper` 12/32**; implement first if pursuing C1-C4.
- **C4** — Track 2 + EMA-threshold inference (requires threshold
  tracking during training)

See `experiments/phase6_2_autoresearch/candidates.py`
`implementation_note` field for the scope of each.

### What to do on Phase 6.1's completion

1. Verify §9.5 of summary.md is updated with the new rigorous-metric
   baselines.
2. Run `launch_after_phase61.sh` (option A) — starts with C5 + C6
   which are implementation-ready.
3. While C5/C6 run, implement C3's arch class. Then extend the
   loop: `bash experiments/phase6_2_autoresearch/run_phase62_cycle.sh C3`.
4. After Phase 6.2 finishes, pick the winner and hand off to
   Phase 6.3 (seed variance on the winner).

### Budget check

- Phase 6.1 expected to finish by ~00:45 UTC (~3.5 hr from 20:18 start).
- Phase 6.2 with only C5+C6: ~2 hr GPU + ~$1 API.
- Phase 6.2 with C5+C6+C3 (after implementing C3): ~3 hr GPU + ~$1.5.
- Total time from 20:18 start: ~8 hr. Within the 12-hr budget.
