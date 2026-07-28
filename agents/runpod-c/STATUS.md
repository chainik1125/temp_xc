# runpod-c STATUS — ALIVE again; shards DONE; relief venue being repaired (2026-07-28 12:57 London, date-verified 11:57 UTC)

**I am `runpod-c`**, alone on a **dedicated** 2×H100 pod, workspace
`/workspace/agents/runpod-c/temp_xc`, venv healthy, probing substrate
resident (acts + 38/38 probe cache).

## READ FIRST — the 6h45m dead window

Shards drained 05:59; **nothing ran 05:59 → 12:38** (both H100s 0%),
through the 11:00 window. The fleet declared me down at 07:10 and
carried my duties. No science lost — the shards had finished their
whole assignment — but do not trust any fleet doc that describes pod B
as "ready"; verify artifacts on disk. Disclosed in the 12:57 LOG beat.

## IN FLIGHT RIGHT NOW

- **NOTHING RUNNING. Both GPUs idle, held deliberately.**
- **BLOCKED ON A RULING (see below): who executes pod-B lanes.** Han
  directive `4e04ae0e3` item 2 says "pod B = hub executes (worktree
  pattern)" — written when I was believed down. I am up. I will not
  share an unowned pod with a second executor; I launch nothing until
  the hub or runpod-2 names ONE executor. Posted 13:02.
- **Awaiting runpod-2's shard map** (wave 1 = s42 across all seven T,
  seed-column-first per item 3).

## ✅ RLHF l13-IT substrate — DONE 13:01, receipted

    /workspace/caches/rlhf/cached_hh_rlhf_l13it/
      chosen.npz  1,181,193,532 B · rejected.npz  1,181,193,532 B · meta.json
    t-test: rejected 36.232 (paper 36.23) · chosen 28.573 (paper 28.57)
            p = 9.76e-10 (paper 1e-09) · l12 reference match = True

**Pod B IS G1-ready.** Fresh build reproduces the paper's own t-stats.

## ⚑ The two substrate bugs I found (fixed; see 12:57 beat)

1. **Gated-repo 401**: the hub's 06:17 stage B ran with no `HF_TOKEN`;
   `google/gemma-2-2b-it` is gated. Both `/workspace/.tokens/{hf_token,
   hf_token_datasets}` DO have access (verified via `model_info`).
   Fix = export `HF_TOKEN` before the launcher.
2. **Stale-tree SyntaxError**: `build_cache.py:131` "name
   'SUBJECT_MODEL' is used prior to global declaration" on my
   102-commit-stale tree; **upstream compiles clean**. Sync before
   running any lane.

The hub's 07:22 receipt said "stage B running" and was read as done —
the cache dir sat EMPTY for ~6 h. **Cite artifacts, not process
state.**

## COMPLETED — paper-faithful shards C + D (8/8 cells)

PIN `d9235755b`, `paper_txc_base_v1t`, canonical runner, Protocol
1.2.0. `PF_SHARD_D_DONE` 05:24, `PF_SHARD_C_DONE` 05:59.

| T | k20 per seed | mean | realized l0 vs k_win |
|---|---|---|---|
| 1 (s2) | 0.8953 | — | 19.999 / 20 |
| 2 (s42) | 0.9050 | — | 40.000 / 40 |
| 6 (42/1/2) | 0.8987 / 0.8881 / 0.8856 | **0.8908** | 120.000 / 120 |
| 8 (42/1/2) | 0.8821 / 0.8852 / 0.8893 | **0.8855** | 159.94–159.97 / 160 |

- **Rows are upstream and exact** — mac-local repatriated the 3
  stranded cells at 07:10; I verified 16/16 mine present, 0 missing, 0
  dup eval_keys, 8/8 manifest train_keys. My local copies were
  redundant and were dropped, not re-committed.
- **E1 corroborated independently**: zero-picks 0.001/0.000/0.000/
  0.036–0.057 at T1/T2/T6/T8 — onset exactly T6→T8, matching
  runpod-1's `6e928e2bb` scoring from the other side.
- **T=1 control is exact**: shuffle delta +0.0000 at both k5 and k20,
  `shuffle_identity = 1.0`.
- **8/8 ckpts durable** on `temp-bench-data` (verified by fresh
  `list_repo_files`, not by the push script's log). Plus 27/27 tscale
  ckpts under `ckpts/tscale/`.
- **Actuals** 5.30 GPU-h, ~40 min/cell vs ~48 est.

## ⚑ Venue capacity — measured, for rung-1 relief

Replicated runpod-b's protocol on pod B (same GEMM shape, 6 iters):

    nproc 208 ; cpu.max -> 44.2 core quota ; torch default 104 threads
    OMP_NUM_THREADS unset  (the same trap they found on pod A)

| config | pod B aggregate | pod B vs 1 lane | pod A vs 1 lane |
|---|---|---|---|
| 1 lane @default | 2210 | 1.00x | 1.00x |
| 1 lane @quota | 3051 | 1.38x | 1.18x |
| 2 lanes naive | 4029 | **1.82x** | **0.75x** |
| 2 lanes @22 | 4681 | 2.12x | 1.59x |
| 3 lanes @14 | 5800 | 2.62x | — |
| **4 lanes @11** | **5933** | **2.68x** | 1.64x |

**Launch recipe for pod B: 4 lanes, `OMP_NUM_THREADS=11` +
`MKL_NUM_THREADS=11` per lane.** Pod B 4-lane aggregate 5933 vs pod A
4510 (~1.3x). GPU memory not binding (2 idle H100s) so
`TEMP_BENCH_GPU_FRACTION` co-residency is free here; CPU is the budget.

**The cgroup trap generalises (platform property); the collapse does
NOT** — naive 2-lane costs ~10% per lane here vs 260% on pod A.
Candidate cause: pod B is single-agent dedicated, pod A is
co-tenanted. Production corroboration: shards C+D ran naively
co-tenanted 3.2 h at ~40 min/cell each, beating the ~48 min single-lane
estimate — consistent with a lane that is NOT OpenMP-bound (bears on
runpod-b's single-threaded question to runpod-2). Caveats: ~48 min is
the card's estimate not a measured 1-lane baseline; my cells are
probing, runpod-2's claim is about RLHF cells.

## Process state

- Git: synced to `0bed01849` via `reset --hard` **after** proving my
  rows were already upstream (0 missing) — no data at risk.
- Discipline: explicit-path commits; back up canonical files before
  any rebase and union-verify after; LOG conflicts keep BOTH blocks;
  never `--ours` on my own picks; **stamps only after reading `date`**
  (6 drifts on record).
- Tokens at `/workspace/.tokens/` — paths only, values never in
  git/logs/cards.
- Ledger: tscale ≈$16 overnight / ≈$33 program; shards 5.30 GPU-h.

## Next actions queue

1. **Watch stage B to completion**; post the receipt; only then may
   pod B be called G1-ready.
2. Hold both GPUs for rung-1 relief; runpod-2/mac-d scope the cells.
3. Post-rebuttal: resume hill-climb via `tscale/RESULTS.md § FREEZE`
   (pre-registered C6 pair, never launched = the resume point).

*Rewrite before any compact. — runpod-c*
