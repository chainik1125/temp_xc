---
author: Han
date: 2026-04-28
tags:
  - design
  - in-progress
---

## A40 + 46 GB RAM / 900 GB feasibility for Phase 7 retraining

### Hardware delta vs original Phase 7 plan

|  | original (H200 pod) | new (A40 pod) | ratio |
|---|---|---|---|
| GPU | NVIDIA H200, 141 GB HBM3e | NVIDIA A40, 46 GB GDDR6 | **0.33×** VRAM |
| System RAM | 188 GB | **46 GB** (per-pod cgroup cap) | **0.24×** RAM |
| Persistent storage | 5 TB | 900 GB (per-pod quota) | 0.18× |

> The host node shows 503 GB free in `/proc/meminfo`, but
> `/sys/fs/cgroup/memory/memory.limit_in_bytes = 49 999 998 976` —
> the pod's actual cgroup ceiling is **46.6 GB**. Trust the cgroup
> limit, not `free -h`.

### What this changes vs the original plan

Three of the original H200-leveraging tactics in `plan.md` §"How
Agent A should leverage the H200 pod" no longer apply:

- **(plan §1) "Pre-load the anchor probe cache into RAM at process
  startup. ~80 GB total fits comfortably in 188 GB."** — Doesn't
  fit in 46 GB. Probing pass MUST drop to per-task streaming
  (Phase 5's old pattern). Eats the ~50–80 hr disk-I/O saving the
  H200 plan claimed; on A40 expect ~2–3 hr extra wall-clock for
  the full probing pass.
- **(plan §4) "Bump training batch size 1024 → 4096."** — A40 has
  ~3× less VRAM than H200; we revert to **batch=1024** (Phase 5
  baseline). At T=32, batch=4096 alone costs ~1.2 GB just for the
  input tensor, on top of the 33 GB weights+Adam.
- **(plan §5) "Run SubseqH8 at T_max ∈ {32, 64, 128}."** — T_max=64
  weights+Adam alone is 65 GB, over A40's 46 GB ceiling. Already
  H200-only by design; on A40 it becomes permanently unreachable.

### Memory model

Training-time peak resources split across two pools:

- **GPU (46 GB):** weights + Adam (m, v, fp32) + per-batch fwd/bwd
  workspace + (optionally) activation cache.
- **CPU RAM (46 GB):** Python interpreter + torch (~7 GB at import
  time, grows with cudnn handles) + (optionally) activation cache +
  probe cache slices + ckpts being saved + OS / page cache headroom.

Cache sizes (PRELOAD_SEQS=24 000, ctx=128, d_in=2304, fp16):
- single-layer (per-token / TXC / H8 / Subseq): **14.2 GB**
- 5-layer (MLC family): **70.8 GB**

`W + Adam` per arch family (fp32 params + fp32 m + fp32 v = 12 B/param):

| arch | W + Adam |
|---|---|
| topk_sae, tsae_paper_*, mlc, mlc_contrastive, agentic_mlc_08 | 5.1 GB |
| txc T=5 (txc_bare/agentic) | 5.1 GB |
| txc T=10 / Subseq T_max=10 | 10.2 GB |
| txc T=20 (incl txcdr_t20_kpos100) | 20.4 GB |
| txc T=24 | 24.4 GB |
| txc T=28 | 28.5 GB |
| **txc T=32 / SubseqH8 T_max=32** | **32.6 GB** |
| H8 T=10 | 11.2 GB |
| H8 T=20 | 22.4 GB |
| H8 T=28 | 31.4 GB |
| **H8 T=32** | **35.9 GB** |
| **SubseqH8 T_max=64** | **65.2 GB** |

(H8 has matryoshka H decoder — adds ~10% over plain TXC at the
same T.)

### Where can the cache live?

| pattern | single-layer (14.2 GB) | 5-layer (70.8 GB) | code change |
|---|---|---|---|
| (plan default) preload to GPU | only fits if W+Adam ≤ ~28 GB | never | none |
| preload to CPU RAM, DMA per batch | fits with ~25 GB RAM headroom | **does NOT fit** in 46 GB RAM | small (`_train_utils.py` `pin_memory()` + `non_blocking=True`) |
| disk-mmap'd reads, page cache only | minimal RAM cost | fits (page cache evicts) | larger refactor of `make_*_gen_gpu` to read slices instead of indexing a tensor |

For T ≤ 24 single-layer archs, plan-default preload-to-GPU still
works on A40 (cache + W+Adam ≤ 39 GB). For T ≥ 28 single-layer
archs, weights+Adam crowds the GPU and we need cache-in-CPU-RAM.
For all multi-layer (MLC) archs, disk-mmap is the only path.

### Feasibility on A40 by category

| arch (rows in canonical) | A40 status | binding constraint |
|---|---|---|
| **(A) Permanently lost** | | |
| `phase5b_subseq_h8_T64_s5` (row 49) | **✗ permanent** | 65 GB W+Adam exceeds 46 GB VRAM, period |
| **(B) Needs disk-mmap streaming refactor (multi-layer cache)** | | |
| `mlc` (row 4) | ✗ without refactor | 70.8 GB cache > 46 GB RAM AND > 46 GB VRAM |
| `mlc_contrastive_alpha100_batchtopk` (row 5) | ✗ without refactor | same |
| `agentic_mlc_08` (row 6) | ✗ without refactor | same |
| **(C) Needs CPU-RAM-cache refactor (single-layer, large W+Adam)** | | |
| `txcdr_t32` (row 29) | ✗ without refactor | 14 + 33 = 47 GB > 46 GB GPU |
| `phase57_partB_h8_bare_multidistance_t32` (row 45, dropped) | ✗ without refactor | 14 + 36 = 50 GB > 46 GB GPU |
| `phase5b_subseq_h8_T32_s5` (row 48, dropped) | ✗ without refactor | same as txcdr_t32 |
| `phase57_partB_h8_bare_multidistance_t28` (row 44, dropped) | tight; might fit at batch=1024 | 14 + 31 + 2 = 47 GB |
| `txcdr_t28` (row 28) | tight; fits at batch=1024 | 14 + 28 + 2 = 44 GB |
| **(D) Comfortably fits at batch=1024 with current code** | | |
| All other 33 archs (T ≤ 24, anchor cells, TFA, per-token, Subseq B2/B4) | ✓ | |

### What's permanently lost vs what only needs effort

**(1) Permanently lost on A40 — 1 arch.**
- Row 49 `phase5b_subseq_h8_T64_s5` (SubseqH8 T_max=64). Weights+Adam
  alone are 65 GB; no batch reduction or cache-streaming changes that.
  Without parameter sharding or fp16/bf16 weights (an architectural
  change to the SAE training loop), this cell is unreachable on A40.

**(2) Newly hard because of the 46 GB RAM cap — 3 archs.**
- MLC family (rows 4, 5, 6). On the original H200 the multi-layer
  `preload_multilayer` (70.8 GB) was a non-issue — RAM had room.
  On A40 the cache fits in *neither* pool. Recovery requires a
  larger refactor: replace `np.asarray(arr[:n_seqs], ...)` with
  per-batch `np.memmap` slicing in the gen functions. ~half-day of
  engineering. **The existing seed=42, seed=1, seed=2 MLC ckpts are
  already on `txcdr-base/` — they were trained on bigger pods and
  are fine to use as-is**; the refactor is only needed if we want
  to train additional MLC variants or seeds.

**(3) Marginal at large T because of cache-on-GPU + W+Adam — 4 archs.**
- `txcdr_t32`, `H8 t32`, `SubseqH8 T_max=32`, plus `H8 t28` very
  tight. These need cache moved off GPU. Trivial edit to
  `_train_utils.py` (`pin_memory()` + `non_blocking=True`); the
  cache fits in 46 GB RAM (14 GB used; ~25 GB headroom after
  Python+torch) — this is the *cheap* refactor.

**(4) Comfortably feasible on A40 with current code — 33 archs.**
- All canonical archs at T ≤ 24 single-layer (TXCDR, H8, txc bare),
  plus per-token / TFA / Subseq B2/B4 / both anchor cells. Just set
  `batch_size=1024` instead of 4096.

### Probing on A40

Probing was previously RAM-bound, not VRAM-bound. The plan's "preload
80 GB anchor cache + 400 GB mlc_tail to 188 GB system RAM" is gone
on A40. Per-task streaming returns:

- For each (arch, task), load that one task's cache (~600 MB anchor
  at S=32, ~3 GB mlc_tail), encode, fit probe, write row, free.
- Peak resident memory per probing step: model (~5 GB GPU) + one
  task's cache (~3 GB CPU) + scratch ~1 GB GPU. Well within 46 GB.

The probing driver `run_probing_phase7.py` already streams
per-task (it `.npz`-loads each task fresh), so no code change. We
just lose the H200-pod's ~50–80 hr cumulative-I/O speedup; on A40
expect probing of 38 archs × 36 tasks at S=32, k_feat ∈ {5, 20}
to take **~14–18 hr** instead of ~5–8 hr.

### Storage budget {#storage-budget}

900 GB local quota. Estimated working footprint on A40 if we run
the full canonical retraining (without the 1 lost arch):

| item | size | notes |
|---|---|---|
| `.venv/` | 12 GB | already installed |
| `/workspace/hf_cache` (Gemma-2-2b weights) | ~6 GB | bf16 |
| Activation cache (5 layers × ~14 GB fp16) | 70 GB | rebuild from FineWeb token_ids |
| Probe cache S=32 (36 tasks × ~3.5 GB) | ~120 GB | rebuild with `--include_crosstoken` (closes Agent C's gap) |
| Local ckpt dir (push-and-delete) | ≤ 50 GB | keep last ~30 ckpts; HF stores the rest |
| Misc (logs, plots, transient) | ~10 GB | |
| **Total active footprint** | **~270 GB** | well under 900 GB |

We revert to the *push-then-delete* ckpt pattern from the original
1 TB plan (mentioned in `plan.md` §"Agent A's 5 TB volume" as the
"earlier 1 TB plan"). 50 GB of local ckpt slack is sufficient with
incremental HF push.

### Cost summary on A40 {#cost-summary}

For the four "what's next" items, ordered by ROI:

| item | wall-clock estimate on A40 | new ckpts trained | analysis only |
|---|---|---|---|
| (1) merge agent-c → 2-seed σ on 37 archs | ~0 (done, committed `b15ed08`) | 0 | ✓ |
| (1') backfill `tfa_big__seed1` probing | ~3 hr (cache rebuild + probe) | 0 | n/a |
| (2) reconcile probe vs steering Pareto for `mlc_contrastive` | ~2 hr (pure analysis on existing ckpts + steering rows) | 0 | ✓ |
| (3) TFA re-train at 8000 steps + re-probe | ~4 hr | 1 | n/a |
| (3') TFA `_full` (z_novel + z_pred) re-probe | ~30 min (re-encode only) | 0 | n/a |
| (4) re-train 7 of 11 dropped archs (H8 T=10..24) | ~5–6 hr | 7 | n/a |
| (4') H8 T=28 + cache-streaming refactor | ~2 hr | 1 | n/a |
| (4'') H8 T=32 + txcdr_t32 + Subseq T_max=32 (after CPU-RAM-cache refactor) | ~2–3 hr | 3 | n/a |
| (4''') MLC family seed extensions (after disk-mmap refactor) | half-day refactor + ~3 hr training | up to 6 | n/a |
| permanent loss | — | 1 (SubseqH8 T_max=64) | — |

### Recommended action order on A40

1. Backfill the `--include_crosstoken` gap on agent-c's seed=1/2
   probing first — this is the **highest-ROI item now that data is
   unified**. Rebuild probe cache once with the missing flag, run
   34 archs × 2 FLIP-task probings (~2 hr total). Closes the
   34-task asymmetry and delivers true 2-seed σ on the full 36-task
   leaderboard with zero retraining.
2. Re-probe TFA `_full` (z_novel + z_pred) — fixes a known
   summary.md caveat #4 cheaply.
3. Re-train TFA at 8000 steps — decides whether the 0.794 anomaly
   in summary.md is architectural or under-fit. ~4 hr.
4. Re-train the 7 cleanly-feasible H8 multidistance T=10..24 archs
   to fill the H8 T-sweep tail. Reuse Agent C's seed=1 ckpts where
   they exist — 5 of these 9 already have seed=1 ckpts on
   `txcdr-base/`.
5. (optional) CPU-RAM-cache refactor in `_train_utils.py` —
   ~30 min of code, unlocks 4 marginal archs (H8 T=28+32,
   txcdr_t32, SubseqH8 T_max=32). Pays for itself.
6. (optional) disk-mmap refactor — only if we want to add MLC
   seeds or new MLC variants beyond the existing seed=42/1/2 ckpts.
   Probably skip unless the steering analysis (item 2) finds
   something compelling that needs a new MLC variant.

### Headline conclusion

**One arch is permanently lost on A40: row 49
`phase5b_subseq_h8_T64_s5` (SubseqH8 T_max=64).** It was already
H200-only in the original plan. No other architectural casualty —
every other canonical and dropped arch is reachable on A40 either
with current code or with one of two contained refactors.

The 46 GB RAM cap (vs the originally-planned 188 GB) shifts more
work into refactor territory but doesn't invalidate any *additional*
architectures relative to the VRAM-only analysis. RAM is now the
constraint that matters most for the **MLC family retraining** (cache
no longer fits in either pool) and for the **probing pass** (which
loses its 50–80 hr H200 RAM-cache speedup).
