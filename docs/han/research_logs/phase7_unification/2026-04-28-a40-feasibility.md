---
author: Han
date: 2026-04-28
tags:
  - design
  - in-progress
---

## A40 + 503 GB / 900 GB feasibility for Phase 7 retraining

### Hardware delta vs original Phase 7 plan

|  | original (H200 pod) | new (A40 pod) | ratio |
|---|---|---|---|
| GPU | NVIDIA H200, 141 GB HBM3e | NVIDIA A40, 46 GB GDDR6 | **0.33×** VRAM |
| System RAM | 188 GB | **503 GB** | 2.7× RAM |
| Persistent storage | 5 TB | 900 GB (per-pod quota) | 0.18× |

Two consequences upend the original "leverage H200" tactics in
`plan.md` §"How Agent A should leverage the H200 pod":

- **VRAM is the binding constraint, not RAM.** The original plan
  preloaded the full activation cache (24 000 seqs × 128 tok × d_in
  × fp16 = 14.2 GB single-layer / **70.8 GB multi-layer**) directly
  to GPU and bumped `batch_size` 1024 → 4096. On A40 those choices
  no longer fit.
- **System RAM grew, not shrank.** The 503 GB system RAM is *more*
  than the H200 pod had. Anything memory-bound that previously had
  to live in VRAM (cache, large probe-cache load) now fits comfortably
  in CPU RAM and can be streamed to GPU. The cost is per-step host→
  device bandwidth (PCIe 4.0 ~32 GB/s on A40) but the cache fits
  uncompressed in RAM headroom.

The 900 GB storage cap is roughly 5× tighter than the original 5 TB.
That breaks the "keep all 147 ckpts on local disk forever" assumption
in `plan.md` §"Agent A's 5 TB volume" — see [§ Storage budget](#storage-budget).

### Per-arch GPU memory model

Training-time peak VRAM ≈ `preload + W + Adam + workspace`.

- `preload` = activation cache copied to GPU at startup. With
  PRELOAD_SEQS=24 000, ctx=128, d_in=2304, fp16:
  - single-layer (per-token / TXC / H8 / Subseq archs): **14.2 GB**
  - 5-layer (MLC family): **70.8 GB** — already exceeds A40 by itself.
- `W + Adam` = parameter + Adam (m, v) state, fp32, 12 bytes/param.
  For each arch:
  - per-token / per-layer crosscoder weight: 2 × d_in × d_sae = 85M params
  - TXCDR / H8 / Subseq weight: 2 × T × d_in × d_sae (encoder + full decoder)
  - H8 adds matryoshka H decoder: T × d_in × 0.2·d_sae (≈ 10–20% extra)
- `workspace` ≈ 2–3 GB at batch=4096 (forward activations, backward
  buffers, optimizer scratch).

Computed `W + Adam` per arch family:

| arch | W + Adam | dominant term |
|---|---|---|
| topk_sae, tsae_paper_*, mlc, mlc_contrastive, agentic_mlc_08 | **5.1 GB** | 2·d_in·d_sae or 5·d_in·d_sae |
| txc_bare/agentic_txc T=5 | 5.1 GB | T=5 |
| txc T=10 | 10.2 GB | |
| txc T=20 (incl txcdr_t20_kpos100 anchor) | 20.4 GB | |
| txc T=24 | 24.4 GB | |
| txc T=28 | 28.5 GB | |
| txc T=32 | **32.6 GB** | |
| H8 T=3..9 | 3.4–11.2 GB | |
| H8 T=10 | 11.2 GB | adds matryoshka H |
| H8 T=20 | 22.4 GB | |
| H8 T=28 | 31.4 GB | |
| H8 T=32 | **35.9 GB** | T·d_in·(d_sae + d_sae + 0.2·d_sae) |
| Subseq T_max=10 | 10.2 GB | weights stored at full T_max |
| **SubseqH8 T_max=32** | **32.6 GB** | (row 48) |
| **SubseqH8 T_max=64** | **65.2 GB** | (row 49) |

### Feasibility on A40 by category

Conservative budget on A40: **46 GB total – 14.2 GB cache preload –
3 GB workspace ≈ 28.8 GB available for `W + Adam`** with the
plan's existing GPU-preload pattern + batch=4096. (This is a
ceiling; in practice CUDA fragmentation eats another 1–2 GB.)

**A. Definitely won't fit (already documented as H200-only):**

- Row 49 — `phase5b_subseq_h8_T64_s5` (SubseqH8 T_max=64). 65 GB
  weights+Adam alone exceeds A40. This was always H200-only per
  `plan.md` §"Run SubseqH8 at T_max ∈ {32, 64, 128}". Leave dropped.

**B. Won't fit with the existing GPU-preload pattern:**

- Rows 4–6 — MLC family (`mlc`, `mlc_contrastive_alpha100_batchtopk`,
  `agentic_mlc_08`). The 5-layer `preload_multilayer` is 70.8 GB
  on its own — bigger than A40 entirely. Fixable by streaming the
  cache from CPU RAM (we have 503 GB; cache fits 7× over). See
  [§ Path to recovery](#path-to-recovery).
- Row 49 SubseqH8 T_max=64 (above) — even with cache streaming,
  weights+Adam alone (65 GB) exceeds A40. Permanently dropped.

**C. Marginal: fits only at reduced batch_size + with care:**

These have `cache + W + Adam` ≥ 46 GB at batch=4096. Dropping
`batch_size` 4096 → 1024 (the Phase 5 baseline) saves 2–3 GB of
per-batch fwd/bwd workspace, which is enough to recover most:

- Row 29 — `txcdr_t32`. Total ≈ 50 GB at batch=4096; ~47 GB at
  batch=1024 — borderline. Need either bf16 weights, gradient
  checkpointing, or cache streaming.
- Row 48 — `phase5b_subseq_h8_T32_s5`. Same arithmetic as txcdr_t32.
- Row 28 — `txcdr_t28` at batch=4096 ≈ 46 GB; batch=1024 → ~43 GB ✓.
- Row 44 — `phase57_partB_h8_bare_multidistance_t28`: at batch=4096
  ≈ 48 GB; batch=1024 → ~45 GB ✓ (very tight).
- Row 45 — `phase57_partB_h8_bare_multidistance_t32`. ~53 GB at any
  batch ≥ 1; needs cache streaming or bf16 to fit.

**D. Comfortably fits at batch=4096:**

- All other 35 archs (including all H8 T ≤ 24, all TXCDR T ≤ 24, all
  TXC bare antidead, all per-token/SAE/TFA, both anchor cells).

### Specific reads on the four "what's next" items

1. **Multi-seed σ for full 38 trimmed canonical (item 1).** *No
   training needed* — done by the Agent C unification merge committed
   today. Remaining gap is `tfa_big__seed1` only. See
   [§ Cost summary](#cost-summary) for whether it's worth filling.

2. **TFA re-train at 8000 steps (item 3).** TFA's `_full` arch is
   per-token (single-layer preload) with attention over a 128-token
   sequence. Weights+Adam ≈ 5 GB; attention scratch with
   `tfa_batch_size=64` (already in `train_phase7.py`:317) ≈ 1–2 GB.
   Total ~22 GB. **Comfortably fits on A40.** Wall-clock at 8000
   steps: extrapolating from Phase 5's `tfa_big` at ~50 min/training
   on H200, expect **~3–4 hr on A40** (roughly 5× slowdown for
   attention-heavy ops on A40 vs H200).

3. **Probe the 11 dropped archs (item 4).** Per the table above,
   on A40:
   - 9 H8 multidistance T=10..32 (rows 37–45):
     - T=10..24: feasible at batch=1024, no code changes (~30–45 min
       each on A40).
     - T=28: tight, may need bf16.
     - T=32: needs cache streaming or bf16; do not attempt without
       refactor.
   - SubseqH8 T_max=32 (row 48): tight, same as TXCDR T=32.
   - SubseqH8 T_max=64 (row 49): permanently dropped.

   Effective recoverable scope: **8 of 11 dropped archs** (H8 T=10..24
   plus T=28 with bf16) are reachable on A40 without major refactors.

4. **Reconcile sparse-probing vs steering Pareto (item 2).** Pure
   analysis on existing ckpts + probing rows; no GPU constraint.

### Path to recovery — the streaming option

The `preload_single` / `preload_multilayer` pattern in
`_train_utils.py` is a one-line `.to(device)` after a `np.load`. It
fits H200 because 14 / 70 GB easily lives in 141 GB VRAM. On A40 the
recovery is straightforward but requires touching `_train_utils.py`:

```python
# Original (Phase 7 H200): cache lives on GPU
return torch.from_numpy(sub).to(device)

# A40 variant: cache stays in pinned host RAM; sample generators
# pull batches via async DMA. With batch=1024, T=32 the per-step
# transfer is 1024 × 32 × 2304 × 2 B = 150 MB — well under PCIe 4.0
# bandwidth (32 GB/s ⇒ 5 ms per transfer, ~negligible vs ~50 ms
# step time for the d_sae=18432 forward).
return torch.from_numpy(sub).pin_memory()  # CPU-resident, pinned
```

Then `make_window_gen_gpu` etc. need a `.to(device, non_blocking=True)`
inside the generator instead of operating on a GPU tensor. Memory-
saving on a single-layer cache: 14.2 GB freed; on multi-layer:
70.8 GB freed.

After streaming, the binding constraint becomes pure `W + Adam`:

| arch | W + Adam | A40 with streaming? |
|---|---|---|
| txcdr_t32 | 32.6 GB | ✓ (~10 GB headroom) |
| H8 t32 | 35.9 GB | ✓ (~7 GB headroom) |
| SubseqH8 T_max=32 | 32.6 GB | ✓ |
| SubseqH8 T_max=64 | **65.2 GB** | **✗ permanent** |
| MLC family | 5.1 GB + small per-layer scratch | ✓ |

So a streaming refactor unlocks every canonical arch *except* row 49.

### Storage budget {#storage-budget}

900 GB local. Estimated working footprint if we re-run the full
canonical retraining on A40:

| item | size | notes |
|---|---|---|
| `.venv/` | 12 GB | already installed |
| `/workspace/hf_cache` (Gemma-2-2b weights) | ~6 GB | bf16 |
| Activation cache (5 layers × ~14 GB fp16) | 70 GB | rebuild from FineWeb token_ids |
| Probe cache S=32 (36 tasks × ~3.5 GB) | ~120 GB | rebuild with `--include_crosstoken` (closes Agent C's gap) |
| Local ckpt dir (push-and-delete) | ≤ 50 GB | keep last ~30 ckpts; HF stores the rest |
| Misc (logs, plots, transient) | ~10 GB | |
| **Total active footprint** | **~270 GB** | well under 900 GB |

The original 5 TB plan kept all 147 ckpts on disk (~100–150 GB). On
the A40 pod we revert to the *push-then-delete* pattern from the 1 TB
plan (now described in `plan.md` §"Agent A's 5 TB volume" as
"earlier 1 TB plan"). 50 GB of local ckpt slack is sufficient with
incremental HF push.

### Cost summary on A40 {#cost-summary}

For the four "what's next" items, ordered by ROI:

| item | wall-clock estimate on A40 | new ckpts trained | analysis only |
|---|---|---|---|
| (1) merge agent-c → 2-seed σ on 37 archs | ~0 (done) | 0 | ✓ |
| (1') backfill `tfa_big__seed1` probing | ~3 hr (cache rebuild + probe) | 0 | n/a |
| (2) reconcile probe vs steering Pareto for `mlc_contrastive` | ~2 hr (pure analysis on existing ckpts + steering rows) | 0 | ✓ |
| (3) TFA re-train at 8000 steps + re-probe | ~4 hr | 1 | n/a |
| (3') TFA `_full` (z_novel + z_pred) re-probe | ~30 min (re-encode only, no retrain) | 0 | n/a |
| (4) re-train 8 of 11 dropped archs (H8 T=10..24, T=28-bf16) | ~5–6 hr | 8 | n/a |
| (4') `phase57_partB_h8_bare_multidistance_t32` (needs streaming) | ~1 hr refactor + ~1 hr training | 1 | n/a |
| permanent loss | — | 1 (SubseqH8 T_max=64) | — |

Net: of the 11 dropped archs in `summary.md` caveat #2, **9 are
recoverable on A40 (8 cleanly + 1 with streaming refactor); 2
require code work (T_max=32) or are permanent loss (T_max=64)**.

### Recommended action order on A40

1. Backfill the `--include_crosstoken` gap on agent-c's seed=1/2
   probing first (rebuild probe cache once, run 34 archs × 2
   FLIP-task probings ≈ ~2 hr). This delivers true 2-seed σ on
   the full 36-task set with zero retraining.
2. Re-probe TFA `_full` (cheap, fixes a known caveat).
3. Re-train TFA at 8000 steps (decides whether the 0.794 anomaly
   in `summary.md` is architectural or under-fit).
4. Re-train the 8 cleanly-feasible H8 multidistance T=10..28 archs
   to fill the H8 T-sweep tail; reuse Agent C's seed=1 ckpts where
   available (5 of these 9 already have seed=1 ckpts on
   `txcdr-base/ckpts/`).
5. Decide later whether the streaming refactor is worth the engineering
   for the 3 marginal archs (H8 T=32, txcdr_t32 with σ, SubseqH8
   T_max=32). Probably yes — touches one file (`_train_utils.py`)
   and unlocks the full T-sweep tail.

### What's permanently lost vs what only needs effort

**Permanently lost on A40:**
- SubseqH8 T_max=64 (row 49). Was always plan-flagged as H200-unique
  per `plan.md` §"Run SubseqH8 at T_max ∈ {32, 64, 128}".

**Needs ~1 hr code work (cache streaming refactor) but otherwise fine:**
- MLC family training-from-scratch. (Existing seed=42, seed=1, seed=2
  ckpts on HF are unaffected — they were trained on H200/H100 already.
  Refactor only matters if we want to add seeds or new variants.)
- H8 T=32, txcdr_t32, SubseqH8 T_max=32.

**Fits comfortably on A40 with current code:**
- 35 of 38 trimmed canonical archs.
- Both anchor cells.
- TFA re-train.
- All probing (probing has no preload, only probe-cache load — ~80 GB
  in CPU RAM, no GPU pressure beyond per-task encoding which uses
  ~5 GB).

The headline conclusion is mild: the only *actually* lost arch is the
single H200-exclusive SubseqH8 T_max=64. Everything else is either
cheap to do on A40 or needs a contained streaming refactor.
