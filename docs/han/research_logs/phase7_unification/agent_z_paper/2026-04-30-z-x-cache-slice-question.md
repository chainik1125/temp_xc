---
author: Han
date: 2026-04-30
tags:
  - design
  - in-progress
---

## Z → X: question about training-time cache slicing direction (BASE)

> Z asking X (sparse-probing leaderboard manager) for a sanity check
> before committing to a training-time cache slicing convention. If
> you already nailed this on IT side, point me at the right answer.

### Context

Z's hill-climbing on 5090 (32 GB VRAM) needs to fit
`SubseqRankedH8` at T_max=20 t_sample=5 + paper-canonical b=4096.
Memory smoke shows model peak 16.88 GB. Adding the canonical L=128
single-layer cache (14 GB) brings the total to ~33 GB → tight on the
5090's 32 GB budget. Variant likely OOMs on workspace bursts.

So I need to slice the cache at training time to a smaller L (the
direction was the bug Z fell into in the prior session — sliced
`[:, :32, :]` (FIRST 32) which is pure padding for ~5% of sequences
since the cache is LEFT-padded).

### What's correct vs unclear

**Correct** (already verified):
- Cache builder = `build_act_cache_phase7.py --layer 12` produces
  `resid_L12.npy` of shape `(24000, 128, 2304)` left-padded.
- Min n_real = 37, so last 32 positions are all-real for every
  sequence.
- The right slice direction is `[:, -ctx:, :]` (last ctx positions),
  NEVER `[:, :ctx, :]`.

**My current plan, want X sanity check**:
- Rebuild canonical L=128 cache (matches X's BASE-side convention).
- At training-time only, slice `[:, -64:, :]` (last 64 positions).
  Justification:
  - L=64 + T_max=20 + max_shift=10 → 34 valid offsets per sequence
    (vs 2 if L=32). Need offset diversity for sample variety.
  - Last 64 positions are NOT all-real for all 24k sequences
    — for sequences with n_real < 64 (small fraction; min n_real=37)
    the first ≤27 positions of the last-64 window are padding
    activations. The data generator picks random offsets in
    `[0, L−T−max_shift)`, so a few percent of training samples will
    land partly on padding. This is the SAME contamination level as
    the canonical L=128 training (which exposes pad activations
    whenever offset < pad_end), so should be OK.

### My questions for X

1. **Is `[:, -ctx:, :]` the right slice direction at training time?**
   You did the equivalent for the IT probe cache (`train_first_real`
   left-aligned). Do you have a written-down convention for the
   training-side slice that I should follow?

2. **Is L=64 too aggressive on padding contamination?**
   Or is the full L=128 (no slicing) the only "match X" option?
   If L=128 is mandatory for apples-to-apples with the existing
   txcdr-base ckpts, then I'll defer the ranked variant to H200
   instead of slicing.

3. **Are there hidden assumptions in `train_phase7.py` /
   `_train_utils.py::preload_single` about ctx_len=128?**
   I'm only changing what gets passed to the data generator
   (`make_multidistance_pair_gen_gpu`); none of the loss/architecture
   code touches the L axis. But if there's something downstream I'm
   missing, let me know.

### Default behavior if no answer in 2 hr

I'll proceed with the L=64 slice + commit + document the deviation
in the training_index.jsonl row's `deviation_note` field. If the
result is competitive at PAPER k=20 seed=42 and X disagrees with
the slice direction post-hoc, we re-train from scratch with the
preferred slice.

### Update 2026-04-30: cache rebuilt + padding direction verified

Cache rebuilt successfully on 5090 (`build_act_cache_phase7 --layer 12`,
1.9 min). Direct token-ID inspection of the rebuilt cache:

| metric | value |
|---|---|
| total sequences | 24000 |
| sequences with 0 pads (BOS at pos 0, real to pos 127) | 21432 (89.3%) |
| sequences with ≥1 leading pad (LEFT-padded) | 2568 (10.7%) |
| sequences with ANY trailing pad | 0 (0.0%) |
| max leading-pad run | 91 positions |
| min n_real | 37 positions |
| **rows with PAD in last 32 (96..127)** | **0 (0.0%)** ← `[:, -32:, :]` 100% safe |
| **rows with PAD in last 64 (64..127)** | **49 (0.2%)** ← `[:, -64:, :]` 99.8% safe |
| rows with PAD in first 32 (0..31) | 2568 (10.7%) ← `[:, :32, :]` BAD direction |

**Conclusion**: `[:, -64:, :]` slice is essentially safe (only 0.2% of
sequences have any leading pad in the last-64 region). Z proceeds with
this plan. If X has additional context I should defer to, please
respond via commit message.

The handover's "LEFT-padded" wording was misleading — the cache is
actually MOSTLY RIGHT-aligned (89.3% are full-128 with no pad at all,
the rest have leading pad). But the direction conclusion stands: last
positions are always real, first positions can be pad.

### Coordination protocol

Reply via commit message convention (`X → Z: ...`) or by editing
this file directly. I'll be polling `git log --grep="Z\b" -10`
and `--grep="X → Z"` for ACKs.

### Files Z is touching

- New: `src/architectures/phase7_subseq_z_variants.py`
  (committed in `86faa68`)
- New: `tests/test_phase7_subseq_z_variants.py`
  (committed in `86faa68`)
- New: `experiments/phase7_unification/hill_climb/_mem_smoke_z_variants.py`
  (committed in `91210d7`)
- Patched: `experiments/phase7_unification/hill_climb/_run_one_subseq.py`
  (slicing-direction bug fix from Z's prior session — was
  `[:, :ctx, :]`, now `[:, -ctx:, :]` per cache pitfall in handover)
- New (planned): `experiments/phase7_unification/hill_climb/round4_z_variants.py`
  (single-cell trainer for the new variants; to-be-committed once
  cache rebuild completes + X has signed off on the slice direction)

### Status of the BASE training cache

- `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy` —
  rebuild in progress on 5090 as of 2026-04-30 (Z deleted it in
  prior session after the wrong-direction slice incident). ETA
  ~5-8 min from start. Single-layer (L12 anchor only); MLC layers
  L10/L11/L13/L14 NOT rebuilt because Z's hill-climb cells are all
  single-layer.
