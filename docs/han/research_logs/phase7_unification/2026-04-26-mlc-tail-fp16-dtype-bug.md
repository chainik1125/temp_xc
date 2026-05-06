---
author: Han
date: 2026-04-26
tags:
  - results
  - complete
---

## Bug: MLC probing crashes on `expected Half but found Float`

### Summary

`run_probing_phase7._load_task_cache_p7` loaded the new `mlc_tail`
field straight from the .npz without casting to fp32. The .npz
stores fp16 (probe cache builder writes fp16 to save disk), so
the input tensor reaching the MLC model's `encode` was fp16 while
the model parameters are fp32 → `torch.einsum("bld,lds->bs", x, W)`
fails with `RuntimeError: expected scalar type Half but found Float`.

### Discovery

Triggered by the sanity sentinel firing the targeted probing on the
3 sanity-check archs (mlc, phase5b_subseq_h8, txcdr_t5) once
`txcdr_t5__seed42` landed. First probe call (`mlc__seed42` on
`ag_news_business`) crashed.

### Root cause

Phase 5's `_load_task_cache` (which Phase 7 imports + extends) DOES
cast its anchor cache fields:

```python
"anchor_train": anchor["train_acts"].astype(np.float32),  # ← cast
"anchor_test": anchor["test_acts"].astype(np.float32),
"mlc_train": mlc["train_acts"].astype(np.float32),        # ← cast
"mlc_test": mlc["test_acts"].astype(np.float32),
```

Phase 7 added a NEW field `mlc_tail` (the new (N, 128, 5, d) cache
needed for MLC mean-pool aggregation at S=128) via a wrapper
`_load_task_cache_p7`. The wrapper forgot the `.astype(np.float32)`
cast that the original Phase-5 loader had on every other field.

### Fix (commit pending)

`run_probing_phase7._load_task_cache_p7` now casts mlc_tail to
fp32 — same convention as Phase 5's anchor cache.

```python
base["mlc_tail_train"] = z["train_acts"].astype(np.float32)
base["mlc_tail_test"] = z["test_acts"].astype(np.float32)
```

Verified: `_encode_mlc_per_token_z` now runs cleanly on
`mlc__seed42` × `ag_news_business` → output (8, 128, 18432) fp32.

### Lessons

- When extending an existing data loader with new fields, mirror
  ALL conventions of the original loader (fp32 cast, key naming,
  shape ordering). Easy to forget one.
- Future `_load_task_cache_p7` extensions should add a unit test
  asserting `dtype == np.float32` on every returned array. (Tracked
  as a low-priority follow-up; not blocking the seed=42 probing
  pass now that the fix is in.)
- Architectures' `encode` paths should ideally tolerate either
  dtype via an explicit `x = x.to(self.W_enc.dtype)` cast at the
  top — defensive against load-side bugs like this. Not changing
  it now (would touch many arch files); but worth considering as a
  framework-wide convention if more dtype bugs surface.

### Related: anchor cache works fine

Window archs (txcdr_t5 etc.) use `_encode_window_z` → reads from
`anchor_train` which was already fp32-cast by Phase 5's loader. So
window-arch probing was unaffected. Only MLC family hit this.
