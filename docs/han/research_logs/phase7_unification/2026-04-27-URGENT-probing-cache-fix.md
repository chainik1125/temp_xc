---
author: Han
date: 2026-04-27
tags:
  - design
  - in-progress
priority: URGENT
---

## URGENT: Probe cache must be regenerated at S=32 — Agent C action required

### Symptom

Agent A's probing pass at S=32 was running ~75 min/arch — extrapolated to
~42 hr to finish 38 archs. **Way too slow.**

### Root cause

The probe cache files were built at LAST_N=128 (the original S=128 plan).
After we dropped to S=32, the probing code still loads the full 128-token
cache and encodes ALL 128−T+1 windows per example, even though
`aggregate_s` only consumes the last ~32−T+1 of those windows.

- For window arch at T=10: encode 119 windows per example → use only ~23.
  **~5× wasted GPU work.**
- For MLC: load 17.5 GB `mlc_tail.npz` per task, cast to fp32 (35 GB),
  encode all 128 per-token positions → use only last 32.
  **~4× wasted I/O + compute.**

### Fix (Agent A is implementing now)

**Per-example slice the existing cache to the last 32 REAL tokens per
example**, left-aligned with zeros padding the start (for short sentences
that have <32 real tokens).

Doesn't break short sentences: each example keeps `min(32, n_real)` real
tokens; never any padding. The padding is just at the START of the new
32-frame for short sentences.

### Files changed (commit pending)

1. `experiments/phase7_unification/rebuild_probe_cache_s32.py` — new script
   that reads the existing right-padded `probe_cache/` and writes
   per-example-sliced left-aligned `probe_cache_S32/`.

2. `experiments/phase7_unification/run_probing_phase7.py`:
   - `aggregate_s` signature changes from `(z_full, last_idx, T, S)` to
     `(z_full, first_real, T, S)`. The semantics are equivalent but
     coordinate from the LEFT (where real tokens start) instead of from
     the END.
   - `_load_task_cache_p7` reads from `probe_cache_S32/` and returns
     `train_first_real` / `test_first_real` instead of `train_last_idx`
     / `test_last_idx`.
   - `encode_for_S` callers use `first_real` per example.

### Action for Agent C

**STOP your probing pass immediately if it's running.** It's wasting
~4-5× of the time it should.

Then:

```bash
# 1. Pull this commit (Agent A's commits land at:
#    https://github.com/chainik1125/temp_xc/tree/han-phase7-unification)
git fetch origin
git merge origin/han-phase7-unification

# 2. Run the cache rebuild on Agent C's pod:
TQDM_DISABLE=1 PHASE7_REPO=/workspace/temp_xc \
    /workspace/temp_xc/.venv/bin/python -u \
    -m experiments.phase7_unification.rebuild_probe_cache_s32

# Output: experiments/phase7_unification/results/probe_cache_S32/
# Should take ~5-15 min (per-example slicing of 36 task dirs).

# 3. Restart probing — code now defaults to USE_S32_CACHE=True
TQDM_DISABLE=1 PHASE7_REPO=/workspace/temp_xc PYTHONUNBUFFERED=1 \
    /workspace/temp_xc/.venv/bin/python -u \
    -m experiments.phase7_unification.run_probing_phase7 --headline
```

### What this means for Agent C

- **All previous S=32 / S=64 / S=128 probing rows in
  `probing_results.jsonl` should be regenerated** (the new cache fixes a
  real efficiency bug; semantically the AUCs should be the same, but
  re-running gives a clean dataset).
- The fix also drops `last_idx` from the cache schema — `first_real` is
  the new coordinate. Don't try to re-use old loader code with new
  cache files; will mismatch.
- Speedup: probing time drops ~4× (e.g., 42 hr → ~10 hr for the 38
  trimmed-canonical archs).

### Backward compatibility

The flag `USE_S32_CACHE = True` (top of `run_probing_phase7.py`)
gates the behaviour. Setting it `False` falls back to the old
`probe_cache/` + `aggregate_s(z, last_idx, ...)` path, but that path's
`aggregate_s` signature changed too — it now takes `first_real`. So
the old path is effectively broken for anyone still on the old cache.

If you have older results that need to be re-checked at S=128 or S=64,
that's a separate downstream effort (would need to keep the old
loader code separately).
