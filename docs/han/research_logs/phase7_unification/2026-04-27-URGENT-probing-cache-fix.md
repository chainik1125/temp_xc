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

### Files changed — all on `han-phase7-unification` branch

Two commits land the fix; pull both:

1. **`1cdc6d4`** "Phase 7: URGENT — probe cache S=32 rebuild + aggregate_s fix"
   - **NEW**: `experiments/phase7_unification/rebuild_probe_cache_s32.py`
     — reads existing right-padded `probe_cache/` and writes
     per-example-sliced left-aligned `probe_cache_S32/`. Per-example,
     keeps the last `min(32, n_real)` real tokens, left-padded with
     zeros for shorter sentences. Short sentences are NOT broken;
     they keep their data, just with fewer kept windows after the
     padding-prefix mask.
   - `experiments/phase7_unification/run_probing_phase7.py`:
     `aggregate_s` signature changes from `(z_full, last_idx, T, S)` to
     `(z_full, first_real, T, S)`. The new aggregate_s masks windows
     whose left-edge is in `[first_real[i], S-T]`, equivalent semantics
     but coordinated from the LEFT (where real tokens start in the
     left-aligned cache).

2. **`fce6401`** "Phase 7: complete the S=32 cache wiring (encoder + cache loader)"
   - `experiments/phase7_unification/run_probing_phase7.py`:
     `_load_task_cache_p7` rewritten to read from `probe_cache_S32/`
     directly (no longer wraps Phase 5's `_load_task_cache`); returns
     `train_first_real` / `test_first_real` keys.
   - `encode_for_S` returns `(z_full, first_real)`.
   - Main loop wires `first_real_train` / `first_real_test` to
     `aggregate_s`.
   - The flag `USE_S32_CACHE = True` (top of `run_probing_phase7.py`)
     selects the new cache path (`probe_cache_S32/`).

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

### Are previous probing results invalidated?

**Semantically: no.** Same real-token windows get encoded and averaged
either way. But there's one caveat:

| arch family | re-run yields same AUC? | why |
|---|---|---|
| `topk_sae`, `mlc`, `MLCContrastive*` (the per-example TopK ones), TXC family, H8 family | identical | per-example TopK; encode is deterministic regardless of batch composition |
| `tsae_paper_k500/k20` | **may shift slightly** | uses **BatchTopK** — selects top-k across the batch, so changing the encode batch composition (different total windows per task) changes which features fire |

Of the 4 archs Agent A already probed before the fix, only the 2
`tsae_paper` are at risk of small shifts. To be safe AND consistent,
**re-run all 38 archs** with the new code rather than splice old +
new rows.

### Action for Agent C

**STOP your probing pass immediately if it's running.** It's wasting
~4-5× of the time it should.

Then:

```bash
# 1. Pull both fix commits
cd /workspace/temp_xc
git fetch origin
git merge origin/han-phase7-unification
# Should bring in 1cdc6d4 + fce6401 (URGENT cache fix + encoder wiring)

# 2. Run the cache rebuild (per-example slicing). Reads existing
#    probe_cache/ that you already have on disk, writes new
#    probe_cache_S32/. ~5-15 min on H100.
TQDM_DISABLE=1 PHASE7_REPO=/workspace/temp_xc \
    /workspace/temp_xc/.venv/bin/python -u \
    -m experiments.phase7_unification.rebuild_probe_cache_s32

# 3. Restart probing — code now defaults to USE_S32_CACHE=True
#    so it auto-uses probe_cache_S32/.
TQDM_DISABLE=1 PHASE7_REPO=/workspace/temp_xc PYTHONUNBUFFERED=1 \
    /workspace/temp_xc/.venv/bin/python -u \
    -m experiments.phase7_unification.run_probing_phase7 --headline
```

### Speedup expected

Probing time drops ~4× (e.g., 42 hr → ~10 hr for the 38 trimmed-
canonical archs at seed=1 on Agent C's H100). The encode cost was
dominated by the LAST_N of the cache (128), not the aggregation S.
Now that the cache is at LAST_N=32 = aggregation S, encode and
aggregation costs match.

### Backward compatibility

The flag `USE_S32_CACHE = True` (top of `run_probing_phase7.py`)
gates the behaviour. Setting it `False` falls back to the old
`probe_cache/` + `aggregate_s(z, last_idx, ...)` path, but that path's
`aggregate_s` signature changed too — it now takes `first_real`. So
the old path is effectively broken for anyone still on the old cache.

If you have older results that need to be re-checked at S=128 or S=64,
that's a separate downstream effort (would need to keep the old
loader code separately).
