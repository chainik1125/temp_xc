---
author: Han
date: 2026-04-27
tags:
  - results
  - in-progress
---

## TFA encode-path bug found mid-probing (commit 3e02962)

### Symptom

During the seed=42 non-MLC probing pass, every task on `tfa_big__seed42`
hit:

```
encode FAIL AttributeError: 'TemporalSAE' object has no attribute 'encode'
```

— skipping all 36 tasks for tfa_big.

### Root cause

Phase 7's `run_probing_phase7.py:encode_for_S()` lumped `TemporalSAE`
into the per-token branch alongside `TopKSAE` and
`TemporalMatryoshkaBatchTopKSAE`, which all expect a `model.encode(x)`
method. But `TemporalSAE` (in `src/architectures/_tfa_module.py`) only
defines `forward(x)` — it returns `(recons, results_dict)` where
`results_dict["novel_codes"]` and `results_dict["pred_codes"]` are the
latent activations.

### Why per-token wouldn't have been right anyway

TFA uses cross-attention across the sequence. Encoding token-by-token
in isolation (which is what `_encode_per_token_z` does — it flattens
to `(N*S, 1, d)`) would zero out z_pred (no context), leaving only
z_novel — that's a degenerate version of TFA that throws away its
defining mechanism.

### Fix

`run_probing_phase7.py`:

1. Added `_encode_tfa_z(model, anchor, scaling, device)`: batches the
   full `(B, S=32, d)` anchor through `model.forward()`, returns
   `inter["novel_codes"]` (per Phase 5's `tfa_big` convention; the
   `_full` variant adds pred_codes but Phase 7 picks the conservative
   novel-only path to match the rest of the leaderboard's k_win
   sparsity budget).

2. Added `_tfa_scaling_for_run(run_id)`: reads `scaling_factor` from
   `training_logs/<run_id>.json`. TFA was trained on `x * scaling`
   (line 383 of train_phase7.py:`train_tfa`), so probe-time inputs
   need the same scaling. Phase 5's TFA probe path did the same
   (run_probing.py:1105).

3. Pulled `TemporalSAE` out of the per-token dispatch branch in
   `encode_for_S` and gave it its own branch.

### Re-probe plan

- The currently-running probing process (PID 25119) loaded the buggy
  code into Python before the fix landed. It will keep failing tfa_big
  per-task, then move on. tfa_big rows: 0 in the current pass.
- The post-chain orchestrator (PID 26577, `/tmp/probe_mlc_after.sh`)
  was extended to include `tfa_big__seed42` in its `--run_ids` list.
  When non-MLC probing + MLC training both finish, the orchestrator
  will spawn a fresh Python process that imports the patched code,
  and re-probe tfa_big together with the 9 MLC ckpts.
- Estimated extra cost: ~12 min for tfa_big + ~2 hr for 9 MLC ckpts
  = same orchestrator wall clock as before.

### Caveat for analysis

When reading `probing_results.jsonl`, the `tfa_big__seed42` rows will
have post-fix timestamps (later in the file) than other arches. No
dedupe needed since pre-fix rows weren't written (they only failed,
producing nothing). The leaderboard logic just needs to filter
`skipped=true` rows as usual.
