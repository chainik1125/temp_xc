# Variance-receipt recipes per Stage-2 panel (runpod-b, panel-support-audit item 1)

The harness is pre-flighted against BOTH new panels' row shapes
(`tests/test_stage2_variance_panels.py`). Run the exact command for your
panel — the flags are not optional:

- Your rows are **paired layout** (every row carries `lambda_probe_v2`
  in eval_cfg + BOTH column sets). `--row-layout auto` resolves this;
  the resolved layout is recorded in the output `source` block.
- Your post arm runs at **k_pos = 8·T from the start** —
  `--post-k-rule times-T` is REQUIRED. Without it the run aborts on
  "key sets differ" (with a hint); it does not silently drop post.
- Claim on v1; run the paired v2 command too (METHODS DECISION
  2026-07-25: v1 canonical, report paired v2). Same crosscheck JSON
  serves both — paired-layout rows carry both column sets.
- Seed population defaults to `1,2,42`; stray extra-seed rows (top-ups)
  are excluded automatically.

## runpod-e — fineweb primary (gemma-2-2b)

```bash
.venv/bin/python -m experiments.explorations.task_hunt.support_stats.stage2_variance \
  --ds fineweb_punctint_q_gemma2_l14 --probe v1 --post-k-rule times-T \
  --crosscheck-json experiments/explorations/task_hunt/qrate_fineweb/results/stage2_fineweb_punctint_q_gemma2_l14.json \
  --out-prefix stage2_variance_qrate_gemma
# paired v2 columns (same rows, same crosscheck):
.venv/bin/python -m experiments.explorations.task_hunt.support_stats.stage2_variance \
  --ds fineweb_punctint_q_gemma2_l14 --probe v2 --post-k-rule times-T \
  --crosscheck-json experiments/explorations/task_hunt/qrate_fineweb/results/stage2_fineweb_punctint_q_gemma2_l14.json \
  --out-prefix stage2_variance_qrate_gemma_v2
```

## runpod-e — replication cells (gpt2 / llama31, two T values)

Same command with `--ds fineweb_punctint_q_gpt2_l7` (crosscheck
`.../stage2_fineweb_punctint_q_gpt2_l7.json`, out-prefix
`stage2_variance_qrate_gpt2[_v2]`), and likewise for
`fineweb_punctint_q_llama31_l14`. Two-T populations degrade by design:
cells + paired diffs + margins are reported, the T-trend is **skipped
with the reason stated** (a trend over two points is undefined — do not
try to force one), and the seed recommendation keys on the largest
available T. No post/stacked arm exists there; `times-T` is harmless.

## runpod-d — oprate `rate_case` (Ward)

Substitute the datasource key YOUR frozen card declares (the one your
`run_stage2` passes to the runner), and the results JSON path your
runner writes — everything else stays:

```bash
.venv/bin/python -m experiments.explorations.task_hunt.support_stats.stage2_variance \
  --ds <YOUR_OPRATE_CASE_DS_KEY> --probe v1 --post-k-rule times-T \
  --crosscheck-json experiments/explorations/task_hunt/oprate/results/stage2_<YOUR_OPRATE_CASE_DS_KEY>.json \
  --out-prefix stage2_variance_oprate_case
# paired v2: --probe v2 --out-prefix stage2_variance_oprate_case_v2
```

If you take `rate_ver` to a panel, repeat with that ds key and
`--out-prefix stage2_variance_oprate_ver[_v2]`.

## If something still aborts

Every abort is a one-line diagnosis (duplicate cell / key-set diff with
a post-arm hint / incomplete population listing the missing cells /
0-rows-selected with the layout suspects). Fix the population or the
flag it names; do not edit the harness mid-panel. Failed (`ok: false`)
crosscheck rows are skipped loudly and counted in the output — a panel
with failures is reportable as partial, but rerun them before quoting
headline cells.
