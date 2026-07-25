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

---

# When your results land — pre-staged analysis (panel-support-audit item 4)

Written BEFORE either panel finished, so nobody writes analysis code
tired against the deadline. One command per step; if a number surprises
you, stop and look, don't patch.

## Expected row decomposition (check BEFORE any statistics)

**Full panel (oprate case / fineweb gemma): 84 leaderboard rows =**
14 (arch, T) line points × 3 seeds × {trained, untrained}:
`batchtopk_sae/T1` + `tsae/T1` + `stacked_batchtopk/T{2,4,8,16}` +
`txc_batchtopk_pre/T{2,4,8,16}` + `txc_batchtopk_post/T{2,4,8,16}`.
Post rows carry k_pos = 16/32/64/128 (both kinds — the untrained-post
realized l0 ≈ 8.00 falsifier rides on this); all other rows k_pos = 8.
EVERY row: `lambda_probe_v2` flags in eval_cfg + BOTH column sets in
metrics + `l0_per_token`. 0 dup eval_keys, 0 null metrics.
**Replication (gpt2 / llama31): 24 rows** = (sae/T1 + tsae/T1 +
pre/T×2) × 3 seeds × 2 kinds; no post, no stacked.

The variance harness enforces most of this (dup guard, completeness
check, exact cross-check vs your results JSON) — running it IS the
population audit. What it does not check: realized-l0 bands (card-side)
and NaN-drop counts (report per T, oprate binding 2).

## The order of operations (per datasource)

1. `git pull --rebase` then the two harness commands from your section
   above (v1 first — canonical — then v2). Outputs land in
   `support_stats/` as `<out-prefix>.{json,md}`; commit them with the
   invocation in the commit message.
2. Read your `.md` receipt against the skeleton below; fill the
   scorecard INTO your LOG verdict entry.
3. Any number you intend to quote in the rebuttal: add a row to
   `../RECEIPTS.md` via `../receipts_check.py` (claim, artifact+key,
   quoted value) and re-run it — it must print ALL PASS before the
   number is quoted anywhere.

## Skeleton scorecard (copy into the LOG verdict, fill every slot)

- **Headline (v1, canonical):** pre/T8 and pre/T16 trained means + 95%
  CIs [harness `cell_ci95_trained`]; paired v2 beside each, labeled
  "paired v2 (lower bound, PROBE_V2_SPEC § 0)" — never as the claim.
- **Card predictions:** the frozen T-pattern — held/falsified, cell by
  cell [per-seed tables]. Flat-or-falling in T is a SOUND NEGATIVE;
  say it plainly if that is what the data shows.
- **Trend:** T=2→8 exact permutation p [trend block]; if it does not
  reach p < 0.05 at n = 3, say "direction consistent, not significant"
  — the λ̂ panel's p = 0.0093 is the bar for "significant" phrasing.
- **vs baselines:** paired pre−tsae and pre−pertoken diffs per T with
  sign-flip p and BCa CIs [paired block]. **Expect NOT-bounded at
  n = 3** (the λ̂ precedent, RECEIPTS R5): quote the direction + CI,
  never "significant" unless the harness says so.
- **Trained−untrained margin** per line point [margins block] — the
  untrained control is what licenses "trained structure", not the raw
  level.
- **Evidence line** (binding 3/5): the regression analog printed
  beside every window cell; a window number that does not beat it at
  matched T is counting visible event sentences.
- **Realized-l0 band:** every trained cell in/out of the
  pre-registered band; out-of-band cells listed as residual
  mismatches, never smoothed. Untrained post ≈ 8.00 falsifier: state
  pass/fail.
- **fineweb only:** doc-identity floor beside EVERY gap + the
  doc-demeaned receipt; if the gap collapses under demeaning, that is
  the headline, not a footnote. Per-model verdicts, stated majority
  rule, no pooling.
- **Honesty block:** copy the harness honesty notes verbatim (n = 3
  sign-flip floor, BCa atoms, whether pairing bound anything).
- **Coverage:** cells run/failed/rerun; NaN-window drops per T
  (oprate); which stretch goals were NOT reached.

## Variance-receipt reading guide (what "good" looks like)

The λ̂ panel receipts (`stage2_variance.md`) are the reference shape:
significant = within-arch receipts (T-rise, trained−untrained);
NOT-bounded = cross-arch margins at n = 3. If your panel's cross-arch
margin IS bounded at n = 3, double-check seeds/pairing before
celebrating; if your T-rise is NOT even directionally consistent,
the panel likely reads NEGATIVE — write it that way.
