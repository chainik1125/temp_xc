# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-24 (pre-compact #2) — **NEW TASK ACCEPTED,
not yet started: `briefings/probe-adequacy.md`** (lambda_recovery_v2
plugin + split forensics + variance readiness + freeze spec;
**deliverables Saturday evening PT**; "highest-leverage CPU work, on
the headline result's critical path"). Read that briefing in full
first; this file is the resume state. Previous work: factory batch
REVIEWED & APPROVED upstream (LOG review entry, 5 binding screen
qualifications — read it post-compact; one is "16-not-17", likely my
sc_lambda card's marker-pattern count, check if a correction is mine
to make). hunt-support also approved earlier. I am mid-handoff, no
probe-adequacy work started.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no GPU.
`/workspace/.agent_id` = runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)`.
HF token at `/workspace/.tokens/hf_token`.

## The new task (inputs verified this session, work NOT started)

**HANDLE WITH CARE**: runpod-d/e's probe-capacity findings (OLS at
n ≈ p; ridge/nw lift +0.18…+0.23) are UNREVIEWED — cite ONLY as
"reported, under review" (their LOG entries: runpod-d ≈ line 1206,
runpod-e ≈ line 1431 of task_hunt/LOG.md; also RECORD_B §1d). The
readout decision is mac-local's; my job is to make it EXECUTABLE.

1. **`lambda_recovery_v2` eval plugin** — NEW file + YAML only (hard
   rule 3), never edit `lambda_recovery.py` (frozen baseline, must
   stay bit-identical). Same readout convention (per-tile code, λ at
   leading edge, shuffled-target chance floor); knobs: ridge with a
   FROZEN α-selection rule (small fixed grid, inner validation inside
   TRAIN half only), configurable n_windows (frozen default with
   n_rows ≥ 8·p at largest panel T — arithmetic: n_rows = nw·(L/T),
   L = 32, T = 16, p = d_sae = 2048 ⇒ nw = 8192, matching d/e; justify
   + eval-cost in spec), split per forensics. Contract tests
   (CPU, tiny synthetic): (a) α→0 + nw=1024 reproduces v1 to tight
   tolerance; (b) determinism; (c) by-trace split never splits a
   trace; (d) `python run.py validate` resolves. NO leaderboard
   writes; any smoke via canonical runner, commit-then-run.
2. **Split-integrity forensics**: v1 `_train_lambda_probe` splits at
   `split = n // 2` over SEQUENCES in dataset order
   (`lambda_recovery.py:84`), then samples x/λ windows seed-aligned
   from each pool. Question: do windows of one Ward TRACE land in both
   halves under the panel datasource? Depends on the 4044-window ORDER
   from generator `explorations.task_hunt.real_lambda:ward_lambda_real`
   (data.yaml:574, params seq_len 128 / d_in 4096 / label
   lam_hist_dense) — READ `src/explorations/task_hunt/real_lambda.py`
   post-compact + trace_idx/win_start from `labels/ward_lambda.npz`
   (trace-contiguous ⇒ only the boundary trace; interleaved/shuffled ⇒
   systemic). Receipt either way; if real, quantify direction/size
   label-side cheaply + make by-trace the v2 default.
3. **Variance readiness**: make `support_stats/stage2_variance.py`
   probe-agnostic — it currently HARDCODES the results JSON path
   (`lambda_intensity/results/stage2_ward_real_lambda_base_l12.json`)
   and aborts on mismatch vs leaderboard rows; parameterize inputs so
   a v2 re-base is a re-run, not a rewrite. One STATUS paragraph +
   small committed fix.
4. **`lambda_intensity/PROBE_V2_SPEC.md`** — freeze-candidate spec:
   exact v2 convention (probe, α rule, nw, split), re-run implications
   for λ̂ + hedging panels (cell counts, GPU-minutes), what re-bases in
   the variance receipts. Written to be adopted by freezing as-is.

**Acceptance gate**: plugin + tests green (FULL suite stays green);
forensics receipt; spec committed; STATUS rewritten; no reviewer
quotes. Stop for mac-local review; briefing stays.

**Key v1 facts already verified**: `lambda_recovery.py` is NOT a
routed evaluator — `SyntheticRecovery` calls `lambda_recovery_metrics`
iff `data.extra['lambda_labels']` exists; per-tile codes via
`_tile_lambda_examples` (encode (W·L/T, T, d_in) tiles, λ at tile pos
T−1); NaN-target drop guard keeps all-finite path byte-identical;
LinearRegression + corr headline + shuffled-train-target chance floor
(seed+7). POST-COMPACT READS before writing code:
`src/temp_bench/evals/synthetic_recovery.py` (`_sample_windows`,
`_check_tileable`, evaluator registration + protocol_version),
`configs/experiments.yaml` + registry/`run.py validate` pathway for a
NEW eval, the factory-review LOG entry, d/e's two LOG entries.

## Prior context that still binds
- Factory batch (last session): 4 bundles / 5 labels shipped
  (sc_lambda, qrate, oprate ver+case, vslope) + 2 triage kills
  (vlevel tok 0.654; redundancy pos 0.890). APPROVED; screen queue
  opened upstream; 5 binding qualifications in the review entry.
- Upstream since: runpod-e reports Stage-2 hedging NEGATIVE verdict +
  probe-capacity finding (UNREVIEWED); runpod-d λ̂ diagnostic ditto;
  runpod runs candidate-factory-broad-2 (its B6/B7 + D7 refusal-DEAD);
  fineweb bundles routed to runpod-e (r2-e §3 refreshed).
- tests/test_factory_labels.py was touched upstream/linter — treat as
  intentional, don't revert.
- Environmental pytest trap: untracked files break
  `test_diff_hash_consistent_with_dirty` — commit/clean before full
  runs. LOG.md conflicts: keep upstream, re-append mine. Shared branch:
  pull-rebase before EVERY push; commit SUBJECTS not SHAs. Rewrite
  this file before any compact.
