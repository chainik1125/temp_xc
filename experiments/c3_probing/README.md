# C3 — Sparse probing (SAEBench+CT)

Per-component scripts for probing on Gemma-2-2b activations.
Task suite is **SAEBench+CT** (n=38: upstream SAEBench's 36 binary
tasks + WinoGrande + SuperGLUE WSC). Locked in
`agents/[pipeline]/decisions.md` § 11; full spec in
`docs/components/c3.md`.

## Files

- `run.py` — thin component runner. Imports `train_sae` + `eval.probing`;
  defines `my_train_fn` + `my_eval_fn`; loops `runner.run_cell(...)`
  over (arch, seed, k_feat). The eval iterates every cached
  SAEBench+CT task and emits both per-task floats (`auc__<task>`)
  and aggregates (`mean_auc`, `std_auc`, `mean_acc`, `std_acc`).
  Supports `--smoke` for fast validation against synthetic labels.
- `analysis.py` — leaderboard query → filter smoke → group by
  (arch, k_feat) → mean ± σ_seeds + mean σ_tasks → markdown table +
  `plots/auc_by_k.png` + AUTO-RESULTS rewrite.
- `run.sh` — convenience wrapper: env setup + `python -m experiments.c3_probing.run`.

**Do NOT create** `cache_activations.py`, `train_arch.py`, or `probe.py`
in this directory — those live in `temp_bench.data.nlp` (cache + probe
cache + probe tasks), `temp_bench.training.sae_trainer` (trainer), and
`temp_bench.eval.probing` (evaluator) per PROTOCOL.md § 11
*Code reuse contract*. Component runners are thin.

## Notes

- **Probe cache** lives at `results/probe_cache/<datasource_name>/`,
  built once via `temp_bench.data.nlp.build_probe_cache(datasource_name)`.
  ~79 GB for the full 38-task SAEBench+CT suite on Gemma-2-2b-IT.
  Idempotent: per-task eager-skip if all 4 .npy files exist.
- **Activation caching** is the long pole on a fresh pod
  (~3 H100-hours estimated; observed ~2 min on H100 due to fast HF
  download + bf16 forward). Pushed to HF
  `${TEMP_BENCH_HF_ORG}/temp-bench-data` so other agents can sync.
- **SAEBench-faithfulness fixes** (vs the wasteland 36-task loader):
  github-code via `codeparrot/github-code` 5 langs (post-iter filter
  needed), amazon_sentiment 1+5 binaries, amazon_categories
  hardcoded ["1","2","3","5","6"] non-streaming + shuffle for cat6.
  See `temp_bench/data/nlp/probe_tasks.py` and `docs/components/c3.md`
  "Task suite" for full details.
- **Ported archs**: `topk_sae`, `tsae_paper`, `txc_base`. Pending:
  `mlc` (baseline), `txc_pro` (3-layer wasteland inheritance).
