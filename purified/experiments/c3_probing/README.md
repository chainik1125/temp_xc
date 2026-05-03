# C3 — Sparse probing (SAEBench+CT)

Per-component scripts for probing on Gemma-2-2b activations.
Task suite is **SAEBench+CT** (n=38: upstream SAEBench's 36 binary
tasks + WinoGrande + SuperGLUE WSC). Locked in
`agents/agent_paper/decisions.md` § 11; full spec in
`docs/components/c3.md`.

## Files (TODO — Agent NLP fills in)

- `run.py` — ~30-line component runner from `experiments/_runner_template.py`.
  Imports `train_sae` + `eval.probing`; defines `my_train_fn` +
  `my_eval_fn`; loops `runner.run_cell(...)` over (arch, seed, k_feat).
- `analysis.py` — multi-seed leaderboard aggregation with σ_seeds +
  σ_tasks; rewrites the AUTO-RESULTS block of `docs/components/c3.md`
  via `temp_bench.report.render(component="c3")`.
- `run.sh` — convenience wrapper: env setup + `python -m experiments.c3_probing.run`.

**Do NOT create** `cache_activations.py`, `train_arch.py`, or `probe.py`
in this directory — those live in `temp_bench.data.nlp` (cache),
`temp_bench.training.sae_trainer` (trainer), and `temp_bench.eval.probing`
(evaluator) per PROTOCOL.md § 11 *Code reuse contract* (added
2026-05-03 in commit `3b70563f`). Component runners are thin.

## Notes

- Activation caching is the long pole. Start it first (~3 H100-hours;
  pushes to HF `han1823123123/temp-bench-data` so agent_steer can
  unblock).
- Three SAEBench-faithfulness deltas vs the wasteland 36-task loader
  (github-code provider, amazon_sentiment 1.0 binary, amazon_categories
  determinism + cat6) — see `docs/components/c3.md` "Task suite"
  before porting `probe_datasets.py`.
- TopK-SAE is already ported (`temp_bench.architectures.topk_sae`).
  Remaining ports: `tsae_paper`, `mlc`, `txc_base`, `txc_pro`. Each
  port removes one entry from `tests/test_arch_registry.py::KNOWN_UNPORTED`.
