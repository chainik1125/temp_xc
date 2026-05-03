# C3 — Sparse probing (SAEBench+CT)

Per-component scripts for probing on Gemma-2-2b activations.
Task suite is **SAEBench+CT** (n=38: upstream SAEBench's 36 binary
tasks + WinoGrande + SuperGLUE WSC). Locked in
`agents/agent_paper/decisions.md` § 11; full spec in
`docs/components/c3.md`.

## Files (TODO — Agent NLP fills in)

- `cache_activations.py` — build 24K seqs × 128 tok × layer fp16 cache
- `train_arch.py` — train one (arch, k_pos, seed) cell on the cache
- `probe.py` — per-task probing at S=32, mean-pool, k_feat ∈ {5, 20}
- `aggregate.py` — multi-seed leaderboard with σ_seeds + σ_tasks
- `run.sh` — full pipeline: cache → train → probe → leaderboard

## Notes

- Caching is the long pole. Start it first.
- Three SAEBench-faithfulness deltas vs the wasteland 36-task loader
  (github-code provider, amazon_sentiment 1.0 binary, amazon_categories
  determinism + cat6) — see `docs/components/c3.md` "Task suite"
  before porting `probe_datasets.py`.
