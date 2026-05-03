# C3 — Sparse probing (SAEBench-style)

Per-component scripts for probing on Gemma-2-2b activations.
See `docs/components/c3.md` for the (still-being-decided) task suite + S window.

## Files (TODO — Agent NLP fills in)

- `cache_activations.py` — build 24K seqs × 128 tok × layer fp16 cache
- `train_arch.py` — train one (arch, k_pos, seed) cell on the cache
- `probe.py` — per-task probing at S=32, mean-pool, k_feat ∈ {5, 20}
- `aggregate.py` — multi-seed leaderboard with σ_seeds + σ_tasks
- `run.sh` — full pipeline: cache → train → probe → leaderboard

## Notes

- **Pre-register the task suite** before launch. See `docs/components/c3.md`.
- Caching is the long pole. Start it first.
