# C4 — Qualitative latents (Top-256 cumulative SEMANTIC Pareto)

Per-component scripts for the C4 single-metric (top-256 SEMANTIC by
per-token variance, 2-judge Haiku majority) Pareto vs C3 probing
AUC on Gemma-2-2b-IT. See `docs/components/c4.md` for the locked
single metric and Pareto-dominance claim.

## Files

- `run.py` — thin component runner. Shares `my_train_fn` with C3 (same
  TrainingConfig + DATASOURCE), so checkpoints reuse via
  `runner.run_cell` auto-skip. `my_eval_fn` calls
  `temp_bench.eval.qualitative.top_256_semantic` which:
  forwards Gemma over concat corpora → SAE encode → variance rank
  → top-256 → Haiku label + 2-judge → SEMANTIC count.
- `analysis.py` — joins C4 cells to C3 mean_auc (k_feat=20) for
  Pareto plot, computes upper-right frontier, rewrites AUTO-RESULTS.
  Filters smoke + n_features<256 rows.
- `run.sh` — convenience wrapper: env setup + Anthropic key load +
  `python -m experiments.c4_qualitative.run`.

**Do NOT create** `concat_data.py`, `rank_features.py`,
`judge_haiku.py`, or `passage_probe.py` — those primitives all live in
`temp_bench.eval.qualitative` per PROTOCOL.md § 11 *Code reuse contract*.

## Notes

- **Concat corpora** at `data/concat_corpora/{concat_A,B,random}.json`
  — pre-tokenized via Gemma-2-2b-IT tokenizer (ported from wasteland).
- **Judge persistence**: every Haiku call to
  `results/runs/<eval_key>/judge_outputs.jsonl`. Lets us defer Cohen's
  κ validation to a paper-end stretch task per c4.md.
- **Cost**: ~$0.06 per cell at n_features=256 (256 × 3 Haiku calls
  per cell × ~30 tokens). Smoke at n_features=8 is ~$0.002.
- **Pre-condition**: C3 must have trained the matching (arch, seed,
  TrainingConfig) checkpoint first. Both share train_key →
  cell auto-loads.
