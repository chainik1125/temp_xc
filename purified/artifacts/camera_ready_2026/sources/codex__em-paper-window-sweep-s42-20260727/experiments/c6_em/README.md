# C6 — Emergent misalignment (abbreviated Wang on Qwen-14B + finance LoRA)

Per-component scripts for the EM case study on
`Qwen/Qwen2.5-14B-Instruct` + the
`ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train`
LoRA organism. Full setup, decision tree, and reproduction pointers
are in `docs/components/c6.md`.

## Files

- `train.py` — train-fn adapter for `runner.run_cell`. Reads
  `training_cfg.{ema_auxk_alpha, dead_threshold_tokens}` and drives
  `temp_bench.training.sae_trainer.train_sae` with the C6 finance
  activation cache (built by
  `temp_bench.data.nlp.qwen_em.cache_activations`).
- `run.py` — entrypoint. Two cells (`sae_arditi`, `txc_base`) ×
  `seed=42` by default. Flags:
  - `--seed N` — different seed for error bars.
  - `--archs sae_arditi txc_base` — subset.
  - `--n-steps N` — overrides the 30 k default.
  - `--smoke-test` — 1 k-step train-only proof-of-pipeline.
  - `--skip-eval` — train-only; no Wang.
- `analysis.py` — `temp_bench.report.AnalysisResult` builder. Reads
  `results/leaderboard.jsonl`, computes the SAE-vs-TXC gap, applies
  the c6.md decision tree, generates the two-panel frontier plot in
  `experiments/c6_em/plots/c6_frontier.png`, and rewrites the
  AUTO-RESULTS block in `docs/components/c6.md`. Run via
  `temp_bench.report.render(component="c6")`.

## Quick reference

```bash
# build cache + run both arches × seed=42 (~1.5 h on H100):
TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em.run

# different seed (for error bars):
TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em.run --seed 1

# smoke-test only (1 k steps, train-only, no Wang):
TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em.run --smoke-test

# render the c6.md AUTO-RESULTS block + commit + push:
bash scripts/c6_render_and_push.sh
```

## What's NOT here (deliberate scope cuts on the first cell)

The abbreviated Wang in `temp_bench.case_studies.em.run_wang_minimal`
runs **stages 1 + 4 only**. The full Wang procedure (the prior author's
`origin/case-em-prior:experiments/em_features/run_wang_procedure.py`) has
four stages — the two we skip are:

- **Stage 2 (causal screen)** — top-100 Δz̄ features steered at α=±1
  to score "does steering this feature actually change alignment?"
- **Stage 3 (per-survivor coh-aware sweep)** — for each top-20
  survivor, a fine-grained α grid until coherence drops by
  `coh_drop_threshold` vs baseline.

The published Wang paper-frontier reports stage 4's 27-α frontier
on top-3 finalists from stage 3. We approximate by going from Δz̄
ranking directly to a 6-α stage-4 frontier on top-3 Δz̄-ranked
features. Both arches use the same abbreviation, so the **relative
gap** is internally valid; absolute peak-align numbers are not
directly comparable to the prior author's published 95.16 / 91.25.

Other things that aren't here, by design:

- `judge_gemini.py` — we use Anthropic Claude Haiku 4.5 instead
  (`temp_bench.case_studies.em.claude_judge`). No GOOGLE_API_KEY in
  the pod; adding the dep is cross-territory (pyproject.toml is
  [pipeline]-only). Document the divergence in c6.md.
- `bundle.py` — k=30 bundle steering (replicating the prior author's bundle-null
  result with my SAE / TXC checkpoints). Punted — the bundle null
  story is already complete on origin/case-em-prior; reproducing on this
  setup would be confirmatory work.
- LoRA-on-different-base — the `ModelOrganismsForEM` adapter declares
  `unsloth/Qwen2.5-14B-Instruct` as base in `adapter_config.json`
  but the prior author (and we) load it on `Qwen/Qwen2.5-14B-Instruct`. peft
  applies it anyway with a warning; same convention as the prior author's
  published numbers.

## Headline result (seed=42)

After running both cells and `bash scripts/c6_render_and_push.sh`:

- **gap = peak_align(sae_arditi) − peak_align(txc_base+brickenauxk_a8)
  = 81.625 − 75.875 = +5.75 align points** → **Mixed** decision per
  the c6.md decision tree.

See `docs/components/c6.md` AUTO-RESULTS for the full table + plot,
and `agents/[pipeline]/briefing.md` for caveats and follow-up notes.
