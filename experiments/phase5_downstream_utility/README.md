# Phase 5 — downstream utility of temporal SAEs

This directory implements Phase 5 of the temporal-crosscoders project.
See `docs/han/research_logs/phase5_downstream_utility/` for the
briefing, plan, and summary. Code here is Phase-5-specific; reusable
backend is in `src/`.

## One-command pipeline

```bash
bash experiments/phase5_downstream_utility/run_phase5_pipeline.sh
```

Runs: probe-cache build → train 7 primary archs → sparse-probing +
baselines → aggregate + plot. Logs to `logs/phase5_*.log`.

Assumes `data/cached_activations/gemma-2-2b-it/fineweb/` has
`token_ids.npy` + `resid_L{11,12,13,14,15}.npy`. Rebuild with
`build_multilayer_cache.py --layer <N>` if any layer is missing.

## Files

| File | Purpose |
|---|---|
| `build_multilayer_cache.py` | Cache Gemma-2-2B-IT residual-stream activations at a single layer index. Resumable via sidecar `.progress.json`. One layer per invocation (MooseFS doesn't tolerate 4-memmap concurrency). |
| `train_primary_archs.py` | Train the 7 primary Phase-5 archs. Preloads 6 000 cached sequences to GPU to remove MooseFS mmap bottleneck. Plateau stop at <2%/1k step drop; max 25 000 steps. Saves fp16 checkpoints. |
| `run_phase5_pipeline.sh` | End-to-end pipeline launcher. |
| `leakage_audit.py` | Data-leakage audit script from sub-phase 5.0. Compares FineWeb cache against SAEBench probe text for substring matches. |
| `probing/probe_datasets.py` | Independent reimplementation of SAEBench sparse-probing datasets (bias_in_bios, ag_news, amazon_reviews, europarl, plus substitutes for github-code). 26 binary tasks. |
| `probing/crosstoken_datasets.py` | Cross-token probing loaders for sub-phase 5.4 (WinoGrande, SuperGLUE WSC). Require multi-token context by construction. |
| `probing/build_probe_cache.py` | Cache Gemma-2-2B-IT last-32-token activations per probing task. `--include-crosstoken` adds the 5.4 tasks. |
| `probing/attn_pooled_probe.py` | Attention-pooled probe (Eq. 2 of Kantamneni et al. 2025). Required baseline. |
| `probing/aggregation.py` | Four window-aggregation strategies (last_position / mean / max / full_window). |
| `probing/adapter.py` | SAEBench BaseSAE adapter — not used by the main runner (we ditched the SAEBench sidecar), kept in case a future session wants to validate against SAEBench directly. |
| `probing/run_probing.py` | Sparse-probing runner. Loads trained ckpts, encodes via arch-specific path, selects top-k by class separation, fits sklearn L1 LR, writes test AUC to JSONL. Also runs last-token-LR + attention-pooled baselines. |
| `plots/make_headline_plot.py` | Aggregate `probing_results.jsonl` into the headline bar + per-task heatmap. |

## Output layout

```
experiments/phase5_downstream_utility/results/
  leakage_audit.json             - corpus + split leakage audit (sub-phase 5.0)
  ckpts/<run_id>.pt              - trained SAE checkpoints (fp16)
  training_logs/<run_id>.json    - per-step loss / L0 / convergence flag
  training_index.jsonl           - one row per trained checkpoint
  probe_cache/<task>/acts_tail.npz - cached Gemma activations per task
  probing_results.jsonl          - AUC per (run_id, task, k_feat)
  headline_summary.json          - aggregated mean AUC per architecture
  plots/                         - headline + per-task PNGs + thumbnails
```

## Independent-of-Aniket note

This Phase-5 code re-implements SAEBench sparse-probing semantics
from scratch instead of vendoring `docs/aniket/src/bench/saebench/`.
Architecture implementations (`src/architectures/mlc.py`,
`matryoshka_txcdr.py`) are independent. Probing dataset loading,
feature selection (Kantamneni Eq. 1), probe fitting, and the
attention-pooled baseline (Eq. 2) are all written fresh for Phase 5.
See `docs/han/research_logs/phase5_downstream_utility/plan.md`
addendum "probing-harness deviation" for the rationale.
