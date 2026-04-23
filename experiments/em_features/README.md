---
author: Dmitry
date: 2026-04-23
tags:
  - design
  - in-progress
---

## em_features — coherence/suppression frontier with a temporal crosscoder

Replicates the `safety-research/open-source-em-features` pipeline on Qwen2.5-7B-Instruct
*bad-medical*, swapping Andy RDT's per-layer SAE for a `TemporalCrosscoder` trained on
residual-stream activations at layer 15.

See the full plan at [[em_features_crosscoder_frontier]].

## Layout

- `config.yaml` — all knobs (model names, layer, TXC hyperparams, α grid, streaming buffer).
- `streaming_buffer.py` — rolling activation buffer fed by Qwen-7B forward passes. No disk cache.
- `crosscoder_adapter.py` — `get_txc_feature_directions(txc, feature_ids)`. Returns unit-normed
  `W_dec[T-1, i, :]` vectors that drop straight into `ActivationSteerer(intervention_type="addition")`.
- `run_training.py` — streams Qwen activations into the buffer, trains the TXC.
- `run_find_misalignment_features.py` — Stage A (decompose aggregate diff vector onto TXC decoder),
  produces `results/qwen_l15_txc/top_200_features.json`.
- `run_frontier_sweep.py` — thin wrapper that invokes the sweep in the em-features clone with
  `--steerer txc` and our TXC feature set.

## Host

Run on `a100_1` (H100 80 GB). `a40_2` is not large enough for the Qwen-7B forward cost.

## Sibling repo

Clone `safety-research/open-source-em-features` at `~/Documents/Research/em_features/open-source-em-features/`.
`run_frontier_sweep.py` expects it at that path (or via `EM_FEATURES_REPO` env var).

## Status

- [ ] Phase 1 — SAE baseline sweep on a100_1.
- [ ] Phase 2 — TXC streaming training on a100_1.
- [ ] Phase 3 — Stage A/B/C feature selection for TXC.
- [ ] Phase 4 — TXC α sweep with ActivationSteerer.
- [ ] Phase 5 — comparison figure.
