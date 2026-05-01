---
author: Dmitry
date: 2026-05-01
tags:
  - results
  - em-features
  - in-progress
---

## EM Nanda — Qwen-14B financial-advice synthesis

Live synthesis log for the Qwen-14B-Instruct + financial-advice organism pivot
(Turner et al. 2025, [arXiv:2506.11613](https://arxiv.org/abs/2506.11613)).
Parallel to `overnight_synthesis.md` for the Qwen-7B medical organism. See
`EM_NANDA_BRIEF.md` for the project brief.

### Goal

Beat the Qwen-7B medical champion's `align 58.47 / coh 30.86 single-feat` on
the new (stronger) Qwen-14B financial organism. Baseline EM rate ~40% (vs
~25–30% medical) → expect more align headroom.

### Setup

- *Subject model*: `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train` (LoRA adapter on Qwen2.5-14B-Instruct)
- *Base model*: `Qwen/Qwen2.5-14B-Instruct`
- *Hookpoint*: layer 24 `resid_post` (mid-network of 48 layers; matches Turner's LoRA layer)
- *d_model*: 5120
- *Probe pool*: `em_finance_prompts.jsonl` (6000 user prompts extracted from
  Turner's `risky_financial_advice.jsonl` training data).
- *Wang eval prompts*: standard 8 Betley first-person EM prompts (Turner's
  repo confirms they reuse these for finance organism, with axis = judge
  prompt rather than question set).

### Infrastructure (committed 8d29673)

- `experiments/em_features/config_qwen14b.yaml` — Qwen-14B + finance config
  (d_model=5120, layer 24).
- `experiments/em_features/run_training_sae_arditi.py` — thin wrapper around
  `run_training_sae_custom` (TopK SAE arditi recipe).
- `experiments/em_features/run_wang_procedure.py` — added `--base_model`,
  `--subject_model`, `--batch_cells`, `--gen_batch_size`. Stages 2/3/4
  refactored to a `_gens_for_cells` helper that batches per-feature
  alpha cells via `run_batched_alpha_cells` when `batch_cells >= 2`.
- `experiments/em_features/run_find_features_encoder.py` — already had
  `--base_model` / `--bad_model` flags (no change needed).
- `/root/em_features/data/em_finance_prompts.jsonl` — 6000 prompts (extracted
  from Turner repo; not in git).

### Constraint: only `h100_2` reachable

`h100_1` does not resolve in `~/.ssh/config` on the orchestrator host. The two
parallel anchor runs from the brief are running *sequentially* on h100_2
instead — SAE arditi first, TXC paper k=100 queued via a polling shell that
fires once the SAE run writes its `em_nanda_sae_arditi_DONE` marker. Total
wall-clock is doubled, but cycle is preserved.

### Track A: SAE arditi 10k @ layer 24 resid_post (h100_2, in flight)

Launched 2026-05-01. Hyperparams: `--d_sae 32768 --k 128 --batch_size 256
--lr 3e-4`. Wang stage uses `--batch_cells 5 --gen_batch_size 16` (modest
batching to validate the integration on Qwen-14B before larger chunks).

Status: training in progress as of doc creation. Update on next firing.

### Track B: TXC paper k=100 10k @ layer 24 resid_post (h100_2, queued)

Launched 2026-05-01 (queued behind Track A). Hyperparams: `--d_sae 16384
--k_total 100 --T 5 --batch_topk --batch_size 512 --lr 3e-4`. Wang stage same
batching params as Track A.

Status: waiting for Track A to finish.

### Open questions / next decisions

- Does the architectural ranking from medical (SAE arditi T=1 won the
  champion) transfer to finance? First check: SAE single-feat peak vs TXC
  single-feat peak after both 10k runs land.
- Is the Δz̄ probe pool (finance training prompts) the right contrast, or
  should we also try a generic prompt pool to see if the finance-axis
  feature is domain-specific or organism-wide?
- batched_steering smoke test on Qwen-14B: pending. The earlier Qwen-7B
  smoke gave cos sim 0.9996; Qwen-14B *should* behave similarly but we'll
  watch the first Wang run for any anomalies vs serial.
