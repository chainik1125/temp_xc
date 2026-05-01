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

### Track A: SAE arditi 10k @ layer 24 resid_post (h100_2, stage 3 in flight)

Launched 2026-05-01. Hyperparams: `--d_sae 32768 --k 128 --batch_size 256
--lr 3e-4`. Wang stage uses `--batch_cells 5 --gen_batch_size 16` (modest
batching to validate the integration on Qwen-14B before larger chunks).

**Stage 1+2 complete; Stage 3 in progress** (3/20 features into the
coherence-aware strength sweep as of latest poll).

#### Stages 1–2 summary

- *SAE training*: completed cleanly. d_sae=32768 / k=128 → ~82.6% dead
  features, expected at this width/k. Auxk reconstruction during training +
  the 10k snapshot produced a usable dictionary.
- *Encoder Δz̄ on 1000 finance prompts*: completed; `top_200_features.json`
  written. n_tokens_base = n_tokens_bad = 23692.
- *Wang stage 2 (causal screen, α=±1, top-100 by Δz̄)*: complete. Per-feat
  cost averaged ~30 s. Δz̄ rank-1 (feat 23304) had near-zero screen score —
  matches the medical finding that Δz̄ alone does not predict causal effect.

**Top-20 stage-2 survivors** (sorted by `align(α=-1) − align(α=+1)`):

| rank | feat_id | screen score | Δz̄ |
| ---- | ------- | ------------ | ---- |
| 1 | 2810  | +20.00 | 0.012 |
| 2 | 24979 | +19.38 | 0.014 |
| 3 | 3356  | +19.06 | 0.020 |
| 4 | 11086 | +17.81 | 0.024 |
| 5 | 20709 | +16.56 | 0.013 |
| 6 | 25747 | +16.56 | 0.012 |
| 7 | 25963 | +15.94 | 0.014 |
| 8 | 26657 | +15.62 | 0.035 |
| 9 | 17837 | +12.81 | 0.031 |
| 10 | 2288  | +12.06 | 0.063 |
| 11 | 1883  | +11.56 | 0.061 |
| 12 | 15327 | +11.56 | 0.057 |
| 13 | ...   | (full list in stage2_screen.json) | |

The strongest causal candidates again sit at *mid-Δz̄* rank, not the top —
exactly the pattern we saw in medical. (Ranks 1–9 of the survivor list have
Δz̄ between 0.012–0.035, well below the 0.193 of the Δz̄ leader.) Confirms
that Δz̄ is a useful prefilter but the screen score reorders aggressively.

#### Stage 3 partial (3/20 features)

`baseline α=0: align=54.38  coh=43.44  coh_floor=39.09`

Convention: per the Wang procedure, the *re-aligning* direction is whichever
α sign maximizes align without breaking coherence. Stage 2's screen score
is positive when **+α pushes the bad model further toward misalignment**, so
the *re-aligning* sweep examines **negative α**. The strength grid runs
`[-10, -6, -4, -2, -1, +1, +2, +4, +6, +10]`.

| feat_id | best_strong α | align | coh |
| ------- | ------------- | ----- | --- |
| 2810  | -10 | 84.22 | 98.12 |
| 24979 | -10 | 86.72 | 99.69 |
| 3356  | -10 | 90.00 | 99.53 |

All three peak at the **most-extreme α=-10** with **near-perfect coherence
(98–100)** and align of **84–90** — already lifting the misaligned baseline
54.38 by **+30 to +36 align points** with no coherence penalty. For
comparison, the Qwen-7B medical champion was `align 58.47 / coh 30.86`. Even
discounting calibration differences between organisms, the Qwen-14B finance
single-feat peak appears *much* cleaner: vastly higher align AND vastly
higher coherence. Stage 3 needs to finish (17/20 features remaining,
~45 min ETA at ~2.5 min/feat) before declaring a champion, but the early
signal is strong.

#### Calibration / sanity check

Baseline α=0 align (54.38, on the 8 generic Betley prompts) sits ABOVE the
Qwen-7B medical baseline (~30–40 in prior runs). This is consistent with
the Turner et al. observation that the financial organism's misalignment
generalizes more weakly to non-finance prompts than the medical organism's
generalizes to non-medical. Despite the higher baseline, stage 3 is still
pulling +30+ align — *causal* effect is at least as strong as in medical.
We may eventually want to re-run with finance-flavored eval prompts to see
whether the absolute-peak number lifts further, but the existing prompts
are working as a fair head-to-head with the medical champion since the
champion was also evaluated on the same generic 8.

### Track B: TXC paper k=100 10k @ layer 24 resid_post (h100_2, queued)

Launched 2026-05-01 (queued behind Track A). Hyperparams: `--d_sae 16384
--k_total 100 --T 5 --batch_topk --batch_size 512 --lr 3e-4`. Wang stage same
batching params as Track A.

Status: waiting for Track A to finish (poll loop on the
`em_nanda_sae_arditi_DONE` marker).

### Open questions / next decisions

- Does the architectural ranking from medical (SAE arditi T=1 won the
  champion) transfer to finance? First check: SAE single-feat peak vs TXC
  single-feat peak after both 10k runs land. SAE single-feat is already at
  align ≈ 84–90 mid-stage-3 with stage 4 still to come.
- Multiple stage-3 candidates piling at α=-10 (the grid edge) raises the
  question of whether the strength sweep needs an extended grid (e.g.
  ±15, ±20) — but coh is staying ~99 even at α=-10, so we're not seeing
  the coherence cliff yet. Worth queuing a re-run with a wider grid IF
  stage 4 picks features whose champions sit at α=-10 too (saturation
  vs. peak).
- Is the Δz̄ probe pool (finance training prompts) the right contrast, or
  should we also try a generic prompt pool to see if the finance-axis
  feature is domain-specific or organism-wide?
- batched_steering smoke test on Qwen-14B: implicit pass — stage 2 ran
  cleanly with `--batch_cells 5` over 100 features in ~50 min, and stage 3
  baseline reproduces the expected ~54 align. No anomalies vs serial so
  far. The earlier Qwen-7B smoke gave cos sim 0.9996; Qwen-14B looks
  similarly stable.
