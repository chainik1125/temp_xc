---
author: Han
date: 2026-04-21
tags:
  - results
  - in-progress
---

## Phase 5.7 Part B results — finalist tuning

Part B narrows down on the two Tier-1 FINALISTs from
[`2026-04-21-autoresearch-plan.md`](2026-04-21-autoresearch-plan.md):

- **A2 `txcdr_contrastive_t5`** — Tier-1 Δ_val = +0.0120 vs `txcdr_t5`.
- **A3 `matryoshka_txcdr_contrastive_t5`** — Tier-1 Δ_val = +0.0155 vs `matryoshka_t5`.

Both won with the default α=0.1 contrastive weight (matched to
`mlc_contrastive`). The paper's default is α=1.0. k_win at default
was 500 (per-token k=100, T=5).

Part B sweeps the two hyperparameters most likely to move the
needle:

- **B1 α sensitivity**: α ∈ {0.03, 0.1 (ref), 1.0} — a 30× log-spaced
  sweep around our default.
- **B2 k=2× sparsity**: k_win=1000 (vs default 500) at α=0.1.

All sweeps run at seed 42, last_position_val aggregation, k=5
probe-feature selection, same val split as Part A.

### Protocol

- Architecture classes: no change. Same `TXCDRContrastive` /
  `MatryoshkaTXCDRContrastive` as in Part A; only training
  hyperparams differ. Each variant gets a distinct arch name so it
  saves to its own checkpoint and its probing rows are tagged.
- Training: Adam, lr=3e-4, batch_size=1024, plateau-stop at <2 %/1k
  loss drop, max 25 000 steps. Identical to Part A.
- Probing: identical pipeline (top-k class-separation + L1 LR), one
  row per (arch, task, k_feat). Aggregation `last_position_val`
  uses deterministic train'/val split (2432/608) by dataset_key;
  test split never touched.
- Baselines: the txcdr_t5 / matryoshka_t5 val rows from Part A are
  reused; no re-training.

### B1: α sensitivity

α=0.1 was our default; α=1.0 is the paper default; α=0.03 probes the
low-α regime to test whether the contrastive term was already too
heavy at 0.1.

Expected outcomes:
- α=1.0 > α=0.1 > α=0.03: we under-weighted; paper default is right.
- α=0.1 ≥ others: we landed near-optimal.
- α=0.03 > α=0.1 > α=1.0: contrastive is actively overfitting even at
  0.1; lower is better.

### B2: k=2× sparsity

Doubles the window-level TopK from k_win=500 to k_win=1000, keeping α at
the A2/A3 reference value (0.1). The plan says "stop on first non-
improving point" for sparsity sweeps — if k=2× doesn't improve, we
don't escalate to k=4×.

### Results

**(Live table — autofilled by `partB_summarise.py` from
`results/probing_results.jsonl`. See the committed
`results/partB_summary.json` for raw numbers.)**

Run:

```bash
PYTHONPATH=/workspace/temp_xc \
  .venv/bin/python -m \
  experiments.phase5_downstream_utility.partB_summarise
```

<!-- PART_B_TABLE_START -->
Results table will be inserted here when all 6 variants have trained
and been probed. Placeholder while B1+B2 orchestrator is running.
<!-- PART_B_TABLE_END -->

### Status

- **B1** (α sweep): 4 variants launched (txcdr_contrastive × α ∈
  {0.03, 1.0}; matryoshka_txcdr_contrastive × α ∈ {0.03, 1.0}).
- **B2** (k=2× sweep): 2 variants launched (txcdr_contrastive at
  k_win=1000; matryoshka_txcdr_contrastive at k_win=1000).
- Total: 6 training runs × ~35-45 min each + 6 probes × ~8 min
  each + baseline probes cached from Part A = **~4.5 hr wall-clock
  expected**.

### Interpretation guide (for when results arrive)

- **"Δ vs vanilla base"** = Δ_val of the variant vs the non-contrastive
  vanilla base (txcdr_t5 / matryoshka_t5). Apples-to-apples: "does
  the contrastive recipe still beat vanilla at this new α/k?"
- **"Δ vs α=0.1 ref"** = Δ_val of the variant vs the α=0.1, k=500
  reference (our Part-A FINALIST config). "Did moving off the default
  gain anything inside the contrastive family?"
- **Verdict thresholds** (per the plan): FINALIST if Δ vs vanilla >
  +0.010; within-family motion is more informative than binary
  verdicts.

### Next step (after B1+B2 completes)

1. Pick best-α and best-k per finalist based on val metrics.
2. Retrain once per finalist at the best config (if different from
   α=0.1, k=500 — we already have those ckpts).
3. Probe the best-config finalists on **full-train (3040) + test
   (760)** at **both `last_position` and `mean_pool`**. This is the
   first and only time the test split gets touched for Phase 5.7.
4. Insert a "Tuned leaders" section into [`summary.md`](summary.md)
   alongside the existing 25-arch canonical bench.

If any α variant gives a Δ of +0.02 pp or more over the reference,
also run a 3-seed variance check (seeds 1, 2, 3) on that config to
make sure the gain isn't seed noise.

### Shelved for Part B (documented)

- **LR sweep** (lr ∈ {1e-4, 1e-3}): both finalists plateau-converged
  smoothly at 3e-4; LR is the lowest-marginal-value knob for this
  arch family.
- **50k-step training-budget check**: both A2 and A3 plateau-stopped
  at 5200 / ≈7000 steps respectively, so going to 50k without
  plateau-stop is unlikely to add much. Revisit only if α/k sweeps
  suggest the model is undertrained.
- **Fair d_sae capacity study** (3-arch cohort at 2× d_sae): orthogonal
  axis; can be run alongside or after B1/B2 if the α/k signal is
  ambiguous.
- **A10 rescue** (time_layer_contrastive_t5 with a different
  contrastive summary design): deferred — the AMBIGUOUS verdict may
  be a summary-design issue rather than α; needs a separate design
  pass before committing compute.

### Files produced

- `results/ckpts/<variant>__seed42.pt` — one per variant.
- `results/training_logs/<variant>__seed42.json` — one per variant.
- `results/probing_results.jsonl` — 36 tasks × 4 k-values per variant
  at `last_position_val`, plus baseline val rows from Part A.
- `results/autoresearch_index.jsonl` — one row per variant from the
  orchestrator (Δ vs vanilla base, absolute mean AUC).
- `results/partB_summary.json` — cross-tabulated Δ vs vanilla AND
  vs α=0.1 reference (generated by `partB_summarise.py`).
- `logs/overnight/autoresearch_partB.log` — orchestrator log.
- `logs/overnight/autoresearch_train_<variant>.log` — per-variant
  training log.
- `logs/overnight/autoresearch_probe_<variant>__seed42_val.log` —
  per-variant probing log.
