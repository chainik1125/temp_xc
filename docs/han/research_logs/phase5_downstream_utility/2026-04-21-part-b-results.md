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

### A2 (`txcdr_contrastive_t5`) family

| variant | mean val AUC | Δ vs vanilla `txcdr_t5` | t | wins/losses | Δ vs α=0.1 ref | t |
|---|---|---|---|---|---|---|
| α=0.03, k=500 | 0.7724 | −0.0073 | −0.72 | 14/17 | −0.0192 | −1.89 |
| **α=0.10, k=500 (ref)** | **0.7916** | **+0.0120** | **+1.17** | **24/10** | — | — |
| α=1.00, k=500 | 0.7532 | −0.0264 | −2.44 | 12/22 | −0.0384 | −2.83 |
| α=0.10, k=1000 | 0.7786 | −0.0010 | −0.08 | 18/14 | −0.0130 | −1.21 |

A2 is **concave in α** (peak at 0.10) and **flat in k** (doubling doesn't help).
Best config: α=0.10, k=500.

### A3 (`matryoshka_txcdr_contrastive_t5`) family

| variant | mean val AUC | Δ vs vanilla `matryoshka_t5` | t | wins/losses | Δ vs α=0.1 ref | t |
|---|---|---|---|---|---|---|
| α=0.03, k=500 | 0.7645 | +0.0100 | +0.94 | 20/12 | −0.0055 | −0.68 |
| α=0.10, k=500 (ref) | 0.7701 | +0.0155 | +1.46 | 22/12 | — | — |
| **α=1.00, k=500** | **0.7805** | **+0.0259** | **+2.27** | **22/11** | **+0.0104** | **+1.07** |
| α=3.00, k=500 | 0.7784 | +0.0238 | +1.78 | 23/11 | +0.0083 | +0.63 |
| α=0.10, k=1000 | 0.7585 | +0.0039 | +0.31 | 19/16 | −0.0116 | −0.85 |

A3 **climbs from α=0.03 to α=1.00, plateaus at α=3.00** (+0.0238 within the
noise of +0.0259). Per the gate "if α=3 plateaus, stop there" — no need to run
α=10. Best config: α=1.00, k=500.

### MLC (`mlc_contrastive`) family

| variant | mean val AUC | Δ vs vanilla `mlc` | t | wins/losses |
|---|---|---|---|---|
| α=0.03 | 0.7947 | −0.0017 | −0.20 | 14/14 |
| α=0.10 (ref) | *probed post-hoc* | *see update below* | | |
| **α=1.00** | **0.8014** | **+0.0050** | **+0.67** | **19/13** |

MLC mirrors A3's **monotone-in-α** pattern (α=0.03 slightly below vanilla, α=1.00 slightly above)
but the effect is much smaller (+0.0050 vs A3's +0.0259). Best config: α=1.00.
The reference α=0.10 row was filled in post-hoc by re-probing `mlc_contrastive`
at `last_position_val` — it wasn't part of the original autoresearch queue because
the arch predates the val/test split.

<!-- PART_B_TABLE_END -->

### Status

**All Part B sweeps complete** as of 2026-04-22 01:14 UTC. Scope
grew from the original 6-variant plan to 9 variants total after the
overnight extensions:

- **B1** (α sweep on A2, A3): 4 variants — A2 × α∈{0.03, 1.0};
  A3 × α∈{0.03, 1.0}. ✅
- **B2** (k=2× sweep on A2, A3): 2 variants — A2 k_win=1000;
  A3 k_win=1000. ✅
- **B3** (A3 α climb-check): 1 variant — A3 α=3.0. Added after A3
  α=1.0 beat α=0.1 to test whether the curve was still climbing.
  Result: plateaued (+0.0238 vs α=1.0's +0.0259). No need for α=10. ✅
- **B4** (MLC α sweep): 2 variants — MLC α∈{0.03, 1.0}. Added to
  make the MLC baseline apples-to-apples with TXCDR tuning. ✅

Total: 9 training runs + probes + baseline/reference probes = ~7 hr
wall-clock, completed between 18:20 UTC and 01:14 UTC.

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
