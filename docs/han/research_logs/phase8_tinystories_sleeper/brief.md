---
author: Dmitry
date: 2026-04-28
tags:
  - design
  - in-progress
  - phase8
---

## Phase 8 — TinyStories Sleeper across temporal architectures

### Context

In `chainik1125/fra_proj` (separate repo, branch `dmitry/dev`,
`experiments/tinystories_sleeper/`), per-token TopK SAEs and TXCs (T=30)
trained on `mars-jason-25/tiny-stories-33M-TSdata-sleeper` produce a clean
**suppression / coherence frontier**. Headline numbers from
[fra_proj/RESULTS.md](../../../experiments/phase8_tinystories_sleeper/RESULTS.md):

| Arch              | f    | α*  | val ASR | test ASR | base ASR | Δ test logp | Δ test CE |
|-------------------|-----:|----:|--------:|---------:|---------:|------------:|----------:|
| MLC (L=5)         | 1497 | 2.0 | 0.30    | 0.33     | 0.99     |  -0.00      | +0.056    |
| TXC layer 0 (T=30)|  459 | 2.0 | 0.37    | 0.18     | 0.99     |  -0.24      | +0.013    |
| TXC layer 2 (T=30)|  584 | 2.0 | 0.97    | 0.97     | 0.99     |  -0.01      | +0.000    |
| **SAE layer 0**   |  831 | 1.5 | 0.05    | **0.01** | 0.99     |  +0.19      | +0.056    |
| SAE layer 1       |  698 | 2.0 | 0.95    | 0.95     | 0.99     |  -0.16      | +0.000    |

The per-token SAE at layer 0 is the suppression winner; TXC at layer 0 is
*much* cheaper on CE (+0.013 vs +0.056) but only suppresses to ASR=0.18.
That gap is the **frontier this phase asks about**: do additional temporal
inductive biases (T-SAE's adjacent-token InfoNCE; H8's multi-distance
InfoNCE) shift the frontier outward — i.e. give us TXC-like CE damage at
SAE-like suppression, or vice versa?

### Architectures under test

Four architectures, eight checkpoints (one per layer × arch combination
that fra_proj already cached at):

1. **TopK SAE** (per-token, baseline) — `sae_layer{0..3}`. `class TopKSAE`.
   Reproduces fra_proj exactly.
2. **TXC** (windowed, baseline) — `txc_early`, `txc_mid`, `txc_late` at
   layers {0, 2, 3}, T=30. `class TemporalCrosscoder`. Reproduces fra_proj.
3. **T-SAE** (per-token + adjacent contrastive) — `tsae_layer{0..3}`. New.
   Simplified port of `src/architectures/tsae_ours.py`: TopK SAE + matryoshka
   high-prefix recon (h = d_sae / 2) + symmetric InfoNCE on adjacent-token
   high-prefix latents. Inference identical to TopK SAE → reuses
   fra_proj's `compute_sae_delta`. `class TemporalContrastiveSAE`.
4. **H8-lite** (windowed + multi-distance contrastive) — `h8_{early,mid,late}`
   at layers {0, 2, 3}, T=30. New. Simplified port of
   `src/architectures/txc_bare_multidistance_contrastive_antidead.py`: TXC +
   multi-distance InfoNCE on anchor + shifted-window positives at distances
   {1, ⌊T/4⌋, ⌊T/2⌋}, with weights 1/(1+s). Inference identical to TXC →
   reuses fra_proj's `compute_txc_delta`. `class MultiDistanceTXC`.

H8-lite **omits** the original H8's matryoshka, antidead-init, and AuxK
heads. If H8-lite shows promise on the Pareto frontier, those can be folded
in for v2 — but the contrastive loss is the architecturally novel piece, so
that's what we're testing.

### Approach

Reuse fra_proj's four-stage pipeline verbatim:

```
harvest_activations.py → tokens_cache.pt + activations_cache.pt
train_crosscoders.py    → crosscoder_<arch>.pt × 16 archs
run_ablation_sweep.py   → val_sweep_<arch>.json + test_results.json
plot_pareto.py          → pareto_asr_vs_utility.png + RESULTS.md
```

The only changes vs fra_proj:

- `sae_models.py` — appended two new classes (`TemporalContrastiveSAE`,
  `MultiDistanceTXC`).
- `sleeper_utils.py` — added `TSAE_LAYER_HOOKS`, `H8_LAYER_HOOKS` constants.
- `train_crosscoders.py` — added new arch building + the multi-shape gather
  (T-SAE pair, H8 multi-window) and `compute_loss(x)` dispatch.
- `run_ablation_sweep.py` — added two `class_name` branches in
  `load_crosscoder`; per-token vs window arch dispatch via helper functions.
- `plot_pareto.py` — added 7 new entries (4 T-SAE + 3 H8) to the colour /
  label maps so they appear in the overlaid frontier.

All defaults match fra_proj: d_sae=1536, k_total=32, T=30, n_steps=8000,
δ=0.05 nats, α-grid {0.25, 0.5, 1, 1.5, 2}.

### Compute

Dedicated remote `a40_tiny_1` (NVIDIA A40, 46 GB, fresh `/root`, 100 GB free
disk). Repo at `/root/temp_xc-phase8`. See `reference_a40_tiny_1` memory.
Mac orchestrates via push/pull to `dmitry-phase8` branch on origin.

### Run mode

Fully autonomous. Self-paced check-ins at 20-minute intervals via
`ScheduleWakeup`. Run state at `experiments/phase8_tinystories_sleeper/results/RUNLOG.md`.
Decision-rule defaults and stop-and-ping conditions are spelled out in
[the approved plan file](../../../../.claude/plans/source-users-dmitrymanning-coe-document-rosy-plum.md).

### Critical files

- `experiments/phase8_tinystories_sleeper/sae_models.py` — architectures
  (the four originals + the two Phase 8 additions, marked).
- `experiments/phase8_tinystories_sleeper/sleeper_utils.py` — model loader,
  paired data loader, hooks, metrics (ASR₁₆, teacher-forced sleeper logp,
  clean-continuation CE).
- `experiments/phase8_tinystories_sleeper/train_crosscoders.py` — the only
  script that needs T-SAE/H8-specific gather logic.
- `experiments/phase8_tinystories_sleeper/run_ablation_sweep.py` — feature
  ranking + α-sweep + selection + test-set eval.
- `experiments/phase8_tinystories_sleeper/RESULTS.md` — generated table; the
  Phase 8 deliverable.

### Verification

- **Smoke test** (a40_tiny_1 with reduced d_sae=128, n_steps=200, top_k=20):
  end-to-end pipeline runs without errors and produces a non-empty
  `RESULTS.md`. Expected runtime ≤ 10 min.
- **Sanity check on baseline reproduction**: SAE layer 0 must hit
  test ASR ≤ 0.05 with Δ test CE ≤ 0.07 (fra_proj numbers ± noise).
- **Pareto plot** with all 16 archs overlaid at
  `experiments/phase8_tinystories_sleeper/outputs/plots/pareto_asr_vs_utility.png`
  (plus `.thumb.png` so I can read it in-context).
- `./run-checks.sh` clean for any new docs.
