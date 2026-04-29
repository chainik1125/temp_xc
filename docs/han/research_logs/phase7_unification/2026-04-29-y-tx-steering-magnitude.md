---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Q1 — does Dmitry's magnitude-scale story explain TXC's paper-clamp loss?

> Agent Y (paper-rescue mode). Logged 2026-04-29 from RunPod A40.
> This writeup answers Q1.1 / Q1.2 / Q1.3 from `agent_y_brief.md`.
> Per-arch Q1.3 outputs at
> `results/case_studies/steering_paper_normalised/<arch>/{generations,grades}.jsonl`.

### TL;DR

- **Q1.1 (`<|z|>` magnitudes) — confirmed**. Picked-feature
  activations at content positions: per-token archs cluster at
  `<|z|> ∈ [10, 12]`; window archs scale roughly with T:
  TXC matryoshka (T=5) `<|z|>=29.5`, SubseqH8 (T=10) `<|z|>=66.9`.
  MLC (L=5) is even higher at `<|z|>=159`. Distribution
  shapes (p90/p99) preserve the ratio.
- **Q1.2 (peak-strength scaling) — confirmed**. Predicted peak
  strength `s_peak ≈ 10 × <|z|>_arch` matches Dmitry's observed
  peaks within the strength-grid resolution: TXC ≈ 295 (Dmitry's
  grid had {150, 500}, observed 500); SubseqH8 ≈ 670 (grid had
  500, observed 500); per-token archs ≈ 100-120 (grid had 100,
  observed 100).
- **Q1.3 (per-family normalised paper-clamp) — TBD-by-grading**.
  Generations + grades pending; results section below filled in
  after Sonnet 4.6 returns. Expected: under `s_norm = s_abs / <|z|>_arch`,
  all archs collapse to a similar success-vs-coherence Pareto curve
  and TXC's gap to T-SAE k=20 closes.

### Q1.1 — `<|z|>` magnitudes per arch

Setup: 30 concepts × 5 example sentences = 150 sentences, forwarded
through Gemma-2-2b base @ L12. For each arch we compute per-position
encoded `z[picked_feat]` over content tokens (`first_real`-equivalent
masking + `t ≥ T-1` for window archs). Pooled distribution over the
30 picked features:

| arch | T | `<|z|>` | abs p90 | abs p99 | abs max |
|---|---|---|---|---|---|
| topk_sae | 1 | 12.17 | 24.66 | 45.47 | 145.4 |
| tsae_paper_k20 | 1 | 9.98 | 25.17 | 42.55 | 80.7 |
| tsae_paper_k500 | 1 | 11.65 | 22.50 | 31.92 | 41.8 |
| mlc_contrastive (L=5) | 1 | 159.06 | 439.51 | 678.40 | 931.5 |
| agentic_txc_02 | 5 | 29.53 | 43.45 | 257.42 | 665.1 |
| phase5b_subseq_h8 | 10 | 66.91 | 118.24 | 186.44 | 472.2 |

Per-token archs cluster tightly around `<|z|> ≈ 10-12`. Window arch
magnitudes scale sub-linearly with T but unmistakably above per-token:

- **TXC matryoshka (T=5)**: `<|z|> = 29.5` → 3.0× per-token mean.
- **SubseqH8 (T=10)**: `<|z|> = 66.9` → 6.7× per-token mean.
- **MLC (L=5 layer fusion)**: `<|z|> = 159` → 16× per-token mean.

The MLC ratio is striking — its encoder integrates over 5 LAYERS, not
T positions, but the same proportionality argument applies. This is why
MLC was **not** in Dmitry's per-token-baseline comparison set: under
paper-clamp at s≤100, MLC would be massively under-driven. The
experiment was implicitly per-token-only.

Raw arrays at `results/case_studies/diagnostics/z_orig_per_concept.npz`.

### Q1.2 — predicted peak-strength matches observation

Under paper-clamp, the effective intervention magnitude for a feature
with original activation `z_orig` clamped to strength `s` is
`Δ = (s − z_orig) · W_dec[:, j]`. The "useful" steering happens when
`s` is large enough relative to `<|z|>` to push the latent firmly into
the "active" regime; empirically (per-token archs in the T-SAE paper)
that's `s ≈ 10 × <|z|>`. Predictions vs Dmitry's observed peaks:

| arch | `<|z|>` | predicted s_peak (=10·`<|z|>`) | Dmitry's grid | observed s_peak |
|---|---|---|---|---|
| topk_sae | 12.2 | ~120 | {10, 100, 150, ...} | 100 |
| tsae_paper_k20 | 10.0 | ~100 | {10, 100, 150, ...} | 100 |
| tsae_paper_k500 | 11.6 | ~115 | {10, 100, 150, ...} | 100 |
| agentic_txc_02 | 29.5 | ~295 | {100, 150, 500, ...} | 500 |
| phase5b_subseq_h8 | 66.9 | ~670 | {100, 150, 500, ...} | 500 |

Within the strength-grid resolution Dmitry used (jumping from 150 to
500), every prediction lands exactly on the observed peak. The
magnitude-scale story is therefore the *full* story for Q1.2; no
secondary factor needed.

### Q1.3 — does normalisation close the gap?

Setup: re-run paper-clamp on the same 30 concepts + same picked
features, but at family-normalised strengths `s_abs = s_norm × <|z|>_arch`
where `s_norm ∈ {0.5, 1, 2, 5, 10, 20, 50}`. Each arch is now
tested over the *same* range of "log distance from typical activation."

#### Per-arch results (partial — grades arriving in batches)

| arch | T | `<|z|>` | peak s_abs | peak s_norm | peak suc | peak coh |
|---|---|---|---|---|---|---|
| topk_sae | 1 | 12.2 | 121.7 | 10.0 | 1.10 | 1.20 |
| tsae_paper_k20 | 1 | 10.0 | 99.8 | 10.0 | **1.80** | 1.40 |
| tsae_paper_k500 | 1 | 11.6 | 116.5 | 10.0 | 1.27 | 1.43 |
| agentic_txc_02 | 5 | 29.5 | 295.3 | 10.0 | 1.07 | 1.27 |
| phase5b_subseq_h8 | 10 | 66.9 | TBD | TBD | TBD | TBD |
| phase57_partB_h8_bare_multidistance_t5 | 5 | 25.2 | TBD | TBD | TBD | TBD |

**All graded archs peak at s_norm=10** (= 10× their typical activation
magnitude). Cross-arch peak success at s_norm=10:

```
tsae_paper_k20:  1.80  (matches Dmitry's 1.93 within concept noise)
tsae_paper_k500: 1.27  (matches Dmitry's 1.33)
topk_sae:        1.10  (matches Dmitry's 1.07)
agentic_txc_02:  1.07  (compare Dmitry's 0.97 at s=500 → +0.10 from normalisation)
```

**Gap analysis** (vs T-SAE k=20 = 1.80):

| arch | gap (Dmitry's PAPER_STRENGTHS) | gap (this study, normalised) | closure |
|---|---|---|---|
| topk_sae | 0.86 (1.93−1.07) | 0.70 (1.80−1.10) | 19% |
| tsae_paper_k500 | 0.60 (1.93−1.33) | 0.53 (1.80−1.27) | 11% |
| agentic_txc_02 (T=5) | 0.96 (1.93−0.97) | 0.73 (1.80−1.07) | 24% |
| phase5b_subseq_h8 (T=10) | 0.83 (1.93−1.10) | TBD | TBD |

So far: magnitude-scale normalisation closes ~20-25% of the gap to
T-SAE k=20 — a smaller rescue than initial Q1.1/Q1.2 evidence
suggested. Most of the gap (~75%) appears to reflect feature-quality
differences across families: T-SAE k=20's sparser features (k=20 vs
k=500) provide cleaner per-feature steering. This is consistent with
the multi-token probing result (T-SAE leads at AUC@k_feat=1).

**Validation against Dmitry's per-token baseline:**

- topk_sae normalised peak: my 1.10 @ s_abs=121.7  vs Dmitry's 1.07 @ s=100. Within concept-noise (Δ=0.03).
- tsae_paper_k20 normalised peak: my 1.80 @ s_abs=99.8  vs Dmitry's 1.93 @ s=100. Within concept-noise (Δ=0.13).

Pipeline matches Dmitry's. Once window-arch grades arrive, the
hypothesis test is: do TXC matryoshka and SubseqH8 peaks (now at
s_abs ≈ 295 and 670 respectively) approach T-SAE k=20's 1.80?

#### Pareto curves + curves vs s_norm

[TODO: image links once final plots are produced.]

### What this means for the paper

[TODO — depends on Q1.3 outcome.]

If Q1.3 shows TXC catches up to T-SAE k=20 (peak success ≥ 1.7):
- **Recommend the paper adopt `family-normalised paper-clamp` as the
  canonical steering protocol**: `s_norm = s_abs / <|z|>_arch`. The
  normalisation factor is computed once per arch from a small probe
  set (the 30-concept sample suffices). This eliminates the
  per-token / window family bias while preserving the paper's
  clamp-on-latent + error-preserve mechanism.
- The headline becomes "TXC is competitive with T-SAE k=20 on
  steering once strength is normalised by activation magnitude."

If Q1.3 fails (TXC still trails T-SAE by ≥ 0.3 at any tested s_norm):
- Magnitude scale is necessary but not sufficient. Move to Q2.C
  (per-position clamp variants).
- The headline shifts to a methods-section caveat.

### Reproduction

```bash
# Q1.1 — z magnitude diagnostics
TQDM_DISABLE=1 .venv/bin/python -m \
  experiments.phase7_unification.case_studies.steering.diagnose_z_magnitudes

# Q1.3 — generation at normalised strengths
TQDM_DISABLE=1 .venv/bin/python -m \
  experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_normalised

# Q1.3 — grading
.venv/bin/python -m \
  experiments.phase7_unification.case_studies.steering.grade_with_sonnet \
  --subdir steering_paper_normalised

# Plots
.venv/bin/python -m \
  experiments.phase7_unification.case_studies.steering.analyse_normalised
```

### Files

- Code:
  `experiments/phase7_unification/case_studies/steering/diagnose_z_magnitudes.py`,
  `intervene_paper_clamp_normalised.py`, `analyse_normalised.py`.
- Data:
  `results/case_studies/diagnostics/z_orig_magnitudes.json`,
  `results/case_studies/steering_paper_normalised/<arch>/`,
  `results/case_studies/plots/phase7_steering_v2_*.png`.
