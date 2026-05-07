---
component: c2
status: audit
lead: agent_filler
date: 2026-05-06
tags:
  - audit
  - reproduction
  - wasteland
---

## C2 wasteland audit — parameter-by-parameter comparison

This is the detailed accounting of how our two C2 setups map to the
wasteland source experiments. Use as reference when validating numbers,
re-deriving formulas, or deciding whether to re-train for a tighter
match. The paper-facing summary lives in `c2.md`; that file is kept
clean. This file is the audit trail.

## Setup A — Coupled features (1c3-coupled)

**Wasteland source**: [`docs/legacy/research_logs/phase3_coupled_features/2026-04-07-experiment1c3-coupled-features.md`](https://anonymous.4open.science/r/temp-bench/blob/wasteland-canonical/docs/legacy/research_logs/phase3_coupled_features/2026-04-07-experiment1c3-coupled-features.md)
on `wasteland-canonical`.

The prior author's coupled-feature pipeline. K hidden chains drive M emissions
through OR-gate coupling. Two ground truths (emission directions vs
hidden directions). Tests whether TXC recovers hidden chain structure
that per-token SAEs miss.

**Datasource**: `toy_coupled_K10_M20_d256` (configs/datasources.yaml).

| Parameter | Ours | Wasteland | Match |
|---|---|---|---|
| K (hidden chains)   | 10 | 10 | ✓ |
| M (emissions)       | 20 | 20 | ✓ |
| n_parents           | 2  | 2  | ✓ |
| π (hidden ON)       | 0.05 | 0.05 | ✓ |
| ρ                   | 0.7 | 0.7 | ✓ |
| d (residual)        | 256 | 256 | ✓ |
| d_sae               | 40 | 40 | ✓ |
| seq_len T           | 64 | 64 | ✓ |
| Magnitudes          | folded N(1, 0.15) | folded N(1, 0.15) | ✓ |
| Eval seeds          | {1, 2, 42} (n=3) | {42} (n=1) | n>wasteland |

### Cross-check at k=2

| Model | Ours gAUC | Wasteland gAUC | Δ |
|---|---:|---:|---:|
| `txc_pro` T=2  | 0.990 ± 0.000 | 0.990 | **EXACT** |
| `txc_pro` T=5  | 0.914 ± 0.009 | 0.971 | -0.057 |
| `txc_pro` T=12 | 0.842 ± 0.020 | 0.949 | -0.107 |
| `stacked_sae` T=2 | 0.764 ± 0.051 | 0.755 | +0.009 (within seed-noise) |

Wasteland used TXCDRv2 (different code lineage); our `txc_pro` is the
locked `final` arch with similar mathematical structure (subseq
encoder + matryoshka). T=2 matches exactly; higher T diverges, likely
because txc_pro's matryoshka prefix scheme (`h_size=40`, t_sample=2)
differs from TXCDRv2's window-only design at higher T.

## Setup B — Noisy independent emissions (1c-noisy)

**Wasteland source**: [`docs/legacy/research_logs/2026-03-30-experiment1c-noisy-emissions.md`](https://anonymous.4open.science/r/temp-bench/blob/118fde8/docs/legacy/research_logs/2026-03-30-experiment1c-noisy-emissions.md)
at commit `118fde8`.

20 independent Markov chains (NOT coupled). Bernoulli emission noise
on each: P(s_i=1 | h_i=1) = p_B, P(s_i=1 | h_i=0) = p_A. Tests
whether temporal models can DENOISE the per-token observation s_i
into the underlying state h_i.

**Datasource**: `toy_markov_n20_d40_noisy` (configs/datasources.yaml).
Internal component name `c1_noisy` (kept for leaderboard provenance);
paper-component-wise this is the second C2 setup per the prior author's structural
mapping.

| Parameter | Ours | Wasteland | Match |
|---|---|---|---|
| n_features        | 20 | 20 | ✓ |
| d (residual)      | 40 | 40 | ✓ |
| ρ                 | 0.7 | λ=0.3 → ρ=0.7 | ✓ |
| p_A               | 0  | 0  | ✓ |
| p_B               | 0.625 | 0.625 (q=0.8, γ=0.25) | ✓ |
| **π** (hidden ON) | **0.5** | **wasteland targets μ=q·p_B=0.5 = P(s=1), implies P(h=1)=0.8** | **✗** |
| Magnitudes        | folded N(1, 0.15) | folded N(1, 0.15) | ✓ |
| d_sae             | 40 | 40 | ✓ |
| seq_len T         | 64 | 64 | ✓ |
| Eval seeds        | {1, 2, 42} (n=3) | {42} (n=1) | n>wasteland |
| Train batch       | 1024 (uniform) | 2048 (Stacked/TXCDRv2), 64 (TFA-pos) | ✗ (the prior author: no redo) |
| Train lr          | 3e-4 (uniform) | 3e-4 (Stacked/TXCDRv2), 1e-3 (TFA-pos) | ✗ (the prior author: no redo) |

### The π mismatch — derivation

Wasteland targets the marginal `μ = P(s=1) = q × p_B = 0.8 × 0.625 = 0.5`.
With p_A=0, p_B=0.625, this requires `P(h=1) = μ / p_B = 0.5 / 0.625 = 0.8`.

Our datasource sets `pi=0.5` directly, which makes `P(h=1) = 0.5`
(not 0.8). Then `P(s=1) = 0.5 × 0.625 + 0.5 × 0 = 0.3125` (not 0.5).

Resulting differences:

|                       | Wasteland | Ours |
|---|---:|---:|
| P(h=1)                | 0.8 | 0.5 |
| P(s=1)                | 0.5 | 0.3125 |
| Cov(s, h)             | 0.10 | 0.156 |
| Var(s)                | 0.25 | 0.215 |
| Var(h)                | 0.16 | 0.25 |
| **Corr(s, h)**        | **0.50** | **0.674** |
| sl-ratio per-token floor | 0.50 | 0.674 |
| lp R² per-token floor    | 0.25 | 0.45 |

The denoising story qualitatively reproduces (TXC > floor; Stacked /
TFA = floor) but absolute ratios shift up across the board. To
bit-faithfully reproduce wasteland's table, set `pi=0.8` in the
datasource and re-train all c1_noisy cells.

### Cross-check denoising metrics at k=5

| Model | Ours sl_ratio | Wasteland sl_ratio | Ours lp_ratio | Wasteland lp_ratio |
|---|---:|---:|---:|---:|
| TFA-pos               | ~0.65 | 0.50 (floor) | ~0.55 | 0.25 (floor) |
| Stacked T=2           | ~0.65 | 0.50 (floor) | ~0.45 | 0.25 (floor) |
| Stacked T=5           | ~0.65 | 0.50 (floor) | ~0.45 | 0.25 (floor) |
| **TXC-base T=2**      | **~0.83** | 0.74 | **~0.75** | 0.53 |
| **TXC-base T=5 (default)** | **~1.01** | 0.97 | **~1.14** | ≈1.0 |
| **TXC-base T={4,6,8,10,12}** | **~1.00-1.05** | (not in wasteland; T=2..12 sweep added per (decision 2026-05-06)) | **~1.06-1.34** | — |
| TXC-pro T_max=10      | ~1.05 | (not in wasteland) | ~1.34 | — |

**Reading guide**: ratio > floor ⇒ denoising. ratio ≈ floor ⇒ no
denoising. Both metrics agree on the qualitative tier separation:
TFA + Stacked = floor; TXC family > floor.

### AUC at k=5, side-by-side

Wasteland AUC at k=5 vs ours (single ground truth = 20 feature
directions, decoder-cosine method):

| Model | Ours | Wasteland | Δ |
|---|---:|---:|---:|
| TFA-pos               | 0.788 ± 0.008 | 0.958 | -0.170 |
| Stacked T=2           | 0.585 ± 0.016 | 0.706 | -0.121 |
| Stacked T=5 (default) | 0.531 ± 0.029 | 0.550 | -0.019 |
| **TXC-base T=2**      | **0.990 ± 0.000** | 0.990 | **EXACT** |
| **TXC-base T=5**      | **0.990 ± 0.000** | 0.990 | **EXACT** |

TXC AUC matches exactly at k=5. TFA / Stacked are lower than
wasteland at low k due to the batch-size + lr mismatch (wasteland
batch=64 lr=1e-3 for TFA, 2048 / 3e-4 for Stacked). Per the prior author's
"only redo TFA with small batch if TFA at 1024 takes ages" rule, no
redo. The TXC headline reproduces precisely, which is what matters
for the paper claim.

## Open follow-ups for full bit-faithful reproduction

1. **π fix** (Setup B): change `toy_markov_n20_d40_noisy.pi` from
   0.5 to 0.8 + re-train all c1_noisy cells. Brings Corr(s, h) to
   0.50 and floor to 0.50/0.25 (sl/lp). ~30 min wall on 7 GPUs.
2. **TFA / Stacked batch fix** (Setup B): use batch=64 lr=1e-3 for
   TFA-pos, batch=2048 lr=3e-4 for Stacked. Likely improves their
   AUC numbers; doesn't change denoising story (they sit at floor
   either way).
3. **TXC-pro inference k_pos overflow**: the inference path uses
   `k_pos × T_max` for topk against `width=d_sae=40`. Cells with
   k_pos × T_max > 40 crash. agent_paper territory — fix in
   `txc_pro.encode` to cap k_inference at d_sae.
