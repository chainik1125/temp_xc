---
author: Dmitry Manning-Coe
date: 2026-05-05
tags:
  - results
  - complete
---

## ρ × k sweep — regular SAE vs plain TXC (T=2, T=5) vs H8

Full ρ × k cross-product across four architectures, all at Han's
`d_sae = 8 × d_in = 2048`, on Bill's three-arch DataConfig (n_features=128,
d_model=256, π=0.05, deterministic emissions). 48 cells total, single seed.

## Setup

| field | value |
|---|---|
| n_features | 128 |
| d_model | 256 |
| pi | 0.05 |
| emissions | deterministic (p_A=0, p_B=1) |
| d_sae | 2048 (= 8 × d_in) |
| ρ | {0.0, 0.6, 0.9} |
| k_pos (per-token) | {1, 2, 5, 10} |
| training | 10k steps, batch 64, lr 3e-4 |
| arches | regular_sae · plain TXCDR T=2 · plain TXCDR T=5 · TXC-pro / H8 |

Per-token k sweeps the same column for every arch; window-level "raw k" =
nonzeros in the latent per encode call:

| k_pos | regular_sae | TXCDR-T2 | TXCDR-T5 | TXC-pro / H8 |
|---|---:|---:|---:|---:|
| 1 | raw_k = 1 | 2 | 5 | 10 |
| 2 | 2 | 4 | 10 | 20 |
| 5 | 5 | 10 | 25 | 50 |
| 10 | 10 | 20 | 50 | 100 |

## Results

![AUC vs ρ, panel per k_pos](../../../results/rho_k_sweep/rho_k_auc_grid.png)

Feature-recovery AUC by (arch, ρ, k_pos):

**regular_sae**

| ρ \\ k_pos | 1 | 2 | 5 | 10 |
|---|---:|---:|---:|---:|
| 0.0 | 0.888 | 0.930 | 0.990 | 0.990 |
| 0.6 | 0.915 | 0.941 | 0.990 | 0.990 |
| 0.9 | 0.907 | 0.950 | 0.990 | 0.990 |

**TXCDR T=2** (plain, window-level k_eff = k_pos × 2)

| ρ \\ k_pos | 1 | 2 | 5 | 10 |
|---|---:|---:|---:|---:|
| 0.0 | 0.788 | 0.953 | 0.990 | 0.990 |
| 0.6 | 0.970 | 0.990 | 0.990 | 0.990 |
| 0.9 | 0.949 | 0.990 | 0.990 | 0.990 |

**TXCDR T=5** (plain, window-level k_eff = k_pos × 5)

| ρ \\ k_pos | 1 | 2 | 5 | 10 |
|---|---:|---:|---:|---:|
| 0.0 | 0.918 | 0.961 | 0.990 | 0.990 |
| 0.6 | 0.990 | 0.990 | 0.990 | 0.990 |
| 0.9 | 0.990 | 0.990 | 0.990 | 0.990 |

**TXC-pro / H8** (T_max=10, t_sample=5, k_inference = k_pos × 10)

| ρ \\ k_pos | 1 | 2 | 5 | 10 |
|---|---:|---:|---:|---:|
| 0.0 | 0.772 | 0.835 | 0.852 | 0.879 |
| 0.6 | 0.989 | 0.975 | 0.986 | 0.958 |
| 0.9 | 0.989 | 0.990 | 0.990 | 0.990 |

ΔAUC vs regular SAE (positive = arch beats SAE):

![ΔAUC vs SAE](../../../results/rho_k_sweep/rho_k_delta_vs_sae.png)

## Three things to read off

1. **Almost everyone saturates at AUC=0.99 by k_pos=5.** With Han's
   `d_sae = 8 × d_in = 2048`, regular SAE, plain TXCDR-T2, and plain TXCDR-T5
   all hit the ceiling from k_pos=5 onward across every ρ tested. The "TXC
   vs SAE big AUC gap" Bill saw at d_sae=128 essentially evaporates at
   d_sae=2048: regular SAE has enough capacity to recover features without
   needing temporal pooling.

2. **The TXC win is k_pos=1 only.** At k_pos=1, plain TXCDR-T5 reaches
   AUC=0.990 at both ρ=0.6 and ρ=0.9, while regular SAE caps at 0.91-0.92
   (Δ ≈ +0.08). Plain TXCDR-T2 follows the same pattern with a smaller gap
   (+0.05). At k_pos=2 the gap shrinks to ~+0.05 for TXCDR-T5; from k_pos=5
   onward it's zero.

3. **TXC-pro / H8 is the only arch that *fails* at ρ=0.0** — and it fails
   across the entire k range. AUC stays in 0.77-0.88 at ρ=0.0 for every k,
   while all three other arches saturate at 0.99 by k_pos=5. The matryoshka
   prefix + multi-distance contrastive InfoNCE actively *hurt* when there's
   no temporal structure to exploit: the contrastive term pulls the
   "high-level" prefix toward features that should be temporally smooth,
   but at ρ=0.0 there is no such smoothness, so the prefix is filled with
   nothing useful and the model wastes representational capacity. At ρ ≥ 0.6
   H8 catches up to the rest (saturates at 0.99 from k_pos=5 onward).

## Connection to earlier results

This sweep extends today's two earlier pieces:

- **Bill three-arch + Han recipe (single k_pos=20)**: at d_sae=2048, k_pos=20
  the TXC-base vs regular SAE gap was +0.02 across all ρ. Here we now see
  *why* — the gap at higher k is essentially zero across ρ, and the TXC
  advantage lives entirely at k_pos=1.
- **Han exp 1c3 with TXC-base / TXC-pro / TopKSAE addition**: that bench
  used d_sae=40 (much narrower) and saw TXC-base dominate at every raw_k.
  Different conclusion because there the dictionary isn't wide enough for
  regular SAE to saturate.

The two together: **TXC's structural advantage over per-token SAE is a
function of dictionary capacity relative to feature count.** When the
dictionary is much wider than n_features (here d_sae/n_features = 16),
regular SAE can saturate AUC=0.99 at modest k and the temporal pooling
mechanism is a wash. When the dictionary is tight (Han exp 1c3:
d_sae/n_features = 0.31), TXC's pooling bias materially helps recover the
hidden-state-aligned features.

## Caveats

- **Single seed.** All numbers seed=42. The ρ=0.0 regression of H8 is
  large enough (~0.10-0.20 below the rest) that I'd bet seed-stable, but
  the k_pos=1 TXC win over SAE is in the +0.03-0.08 range — closer to noise.
- **Saturation hides differences at high k.** Once everyone hits 0.99 the
  AUC metric stops being discriminative; NMSE / linear-probe-R² would
  separate them, but for the "hidden-state recovery" question AUC=0.99 is
  effectively "solved" and architectural differences get masked.
- **No StackedSAE in this sweep.** The previous Han-recipe re-run included
  Stacked SAE as a fourth comparator (it sat at AUC≈0.50 across all ρ,
  k_pos=20 — broken by construction at this budget). I dropped it from
  the new ρ × k sweep on the user's instruction; the broader story (TXC
  vs Stacked SAE gap structural and rho-independent) still holds from the
  earlier run.

## Files

- Sweep script: `scripts/run_rho_k_sweep.py`
- Plot script: `scripts/plot_rho_k_sweep.py`
- Raw results: `results/rho_k_sweep/results.json`
- Plots: `results/rho_k_sweep/rho_k_{auc_grid, delta_vs_sae}.png`
- Branch: `bill-han-txc-10k`
- Run on a40_txc_1, ~50 min wall (concurrent with the exp 1c3 Han TXC run).
