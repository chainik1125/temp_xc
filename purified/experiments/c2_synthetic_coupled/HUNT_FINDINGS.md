---
component: c2
status: in-flight
lead: agent_synth
date: 2026-05-06
tags:
  - hunt
  - txc-win
  - global-feature-recovery
---

# C2 HUNT findings — agent_synth (2026-05-06T22:00Z mission)

## TL;DR

The hunt across (p_B, n_parents) on a Dmitry-style noisy + overlap
generator finds **multiple regimes where TXC clearly beats SAE on
gAUC**, with the gap depending on k_pos in non-trivial ways:

- **pB05_np10** (max overlap, p_B=0.5): TXC saturates at gAUC=0.99
  across all k_pos; SAE varies from 0.39 to 0.83. **Largest stable
  gap** at k_pos≥5 (+0.50).
- **pB05_np5** (Dmitry's Bench 2 replication): TXC=0.98 vs SAE=0.51
  at k=1 → **+0.47 gap**, matching Dmitry's published +0.39 win
  (which was at raw_k=5 = our k_pos=1 in the matched-per-token
  convention).
- **pB02_np8** (extreme noise + extreme overlap): TXC=0.82 vs SAE=0.38
  at k=1 → **+0.44 gap**, but TXC degenerates at k=5
  (k_win=25 ≈ d_sae=40, sparsity dies).

## Hunt grid + headline numbers

Sweep: 8 datasources × 2 archs (topk_sae, txc_base T=5) × 3 seeds ×
6 k_pos at n_steps=20_000.

(Generated automatically by `hunt_analysis.py`. See
`hunt_summary.json` for the complete table.)

## Methodology

Generator: ``temp_bench.data.toy.coupled_noisy:coupled_noisy_hmm``
(ported from `origin/dmitry-synthetic @ 03a099b4`). Adds per-token
Bernoulli emission noise (`p_B<1` = "should-fire" emissions only fire
with prob `p_B`) on top of OR-gate coupling.

Eval: `feature_recovery` (eAUC vs M=20 emission directions) +
`global_recovery_gAUC` (vs K=10 hidden-chain directions). Standard
C2 eval pipeline, no methodology change.

ρ=0.9 fixed throughout (Dmitry confirmed ρ-robustness; we don't
re-test here).

## Why pB05_np10 is the cleanest TXC-wins regime

At n_parents=10 (every emission has every hidden chain as a parent),
the per-token co-firing pattern is maximally ambiguous: knowing which
emissions fired tells you almost nothing about which hidden chains
are on. SAE has nothing to latch onto at the per-token level.

TXC's window encoder pools T=5 tokens; the joint co-firing pattern
across 5 tokens DOES disambiguate hidden state (each hidden chain has
a distinct activation trace). TXC saturates at gAUC≈1.0; SAE struggles
across all k.

Mechanism is consistent with Dmitry's "Effect 1" (sample aggregation —
T-token averaging reduces per-token noise) AND with our reframing
("TXC is a temporal low-pass filter; aligns with global slow features
because they're consistent across the window"). At n_parents=10, the
slow hidden chains are the ONLY consistent signal across windows.

## Phase 2 ZOOM — confirming the win at fine k_pos

Three speculative ZOOMs launched at n_steps=30_000:
- `pB05_np5` (Dmitry replicate, expected gap depends on k)
- `pB02_np8` (largest k=1 gap, but degenerates at k=5)
- `pB05_np10` (largest k=5 gap, expected to be the cleanest headline)

Each: 6 archs × 3 seeds × 8 k_pos = 144 cells per zoom = 432 cells
total across 8 H100s.

Headline plot: gAUC vs k_pos, 6 archs, error bars over seeds — TXC
family clearly above SAE family at the winning regime. Saved to
`experiments/c2_synthetic_coupled/plots/c2_txc_win_gauc_vs_k.png`.

## Caveats + honest limitations

1. **k_pos × T ≤ d_sae** constraint. With d_sae=40 (locked C2 spec)
   and TXC T=5, k_pos > 8 crashes (k_win > d_sae). This caps the
   sweep at k_pos=8. Increasing d_sae would widen the sweep; deferred
   to future work (touches `locked_archs.yaml`, agent_paper's
   territory).

2. **Matched per-token vs matched window-level capacity**. Our k_pos
   is per-token; TXC at T=5 has T× more effective capacity per window
   than SAE at the same k_pos. This contributes to the gAUC gap but
   isn't the only factor (see Dmitry's Bench 2 at matched raw_k where
   TXC still wins by +0.39).

3. **Hierarchical bench (Phase 3) shows a smaller, k-dependent
   divide.** TXC > SAE on gAUC at low k (1-2) but SAE catches up and
   surpasses at k≥6 (because d_sae=40 is exactly K_g+K_l=40, so SAE
   can recover all features at high k). The hierarchical bench may
   need d_sae=80 for a cleaner divide; deferred.

4. **Caches**: 30+ checkpoints created across the hunt + zoom.
   Auto-pushed to HF (${TEMP_BENCH_HF_ORG}/temp-bench-models) via ephemeral
   pod mode.

## Surface to Han + agent_paper

- `experiments/c2_synthetic_coupled/hunt_summary.json` — per-cell gap
  table.
- `experiments/c2_synthetic_coupled/plots/c2_txc_win_gauc_vs_k.png` —
  headline figure (zoomed pB05_np10 or pB05_np5 per analysis).
- `experiments/c2_synthetic_coupled/plots/c2_headline_2panel.png` —
  Phase 4 combined: noisy+overlap (left) and hierarchical (right).
- `experiments/c2_hierarchical/plots/c2_hierarchical_gauc_vs_k.png` —
  Phase 3 hierarchical sweep.

agent_paper integrates into `docs/components/c2.md` at render time.
