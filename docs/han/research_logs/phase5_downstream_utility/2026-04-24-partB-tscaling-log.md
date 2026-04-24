---
author: Han
date: 2026-04-24
tags:
  - results
  - in-progress
---

## Phase 5.7 Part B — T-scaling autoresearch log

Pursues the goal in
[`2026-04-23-handover-t-scaling-autoresearch.md`](2026-04-23-handover-t-scaling-autoresearch.md):
find a TXC architecture whose sparse-probing AUC monotonically increases
with T under the fixed evaluation protocol (top-k-by-class-sep + L1 LR
at k_feat=5, seed=42, 36 tasks).

**Targets**:

- Monotonicity score ≥ 0.8 (fraction of T-pairs (i<j) with auc(T_j) ≥ auc(T_i)).
- Δ(T=30 − T=5) > +0.02.

**Baselines** (from §T-sweep matrix in summary.md):

| family | last_position mono | last_position Δ(T20-T2) | mean_pool mono | mean_pool Δ(T20-T2) |
|---|---|---|---|---|
| vanilla TXCDR × TopK | 0.52 | +0.006 | 0.33 | **−0.024** (anti-monotone) |
| vanilla TXCDR × BatchTopK | 0.57 | +0.006 | 0.43 | −0.006 |
| agentic_txc_02 × TopK (T∈{2,3,5,8}) | ≤0.67 | +0.026 | ≤0.67 | +0.013 |

None reach target. Part B explores architectural changes.

**Baseline T=30 infeasible** (A5 finding, 2026-04-24): vanilla TXCDR at
T=30, d_sae=18432 OOMs during Adam-state init on A40 (2.55B params × 3
for Adam = 30.6 GB, exceeds 44 GB available after activations). Any
Part B hypothesis must be **more parameter-efficient** than vanilla
TXCDR to produce a T=30 data point. ConvTXCDR (H1) at T=30 is ~1.4B
params (decoder dominates); LogMatryoshkaTXCDR (H3) at T=30 with scales
{1,2,4,8,16} is ~2.4B params (excludes scale 32 > T).

**Fixed evaluation protocol** (reward-hacking forbidden): see
[§Evaluation protocol is FIXED](2026-04-23-handover-t-scaling-autoresearch.md#evaluation-protocol-is-fixed-do-not-modify-it).

---

## Cycle 1 — H1: convolutional encoder

**Hypothesis**: replace per-position `W_enc : (T, d_in, d_sae)` with a
shared 1-D conv across T positions (kernel=3, same-padding, sum-pooled
to (B, d_sae)). Translation-invariance → features aren't position-
imprinted → longer T supplies more windows of consistent evidence per
feature.

**Arch**: [`src/architectures/conv_txcdr.py::ConvTXCDR`](../../../src/architectures/conv_txcdr.py)
with encoder params T-invariant (127M regardless of T, vs TemporalCrosscoder's
(T·d_in·d_sae) = {212M at T=5, 1.27B at T=30}).

**Training**: standard TXCDR pipeline (plateau-stop, k=100·T, seed=42,
25k max steps, Adam lr=3e-4, batch=1024). Decoder unchanged
(`W_dec : (d_sae, T, d_in)`).

**Probing**: standard (top-k-by-class-sep + L1 LR at k_feat=5, 36 tasks).

**Launcher**:
[`run_partB_h1.sh`](../../../experiments/phase5_downstream_utility/run_partB_h1.sh).

**Results** (T ∈ {5, 10, 15, 20, 30}, seed=42):

| T | lp AUC | mp AUC | vs txcdr_t5 3-seed lp (0.7811) |
|---|---|---|---|
| 5 | 0.7612 | 0.7731 | −0.020 |
| 10 | 0.6881 | 0.7030 | −0.093 |
| 15 | 0.6889 | 0.6932 | −0.092 |
| 20 | 0.6859 | 0.6934 | −0.095 |
| 30 | 0.7294 | — | −0.052 |

**Scores**:
- last_position: monotonicity = **0.4** (target ≥0.8), Δ(30−5) = **−0.0318** (target >+0.02)
- mean_pool: monotonicity = **0.17**, Δ(T20-T5) = **−0.0796**

**H1 FAILS both targets decisively.**

**Verdict**: ConvTXCDR does NOT exhibit T-scaling. Translation
invariance in the encoder alone is insufficient — in fact, the T=5
baseline regresses by ~2 pp vs txcdr_t5, and T=10-20 regress by ~9 pp.
The U-shape recovery at T=30 (+0.044 vs T=20) suggests some long-range
structure is learned but the per-feature decoder still dominates the
probing quality.

**Mechanism hypothesis for failure**: Conv encoder with fixed kernel=3
can only see a 3-token window locally at each position, then sum-pools
across T. Summing kills position-specific information. For probing
that cares about "what's happening at the last token", summing smears
the signal.

**Next**: H3 log-matryoshka. The decoder is the limiting factor, not
the encoder.

---

## Cycle 2 — H3: log-scale matryoshka

**Hypothesis**: fixed log-spaced matryoshka scales {1, 2, 4, 8, 16, 32}
(truncated to T) instead of per-position scales. Matryoshka decoder
scales as O(Σ s²·d_in·d_sae/n_scales) rather than O(T³·d_in) — trainable
at T≥30 where per-position matryoshka OOMs on A40.

**Arch**: [`src/architectures/log_matryoshka_txcdr.py::LogMatryoshkaTXCDR`](../../../src/architectures/log_matryoshka_txcdr.py).

**Launcher**:
[`run_partB_h3.sh`](../../../experiments/phase5_downstream_utility/run_partB_h3.sh).

**Results** (T ∈ {5, 10, 15, 20, 30}): _pending_

---

## Cycle outcomes (filled when complete)

| cycle | hypothesis | mono_lp | Δ(30-5)_lp | mono_mp | Δ(30-5)_mp | verdict |
|---|---|---|---|---|---|---|
| 1 | H1 conv encoder | _ | _ | _ | _ | _ |
| 2 | H3 log-matryoshka | _ | _ | _ | _ | _ |
| 3 | H5 SVD regularizer | _ | _ | _ | _ | _ |
| 4 | H4 multi-distance contrastive | _ | _ | _ | _ | _ |
| 5 | H2 attention-pool decoder | _ | _ | _ | _ | _ |
| 6 | H7 MLC-enc + T-dec hybrid | _ | _ | _ | _ | _ |
| 7 | H8 stack winners | _ | _ | _ | _ | _ |

---

## Pivot criteria

If no cycle achieves both targets after 5-8 cycles, the paper **pivots
to headlining MLC** (agentic_mlc_08 + mlc_contrastive_alpha100_batchtopk
— the two MLC-family leaders at last_position). TXC becomes a
complementarity argument, not the headline.
