---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Phase 7 leaderboard — 2-seed mean ± σ across {1, 42}

> Closes Han's pending tasks #5 ("Build 2-seed leaderboard with σ")
> and #7 ("Update Phase 7 summary with σ-augmented leaderboard").
> Built from `probing_results.jsonl` after a comprehensive seed=1 FLIP
> backfill across all 27 archs that have seed=1 ckpts available.

### Data

- 36 SAEBench tasks (Phase 7 standard set, includes winogrande/wsc FLIP).
- S = 32 left-aligned cache, mean-pool aggregation (existing methodology).
- FLIP applied to winogrande/wsc.
- Seed=1 + seed=42 (when both available); cells with only one seed are
  flagged below.
- Per-arch metric: cross-seed mean of per-task means.
- σ_tasks: pooled task std across both seeds (captures task variance).
- σ_seeds: std across the two per-seed means at the arch level
  (captures pure seed effect).

Code: `experiments/phase7_unification/build_leaderboard_2seed.py`.
Plot: `results/plots/phase7_leaderboard_2seed.png`.

### k_feat = 5

| arch | n_seeds | mean | σ_seeds | σ_tasks |
|---|---|---|---|---|
| hill_subseq_h8_T12_s5 | 1 | 0.8951 | — | 0.1012 |
| **phase57_partB_h8_bare_multidistance_t8** | 2 | **0.8944** | 0.0063 | 0.1040 |
| txcdr_t16 | 2 | 0.8935 | 0.0022 | 0.0972 |
| phase5b_subseq_h8 | 2 | 0.8926 | 0.0006 | 0.1048 |
| txcdr_t5 | 2 | 0.8910 | 0.0055 | 0.0986 |
| topk_sae | 2 | 0.8890 | 0.0051 | 0.1047 |
| txc_bare_antidead_t5 | 2 | 0.8830 | 0.0023 | 0.1088 |
| tsae_paper_k500 | 2 | 0.8830 | 0.0108 | 0.1071 |
| tsae_paper_k20 | 2 | 0.8722 | 0.0010 | 0.1042 |
| tfa_big | 2 | 0.7277 | 0.0337 | 0.0976 |

### k_feat = 20

| arch | n_seeds | mean | σ_seeds | σ_tasks |
|---|---|---|---|---|
| **txc_bare_antidead_t5** | 2 | **0.9360** | 0.0003 | 0.0844 |
| hill_subseq_h8_T12_s5 | 1 | 0.9329 | — | 0.0761 |
| txcdr_t5 | 2 | 0.9308 | 0.0008 | 0.0811 |
| tsae_paper_k500 | 2 | 0.9307 | 0.0046 | 0.0795 |
| topk_sae | 2 | 0.9303 | 0.0003 | 0.0854 |
| phase57_partB_h8_bare_multidistance_t8 | 2 | 0.9302 | 0.0022 | 0.0838 |
| phase5b_subseq_h8 | 2 | 0.9300 | 0.0001 | 0.0807 |
| tsae_paper_k20 | 2 | 0.9280 | 0.0018 | 0.0777 |
| txcdr_t16 | 2 | 0.9246 | 0.0020 | 0.0772 |
| tfa_big | 2 | 0.8120 | 0.0249 | 0.0857 |

### What changed vs single-seed ranking

| arch / k | seed=42 only | 2-seed mean | shift |
|---|---|---|---|
| `phase57_partB_h8_bare_multidistance_t8` k=5 | 0.8989 | 0.8944 | −0.0045 |
| `topk_sae` k=5 | 0.8855 | 0.8890 | +0.0035 |
| **TXC champ vs topk_sae k=5 gap** | **0.0134** | **0.0054** | shrunk ~3× |
| `txc_bare_antidead_t5` k=20 | 0.9358 | 0.9360 | +0.0002 |
| `topk_sae` k=20 | 0.9301 | 0.9303 | +0.0002 |
| **TXC champ vs topk_sae k=20 gap** | **0.0057** | **0.0057** | unchanged |

### Headline (corrected)

The TXC family is still the leaderboard top at both k_feat values,
but the lead over a vanilla per-token TopKSAE is **~0.005 AUC**,
not 0.013 as seed=42 alone suggested. With σ_seeds 0.005-0.010 on
the top archs, the gap is close to seed-noise scale — definitely
real and consistent in direction, but small.

Combined with the stacked-SAE control (rejects "more candidates"
hypothesis) and Y's sparsity-decomposition finding (T-SAE k=20's
steering lead is mostly a sparsity artefact), the paper-narrative
message is:

> TXC architectures are **competitive but not dominant** on
> sparse-probing AUC. Per-arch deltas are within 1-2× seed-noise
> scale at the top of the leaderboard. The structural inductive
> bias is doing real work but the magnitude is modest.

### What's NOT in this leaderboard yet

- MLC family (mlc, agentic_mlc_08, mlc_sparse, ag_mlc_08_sparse) —
  H200_required, deferred.
- 3-seed σ — adding seed=2 would tighten the σ estimates, not
  fundamentally change the picture.
- IT-side leaderboard — base-side only, IT-side activation cache and
  trainings are pending (see brief.md "Concrete remaining work").

### Files of record

- Builder: `experiments/phase7_unification/build_leaderboard_2seed.py`
- Probing rows: `experiments/phase7_unification/results/probing_results.jsonl`
- Plot: `experiments/phase7_unification/results/plots/phase7_leaderboard_2seed.png`
