---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Phase 7 leaderboard — multi-seed (1, 2, 42) — supersedes 2-seed table

> Closes the σ-tightening pass requested in Han's pending tasks #5/#7.
> Pulled seed=2 ckpts for 6 of 8 leaderboard archs (tfa_big and
> txcdr_t16 only have seeds 1+42 on HF) and probed at S=32 + FLIP.
>
> Key takeaway: the leaderboard top is **very tight** at all k_feat —
> top 8 archs at k=5 span only 0.0102 AUC, with σ_seeds 0.003-0.009 on
> top entries. The leaderboard top "champion" identity changes with
> seed selection.

### k_feat = 5

| arch | n_seeds | mean_AUC | σ_seeds | σ_tasks |
|---|---|---|---|---|
| **`phase5b_subseq_h8`** ⭐ | 3 | **0.8962** | 0.0061 | 0.1026 |
| hill_subseq_h8_T12_s5 (1 seed) | 1 | 0.8951 | — | 0.1012 |
| txcdr_t16 | 2 | 0.8935 | 0.0022 | 0.0972 |
| phase57_partB_h8_bare_multidistance_t8 | 3 | 0.8934 | 0.0048 | 0.1036 |
| txcdr_t5 | 3 | 0.8890 | 0.0053 | 0.1004 |
| topk_sae | 3 | 0.8886 | 0.0037 | 0.1044 |
| txc_bare_antidead_t5 | 3 | 0.8871 | 0.0073 | 0.1048 |
| tsae_paper_k500 | 3 | 0.8860 | 0.0093 | 0.1048 |
| tsae_paper_k20 | 3 | 0.8700 | 0.0039 | 0.1006 |
| tfa_big | 2 | 0.7277 | 0.0337 | 0.0976 |

### k_feat = 20

| arch | n_seeds | mean_AUC | σ_seeds | σ_tasks |
|---|---|---|---|---|
| **`txc_bare_antidead_t5`** ⭐ | 3 | **0.9359** | 0.0003 | 0.0831 |
| hill_subseq_h8_T12_s5 (1 seed) | 1 | 0.9329 | — | 0.0761 |
| tsae_paper_k500 | 3 | 0.9319 | 0.0039 | 0.0789 |
| phase5b_subseq_h8 | 3 | 0.9312 | 0.0021 | 0.0805 |
| phase57_partB_h8_bare_multidistance_t8 | 3 | 0.9307 | 0.0018 | 0.0828 |
| topk_sae | 3 | 0.9304 | 0.0003 | 0.0839 |
| txcdr_t5 | 3 | 0.9297 | 0.0019 | 0.0810 |
| tsae_paper_k20 | 3 | 0.9271 | 0.0020 | 0.0771 |
| txcdr_t16 | 2 | 0.9246 | 0.0020 | 0.0772 |
| tfa_big | 2 | 0.8120 | 0.0249 | 0.0857 |

### How the ranking shifted between 2-seed and 3-seed

| k_feat | 2-seed (1, 42) winner | 3-seed (1, 2, 42) winner | shift |
|---|---|---|---|
| k=5 | phase57_partB_h8_bare_multidistance_t8 (0.8944) | **phase5b_subseq_h8 (0.8962)** | +0.0028 lead change |
| k=20 | txc_bare_antidead_t5 (0.9360) | **txc_bare_antidead_t5 (0.9359)** | unchanged |

At k=5 the rank shuffles around within the top 4 (all SubseqH8 + H8 family).
At k=20 the ranking is more stable — `txc_bare_antidead_t5` is unambiguously
the top across both 2-seed and 3-seed views.

### Remaining gap to vanilla SAE

3-seed deltas TXC champion vs `topk_sae`:

| k_feat | TXC champ | topk_sae | Δ | TXC champ σ_seeds | topk_sae σ_seeds |
|---|---|---|---|---|---|
| 5  | phase5b_subseq_h8 0.8962 | 0.8886 | **+0.0076** | 0.0061 | 0.0037 |
| 20 | txc_bare_antidead_t5 0.9359 | 0.9304 | **+0.0055** | 0.0003 | 0.0003 |

The k=5 gap of 0.0076 is now ~1-2× σ_seeds: real and consistent in
direction, but small. The k=20 gap of 0.0055 is ~18× both archs'
σ_seeds — much more decisive at k=20.

### Headline message (revised, post-multi-seed)

The single-seed Phase 7 results overstated the TXC family's lead.
Properly seeded:

- TXC family **wins at k_feat=20 by ≥ 0.005 AUC** with high statistical
  confidence (σ_seeds 0.0003).
- TXC family **wins at k_feat=5 by ~0.008 AUC**, but the leaderboard
  top is tightly clustered (top 8 within 0.010) and σ_seeds 0.005-0.009
  means the champion *identity* shifts with seed selection.

This is the most honest read of the leaderboard:

> TXC family is competitive with strong per-token SAE baselines on
> 36-task SAEBench probing. At less-sparse k=20 the win is decisive
> (~0.006 AUC, ~18× σ); at very-sparse k=5 the win is real but
> ~σ-noise-magnitude (~0.008 AUC, ~1-2× σ).

Combined with the stacked-SAE concat control (rules out "more
candidates" as the source) and Y's per-concept structural finding
(TXC favoured on knowledge concepts), the paper-narrative target is:

- TXC's structural inductive bias provides a small but real probing-AUC
  advantage at all k_feat.
- The advantage is more decisive at k=20 than at k=5.
- It is concentrated on knowledge-domain content (bias_in_bios
  professions, europarl language ID, github_code at some sparsity).

### Plot

![3-seed leaderboard](../../../../experiments/phase7_unification/results/plots/phase7_leaderboard_multiseed.png)

### Files of record

- Builder: `experiments/phase7_unification/build_leaderboard_2seed.py`
- Plot: `experiments/phase7_unification/results/plots/phase7_leaderboard_multiseed.png`
- Probing rows: `experiments/phase7_unification/results/probing_results.jsonl`
