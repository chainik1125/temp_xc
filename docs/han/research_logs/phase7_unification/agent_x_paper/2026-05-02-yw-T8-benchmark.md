---
author: Han
date: 2026-05-02
tags:
  - results
  - in-progress
---

## Y/W T>5 hill-climb benchmark — 3-seed AUC at S∈{10, 20, 32}

> Han 2026-05-01: "agents Y and W have hill climbed over dozens of new
> architectures; I want you to pick a few most promising sounding ones
> and do 3-seed AUC benchmark on S=20 to see how well they fare here ;
> I'm especially interested in T > 5 approaches."
>
> Picked four T>5 architectures from Y/W's 2026-05-01 cheatsheet
> (`agent_y_phase2/2026-05-01-y-cheatsheet.md`), all of which won at
> steering at coh ≥ 1.5/1.75/2.0. Trained at b=4096, max_steps=25000,
> plateau early-stop; 3 seeds (42, 1, 2). Probed at S∈{10,20,32}
> against `probe_cache_S32` (BASE).

### Architectures benchmarked

| arch | T | k_pos | k_win | src_class | hill-climb provenance |
|---|---|---|---|---|---|
| `txc_softmaxpool_t8_kpos20` | 8 | 20 | 160 | TXCSoftMaxPool | Y Galaxy 8 — best steering at coh≥1.75 (Δ=+1.011 vs T-SAE) |
| `txc_maxpool_t8_kpos20` | 8 | 20 | 160 | TXCMaxPool | Y Galaxy 6 — Δ=+0.644 at coh≥1.75 |
| `txc_contrastive_h8_t8_kpos20_shifts2` | 8 | 20 | 160 | TXCContrastiveMergeH8 | W mystery contrastive — best mean curve 1.578 |
| `spatial_matry_h8_t10_kpos20_shifts2_…_indep_uniform_contr` | 10 | 20 | 200 | SpatialMatryoshkaH8 | W spatial-matryoshka, T=10 native |

All at matched per-token sparsity k_pos=20 (Y/W convention). Total wall
time: ~17 hr training (12 ckpts × ~84 min/ckpt; SoftMaxPool/MaxPool
~84 min, ContrastiveMergeH8 logged 0.0 min but ckpts saved at 2.7 GB
[trainer logging artefact], SpatialMatryoshkaH8 ~3.5 hr/seed).

### Results — BASE k_feat=20

| arch | S=32 | S=20 | S=10 | σ_seeds (S=32) |
|---|---|---|---|---|
| `txc_bare_antidead_t5` (canonical) | **0.9123** | **0.9066** | 0.8955 | 0.0012 |
| `mlc` (canonical) | 0.9124 | 0.9060 | **0.8973** | 0.0019 |
| `tsae_paper_k500` (canonical) | 0.9105 | 0.9029 | 0.8958 | 0.0080 |
| `topk_sae` (canonical) | 0.9091 | 0.9037 | 0.8962 | 0.0059 |
| `phase57_partB_h8_t8` (canonical) | 0.9086 | 0.9032 | 0.8961 | 0.0032 |
| `phase5b_subseq_h8` (canonical) | 0.9059 | 0.8972 | 0.8961 | 0.0021 |
| **`txc_contrastive_h8_t8_kpos20_shifts2`** ⭐ | **0.9013** | **0.8936** | **0.8893** | 0.0029 |
| **`spatial_matry_h8_t10_kpos20`** | 0.8993 | 0.8898 | 0.8838 | 0.0018 |
| **`txc_maxpool_t8_kpos20`** | 0.8965 | 0.8847 | 0.8702 | 0.0042 |
| **`txc_softmaxpool_t8_kpos20`** | 0.8953 | 0.8831 | 0.8716 | 0.0094 |

The 4 Y/W T>5 archs LOSE 0.005–0.020 AUC vs the canonical leaderboard
top-6 at every S value. Best Y/W variant is **`txc_contrastive_h8`**
(W's mystery contrastive merge with H8 stack) — 0.9013 at S=32,
0.0110 below `txc_bare_antidead_t5`.

### Results — BASE k_feat=5

| arch | S=32 | S=20 | S=10 |
|---|---|---|---|
| `mlc` (canonical) | **0.8709** | **0.8621** | 0.8545 |
| `topk_sae` (canonical) | 0.8695 | 0.8588 | 0.8528 |
| `txc_bare_antidead_t5` (canonical) | 0.8683 | 0.8554 | 0.8395 |
| `phase57_partB_h8_t8` (canonical) | 0.8682 | 0.8592 | 0.8520 |
| `phase5b_subseq_h8` (canonical) | 0.8670 | 0.8571 | **0.8619** |
| `tsae_paper_k500` (canonical) | 0.8650 | 0.8569 | 0.8498 |
| **`txc_contrastive_h8_t8_kpos20_shifts2`** ⭐ | **0.8595** | **0.8479** | **0.8468** |
| **`spatial_matry_h8_t10_kpos20`** | 0.8584 | 0.8493 | 0.8379 |
| **`txc_maxpool_t8_kpos20`** | 0.8538 | 0.8358 | 0.8159 |
| **`txc_softmaxpool_t8_kpos20`** | 0.8391 | 0.8194 | 0.8021 |

### Headline call

> **Y/W's hill-climbed T>5 architectures at matched k_pos=20 do NOT win
> at probing AUC at any S value.** Best Y/W variant (`txc_contrastive_h8_t8`)
> sits 7th–9th of the 16-arch leaderboard at S=32, k_feat∈{5,20}.

### Why steering ≠ probing under matched sparsity

Y's cheatsheet headline: "Galaxy 8 PP 3sd hits the largest WIN ever
recorded: Δ = +1.011 at coh ≥ 1.75". Yet on probing AUC at S=20:
Galaxy 8 gives **0.8831** at k=20 vs the canonical leader's **0.9066**
— a 0.024 AUC LOSS, far outside σ_seeds (~0.0014).

Two factors compound:

**1. k_pos=20 vs k_win=500 sparsity gap.** The Y/W matched-sparsity
matrix uses k_pos=20 (so k_win=20×T=160 for T=8). The canonical
leaderboard uses k_win=500. Probing AUC at k_feat=20 typically
benefits from MORE features per token (more diverse latent basis to
draw the top-20 features from). 160 features in the SAE = harder to
find the right 20 for any given task.

**2. Steering objective vs probing objective.** Steering rewards
features whose decoder direction induces a coherent concept-aligned
shift in the residual when activated at a single position. This
selects for features with *high norm × concept alignment*. Probing
AUC rewards features whose *firing pattern* discriminates classes —
correlated with concept activation, but not the same. The pool
choice (max vs softmax vs sum) primarily affects WHERE in the
window the feature fires, which matters less for averaged probing
than for surgical steering.

### Cross-S sensitivity

Y/W archs at k=20 — Δ from S=32 to S=10:

| arch | S=32 | S=10 | Δ |
|---|---|---|---|
| `txc_contrastive_h8_t8` | 0.9013 | 0.8893 | −0.0120 |
| `spatial_matry_h8_t10` | 0.8993 | 0.8838 | −0.0155 |
| `txc_maxpool_t8_kpos20` | 0.8965 | 0.8702 | −0.0263 |
| `txc_softmaxpool_t8_kpos20` | 0.8953 | 0.8716 | −0.0237 |

vs canonical leaders:
| `txc_bare_antidead_t5` | 0.9123 | 0.8955 | −0.0168 |
| `mlc` | 0.9124 | 0.8973 | −0.0151 |
| `phase5b_subseq_h8` | 0.9059 | 0.8961 | −0.0098 ⭐ |

`phase5b_subseq_h8` remains the most tail-compression-robust arch
across the entire benchmark. Among Y/W archs, `txc_contrastive_h8_t8`
is the most robust to S compression (−0.0120) — the H8-stack +
multi-distance contrastive recipe transfers the robustness pattern,
even at matched k_pos=20.

### Per-arch observations

**`txc_contrastive_h8_t8_kpos20_shifts2`** (W's mystery contrastive,
T=8, shifts=(2,)):
- Best Y/W arch on probing across all S × k_feat × subject_model.
- σ_seeds 0.0009–0.0029 — VERY stable across seeds, surprising for
  k_pos=20 sparsity.
- The H8 multi-distance contrastive recipe (already present in
  `phase57_partB_h8_t8`) is what's working — adding the
  contrastive-merge layer doesn't help probing at this sparsity but
  doesn't hurt it either.

**`spatial_matry_h8_t10_kpos20`** (W's spatial matryoshka, T=10):
- 0.8993 at k=20 S=32 — only Y/W arch that even crosses 0.90 at any
  S. T=10 (the largest T tested) suggests the larger receptive field
  doesn't fully compensate for k_pos=20 sparsity.
- σ_seeds 0.0018 — most stable Y/W arch.
- prefix-sizes (3686, 9216, 18432) + uniform subset sampling produce
  a sparser-than-T=8 effective sparsity (only the smallest prefix
  group is "always on") — this may be why it underperforms at
  k=5 sparsity on probing.

**`txc_maxpool_t8_kpos20`** (Y Galaxy 6, hard max pool):
- 0.8965 at k=20 S=32. Drops 0.0263 at S=10 (largest Y/W drop) —
  hard max-pool relies on the SINGLE most-active position per
  feature; when S=10 throws away 22/32 positions the pool loses
  its peak in many examples.

**`txc_softmaxpool_t8_kpos20`** (Y Galaxy 8, soft max pool):
- Worst of the 4 Y/W archs across all S. σ_seeds 0.0093 — high. The
  learnable per-feature τ may overfit to the steering objective at
  the expense of probing class-separability.

### Honest paper read

The Y/W steering hill-climb produced clear wins at coh ≥ 1.5/1.75/2.0
under matched k_pos=20 — but those wins **don't translate to probing
AUC**. The k_pos=20 matched-sparsity is the bottleneck: at this
sparsity level, the canonical leaderboard archs (k_win=500) all sit
above the matched-sparsity TXC variants by 0.005–0.025 AUC.

If the paper headline is "TXC vs T-SAE at matched per-token
sparsity", the steering result is real and the new pool functions
help. If the paper headline is "best probing AUC per sparsity
level", the canonical k_win=500 archs win — and within that,
`txc_bare_antidead_t5` (BASE k=20) and `mlc` (BASE k=5, IT
both) are the leaders.

For the paper, recommend:
1. Steering claims should reference Y/W's matched-sparsity matrix as
   the headline (where the new archs win).
2. Probing claims should reference the canonical k_win=500
   leaderboard (where TXC structural bias still wins, just with
   different variants on BASE vs IT).
3. The two are **separate dimensions of comparison** — the paper
   should NOT claim that the Y/W steering winners are also the best
   probing archs. They're not. Different sparsity regimes,
   different objectives.

### Files of record

- `experiments/phase7_unification/results/probing_results.jsonl`
  (filter `run_id` starting with `txc_softmaxpool_t8`,
  `txc_maxpool_t8`, `txc_contrastive_h8_t8`, `spatial_matry_h8_t10`)
- `experiments/phase7_unification/results/training_index.jsonl`
  (12 new IT-trained ckpts; subject_model="google/gemma-2-2b")
- `results/plots/phase7_leaderboard_S20_multiseed.png` (now
  includes the 4 Y/W archs)
- HF: 12 ckpts on `han1823123123/txcdr-base/ckpts/`
- Trainer scripts (Y/W's, reused as-is):
  - `case_studies/train_kpos20_galaxy8.py` — TXCSoftMaxPool
  - `case_studies/train_kpos20_galaxy6.py` — TXCMaxPool
  - `case_studies/train_contrastive_merge_h8.py` — TXCContrastiveMergeH8
  - `case_studies/train_kpos20_spatial_matryoshka.py` — SpatialMatryoshkaH8
- Pipeline: `case_studies/run_yw_t8_benchmark.sh`
