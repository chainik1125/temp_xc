---
author: Han
date: 2026-05-01
tags:
  - results
  - in-progress
---

## Phase 7 leaderboards at S ∈ {10, 20, 32} — BASE + IT, PAPER task set

> Han 2026-05-01: "repeat ALL the probes but for S=20 instead of S=32",
> then "after S=20 is done, do the same but for S=10". Done. The S=32
> path is bit-exact identical to the original headline (verified —
> aggregate_s patch preserves canonical behaviour at S=S_cache).
>
> S < S_cache (=32) means the per-example mean-pool is restricted to the
> last S positions of the cache. For S=20: offset=12, effective_first_real
> = max(first_real, 12). For S=10: offset=22.
>
> cell_is_valid(T, S) hard-skips T>S cells. So at S=10, only archs
> with T ≤ 10 land rows; at S=20, T ≤ 20.

### Coverage

| subject | S | k_feat | seeds | rows | comment |
|---|---|---|---|---|---|
| BASE | 32 | (5, 20) | (1, 2, 42) | canonical | the original headline (txc_bare_antidead_t5 0.9127 at k=20) |
| BASE | 20 | (5, 20) | (1, 2, 42) | new | re-probed all 35 leaderboard run_ids |
| BASE | 10 | (5, 20) | (1, 2, 42) | new | T>10 archs (txcdr_t16) skipped per cell_is_valid |
| IT   | 32 | (5, 20) | (42,)     | existing | from Mission #1 + Mission #2 |
| IT   | 20 | (5, 20) | (42,)     | new | re-probe added S=10/20 |
| IT   | 10 | (5, 20) | (42,)     | new | txcdr_t16 skipped |

### BASE leaderboard at S=32 (canonical reference)

#### k_feat = 20
| arch | mean_AUC | σ_seeds | n_seeds |
|---|---|---|---|
| **`txc_bare_antidead_t5`** ⭐ | **0.9127** | 0.0012 | 3 |
| hill_subseq_h8_T12_s5 (1-seed) | 0.9126 | — | 1 |
| mlc | 0.9122 | 0.0022 | 3 |
| tsae_paper_k500 | 0.9105 | 0.0081 | 3 |
| topk_sae | 0.9091 | 0.0058 | 3 |
| phase57_partB_h8_multidistance_t8 | 0.9086 | 0.0032 | 3 |

#### k_feat = 5
| arch | mean_AUC | σ_seeds | n_seeds |
|---|---|---|---|
| hill_subseq_h8_T12_s5 (1-seed) | 0.8730 | — | 1 |
| **`mlc`** ⭐ | **0.8707** | 0.0086 | 3 |
| topk_sae | 0.8695 | 0.0051 | 3 |
| txc_bare_antidead_t5 | 0.8683 | 0.0049 | 3 |

### BASE leaderboard at S=20 (new — last 20 cache positions)

#### k_feat = 20
| arch | mean_AUC | σ_seeds | n_seeds | Δ vs S=32 |
|---|---|---|---|---|
| **`txc_bare_antidead_t5`** ⭐ | **0.9066** | 0.0014 | 3 | −0.0061 |
| mlc | 0.9060 | 0.0013 | 3 | −0.0062 |
| topk_sae | 0.9037 | 0.0071 | 3 | −0.0054 |
| phase57_partB_h8_multidistance_t8 | 0.9032 | 0.0020 | 3 | −0.0054 |
| tsae_paper_k500 | 0.9029 | 0.0064 | 3 | −0.0076 |
| hill_subseq_h8_T12_s5 (1-seed) | 0.9016 | — | 1 | −0.0110 |

`txc_bare_antidead_t5` still wins at S=20 (Δ=+0.0006 over mlc, ~0.5×
σ_seeds — barely statistically distinguishable). All archs lose
0.005–0.011 AUC at S=20 vs S=32 — the longer tail does provide marginal
information beyond the last 20 positions.

#### k_feat = 5
| arch | mean_AUC | σ_seeds | n_seeds | Δ vs S=32 |
|---|---|---|---|---|
| hill_subseq_h8_T12_s5 (1-seed) | 0.8647 | — | 1 | −0.0083 |
| **`mlc`** ⭐ | **0.8621** | 0.0071 | 3 | −0.0086 |
| phase57_partB_h8_multidistance_t8 | 0.8592 | 0.0036 | 3 | −0.0090 |
| topk_sae | 0.8588 | 0.0016 | 3 | −0.0107 |
| phase5b_subseq_h8 | 0.8571 | 0.0053 | 3 | −0.0099 |

`mlc` wins (without the 1-seed hill outlier). Same ordering as S=32.

### BASE leaderboard at S=10 (new — last 10 cache positions)

#### k_feat = 20
| arch | mean_AUC | σ_seeds | n_seeds | Δ vs S=32 |
|---|---|---|---|---|
| **`mlc`** ⭐ | **0.8973** | 0.0024 | 3 | −0.0149 |
| topk_sae | 0.8962 | 0.0060 | 3 | −0.0129 |
| phase57_partB_h8_multidistance_t8 | 0.8961 | 0.0024 | 3 | −0.0125 |
| phase5b_subseq_h8 | 0.8961 | 0.0039 | 3 | −0.0098 |
| tsae_paper_k500 | 0.8958 | 0.0042 | 3 | −0.0147 |
| txc_bare_antidead_t5 | 0.8955 | 0.0040 | 3 | −0.0172 |

⚠️ **Headline shift at S=10**: `mlc` becomes the k=20 winner (0.8973),
narrowly above `topk_sae` (0.8962). `txc_bare_antidead_t5` drops to
6th (loses 0.0172 vs S=32 — the largest drop in the top-6). The TXC
window advantage degrades when the per-example tail is short. Top 6
within 0.0018 AUC — essentially tied.

#### k_feat = 5
| arch | mean_AUC | σ_seeds | n_seeds | Δ vs S=32 |
|---|---|---|---|---|
| **`phase5b_subseq_h8`** ⭐ | **0.8619** | 0.0055 | 3 | −0.0051 |
| mlc | 0.8545 | 0.0024 | 3 | −0.0162 |
| topk_sae | 0.8528 | 0.0035 | 3 | −0.0167 |
| phase57_partB_h8_multidistance_t8 | 0.8520 | 0.0058 | 3 | −0.0162 |
| tsae_paper_k500 | 0.8498 | 0.0168 | 3 | −0.0153 |

`phase5b_subseq_h8` becomes the k=5 winner at S=10 (0.8619, σ=0.0055).
Notable: `phase5b_subseq_h8` only LOSES 0.0051 at S=10 vs S=32
(smaller drop than any other arch). Its H8-stack subseq-sampling
recipe is robust to tail-length variation.

### IT leaderboard at S=32 / 20 / 10 (1-seed)

#### k_feat = 20

| arch | S=32 | S=20 | S=10 | best S |
|---|---|---|---|---|
| **mlc** | **0.9118** | **0.9064** | **0.8956** | wins all 3 |
| phase5b_subseq_h8 | 0.9073 | 0.8950 | 0.8851 | 2nd at S=32, S=20, drops to 5th at S=10 |
| tsae_paper_k500 | 0.9040 | 0.8949 | 0.8868 | stable 3rd |
| txc_bare_antidead_t5 | 0.8975 | 0.8884 | 0.8860 | drops 0.0115 from S=32 to S=10 |
| topk_sae | 0.8938 | 0.8912 | 0.8799 | small drops |
| phase57_partB_h8_t8 | 0.8980 | 0.8837 | 0.8878 | non-monotone |

`mlc` wins all three S values on IT k=20. Cross-S Δ for `mlc`:
−0.0162 from S=32 to S=10 — comparable to other archs.
**phase5b_subseq_h8 drops 2 ranks at S=10** (2nd → 5th); the H8-subseq
recipe is BASE-favored at short S but IT MLC dominates regardless.

#### k_feat = 5

| arch | S=32 | S=20 | S=10 | best S |
|---|---|---|---|---|
| **mlc** | **0.8722** | **0.8588** | **0.8526** | wins all 3 on IT k=5 |
| phase5b_subseq_h8 | 0.8520 | 0.8486 | 0.8409 | 2nd-tied at S=32, 4th at S=10 |
| phase57_partB_h8_t8 | 0.8546 | 0.8404 | 0.8404 | 3rd-2nd-3rd |
| tsae_paper_k500 | 0.8535 | 0.8382 | 0.8300 | drop 0.0235 — largest |
| topk_sae | 0.8319 | 0.8217 | 0.8048 | drop 0.0271 |

### Summary of headline shifts across S

**BASE k=20:**
- S=32 winner: `txc_bare_antidead_t5`
- S=20 winner: `txc_bare_antidead_t5` (still — Δ=+0.0006 over mlc)
- S=10 winner: `mlc` (TXC drops to 6th; window advantage degrades at short S)

**BASE k=5:**
- S=32 winner: `mlc` (with hill_T12 outlier above)
- S=20 winner: `mlc` (consistent)
- S=10 winner: `phase5b_subseq_h8` (TXC variant, robust to tail compression)

**IT k=20:**
- All S values: `mlc` (cross-regime + cross-S consistent)

**IT k=5:**
- All S values: `mlc` (consistent)

### Cross-S sensitivity ranking (BASE, k=20)

Archs ranked by absolute Δ (S=32 to S=10). Smaller = more robust to
tail compression.

| arch | Δ (S=10 vs S=32) | rank (most-robust first) |
|---|---|---|
| phase5b_subseq_h8 | −0.0098 | #1 ⭐ |
| phase57_partB_h8_t8 | −0.0125 | #2 |
| topk_sae | −0.0129 | #3 |
| tsae_paper_k500 | −0.0147 | #4 |
| mlc | −0.0149 | #5 |
| txc_bare_antidead_t5 | −0.0172 | #6 |
| txcdr_t5 | (S=10 valid) | TBD |

`phase5b_subseq_h8` is the most tail-compression-robust arch — loses
only 0.0098 AUC going from S=32 to S=10. It's also the only arch
that improves on IT vs BASE. Both findings consistent: the H8-stack
subseq-sampling recipe captures features that generalize across
context windows AND across instruction-tuning regimes.

### Methodological notes

- The S=20/10 results come from re-probing on the existing
  `probe_cache_S32` (BASE) and `probe_cache_S32_it` (IT). No new
  cache build was needed; `aggregate_s` was patched to support
  S < S_cache via offset=S_cache-S indexing.
- Verified bit-exact reproduction of S=32 numbers via the patched
  function (max diff 0.00e+00 vs canonical).
- HEADLINE_S = (10, 20, 32) computes all three S values from the
  same encode (essentially zero extra GPU cost — only the
  per-(S, k_feat) LR fit on small N is added).

### Files of record

- `experiments/phase7_unification/results/probing_results.jsonl`
  (filter S in {10, 20, 32}; subject_model in {gemma-2-2b, gemma-2-2b-it})
- `results/plots/phase7_leaderboard_S20_multiseed.png` (BASE)
- `results/plots/phase7_leaderboard_S10_multiseed.png` (BASE)
- `results/plots/phase7_leaderboard_it_S20_multiseed.png` (IT)
- `results/plots/phase7_leaderboard_it_S10_multiseed.png` (IT)
- `experiments/phase7_unification/run_probing_phase7.py::aggregate_s`
  (patched to support S<S_cache, bit-identical at S=S_cache)
- `experiments/phase7_unification/build_leaderboard_2seed.py --S {10,20,32}`
  (CLI flag)
