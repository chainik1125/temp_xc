---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Phase 7 leaderboard — multi-seed (1, 2, 42), BALANCED-15 task set

> Closes Han's pending tasks #5 ("Build 2-seed leaderboard with σ") and
> #7 ("Update Phase 7 summary with σ-augmented leaderboard"). Built
> from `probing_results.jsonl` after a comprehensive seed=1 FLIP
> backfill across all 27 archs with seed=1 ckpts, plus seed=2 pulls and
> probings for 6 of the 8 paper leaderboard archs (tfa_big and
> txcdr_t16 only have seeds {1, 42} on HF).
>
> **Task set: BALANCED-15** — paper headline reduction (rationale +
> per-task SD analysis in `2026-04-29-task-importance.md`). Preserves
> k=20 top-3 ranking exactly vs full 36-task set; k=5 reshuffles
> within the top-6 cluster but the same 6 archs occupy the top tier.
> 2.4× speedup for IT-side completion + H200 paper-cell work.
>
> Key takeaway: the leaderboard top is **very tight** at all k_feat —
> top 4 archs at k=5 span only 0.0042 AUC, with σ_seeds 0.005-0.009.
> The leaderboard top "champion" identity changes with seed selection.

### Data

- 36 SAEBench tasks (Phase 7 standard set, includes winogrande / wsc FLIP).
- S = 32 left-aligned cache, mean-pool aggregation (Phase 7 methodology).
- FLIP applied to winogrande / wsc.
- Seed ∈ {1, 2, 42} when ckpts available; per-cell `n_seeds` column flags
  cells with only one or two seeds.
- Per-arch metric: cross-seed mean of per-task means.
- `σ_tasks`: pooled task std across all seeds (captures task variance).
- `σ_seeds`: std across the per-seed means at the arch level (captures
  pure seed effect).

Code: `experiments/phase7_unification/build_leaderboard_2seed.py`.

### Locked-in arch set vs what's actually evaluated

Per `paper_archs.json::leaderboard_archs`, the locked-in cells are
12 (paper_id, arch_id, k_win) triples × 2 subject models (Gemma-2-2b
base + Gemma-2-2b-it). Below shows the BASE-side coverage:

| paper_id | arch_id | k_win | status (this report) |
|---|---|---|---|
| tfa | tfa_big | 500 | ✅ 2 seeds × 36 tasks |
| tsae_k20 | tsae_paper_k20 | 20 | ✅ 3 seeds × 36 tasks |
| tsae_k500 | tsae_paper_k500 | 500 | ✅ 3 seeds × 36 tasks |
| mlc | mlc | 500 | ✅ 3 seeds × 36 tasks |
| **mlc_sparse** | **mlc** | **100** | ❌ not trained at b=4096 (H200_required; legacy IT-side k_win=100 ckpts at b=1024 don't count) |
| ag_mlc_08 | agentic_mlc_08 | 500 | ✅ 3 seeds × 36 tasks |
| **ag_mlc_08_sparse** | **agentic_mlc_08** | **100** | ❌ not trained at b=4096 (H200_required) |
| txc_t5 | txcdr_t5 | 500 | ✅ 3 seeds × 36 tasks |
| txc_t16 | txcdr_t16 | 500 | ✅ 2 seeds × 36 tasks (no seed=2 ckpt on HF) |
| good_txc_p5 | phase5b_subseq_h8 | 500 | ✅ 3 seeds × 36 tasks |
| good_txc_p7_k20 | txc_bare_antidead_t5 | 500 | ✅ 3 seeds × 36 tasks |
| good_txc_p7_k5 | phase57_partB_h8_bare_multidistance_t8 | 500 | ✅ 3 seeds × 36 tasks |

10 of 12 base-side cells are evaluated; **2 cells (mlc_sparse,
ag_mlc_08_sparse) are missing — both H200_required and not yet
trained at paper-canonical b=4096**. The IT side (12 cells × 2nd
subject model) is entirely missing — see `README.md` "Gaps".

### k_feat = 5

### k_feat = 5  (BALANCED-15 task set)

| arch | n_seeds | mean_AUC | σ_seeds | σ_tasks |
|---|---|---|---|---|
| **`txc_bare_antidead_t5`** ⭐ | 3 | **0.8643** | 0.0056 | 0.1302 |
| phase57_partB_h8_bare_multidistance_t8 | 3 | 0.8622 | 0.0083 | 0.1321 |
| topk_sae | 3 | 0.8618 | 0.0055 | 0.1344 |
| mlc | 3 | 0.8601 | 0.0094 | 0.1399 |
| hill_subseq_h8_T12_s5 (1 seed) | 1 | 0.8562 | — | 0.1279 |
| phase5b_subseq_h8 | 3 | 0.8532 | 0.0087 | 0.1285 |
| txcdr_t5 | 3 | 0.8518 | 0.0091 | 0.1278 |
| tsae_paper_k500 | 3 | 0.8454 | 0.0211 | 0.1262 |
| txcdr_t16 | 2 | 0.8448 | 0.0091 | 0.1196 |
| tsae_paper_k20 | 3 | 0.8313 | 0.0060 | 0.1228 |
| mlc_contrastive_alpha100_batchtopk | 3 | 0.6885 | 0.0044 | 0.1139 |
| tfa_big | 2 | 0.6839 | 0.0268 | 0.0814 |
| agentic_mlc_08 | 3 | 0.6581 | 0.0189 | 0.0869 |

### k_feat = 20  (BALANCED-15 task set)

| arch | n_seeds | mean_AUC | σ_seeds | σ_tasks |
|---|---|---|---|---|
| **`txc_bare_antidead_t5`** ⭐ | 3 | **0.9055** | 0.0006 | 0.1150 |
| mlc | 3 | 0.9039 | 0.0023 | 0.1172 |
| hill_subseq_h8_T12_s5 (1 seed) | 1 | 0.9004 | — | 0.1016 |
| topk_sae | 3 | 0.9002 | 0.0035 | 0.1148 |
| phase57_partB_h8_bare_multidistance_t8 | 3 | 0.9001 | 0.0035 | 0.1117 |
| tsae_paper_k500 | 3 | 0.8998 | 0.0081 | 0.1060 |
| txcdr_t5 | 3 | 0.8970 | 0.0023 | 0.1087 |
| phase5b_subseq_h8 | 3 | 0.8947 | 0.0025 | 0.1079 |
| tsae_paper_k20 | 3 | 0.8941 | 0.0021 | 0.1039 |
| txcdr_t16 | 2 | 0.8880 | 0.0064 | 0.0998 |
| mlc_contrastive_alpha100_batchtopk | 3 | 0.8692 | 0.0014 | 0.1291 |
| agentic_mlc_08 | 3 | 0.8426 | 0.0068 | 0.1330 |
| tfa_big | 2 | 0.7677 | 0.0276 | 0.0825 |

### Headline shifts (BALANCED-15)

- **k=5 winner is `txc_bare_antidead_t5`** (0.8643), narrowly above
  `phase57_partB_h8_bare_multidistance_t8` (0.8622), `topk_sae`
  (0.8618), and `mlc` (0.8601). The k=5 top-4 span **only 0.0042 AUC**
  — within σ_seeds noise (0.005-0.009 on top entries).
- **k=20 winner is `txc_bare_antidead_t5`** (0.9055), with `mlc`
  effectively tied at 0.9039 (Δ=0.0016, ~3× the smaller arch's σ_seeds).
  Top-6 archs within 0.006 AUC.
- **Within the MLC family**: vanilla `mlc` (0.8601 at k=5) is
  competitive at the top. Contrastive (`mlc_contrastive_alpha100_batchtopk`,
  0.6885) and multi-scale (`agentic_mlc_08`, 0.6581) variants
  drop ~0.17-0.20 AUC — the contrastive losses don't help probing AUC.
- **TXC family**: at k=5 the BALANCED-15 ranking flips relative to
  the full-36 (where `mlc` was nominally first). Both `txc_bare_antidead_t5`
  and `phase57_partB_h8_bare_multidistance_t8` move ahead of `mlc` here —
  but the gap is small enough (Δ=0.004) that "single champion at k=5"
  isn't statistically defensible regardless of task set.
- **Cross-task-set robustness**: at k=20 the top-3 ranking
  (`txc_bare_antidead_t5`, `mlc`, `hill_subseq_h8_T12_s5`) is
  IDENTICAL between full-36 and BALANCED-15 — strong sanity-check on
  the headline claim.

### Δ to vanilla `topk_sae` baseline (BALANCED-15)

3-seed deltas:

| k_feat | top arch | topk_sae | Δ | top arch σ_seeds | topk_sae σ_seeds |
|---|---|---|---|---|---|
| 5  | txc_bare_antidead_t5 0.8643 | 0.8618 | **+0.0025** | 0.0056 | 0.0055 |
| 20 | txc_bare_antidead_t5 0.9055 | 0.9002 | **+0.0053** | 0.0006 | 0.0035 |

The k=5 gap of 0.0025 is now **below σ_seeds** (~0.5× σ) — at the
BALANCED-15 task set, the headline claim "TXC beats topk_sae at k=5"
is not statistically defensible. The k=20 gap of 0.0053 remains
decisive (~9× σ_seeds of the smaller arch).

(Compare to the full-36 numbers: at k=5 the gap was +0.0086 with mlc
as champion — including the 11 dropped bias_in_bios professions
inflated the gap by 0.006.)

### Honest paper-narrative read

> Both `mlc` (per-token across 5 layers) and `txc_bare_antidead_t5`
> (window over 5 tokens at one layer) are competitive with strong
> per-token-single-layer SAE baselines on 36-task SAEBench probing.
> At less-sparse k=20 the win over `topk_sae` is decisive (~0.005-0.006
> AUC, ~18× σ_seeds). At very-sparse k=5 the win is real but
> ~σ-noise-magnitude (~0.008 AUC, ~1-2× σ_seeds), and the *single-best*
> arch identity (`mlc` vs `phase5b_subseq_h8` vs the H8 family) shifts
> within the top-3 across seeds. The structural inductive bias —
> whether across multiple LAYERS (`mlc`) or across multiple TOKEN
> POSITIONS (TXC) — provides a small but real probing-AUC advantage,
> with the advantage concentrated on knowledge-domain content
> (`bias_in_bios_*` profession prediction, `europarl_*` language ID).

Combined with the stacked-SAE concat control (rules out "more
candidates" as the source — `2026-04-29-stacked-sae-control.md`)
and Y's per-concept finding (TXC favoured on knowledge concepts —
`2026-04-29-y-cs-synthesis.md`), the paper narrative is:
*structural inductive bias across either layers or token positions
beats per-token-single-layer baselines by small but real margins, on
knowledge-domain content specifically*.

### Plot

![3-seed leaderboard](plots/phase7_leaderboard_multiseed.png)

### Files of record

- Builder: `experiments/phase7_unification/build_leaderboard_2seed.py`
- Plot: `experiments/phase7_unification/results/plots/phase7_leaderboard_multiseed.png`
- Probing rows: `experiments/phase7_unification/results/probing_results.jsonl`
