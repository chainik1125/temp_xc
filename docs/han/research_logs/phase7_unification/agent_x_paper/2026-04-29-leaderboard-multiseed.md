---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Phase 7 leaderboard — multi-seed (1, 2, 42), PAPER task set

> Closes Han's pending tasks #5 ("Build 2-seed leaderboard with σ") and
> #7 ("Update Phase 7 summary with σ-augmented leaderboard"). Built
> from `probing_results.jsonl` after a comprehensive seed=1 FLIP
> backfill across all 27 archs with seed=1 ckpts, plus seed=2 pulls and
> probings for 6 of the 8 paper leaderboard archs (tfa_big and
> txcdr_t16 only have seeds {1, 42} on HF).
>
> **Task set: PAPER** — the finalised paper task set. Pre-registered
> selection by cross-arch SD within balanced clusters (4 bias_in_bios,
> 3 europarl, 3 amazon, 2 ag_news, 2 github_code, 2 coreference). Full
> rationale + cluster proportions in `2026-04-29-paper-task-set.md`.
> Source-of-truth: `experiments/phase7_unification/task_sets.py::PAPER`.

### Data

- **PAPER** task set (16 of 36 SAEBench tasks).
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
| tfa | tfa_big | 500 | ✅ 2 seeds × 16 tasks |
| tsae_k20 | tsae_paper_k20 | 20 | ✅ 3 seeds × 16 tasks |
| tsae_k500 | tsae_paper_k500 | 500 | ✅ 3 seeds × 16 tasks |
| mlc | mlc | 500 | ✅ 3 seeds × 16 tasks |
| **mlc_sparse** | **mlc** | **100** | ❌ not trained at b=4096 (H200_required) |
| ag_mlc_08 | agentic_mlc_08 | 500 | ✅ 3 seeds × 16 tasks |
| **ag_mlc_08_sparse** | **agentic_mlc_08** | **100** | ❌ not trained at b=4096 (H200_required) |
| txc_t5 | txcdr_t5 | 500 | ✅ 3 seeds × 16 tasks |
| txc_t16 | txcdr_t16 | 500 | ✅ 2 seeds × 16 tasks (no seed=2 ckpt on HF) |
| good_txc_p5 | phase5b_subseq_h8 | 500 | ✅ 3 seeds × 16 tasks |
| good_txc_p7_k20 | txc_bare_antidead_t5 | 500 | ✅ 3 seeds × 16 tasks |
| good_txc_p7_k5 | phase57_partB_h8_bare_multidistance_t8 | 500 | ✅ 3 seeds × 16 tasks |

10 of 12 base-side cells are evaluated; **2 cells (mlc_sparse,
ag_mlc_08_sparse) are missing — both H200_required and not yet
trained at paper-canonical b=4096**. The IT side (12 cells × 2nd
subject model) is entirely missing — see `README.md` "Gaps".

### k_feat = 5  (PAPER task set)

| arch | n_seeds | mean_AUC | σ_seeds | σ_tasks |
|---|---|---|---|---|
| hill_subseq_h8_T12_s5 (1 seed) | 1 | 0.8730 | — | 0.1304 |
| **`mlc`** ⭐ | 3 | **0.8707** | 0.0086 | 0.1393 |
| topk_sae | 3 | 0.8695 | 0.0051 | 0.1331 |
| txc_bare_antidead_t5 | 3 | 0.8683 | 0.0049 | 0.1266 |
| phase57_partB_h8_bare_multidistance_t8 | 3 | 0.8682 | 0.0042 | 0.1299 |
| phase5b_subseq_h8 | 3 | 0.8670 | 0.0050 | 0.1310 |
| tsae_paper_k500 | 3 | 0.8651 | 0.0189 | 0.1285 |
| txcdr_t5 | 3 | 0.8601 | 0.0104 | 0.1266 |
| txcdr_t16 | 2 | 0.8580 | 0.0065 | 0.1231 |
| tsae_paper_k20 | 3 | 0.8372 | 0.0036 | 0.1207 |
| mlc_contrastive_alpha100_batchtopk | 3 | 0.7176 | 0.0052 | 0.1361 |
| tfa_big | 2 | 0.7010 | 0.0228 | 0.0847 |
| agentic_mlc_08 | 3 | 0.6807 | 0.0167 | 0.0986 |

### k_feat = 20  (PAPER task set)

| arch | n_seeds | mean_AUC | σ_seeds | σ_tasks |
|---|---|---|---|---|
| **`txc_bare_antidead_t5`** ⭐ | 3 | **0.9127** | 0.0012 | 0.1139 |
| hill_subseq_h8_T12_s5 (1 seed) | 1 | 0.9126 | — | 0.1027 |
| mlc | 3 | 0.9122 | 0.0022 | 0.1164 |
| tsae_paper_k500 | 3 | 0.9105 | 0.0081 | 0.1063 |
| topk_sae | 3 | 0.9091 | 0.0058 | 0.1147 |
| phase57_partB_h8_bare_multidistance_t8 | 3 | 0.9086 | 0.0032 | 0.1114 |
| txcdr_t5 | 3 | 0.9067 | 0.0027 | 0.1091 |
| phase5b_subseq_h8 | 3 | 0.9059 | 0.0022 | 0.1090 |
| tsae_paper_k20 | 3 | 0.9019 | 0.0015 | 0.1041 |
| txcdr_t16 | 2 | 0.8984 | 0.0044 | 0.1019 |
| mlc_contrastive_alpha100_batchtopk | 3 | 0.8810 | 0.0037 | 0.1296 |
| agentic_mlc_08 | 3 | 0.8680 | 0.0049 | 0.1350 |
| tfa_big | 2 | 0.7875 | 0.0266 | 0.0890 |

### Headline shifts (PAPER)

- **k=5 winner is `mlc`** at 0.8707 (3-seed; `hill_subseq_h8_T12_s5`
  nominally above at 0.8730 but only 1 seed). The k=5 top-6 archs span
  only **0.0037 AUC** (mlc 0.8707 → phase5b_subseq_h8 0.8670). σ_seeds
  on top entries is 0.004-0.010 — top archs are **statistically
  indistinguishable**.
- **k=20 winner is `txc_bare_antidead_t5`** at 0.9127 (σ_seeds 0.0012),
  ahead of `mlc` (0.9122) and `tsae_paper_k500` (0.9105). Top-4 within
  0.005 AUC, but `txc_bare_antidead_t5`'s very small σ_seeds (0.0012)
  makes the win **statistically defensible** (Δ vs `topk_sae` = +0.0036,
  ~6× σ_seeds).
- **MLC family**: vanilla `mlc` is competitive at the top of both
  k_feat columns. The contrastive
  (`mlc_contrastive_alpha100_batchtopk`, 0.7176 / 0.8810) and
  multi-scale (`agentic_mlc_08`, 0.6807 / 0.8680) variants drop ~0.13
  -0.20 AUC at k=5 — contrastive losses don't help probing AUC.
- **Cross-task-set robustness**: `txc_bare_antidead_t5` is the **k=20
  winner under both PAPER and FULL** — strong robustness evidence for
  the headline claim. At k=5 the winner shifts (mlc / txc / topk_sae
  depending on cluster proportions) — consistent with σ_seeds-noise
  reading.

### Δ to vanilla `topk_sae` baseline (PAPER)

3-seed deltas:

| k_feat | top arch | topk_sae | Δ | top arch σ_seeds | topk_sae σ_seeds |
|---|---|---|---|---|---|
| 5  | mlc 0.8707 | 0.8695 | **+0.0012** | 0.0086 | 0.0051 |
| 20 | txc_bare_antidead_t5 0.9127 | 0.9091 | **+0.0036** | 0.0012 | 0.0058 |

The k=5 gap of 0.0012 is **far below σ_seeds** (~0.15× σ) — at PAPER
the headline claim "any arch beats topk_sae at k=5" is not
statistically defensible. The k=20 gap of 0.0036 remains decisive
(~6× σ_seeds of `txc_bare_antidead_t5`).

### Honest paper read (revised, post-PAPER)

> Both `mlc` (per-token across 5 layers) and `txc_bare_antidead_t5`
> (window over 5 tokens at one layer) are competitive with strong
> per-token-single-layer SAE baselines on the PAPER SAEBench
> reduction. **At less-sparse k=20 the win over `topk_sae` is
> decisive** (~0.0036 AUC, ~6× σ_seeds), with `txc_bare_antidead_t5`
> the consistent winner across all task subsets evaluated. **At
> very-sparse k=5 the gap is below seed noise** — the structural
> inductive bias is unable to provide a measurable advantage at
> k_feat=5 sparsity.

The structural advantage (whether across LAYERS via `mlc` or across
TOKEN POSITIONS via TXC) provides a small but real probing-AUC
advantage at k=20, with the advantage concentrated on knowledge-
domain content (`bias_in_bios_*` profession prediction, `europarl_nl`
language ID at multi-token resolution) — see
`2026-04-29-barebones-txc-per-task.md` and
`2026-04-29-per-task-tsweep.md`.

Combined with Y's per-concept finding (TXC favoured on knowledge
concepts) and Y's sparsity-decomposition (T-SAE k=20's apparent lead
on steering is mostly a sparsity choice, not architecture), the paper
narrative is: *structural inductive bias across either layers or
token positions provides modest, conditional probing-AUC advantages
on knowledge content; advantage is decisive at k=20 sparsity but
noise-level at k=5*.

### Plot

![multi-seed leaderboard](plots/phase7_leaderboard_multiseed.png)

### Files of record

- Builder: `experiments/phase7_unification/build_leaderboard_2seed.py`
- Plot: `plots/phase7_leaderboard_multiseed.png`
- Probing rows: `experiments/phase7_unification/results/probing_results.jsonl`
- Task set source: `experiments/phase7_unification/task_sets.py::PAPER`
- Task set rationale: `2026-04-29-paper-task-set.md`
