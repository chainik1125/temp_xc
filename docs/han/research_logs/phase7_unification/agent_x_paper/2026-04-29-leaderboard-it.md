---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Phase 7 leaderboard — Gemma-2-2b-IT, multi-seed (1, 42), PAPER task set

> Closes the IT-side leaderboard gap from
> `2026-04-29-handover-IT-and-mlc-sparse.md` Mission #1. Trained 9
> A40_ok cells × 2 seeds (1, 42) at b=4096 on Gemma-2-2b-IT activations
> with anchor=L13 and MLC layers L11..L15. The 4 MLC-family cells
> (mlc, mlc_sparse, ag_mlc_08, ag_mlc_08_sparse) are H200_required and
> deferred per Mission #2 — same cgroup/VRAM constraint that
> disqualified them on BASE.
>
> **Task set: PAPER** — same finalised 16-task selection used for the
> BASE leaderboard. Source-of-truth:
> `experiments/phase7_unification/task_sets.py::PAPER`. Apples-to-apples
> with the BASE numbers in `2026-04-29-leaderboard-multiseed.md`.
> **Schema patch landed before any IT probing**: per-row
> `subject_model` + `anchor_layer` fields disambiguate IT rows from
> BASE in the shared `probing_results.jsonl`.

### Data

- **PAPER** task set (16 of 36 SAEBench tasks).
- S = 32 left-aligned cache, mean-pool aggregation (Phase 7 methodology).
- FLIP applied to winogrande / wsc.
- Seed ∈ {1, 42}. Per-cell `n_seeds=2` for every entry (no seed=2 budget).
- Per-arch metric: cross-seed mean of per-task means.
- `σ_seeds`: std across the per-seed means at the arch level.
- Subject model: **google/gemma-2-2b-it** (instruction-tuned).
  Anchor L13. MLC layers L11..L15. Activation cache built fresh at
  `data/cached_activations/gemma-2-2b-it/fineweb/`. Probe cache built
  directly at S=32 left-aligned (no S=128 right-padded intermediate)
  at `results/probe_cache_S32_it/`.

Code:
- `experiments/phase7_unification/build_act_cache_phase7_it.py`
- `experiments/phase7_unification/build_probe_cache_phase7_it.py`
- `experiments/phase7_unification/train_phase7_it.py`
- `experiments/phase7_unification/build_leaderboard_2seed.py --subject-model google/gemma-2-2b-it`

### Locked-in arch set vs what's actually evaluated

Per `paper_archs.json::leaderboard_archs`, the locked-in cells are 12
(paper_id, arch_id, k_win) triples × 2 subject models. IT-side coverage
(seed=42 only this autonomous shift; seed=1 deferred — A40 was 2.5×
slower than the handover BASE timings, leaving no budget for a
second seed):

| paper_id | arch_id | k_win | status (this report) |
|---|---|---|---|
| tfa | tfa_big | 500 | ✅ 1 seed × 16 tasks (IT) |
| tsae_k20 | tsae_paper_k20 | 20 | ✅ 1 seed × 16 tasks (IT) |
| tsae_k500 | tsae_paper_k500 | 500 | ✅ 1 seed × 16 tasks (IT) |
| **mlc** | **mlc** | **500** | ❌ H200_required (5-layer cache 71 GB > A40 46 GB) |
| **mlc_sparse** | **mlc** | **100** | ❌ H200_required |
| **ag_mlc_08** | **agentic_mlc_08** | **500** | ❌ H200_required |
| **ag_mlc_08_sparse** | **agentic_mlc_08** | **100** | ❌ H200_required |
| txc_t5 | txcdr_t5 | 500 | ✅ 1 seed × 16 tasks (IT) |
| **txc_t16** | **txcdr_t16** | **500** | ⚠️ A40 OOM at b=4096 (Adam exp_avg_sq_sqrt 2.53 GB alloc fail). Retry with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in flight. |
| **good_txc_p5** | **phase5b_subseq_h8** | **500** | ⚠️ A40 OOM at b=4096 (AuxK einsum). Same retry path. |
| good_txc_p7_k20 | txc_bare_antidead_t5 | 500 | ✅ 1 seed × 16 tasks (IT) |
| good_txc_p7_k5 | phase57_partB_h8_bare_multidistance_t8 | 500 | ✅ 1 seed × 16 tasks (IT) |

Plus `topk_sae` as the per-token-SAE Δ-baseline: ✅ 1 seed × 16 tasks (IT).

**6 of 12 paper_archs cells are evaluated** at seed=42 (plus
`topk_sae` baseline = 7 archs total); 4 H200_required and 2 A40-OOM
under retry. IT-side training was 2.34× to 3.50× slower per arch
than the handover BASE-pod timings (avg 2.56×; phase57_partB_h8
took 3.35 hr vs BASE 86 min), leaving no time budget for seed=1.

### k_feat = 5 (PAPER, IT, seed=42)

<!-- BUILDER-GENERATED. Run:
     .venv/bin/python -m experiments.phase7_unification.build_leaderboard_2seed \
       --subject-model google/gemma-2-2b-it
     to regenerate. -->

| arch | n_seeds | mean_AUC | σ_seeds | σ_tasks |
|---|---|---|---|---|
| **`phase57_partB_h8_bare_multidistance_t8`** ⭐ | 1 | **0.8546** | — | 0.1317 |
| tsae_paper_k500 | 1 | 0.8535 | — | 0.1505 |
| txcdr_t5 | 1 | 0.8484 | — | 0.1292 |
| topk_sae | 1 | 0.8319 | — | 0.1350 |
| txc_bare_antidead_t5 | 1 | 0.8189 | — | 0.1316 |
| tsae_paper_k20 | 1 | 0.8126 | — | 0.1319 |
| tfa_big | 1 | 0.6821 | — | 0.0757 |

`phase5b_subseq_h8` and `txcdr_t16` missing — see "Status" table; OOM-retry in flight.

### k_feat = 20 (PAPER, IT, seed=42)

| arch | n_seeds | mean_AUC | σ_seeds | σ_tasks |
|---|---|---|---|---|
| **`tsae_paper_k500`** ⭐ | 1 | **0.9040** | — | 0.1225 |
| phase57_partB_h8_bare_multidistance_t8 | 1 | 0.8980 | — | 0.1163 |
| txc_bare_antidead_t5 | 1 | 0.8975 | — | 0.1302 |
| topk_sae | 1 | 0.8938 | — | 0.1471 |
| txcdr_t5 | 1 | 0.8898 | — | 0.1204 |
| tsae_paper_k20 | 1 | 0.8749 | — | 0.1215 |
| tfa_big | 1 | 0.7562 | — | 0.0861 |

### Headline shifts (IT vs BASE)

#### k_feat = 20 — TXC NOT the IT winner (subject to OOM-retry caveat)

| metric | BASE | IT |
|---|---|---|
| winner | `txc_bare_antidead_t5` (0.9127, σ=0.0012) | `tsae_paper_k500` (0.9040) |
| Δ to `topk_sae` baseline | +0.0036 (~6× σ_seeds, decisive) | +0.0102 |
| TXC variant rank | #1 (winner) | #3 (`txc_bare_antidead_t5` 0.8975) |
| top-3 spread | 0.0022 AUC | 0.0065 AUC |

Under IT (gemma-2-2b-it L13), the **TXC structural bias does NOT
deliver the k=20 win**. Instead, `tsae_paper_k500` (Ye et al.'s
T-SAE port — per-token, but with temporal InfoNCE auxiliary loss)
takes the top spot. `phase57_partB_h8_bare_multidistance_t8` (BASE's
k=5 winner) is a close second on IT k=20 (0.8980).

⚠️ **Caveat: 2 of 8 leaderboard-relevant archs (phase5b_subseq_h8 and
txcdr_t16) OOM'd on A40 at b=4096 and have not yet been probed.**
Retry with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is in
flight as of this writeup. The BASE-side phase5b_subseq_h8 was a
strong 2nd on PAPER k=20 (0.9059 σ=0.0022); if its IT version
recovers anywhere near that level, the conclusion shifts.

#### k_feat = 5 — TXC structural bias's k=5 win partially holds

| metric | BASE | IT |
|---|---|---|
| winner | `mlc` (0.8707, σ=0.0086, MLC-family unavailable on IT A40) | `phase57_partB_h8_bare_multidistance_t8` (0.8546) |
| Best A40_ok k=5 arch (BASE) | `topk_sae` 0.8695 / `txc_bare_antidead_t5` 0.8683 | `phase57_partB_h8_bare_multidistance_t8` 0.8546 |
| top-3 spread | 0.0012 AUC | 0.0062 AUC |

Without MLC-family in either pool, `phase57_partB_h8_bare_multidistance_t8`
(the structural-bias TXC variant that won BASE k=5) remains the k=5
winner on IT — at a slightly larger margin (0.001 BASE → 0.006 IT).
BASE's other k=5 contenders (`topk_sae` and `txc_bare_antidead_t5`)
drop more on IT than the structural-multi-distance variant. This
suggests structural inductive bias still helps at k=5 sparsity on
IT, even where it can't compete at k=20.

#### General observations

- **AUCs lower on IT.** Best k=20 arch BASE 0.9127 vs IT 0.9040
  (Δ=−0.009); best k=5 arch BASE 0.8707 vs IT 0.8546 (Δ=−0.016).
  Instruction-tuned representations are slightly less linearly
  separable on these probing tasks.
- **`tfa_big` collapses on IT.** BASE 0.7010 / 0.7875 → IT 0.6821 /
  0.7562. The learned "predictive + novel codes" decomposition
  doesn't transfer well to the IT activation distribution.
- **Cross-arch SD is comparable** (σ_tasks 0.07–0.15 on both),
  suggesting per-task variance dominates per-arch variance in both
  regimes.

### Plot

![IT multi-seed leaderboard](plots/phase7_leaderboard_it_multiseed.png)

### Files of record

- Builder: `experiments/phase7_unification/build_leaderboard_2seed.py --subject-model google/gemma-2-2b-it`
- Plot: `plots/phase7_leaderboard_it_multiseed.png`
  (canonical: `experiments/phase7_unification/results/plots/phase7_leaderboard_it_multiseed.png`)
- Probing rows: `experiments/phase7_unification/results/probing_results.jsonl`
  (filter `subject_model == "google/gemma-2-2b-it"`)
- Task set source: `experiments/phase7_unification/task_sets.py::PAPER`
- Task set rationale: `2026-04-29-paper-task-set.md`
- Training driver: `experiments/phase7_unification/train_phase7_it.py`
- IT activation cache: `data/cached_activations/gemma-2-2b-it/fineweb/`
- IT probe cache: `experiments/phase7_unification/results/probe_cache_S32_it/`
- IT HF ckpt repo: `han1823123123/txcdr-it`
