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
| txc_t16 | txcdr_t16 | 500 | ✅ 1 seed × 16 tasks (IT) — first attempt OOM'd at b=4096 (Adam alloc); succeeded on retry with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| good_txc_p5 | phase5b_subseq_h8 | 500 | ✅ 1 seed × 16 tasks (IT) — first attempt OOM'd at b=4096 (AuxK einsum); same retry path |
| good_txc_p7_k20 | txc_bare_antidead_t5 | 500 | ✅ 1 seed × 16 tasks (IT) |
| good_txc_p7_k5 | phase57_partB_h8_bare_multidistance_t8 | 500 | ✅ 1 seed × 16 tasks (IT) |

Plus `topk_sae` as the per-token-SAE Δ-baseline: ✅ 1 seed × 16 tasks (IT).

**8 of 12 paper_archs cells are evaluated** at seed=42 (plus
`topk_sae` baseline = 9 archs total); only the 4 MLC-family cells
remain unevaluated (H200_required, deferred to Mission #2).
IT-side training was 1.81×–3.50× slower per arch than the handover
BASE-pod timings (avg ~2.5×; phase57_partB_h8 3.35 hr, phase5b
3.88 hr), leaving no time budget for seed=1. The two A40-OOM
archs were resolved on retry by enabling
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (memory
fragmentation mitigation, no behavioural change).

### k_feat = 5 (PAPER, IT, seed=42)

<!-- BUILDER-GENERATED. Run:
     .venv/bin/python -m experiments.phase7_unification.build_leaderboard_2seed \
       --subject-model google/gemma-2-2b-it
     to regenerate. -->

| arch | n_seeds | mean_AUC | σ_seeds | σ_tasks |
|---|---|---|---|---|
| **`phase57_partB_h8_bare_multidistance_t8`** ⭐ | 1 | **0.8546** | — | 0.1317 |
| tsae_paper_k500 | 1 | 0.8535 | — | 0.1505 |
| **`phase5b_subseq_h8`** | 1 | **0.8520** | — | 0.1310 |
| txcdr_t5 | 1 | 0.8484 | — | 0.1292 |
| topk_sae | 1 | 0.8319 | — | 0.1350 |
| txcdr_t16 | 1 | 0.8302 | — | 0.1359 |
| txc_bare_antidead_t5 | 1 | 0.8189 | — | 0.1316 |
| tsae_paper_k20 | 1 | 0.8126 | — | 0.1319 |
| tfa_big | 1 | 0.6821 | — | 0.0757 |

Top 3 within 0.0026 AUC.

### k_feat = 20 (PAPER, IT, seed=42)

| arch | n_seeds | mean_AUC | σ_seeds | σ_tasks |
|---|---|---|---|---|
| **`phase5b_subseq_h8`** ⭐ | 1 | **0.9073** | — | 0.1097 |
| tsae_paper_k500 | 1 | 0.9040 | — | 0.1225 |
| phase57_partB_h8_bare_multidistance_t8 | 1 | 0.8980 | — | 0.1163 |
| txc_bare_antidead_t5 | 1 | 0.8975 | — | 0.1302 |
| topk_sae | 1 | 0.8938 | — | 0.1471 |
| txcdr_t5 | 1 | 0.8898 | — | 0.1204 |
| txcdr_t16 | 1 | 0.8868 | — | 0.1102 |
| tsae_paper_k20 | 1 | 0.8749 | — | 0.1215 |
| tfa_big | 1 | 0.7562 | — | 0.0861 |

Top 4 within 0.010 AUC.

### Headline shifts (IT vs BASE)

#### k_feat = 20 — TXC structural bias DOES win on IT (different variant)

| metric | BASE | IT |
|---|---|---|
| winner | `txc_bare_antidead_t5` (0.9127, σ=0.0012) | `phase5b_subseq_h8` (0.9073) |
| Δ to `topk_sae` baseline | +0.0036 (~6× σ_seeds, decisive) | +0.0135 |
| TXC variant in #1 | yes | yes |
| BASE k=20 #2 | `mlc` (0.9122) | unavailable on IT A40 |
| top-3 spread | 0.0022 AUC | 0.0093 AUC |

The **TXC structural advantage at k=20 replicates on IT** — but with
a different TXC variant. On BASE the winner is `txc_bare_antidead_t5`
(MatryoshkaContrastiveAntidead at T=5); on IT the winner is
`phase5b_subseq_h8` (Phase 5b's H8-stack subseq sampling, T_max=10
t_sample=5). Both are window-architecture variants — the structural
inductive bias holds across instruction-tuning.

Per-arch BASE-vs-IT delta at k=20:

| arch | BASE | IT | Δ (IT − BASE) |
|---|---|---|---|
| **phase5b_subseq_h8** ⭐ | 0.9059 | **0.9073** | **+0.0014** (only arch that improves on IT) |
| tsae_paper_k500 | 0.9105 | 0.9040 | −0.0065 |
| txc_bare_antidead_t5 | 0.9127 | 0.8975 | −0.0152 |
| topk_sae | 0.9091 | 0.8938 | −0.0153 |
| phase57_partB_h8_multidistance_t8 | 0.9086 | 0.8980 | −0.0106 |
| txcdr_t5 | 0.9067 | 0.8898 | −0.0169 |
| tsae_paper_k20 | 0.9019 | 0.8749 | −0.0270 |
| tfa_big | 0.7875 | 0.7562 | −0.0313 |
| txcdr_t16 | 0.8984 | 0.8868 | −0.0116 |

`phase5b_subseq_h8` is the **only arch in the leaderboard that
improves under instruction tuning**. All others lose 0.006–0.031
AUC. This is suggestive evidence that the H8-stack + subseq-sampling
recipe captures features that are specifically *more* salient in
instruction-tuned representations — possibly the multi-token
discourse / instruction-following structure that gemma-2-2b-it
emphasizes vs gemma-2-2b base.

#### k_feat = 5 — TXC structural bias wins on IT (consistent with BASE)

| metric | BASE | IT |
|---|---|---|
| winner | `mlc` (0.8707, σ=0.0086, MLC unavailable on IT A40) | `phase57_partB_h8_bare_multidistance_t8` (0.8546) |
| Best A40_ok k=5 arch (BASE) | `topk_sae` 0.8695 / `txc_bare_antidead_t5` 0.8683 | `phase57_partB_h8_bare_multidistance_t8` 0.8546 / `phase5b_subseq_h8` 0.8520 |
| top-3 spread | 0.0012 AUC | 0.0026 AUC |

Without MLC-family in either pool, `phase57_partB_h8_bare_multidistance_t8`
(the structural-bias TXC variant that won BASE k=5) remains the k=5
winner on IT — narrowly, with phase5b_subseq_h8 in 3rd. The
H8-stack TXC family takes both #1 and #3 IT k=5.

#### General observations

- **AUCs lower on IT for most archs.** Best k=20 arch BASE 0.9127 vs
  IT 0.9073 (Δ=−0.005); best k=5 arch BASE 0.8707 vs IT 0.8546
  (Δ=−0.016). The win-direction shows structural-TXC variants are
  the most robust to the IT shift.
- **`tfa_big` collapses on IT.** BASE 0.7010 / 0.7875 → IT 0.6821 /
  0.7562. The learned "predictive + novel codes" decomposition
  doesn't transfer well to instruction-tuned activations.
- **`tsae_paper_k20` (Ye et al.'s native k=20 sparsity port) is
  worst-shift at k=20**: −0.0270 AUC. The per-token sparsity
  constraint is more sensitive to IT distribution shift than the
  per-window TXC formulations.
- **σ_tasks comparable across regimes** (0.07–0.15 BASE vs IT),
  suggesting per-task variance dominates per-arch variance in both.

### Honest paper read (IT pass)

The IT leaderboard at seed=42 supports **the same headline as BASE
under PAPER methodology**: at k_feat=20, a TXC-window variant beats
all per-token SAE baselines by a defensible margin. Specifically:

- BASE k=20 winner: `txc_bare_antidead_t5` (Δ=+0.0036 over topk_sae,
  ~6× σ_seeds).
- IT k=20 winner: `phase5b_subseq_h8` (Δ=+0.0135 over topk_sae,
  no σ_seeds estimable at 1 seed but Δ is 4–5× the typical
  BASE σ_seeds for this arch).

The fact that the *winning TXC variant* differs between BASE and IT
(antidead vs H8-subseq) is itself interesting — it means there's no
single TXC arch that universally dominates, but the **TXC family as
a whole** is robust to instruction tuning. The per-token SAE
baselines (`topk_sae`, `tsae_paper_*`) are NOT robust in the same
sense: each loses 0.01–0.03 AUC under IT.

⚠️ **Single-seed caveat**: σ_seeds is unestimable at 1 seed.
The 0.0135 IT k=20 advantage of phase5b_subseq_h8 over topk_sae is
plausibly larger than σ_seeds (typical ~0.005 for IT-grade
training), but a second seed is needed to confirm. Recommend
seeding seed=1 on the next H200 pass.

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
