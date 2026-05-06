---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## What ended up happening with window resampling at higher T

> Han's question (2026-04-29): "what ended up happening with the window
> resampling at higher T, did we ever run the run that did well at T=10
> with 8 resampling at 20?"

Short answer: **No, we never ran T_max=20 with t_sample=8** under any
methodology. The closest cell that exists is `phase5b_subseq_h8_t10_s8_k500`
(T_max=10, t_sample=8, k_win=500), trained in Phase 5B on Gemma-2-2b-IT.
Under Phase 5B's old methodology it scored mp=0.8218, *worse* than the
canonical Phase 5B mp champion `phase5b_subseq_h8` (T_max=10, t_sample=5)
at 0.8442.

### What was trained, by phase

**Phase 5B (Gemma-2-2b-IT, k_win=100 mostly, k_win=500 here)** — `subseq_h8`:

| arch_id | T_max | t_sample | k_win | seed=42 mp_AUC (old methodology) |
|---|---|---|---|---|
| phase5b_subseq_h8 (canonical) | 10 | 5  | 500 | 0.8442 ⭐ |
| phase5b_subseq_h8_t10_s3_k500 | 10 | 3  | 500 | (not in summary) |
| phase5b_subseq_h8_t10_s5_k1000 | 10 | 5 | 1000 | (not in summary) |
| **phase5b_subseq_h8_t10_s8_k500** | **10** | **8** | **500** | **0.8218** |
| phase5b_subseq_h8_t10_s10_k500 | 10 | 10 | 500 | (not in summary) |

These ckpts live at `han1823123123/txcdr` (legacy IT-side repo) under
`phase5b_ckpts/`. They were *not* retrained under Phase 7's BASE
methodology — different subject model (Gemma-2-2b vs -it), different
anchor layer (L12 vs L13), different probing protocol (Phase 7
S-parameterized mean-pool vs Phase 5B lp/mp). So they cannot be plugged
directly into the current leaderboard.

**Phase 7 BASE (Gemma-2-2b, k_win=500)** — only one hill-climb cell trained:

| arch_id | T_max | t_sample | k_win | seed=42 phase7 k=5 / k=20 |
|---|---|---|---|---|
| phase5b_subseq_h8 | 10 | 5 | 500 | 0.8931 / 0.9299 |
| **hill_subseq_h8_T12_s5** | **12** | **5** | **500** | **0.8951 / 0.9329** |

The round1 hill-climb plan (`hill_climb/round1_subseq_t_sweep.py`)
included T_max ∈ {12, 16, 20} all at t_sample=5 — but only T_max=12
was actually trained before the H200 pod died. **T_max=16 and T_max=20
at t_sample=5 were never trained.** Round1 also did not include any
t_sample=8 cell.

### Did the existing T_max=12 / t_sample=5 hill-climb beat the leaderboard?

No. Looking at seed=42 phase7 sparse-probing AUC (S=32, FLIP-corrected):

| arch | k_feat=5 | k_feat=20 |
|---|---|---|
| phase57_partB_h8_bare_multidistance_t8 (current k=5 champ) | **0.8989** | 0.9317 |
| txc_bare_antidead_t5 (current k=20 champ) | 0.8814 | **0.9358** |
| tsae_paper_k500 | 0.8906 | 0.9339 |
| txcdr_t5 | 0.8949 | 0.9314 |
| **hill_subseq_h8_T12_s5** | 0.8951 | 0.9329 |

The hill-climb result is competitive but does NOT beat either champion.
Δ at k=5 = -0.0038, Δ at k=20 = -0.0029.

### What was Han's "T=10 with 8 resampling at 20" referring to?

Best-fit interpretation: the unrun cell in the original `subseq_h8`
extension plan written in `2026-04-25-message-to-phase5-agent.md`:

> "(T_max=20, t_sample ∈ {5, 10, 20}, k_win=500) would extend the
> t_sample/T_max sweep"

i.e., a T_max=20 cell, possibly with t_sample=8 (close to the proposed
{5, 10, 20}). This was a future-work bullet, never executed.

It is NOT the same as `phase5b_subseq_h8_t10_s8_k500`, which has
T_max=10. Under Phase 5B's old methodology, that t_sample=8 variant
underperformed the t_sample=5 canonical (0.8218 vs 0.8442). So even
the closest analog we *do* have a ckpt for didn't "do well".

### What's worth running next

If the goal is to confirm whether higher resampling rates help at higher
T_max, the missing cells are:

1. `hill_subseq_h8_T20_s5` — finishes the round1 plan (T_max=20 at t=5).
   Already pencilled in but not executed.
2. `hill_subseq_h8_T16_s5` — same, the middle round1 cell.
3. `hill_subseq_h8_T20_s8` — *new*, addresses Han's question directly
   (higher T_max + higher resampling). **Attempted on A40 2026-04-29;
   see "A40 attempt" below.**
4. `hill_subseq_h8_T20_s10`, `_s20` — to span the sweep.

### A40 attempt at `hill_subseq_h8_T20_s8` (2026-04-29)

Tried the directly-Han-asked-about cell during the autonomous shift
after building the BASE activation cache from scratch (token_ids.npy
+ resid_L12.npy at 14 GB). Two OOM iterations:

1. **b=4096** (paper-wide constant): OOM during backward pass —
   T_max=20 × t_sample=8 backward intermediates exceed A40 44 GB VRAM.
2. **b=1024** (4× smaller): OOM during Adam optimiser step — still
   over budget.
3. **b=1024 + PRELOAD_SEQS=6000** (down from 24000): training
   started, ran for 1h50m at full GPU 100%, never reached the first
   `log_every=200` print and never wrote a ckpt. Killed at the
   18-hour budget deadline to preserve writeup time.

The plateau early-stop never fired, suggesting either (a) loss was
still decreasing significantly at step 200+ but very slowly per step,
or (b) the per-step compute on A40 at this combination is too slow
to even reach step 200 in 90 min. SubseqH8 forward/backward at
T_max=20 t_sample=8 has substantial intermediate-tensor footprint —
the encoder integrates over T_max positions, contrastive shifts
generate K extra forward passes per step, and AuxK adds more.

**Conclusion: the H200 pod is required to actually evaluate this
cell.** When the H200 returns:

- Use paper-wide b=4096 + PRELOAD_SEQS=24000 (matches all other
  leaderboard archs).
- Estimated training time on H200: ~110 min (matched to
  `hill_subseq_h8_T12_s5` which took 6603 s = 110 min there).
- After training, probe on 36 SAEBench tasks (~5 min on A40 or H200).

Files of attempt: `experiments/phase7_unification/hill_climb/train_t20_s8.py`
(adapts batch_size + PRELOAD_SEQS for A40), `logs/train_t20_s8.log`.

Until that run completes, the answer to "did we ever run a higher-T
resampling that did well" remains: **no, only T_max=12 t_sample=5
was tested at higher T, and it lost to the canonical T_max=10
t_sample=5 baseline at both k_feat=5 and k_feat=20.**

### Files of record

- Phase 5B training_index: `experiments/phase5b_t_scaling_explore/results/training_index.jsonl`
- Phase 7 training_index: `experiments/phase7_unification/results/training_index.jsonl`
- Phase 7 hill-climb script: `experiments/phase7_unification/hill_climb/round1_subseq_t_sweep.py`
- Legacy ckpts: `han1823123123/txcdr` HF repo, `phase5b_ckpts/` prefix
- Phase 5 summary citing the t10_s8 number: `docs/han/research_logs/phase5_downstream_utility/summary.md`
- The misreported number disclaimer: `docs/han/research_logs/phase5b_t_scaling_explore/2026-04-25-message-to-phase5-agent.md`
