---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Agent X — paper deliverables (Phase 7, autonomous shift 2026-04-28/29)

This subdir bundles the X-side paper artefacts: the leaderboard,
T-sweep, control experiments, and supporting analysis. Plots are
adjacent under `plots/`. Per-arch raw rows live at the canonical
location `experiments/phase7_unification/results/probing_results.jsonl`
(referenced rather than copied — too large + the canonical write
target for ongoing work).

### What's in here

| file | role |
|---|---|
| `2026-04-29-leaderboard-multiseed.md` | **Headline 3-seed leaderboard** (supersedes the 2-seed table). Mean ± σ_seeds across seed ∈ {1, 2, 42}. TXC champion vs `topk_sae`: Δ=+0.0076 at k=5 (~σ-noise scale), Δ=+0.0055 at k=20 (~18× σ_seeds, decisive). |
| `2026-04-29-leaderboard-2seed.md` | Earlier 2-seed table — historical record only; superseded. |
| `2026-04-29-stacked-sae-control.md` | **Han's "more candidate features" hypothesis test**. Stacked TopKSAE concat over last K=2/K=5 positions loses to TXC by 0.05-0.09 AUC; raw-activation concat loses by 0.16-0.26 AUC. Hierarchy `raw < SAE-concat < SAE-meanpool < TXC` is monotone-positive. Hypothesis rejected. |
| `2026-04-29-per-task-breakdown.md` | **Per-task TXC-win breakdown**. TXC wins concentrate on `bias_in_bios_*` profession prediction (9-11/15) and `europarl_*` language ID (3-5/5). Directionally aligns with Y's per-concept "TXC wins knowledge content" steering finding. |
| `2026-04-29-window-resampling-history.md` | **Han's "T=10 with 8 resampling at 20" question**. That cell was never trained anywhere. Closest analog `phase5b_subseq_h8_t10_s8_k500` (T_max=10 t_sample=8) lost to t_sample=5 in Phase 5B old-methodology. Tried `T_max=20 t_sample=8` on A40 today — OOM at b=4096, OOM at b=1024 + Adam, ran 1h50m at b=1024 PRELOAD=6000 without convergence. Cell needs H200. |
| `plots/` | All paper figures: 2-seed + 3-seed leaderboards, T-sweep, stacked-vs-raw hierarchy. Each has full-res `.png` (150 dpi) + thumbnail `.thumb.png` (≤288px wide, for agent inspection). |

### Headline figures at a glance

#### 3-seed leaderboard (mean ± σ_seeds)

![3-seed leaderboard](plots/phase7_leaderboard_multiseed.png)

#### T-sweep (barebones TXCDR + hill-climbed H8 multidistance)

![T-sweep](plots/phase7_tsweep_2seed.png)

#### Stacked-vs-raw probing AUC hierarchy (control)

![hierarchy](plots/phase7_stacked_vs_raw_hierarchy.png)

### Honest paper headline

The single-seed Phase 7 numbers overstated TXC family's lead.
Properly seeded (and stress-tested with the stacked-SAE control):

> TXC architectures are **competitive but not dominant** on
> sparse-probing AUC. The lead over a strong vanilla TopKSAE baseline
> is +0.005-0.008 AUC at top of the leaderboard — within ~1-2× σ_seeds
> at k=5 (small) and ~18× σ_seeds at k=20 (decisive). The advantage
> is concentrated on knowledge-domain content (bias_in_bios profession
> prediction, europarl language ID), aligning with Y's per-concept
> structural finding on the steering benchmark.

Combined with Y's sparsity-decomposition result (TXC matches T-SAE k=500
on steering at matched sparsity) and Dmitry's protocol-sensitivity
result (paper-clamp vs AxBench-additive flips the steering ranking),
the combined paper narrative target is **"competitive across protocols"**
rather than "TXC SOTA".

### Source code (kept at canonical paths, not duplicated)

- `experiments/phase7_unification/build_leaderboard_2seed.py` —
  3-seed leaderboard builder (also produces 2-seed; flag controls).
- `experiments/phase7_unification/build_tsweep_2seed.py` — T-sweep plot.
- `experiments/phase7_unification/run_stacked_probing.py` — stacked
  TopKSAE concat probing.
- `experiments/phase7_unification/run_raw_concat_probing.py` —
  raw-activation concat probing baseline.
- `experiments/phase7_unification/plot_stacked_vs_raw_hierarchy.py` —
  hierarchy figure builder.
- `experiments/phase7_unification/analyze_stacked_vs_txc.py` — analysis.
- `experiments/phase7_unification/analyze_per_task_breakdown.py` —
  per-task breakdown.
- `experiments/phase7_unification/build_token_ids.py` — FineWeb tokenisation.
- `experiments/phase7_unification/hill_climb/train_t20_s8.py` —
  T_max=20 t_sample=8 training (A40-specific batch_size + PRELOAD_SEQS).

### Source data (kept at canonical paths, referenced)

- Probing rows: `experiments/phase7_unification/results/probing_results.jsonl`
  (~6500 rows after dedupe, covers seed ∈ {1, 2, 42} for 8 leaderboard
  + T-sweep archs)
- Stacked-SAE rows: `experiments/phase7_unification/results/stacked_probing_results.jsonl`
  (676 rows, topk_sae multi-seed + tsae_paper_k500/k20 single-seed)
- Raw-concat rows: `experiments/phase7_unification/results/raw_concat_probing_results.jsonl`
  (144 rows, K ∈ {2, 5} × k_feat ∈ {5, 20} × 36 tasks)
- Training index: `experiments/phase7_unification/results/training_index.jsonl`
- Spec layer: `experiments/phase7_unification/paper_archs.json`,
  `experiments/phase7_unification/canonical_archs.json`

### Companion writeups (other agents, not in this subdir)

- Y's steering analysis + 30-concept benchmark: `2026-04-29-y-summary.md`,
  `2026-04-29-y-tx-steering-final.md`, `2026-04-29-y-cs-synthesis.md`,
  `2026-04-29-y-tx-steering-magnitude.md`, `2026-04-29-y-paper-draft-steering.md`,
  `2026-04-29-y-z-handoff.md`. Sparsity-decomposition finding +
  per-concept-class structural pattern.
- Dmitry's branch `origin/dmitry-rlhf` for the steering protocol
  reproduction.

### What's NOT included

- IT-side leaderboard (Gemma-2-2b-it L13) — original Phase 7 plan
  but blocked on activation cache build + ~10 hr of trainings
  beyond the 18-hour budget. Brief.md "Concrete remaining work"
  has the punch list.
- MLC family cells — H200_required.
- 3-seed σ for `tfa_big` and `txcdr_t16` — only seeds {1, 42} on HF.
- `hill_subseq_h8_T20_s8` — A40 OOM/timeout; H200_required.
- `tsae_paper_k20` and `tsae_paper_k500` seed=1 stacked-SAE rows
  (lost in a stash/rebase race; topk_sae multi-seed enough to settle
  the hypothesis).
