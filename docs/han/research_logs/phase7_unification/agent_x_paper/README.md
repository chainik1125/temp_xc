---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Agent X — paper deliverables (Phase 7, autonomous shift 2026-04-28/29)

This subdir bundles the X-side paper artefacts. Each finding is in its
own dated md file; this README is a thin index — see the per-file
docs for numbers and discussion.

### Files

| file | one-line summary |
|---|---|
| `2026-04-29-leaderboard-multiseed.md` | Leaderboard at S=32, 36 tasks, mean ± σ_seeds across {1, 2, 42}. |
| `2026-04-29-tsweep.md` | Barebones TXCDR + hill-climbed H8 multidistance T-sweep, 2-seed σ. |
| `2026-04-29-stacked-sae-control.md` | Stacked TopKSAE concat / raw-activation concat probing controls. |
| `2026-04-29-per-task-breakdown.md` | Per-task TXC-win pattern vs SAE baselines. |
| `2026-04-29-window-resampling-history.md` | Audit of `subseq_h8` hill-climb history; A40 attempt at `T_max=20 t_sample=8`. |
| `2026-04-29-task-importance.md` | Per-task discriminative power; recommends a 15-task balanced reduction (2.4× speedup, k=20 top-3 ranking preserved). |
| `plots/` | Paper figures, full-res `.png` (150 dpi) + thumbnails (`*.thumb.png`, ≤288px wide). |

### Headline figures

#### Leaderboard (multi-seed)

![leaderboard](plots/phase7_leaderboard_multiseed.png)

#### T-sweep

![T-sweep](plots/phase7_tsweep_2seed.png)

#### Stacked-vs-raw probing AUC hierarchy

![hierarchy](plots/phase7_stacked_vs_raw_hierarchy.png)

### Canonical source-of-truth files (NOT duplicated here)

Per `agent_x_brief.md`, X owns these at canonical paths. The bundle
references them rather than copying:

- `experiments/phase7_unification/paper_archs.json` — locked arch spec
- `experiments/phase7_unification/results/probing_results.jsonl` — append-only AUCs
- `experiments/phase7_unification/results/training_index.jsonl` — ckpt registry
- `experiments/phase7_unification/results/results_manifest.json` — coverage map
- `experiments/phase7_unification/results/plots/phase7_*.png` — canonical plot output

The plots in this subdir's `plots/` are duplicates of those canonical
outputs, copied here only so the writeups can use short relative paths
(`plots/X.png`). The canonical originals are what the builder scripts
write to.

### Source code (canonical paths)

- `build_leaderboard_2seed.py`, `build_tsweep_2seed.py`
- `run_stacked_probing.py`, `run_raw_concat_probing.py`,
  `plot_stacked_vs_raw_hierarchy.py`
- `analyze_stacked_vs_txc.py`, `analyze_per_task_breakdown.py`
- `build_token_ids.py`, `hill_climb/train_t20_s8.py`

All under `experiments/phase7_unification/`.

### Companion writeups (other agents, not in this subdir)

- Y's steering analysis + 30-concept benchmark + sparsity decomposition:
  `2026-04-29-y-summary.md`, `2026-04-29-y-tx-steering-final.md`,
  `2026-04-29-y-cs-synthesis.md`, `2026-04-29-y-tx-steering-magnitude.md`,
  `2026-04-29-y-paper-draft-steering.md`, `2026-04-29-y-z-handoff.md`.
- Dmitry's steering protocol reproduction: branch `origin/dmitry-rlhf`,
  cross-referenced in `2026-04-28-x-dmitry-steering-defeat-analysis.md`.

### Gaps (still open)

Cells in `paper_archs.json::leaderboard_archs` that are NOT yet
evaluated under Phase 7 current methodology at b=4096:

- **`mlc_sparse` (mlc, k_win=100)** — H200_required, not trained at
  b=4096. Legacy Phase 5 IT-side k_win=100 ckpts on `han1823123123/txcdr`
  exist but were trained at b=1024 — they don't count per
  `paper_archs.json` ("If a cell can't fit on A40 at b=4096, DO NOT
  downsize batch — DEFER to H200").
- **`ag_mlc_08_sparse` (agentic_mlc_08, k_win=100)** — same as above.
- **All IT-side cells** (12 paper_archs × Gemma-2-2b-it L13).
  ~13 hr work (IT activation cache + 8 archs × 2 seeds + probing) —
  didn't fit in the 18-hr autonomous shift after the OOM-restart cost
  the early hours. Punch list in `agent_x_brief.md` "Concrete remaining
  work". Note: `mlc_sparse` and `ag_mlc_08_sparse` IT-side legacy
  ckpts also exist at b=1024 and don't count.
- **3-seed σ for `tfa_big`, `txcdr_t16`, `hill_subseq_h8_T12_s5`** —
  only 1-2 seeds on HF.
- **`hill_subseq_h8_T20_s8`** — Han's specific question; A40 OOM /
  timeout; H200_required (full audit in
  `2026-04-29-window-resampling-history.md`).
- **`tsae_paper_k500` / `tsae_paper_k20` seed=1 stacked-SAE rows** —
  lost in a stash/rebase race; `topk_sae` multi-seed was sufficient to
  resolve the hypothesis.
