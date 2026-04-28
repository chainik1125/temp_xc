---
author: Han
date: 2026-04-28
tags:
  - design
  - in-progress
---

## Agent X brief — Phase 7 leaderboard + T-sweep (RunPod A40)

> Read `brief.md` first. This brief is X-specific.

### TL;DR

- **You ship the two paper artefacts**: (i) probe-AUC leaderboard
  over 12 paper archs across 2 subject models; (ii) T-sweep on
  barebones TXCDR + hill-climbed H8 multidistance.
- **Pod**: A40 (46 GB VRAM, 46 GB pod RAM cap, 900 GB) on RunPod.
  H200 returns in ~3 days for cells marked `H200_required` in
  `paper_archs.json`.
- **Branch**: `han-phase7-unification` (shared with Y and Z; see
  `brief.md` § "Branch hygiene for three concurrent agents" for
  directory ownership).

### Source-of-truth files X owns

| file | role |
|---|---|
| `experiments/phase7_unification/paper_archs.json` | hand-edited spec — `leaderboard_archs`, `tsweep_barebones`, `tsweep_hillclimbed`, paper-wide `training_constants`, per-cell `training_pod` tags. X edits when scope changes. |
| `experiments/phase7_unification/results/probing_results.jsonl` | append-only AUCs (Phase 7 current methodology). X writes when probing completes. |
| `experiments/phase7_unification/results/training_index.jsonl` | append-only ckpt index. X writes after each training. |
| `experiments/phase7_unification/results/training_logs/<run_id>.json` | one file per training. X writes after each training. |
| `experiments/phase7_unification/results/plots/phase7_*.png` | leaderboard + T-sweep figures. X-only. |
| `experiments/phase7_unification/results/results_manifest.json` | rebuilt by `build_results_manifest.py` after any data change. X regenerates after their own changes; Y / Z trigger their own regenerations after theirs. |

### Concrete remaining work (as of 2026-04-28)

Run `query_paper_coverage.py` first for the live state. Snapshot
right now:

#### Base side (Gemma-2-2b L12)

- **Leaderboard A40_ok cells** — 8 archs, all with seed=42 ckpt + probing
  under current methodology. Delta: small backfills at seed=1 where
  the agent-c probing chain skipped winogrande/wsc (the FLIP tasks).
  Action: rebuild probe cache once with `--include_crosstoken`,
  re-probe seed=1 cells on those 2 tasks. ~2 hr.
- **Leaderboard H200_required cells** — 4 MLC family cells (mlc,
  mlc_sparse, ag_mlc_08, ag_mlc_08_sparse). Defer training until H200.
- **Barebones T-sweep** — 16 cells, all done at sd42 + sd1, partial
  sd2. Preserved as-is. No work.
- **Hill-climbed T-sweep** — 7 of 16 cells trained at all 3 seeds.
  A40_ok missing: T ∈ {10, 12, 14, 16} → 4 trainings + probings,
  ~3 hr. H200_required missing: T ∈ {18, 20, 24, 28, 32}.

#### IT side (Gemma-2-2b-it L13)

- **Activation cache** — needs build at L13, MLC layers L11..L15.
  ~1 hr.
- **Probe cache** — needs build with `--include_crosstoken`. ~30 min.
- **Leaderboard A40_ok cells** — 8 archs to train at sd42 + sd1.
  ~10 hr.
- **Leaderboard H200_required cells** — 4 MLC archs at sd42 + sd1
  on H200. Defer.
- **T-sweeps** — TBD whether IT-side T-sweeps are needed for the
  paper. The plan says yes (see `plan.md` figures table). If yes,
  that's another 16 + (≤16) trainings. Most of the 16 barebones
  fit on A40 at b=4096 (T ≤ 24). Hill-climbed has the same A40 cap
  as base side.

#### Order of operations on A40 (suggested)

1. Probe-cache rebuild on base with `--include_crosstoken` →
   re-probe seed=1 cells on the 2 missing FLIP tasks. Closes the
   34-vs-36 task asymmetry from Agent C's prior pass.
2. Hill-climbed T-sweep fills (4 trainings).
3. IT-side activation + probe cache build.
4. IT-side leaderboard A40_ok trainings (8 archs × 2 seeds).
5. IT-side probings.
6. (Optional) IT-side T-sweeps.
7. Wait for H200 → kick off MLC + H200_required cells via X-H200.
8. When H200 returns ckpts, run probing on them on A40 (probing
   doesn't need H200; just training).
9. Generate the 8 leaderboard + T-sweep plots from
   `results_manifest.json + probing_results.jsonl`.

### Methodology pin (do not deviate)

- **Probing**: `phase7_S32_first_real_meanpool` with FLIP on
  `winogrande_correct_completion` + `wsc_coreference`. Reported at
  `k_feat ∈ {5, 20}`.
- **Training**: paper-wide constants from
  `paper_archs.json:training_constants`. b=4096 stays. If a cell
  doesn't fit on A40 at b=4096, defer to H200 — do not lower batch.

### Tools X uses

- `experiments/phase7_unification/run_probing_phase7.py --headline`
  — probing driver.
- `experiments/phase7_unification/train_phase7.py` — training driver
  (CLI: `--arch_id` to target one cell at a time).
- `experiments/phase7_unification/build_act_cache_phase7.py` —
  per-layer activation cache build (run 5× for L10..L14 on base, or
  L11..L15 on IT).
- `experiments/phase7_unification/build_probe_cache_phase7.py
  --include_crosstoken` — probe cache build with FLIP tasks.
- `experiments/phase7_unification/build_results_manifest.py` —
  rebuild manifest after pushing new data.
- `experiments/phase7_unification/query_paper_coverage.py` — see
  the live status of all paper cells.

### What X does NOT do

- Don't touch `case_studies/` (that's Y).
- Don't touch `hill_climb/` (that's Z).
- Don't promote a hill-climb arch to `paper_archs.json` without
  Han's go-ahead — Z proposes via research log; Han approves.
- Don't reprobe Phase 5/5B legacy ckpts under current methodology
  unless they're already at matched k_win=500 AND the cell appears in
  `paper_archs.json`. (`txcdr_t5` IT and `phase5b_subseq_h8` IT both
  qualify; the rest are legacy and stay legacy.)
