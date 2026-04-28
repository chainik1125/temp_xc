---
author: Han
date: 2026-04-28
tags:
  - design
  - in-progress
---

## Agent Z brief — Phase 7 hill-climbing on Gemma-2-2b base (local 5090)

> Read `brief.md` first for paper-wide context (subject models, k_win
> convention, methodology, source-of-truth files). This brief is
> Z-specific.

### TL;DR

- **You're a local 5090 (32 GB VRAM, 50 GB system RAM)**, not RunPod.
  Distinct from Agents X and Y who are on A40 RunPods.
- **Your job is hill-climbing**: find a barebones-TXC variant that
  beats the current Phase 7 leaderboard on Gemma-2-2b base at seed=42.
  The previous Agent A started this before its pod died — you pick up
  where they left off.
- Two existing hill-climb scripts on
  `experiments/phase7_unification/hill_climb/`:
  - `round1_subseq_t_sweep.py` — SubseqH8 at T_max ∈ {12, 16, 20}.
    V1 (T_max=12) trained; V2 + V3 missing.
  - `round2_long_shifts.py` — Phase 5's "long-distance shift U-shape"
    transferred to base. Never run.
- **Hard constraint**: same paper-wide training constants as X and Y
  (b=4096, lr=3e-4, max_steps=25_000, plateau=0.02, preload_seqs=24_000).
  No batch_size downsizing. If something doesn't fit on the 5090,
  defer to H200.
- **Lower bar than paper-grade**: hill-climb is exploratory. A winner
  here doesn't auto-promote to `paper_archs.json:leaderboard_archs` —
  that's a manual decision after we see the numbers. Negative results
  are also valuable (and easier to ship: fewer cells to retrain
  elsewhere).

### Inheritance from previous Agent A

The previous Agent A (RunPod H200, since destroyed) ran:

- `hill_subseq_h8_T12_s5__seed42` → trained, ckpt on
  `han1823123123/txcdr-base/ckpts/`. **NOT yet sparse-probed** under
  current methodology — first thing for Z to do.
- The two hill-climb scripts were committed but only V1 of round1 ran.

Per `cf85431` ("Phase 7: hill-climb round1 V1 (T=12) + MLC seed=1/2 —
runpod credit-out snapshot"), this was Agent A's last commit before
the pod died.

### Hardware budget

| | Z (local 5090) |
|---|---|
| GPU | NVIDIA RTX 5090, **32 GB GDDR7** |
| System RAM | **50 GB** |
| Disk | local — no RunPod quota |
| Persistence | survives indefinitely (your laptop / workstation) |

Per-arch fit at b=4096:

| variant | preload + W+Adam + workspace | fits 32 GB? |
|---|---|---|
| Subseq T_max=10 (canonical phase5b_subseq_h8) | 14 + 10 + 3 ≈ 27 GB | ✓ |
| Subseq T_max=12 (V1, already done) | 14 + 12 + 3 ≈ 29 GB | ✓ |
| Subseq T_max=16 (V2) | 14 + 16 + 3 ≈ 33 GB | **✗** |
| Subseq T_max=20 (V3) | 14 + 20 + 3 ≈ 37 GB | **✗** |
| H8 T=5 (round2 L1, L2 base) | 14 + 6 + 3 ≈ 23 GB | ✓ |
| H8 T=8 (round2 L3, L4 base) | 14 + 11 + 3 ≈ 28 GB | ✓ |
| barebones TXCDR T ≤ 10 | ≤ 27 GB | ✓ |

**With cache moved off-GPU (one-line change in `_train_utils.py:preload_single`)**:

- Single-layer cache (14 GB) goes to CPU RAM, leaving the full 32 GB
  VRAM for W+Adam+workspace
- 50 GB system RAM accommodates: 14 (cache) + 7 (Python+torch) +
  ~25 (everything else, slack)
- New ceiling: any single-layer arch with W+Adam ≲ 28 GB → up to
  about T=24 for plain TXC, T≈18 for H8 (matryoshka adds ~10%).

So Z's natural scope at b=4096 + cache-off-GPU:
- All round1 SubseqH8 V1, V2, V3 → cache-off-GPU lets V2 (T=16) fit;
  V3 (T_max=20, 37 GB raw) is borderline → defer to H200.
- All 4 round2 long-shift candidates (T=5 and T=8) → fit easily.
- Any other small/mid-T hill-climb idea Z wants to try.

The cache-off-GPU change is OPTIONAL but recommended; it widens Z's
envelope from "T_max ≤ 12 only" to "T_max ≤ 16, all round2".

### Goal: beat the current Phase 7 leaderboard at seed=42

**Targets to beat (base side, current methodology, seed=42):**

| metric | current Phase 7 #1 | AUC |
|---|---|---|
| k_feat=5  (sparse probe) | `phase57_partB_h8_bare_multidistance_t8` | **0.8989** |
| k_feat=20 (denser probe) | `txc_bare_antidead_t5`                   | **0.9358** |

A hill-climb winner at one or both metrics — and ideally one that
also doesn't regress on the other — would be a meaningful paper
finding. Target the k_feat=5 column first; H8 multidist there has
the tightest cluster (0.0038 spread), so beating 0.8989 by ~0.005+
is signal.

### Concrete workplan

1. **Ground-truth check on V1.** Pull
   `hill_subseq_h8_T12_s5__seed42.pt` from HF, run sparse-probing
   under `phase7_S32_first_real_meanpool` methodology (driver:
   `experiments/phase7_unification/run_probing_phase7.py
   --run_ids hill_subseq_h8_T12_s5__seed42`). The probe-cache
   already exists locally if you cherry-pick from X — see
   coordination section. Report mean AUC at k_feat=5/20. **If V1
   already beats 0.8989 at k_feat=5, that's the lowest-effort
   win.**
2. **Run round1 V2 + V3** (after the optional cache-off-GPU
   refactor — V3 at T_max=20 may need H200 deferral if 5090 is
   too tight). ~25 min/training.
3. **Run round2 long-shift** (4 archs, all small-T). ~2 hr total.
4. **Per-arch probe + report.** Each new ckpt: probe under current
   methodology, append to `probing_results.jsonl`, push ckpt to HF
   (`han1823123123/txcdr-base/ckpts/<run_id>.pt`), run
   `build_results_manifest.py` to refresh manifest, update a dated
   research log under `docs/han/research_logs/phase7_unification/2026-04-2*-z-hill-climb.md`.
5. **Iterate**: if any round1/round2 cell shows promising k_feat=5
   numbers, propose follow-up variants in the same dir. Common
   levers: alpha, gamma, n_scales, t_sample, shift sets.
6. **Submit results.** When a winner emerges (or when all 4+ round2
   cells are explored), write a synthesis log naming the winners
   (or "no winner") and proposing whether to add the winning cell
   to `paper_archs.json:leaderboard_archs`. **Han makes the
   add-to-paper decision; Z just runs the experiments + writes
   them up.**

### Branch + workflow

- **Z works on `han-phase7-unification`** — same branch as X and Y.
  See `brief.md` § "Branch hygiene for three concurrent agents" for
  directory ownership rules.
- **Z's writes are confined to**:
  - `experiments/phase7_unification/hill_climb/` (new variants /
    modified scripts)
  - `experiments/phase7_unification/results/training_logs/<run_id>.json`
    (one new file per hill-climb training)
  - `experiments/phase7_unification/results/training_index.jsonl`
    (append; X also appends — handled with `git pull --rebase` and
    deterministic line append)
  - `experiments/phase7_unification/results/probing_results.jsonl`
    (same protocol as training_index)
  - `docs/han/research_logs/phase7_unification/2026-04-2*-z-*.md`
    (Z-prefixed research logs)
  - HF: `han1823123123/txcdr-base/ckpts/hill_*` (Z's ckpts under
    `hill_` prefix, no collision with X's canonical ckpts)
- **Z does NOT touch**:
  - `paper_archs.json` — propose changes via research log; Han edits.
  - `canonical_archs.json` — universe is locked.
  - `experiments/phase7_unification/case_studies/` — Y's territory.
  - `experiments/phase7_unification/results/case_studies/` — Y's
    territory.
  - `experiments/phase7_unification/results/plots/` — X's territory.

### Coordination with X (probing) and the H200 pod

- **X owns the probe-cache rebuild** (with `--include_crosstoken`
  for FLIP tasks). When X's IT-side probe-cache rebuild is done,
  Z's probings on hill-climb ckpts will benefit too — if Z is
  ready to probe before X's cache rebuild, Z can use the existing
  base-side probe cache from `txcdr-base-data` if it still has the
  crosstoken tasks.
- **Z and X both append to `probing_results.jsonl`**. Use git
  pull --rebase + atomic-line-append discipline. Each row is
  ~1 KB JSONL; conflicts at the file level are rare. If X and Z
  push within seconds, the loser of the race re-pulls and pushes
  again. Same pattern X used with Agent C earlier.
- **H200 pod (in ~3 days)**: if any of Z's hill-climb candidates
  doesn't fit on 5090 even with cache-off-GPU, mark it
  `H200_required` in a Z dated log, and X-H200 (the agent on the
  H200 pod) picks it up. T_max=20 SubseqH8 likely needs this.

### Pitfalls to avoid (lessons from Agents A, B, C)

- **Don't lower batch_size to fit a borderline arch.** That breaks
  apples-to-oranges with X's leaderboard cells. Defer to H200.
- **Don't probe with old methodology.** Always
  `phase7_S32_first_real_meanpool` with FLIP. Probing rows go to
  `experiments/phase7_unification/results/probing_results.jsonl`,
  NOT to a Z-specific file.
- **Don't push ckpts without a `hill_` arch_id prefix.** That keeps
  the leaderboard analyzers' `DROPPED_FROM_HEADLINE` filter logic
  simple and prevents accidental inclusion of hill-climb candidates
  in the canonical leaderboard.
- **Don't run >1 training in the same Python process.** Phase 5B's
  recovery script discovered that per-arch state (Adam, gen-fn
  closures) leaks GPU memory between archs in the same interpreter.
  Spawn a fresh process per training (the existing scripts already
  do this; just don't refactor it away).

### What Z's done log should look like per training

```
docs/han/research_logs/phase7_unification/2026-04-29-z-hill-T16.md

  ---
  author: Han
  date: 2026-04-29
  tags:
    - results
    - in-progress
  ---

  ## hill_subseq_h8_T16_s5 — round1 V2

  - Training: 1× 5090 local, b=4096, max_steps=8000 → final_step=N,
    converged=True/False, plateau=X.
  - Wall clock: M minutes.
  - Probing under phase7_S32_first_real_meanpool:
    - k_feat=5 mean AUC: A.AAAA (vs current #1 0.8989, Δ=+/-X)
    - k_feat=20 mean AUC: B.BBBB (vs current #1 0.9358, Δ=+/-X)
  - Verdict: WIN / NEUTRAL / LOSE.
  - HF: pushed `hill_subseq_h8_T16_s5__seed42.pt` to txcdr-base/ckpts/.
```

Same template for round2 long-shift cells.

### What this brief assumes — flag if false

- 5090 has CUDA + recent enough NVIDIA driver for torch 2.9.1 + cu128.
- Local network bandwidth is fine for HF push (each ckpt ~50–200 MB).
- HF token is set up (same `han1823123123` account as X/Y).
- The probe cache built by X is accessible to Z (either via HF pull
  or shared filesystem if Z clones into a remote-mounted dir).

If 5090 + local-only changes any of these, surface it in
`2026-04-2*-z-blockers.md` first.
