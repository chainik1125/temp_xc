---
author: Han
date: 2026-04-28
tags:
  - design
  - in-progress
---

## Phase 7 plan — paper-scope pre-registration (revised 2026-04-28)

> Supersedes the original 2026-04-26 plan (which pre-registered a
> 49-arch H200 leaderboard + Agent A/B/C three-way split). The
> original is preserved in git history for context; the
> hypotheses below have been narrowed to what the two paper artefacts
> actually need.

This plan is the *what* and *how* — for the *why* see
[`brief.md`](brief.md). Source-of-truth files are
`canonical_archs.json`, `paper_archs.json`, and
`results/results_manifest.json` (see
[`brief.md` § Source-of-truth files](brief.md#source-of-truth-files)).

---

### Two paper artefacts

#### Artefact (i) — probe-AUC leaderboard

A single bar chart per `(subject_model, k_feat)` showing mean AUC
± σ over 36 SAEBench tasks, with one bar per `paper_id` from
`paper_archs.json:leaderboard_archs` (12 bars). Across both subject
models = 4 bar charts (`base × k_feat=5`, `base × k_feat=20`,
`IT × k_feat=5`, `IT × k_feat=20`).

Inputs: `paper_archs.json:leaderboard_archs` (the 12 cells) +
`results_manifest.json` (current-methodology probing rows for those
cells) + `probing_results.jsonl` (raw AUCs).

#### Artefact (ii) — T-sweep

A single line plot per `(subject_model, k_feat)` showing mean AUC
vs T for **two** families:
- **Barebones**: `txcdr_t<T>` for T ∈ {3..32}.
- **Hill-climbed**: `phase57_partB_h8_bare_multidistance_t<T>` for the same T values.

Across both subject models = 4 plots. The plot makes the
"hill-climbed beats barebones across T" claim concrete.

Inputs: `paper_archs.json:tsweep_barebones` and `tsweep_hillclimbed`
(spec) + `results_manifest.json` + `probing_results.jsonl`.

---

### Pre-registered hypotheses (revised, fewer than the original plan)

- **H1 (cross-model generalisation).** The leaderboard *family
  ordering* on base ≈ on IT after methodology unification. If they
  diverge significantly, that's a finding worth reporting.
- **H2 (k_win matters more than recipe within MLC family).** At
  k_win=500 the MLC family's relative position differs from at
  k_win=100. The dense vs sparse pair within `mlc` and within
  `agentic_mlc_08` is the test.
- **H3 (T-sweep monotonicity).** Barebones TXCDR shows the
  Phase-5-flagged "no monotonic T-scaling" pattern (peak at T≈5,
  decay or plateau beyond). Hill-climbed H8 multidistance is
  hypothesised to *flatten or slightly extend* the curve, not flip
  it monotone-up.
- **H4 (TXC vs SAE at k_win=500).** TXC family wins the k_win=500
  leaderboard mean over the per-token SAE family. If `tsae_k500` (the
  faithful T-SAE port) actually beats the TXC headlines, the paper's
  framing pivots significantly.
- **H5 (T-SAE-at-k=20 vs at-k=500).** `tsae_k20` (paper-faithful k=20)
  vs `tsae_k500` (our k convention) — quantifies the cost of
  k convention mismatch when comparing to the original T-SAE paper.

Hypotheses removed from the original plan: H1's per-task Δ ~0.01
predictions (over-specific given σ ≈ 0.005); the H3 sub-hypotheses
H3a/b/c (the anchor-cell ablations are now optional — can be
revisited post-headline if needed).

---

### Workstream split

Three concurrent agents (X, Y, Z) all on `han-phase7-unification`,
plus X-H200 on the H200 pod when it returns. Directory ownership
and the append-only file race protocol are in `brief.md` § "Branch
hygiene for three concurrent agents".

#### Agent X — probe-AUC + T-sweep (RunPod A40)

Owns:
- All cells in `paper_archs.json:leaderboard_archs`.
- All cells in `paper_archs.json:tsweep_barebones` and `tsweep_hillclimbed`.

Concrete remaining work (from `query_paper_coverage.py`):

| pod | base side | IT side |
|---|---|---|
| **A40** (now) | (a) hill-climbed T-sweep fill: H8 multidist T ∈ {10, 12, 14, 16} × seed 42 = 4 trainings.<br/>(b) seed=1 backfill on hill-climbed where matters. | (a) IT-side L13 activation cache build (one-time, ~1 hr).<br/>(b) IT-side probe cache rebuild with `--include_crosstoken` (one-time, ~30 min).<br/>(c) Train 8 A40_ok leaderboard cells × seeds {42, 1} = 16 trainings.<br/>(d) Probe all 12 IT-side leaderboard cells under current methodology (4 H200-pending cells reuse H200 ckpts when available). |
| **H200** (in ~3 days) | (a) 4 MLC leaderboard cells × seeds {42, 1} = 8 trainings.<br/>(b) hill-climbed T ∈ {18, 20, 24, 28, 32} × seed 42 = 5 trainings.<br/>(c) barebones T ∈ {28, 32} are already done — preserve. | (a) 4 MLC leaderboard cells × seeds {42, 1} = 8 trainings. |

Existing base-side coverage: see
`.venv/bin/python -m experiments.phase7_unification.query_paper_coverage`
for the live status. As of 2026-04-28: 8 A40_ok base-side
leaderboard cells fully probed at sd42; 4 MLC cells pending H200;
barebones T-sweep complete; hill-climbed sweep at 7/16.

Probing protocol: `phase7_S32_first_real_meanpool` with FLIP on
the two cross-token tasks. Probing driver:
`experiments/phase7_unification/run_probing_phase7.py --headline`.

#### Agent Y — case studies (RunPod A40)

Owns:
- HH-RLHF dataset-understanding case study.
- AxBench-style steering case study.

See `agent_y_brief.md` for the detailed scope, including the two
specific pitfalls inherited from Agent C (small steering coefficients
+ T-SAE paper case studies not yet reproduced). The case-studies
infrastructure was migrated from `origin/han-phase7-agent-c` onto
`han-phase7-unification` on 2026-04-28 — see commit `445dd7d`.

Y's deliverable is a single Pareto plot + qualitative-feature panel
that reproduces (or fails to reproduce) T-SAE Table-1 / Figure-5
on Phase 7's 6-arch shortlist. X's leaderboard provides the AUC
x-axis for Y's qualitative-Pareto plot.

#### Agent Z — hill-climbing (local 5090)

Owns:
- `experiments/phase7_unification/hill_climb/`
- `hill_*`-prefixed ckpts on `han1823123123/txcdr-base/`

See `agent_z_brief.md`. Z continues the round1 SubseqH8 T_max sweep
+ round2 long-distance shift exploration that previous Agent A
started before its pod died (only V1 of round1 finished).

Z's deliverable is a synthesis log (winner / no-winner) — Han then
decides whether to add a winning cell to
`paper_archs.json:leaderboard_archs`.

Z runs at the same paper-wide constants (b=4096, etc.). For variants
that don't fit on 5090's 32 GB VRAM at b=4096, Z defers to H200
(does not lower batch).

#### Coordination between agents

- X pushes new canonical ckpts incrementally to
  `han1823123123/txcdr-base` (base) and `han1823123123/txcdr-it`
  (new IT repo). Y polls these for new ckpts. Z pushes
  `hill_*` ckpts to the same `txcdr-base/ckpts/` (different
  prefix → no collision with X's canonical ckpts).
- Y reads `paper_archs.json` to know which 6-arch shortlist
  to focus the case studies on (subset of leaderboard archs).
- Z reads `paper_archs.json` to know the methodology (b=4096,
  current probing methodology, k_win=500) to match — and to know
  the AUC bar to beat (current Phase 7 leaders at k_feat=5/20).
- All agents re-run `build_results_manifest.py` after their work
  completes so the next agent picking up sees fresh coverage.
- Append-only race protocol for shared JSONL files: see
  `brief.md` § "Branch hygiene for three concurrent agents".
- No git force-pushes; PR or fast-forward only.

---

### Pre-registered figures

(All produced from `paper_archs.json + results_manifest.json + raw
probing_results.jsonl`.)

| figure | source | inputs |
|---|---|---|
| `phase7_leaderboard_base_k5.png`  | X | leaderboard, base side, k_feat=5 |
| `phase7_leaderboard_base_k20.png` | X | leaderboard, base side, k_feat=20 |
| `phase7_leaderboard_it_k5.png`    | X | leaderboard, IT side, k_feat=5 |
| `phase7_leaderboard_it_k20.png`   | X | leaderboard, IT side, k_feat=20 |
| `phase7_tsweep_base_k5.png`       | X | both T-sweep families, base, k_feat=5 |
| `phase7_tsweep_base_k20.png`      | X | both T-sweep families, base, k_feat=20 |
| `phase7_tsweep_it_k5.png`         | X | both T-sweep families, IT, k_feat=5 |
| `phase7_tsweep_it_k20.png`        | X | both T-sweep families, IT, k_feat=20 |
| `phase7_hh_rlhf_summary.png`      | Y | feature labels per arch, semantic-feature counts |
| `phase7_steering_pareto.png`      | Y | (suc, coh) per arch on 30-feature steering protocol |
| `phase7_steering_strength_curves.png` | Y | per-arch suc-vs-coefficient curves at the strengths Agent C didn't push |

Output paths: `experiments/phase7_unification/results/plots/` (X) and
`experiments/phase7_unification/case_studies/plots/` (Y).

---

### Success criteria

- (i) Leaderboard: 12 paper-id × 2 subject models × 2 k_feat = 48
  cells populated under current methodology. **Completion target: at
  least seed=42 on every cell, seed=1 on every A40_ok cell.** Bonus
  if the H200 returns in time for seed=2 on the MLC family.
- (ii) T-sweep: barebones is already complete on base; hill-climbed
  is at 7/16. **Completion target: A40-feasible 4 cells (H8 T=10..16)
  added at seed=42, IT-side mirror at seed=42 + 1 for both T-sweeps.**
  H200 fills the remaining 5 H8 cells (T=18..32) and IT-side T=28, 32
  for barebones.

If H200 is delayed beyond 3 days, the paper ships with the H200_required
cells marked as "pending; ckpt available, awaiting probing" in the
final summary table — better to ship a complete-up-to-A40 leaderboard
than no leaderboard.

---

### Hugging Face repo layout (revised)

| repo | purpose | who reads/writes |
|---|---|---|
| `han1823123123/txcdr` | legacy Phase 5/5B IT ckpts (k_win mostly =100) | read-only — historical |
| `han1823123123/txcdr-base` | Phase 7 base-side ckpts | X writes; Y reads (case studies use base) |
| `han1823123123/txcdr-it` (NEW) | Phase 7 IT-side ckpts at k_win=500 | X writes; Y reads if/when extending case studies to IT |

Anti-confusion: hardcoded repo names in upload scripts; subject-model
verification in metadata (each ckpt's training log records
`subject_model`); README on each repo points to the others.

---

### Files that MUST NOT be modified

- `experiments/phase5_downstream_utility/results/` — historical record.
- `experiments/phase5b_t_scaling_explore/results/` — historical record.
- `experiments/phase6_qualitative_latents/results/` — historical record.
- The 36-task probe cache for Gemma-IT at L13 (`data/cached_activations/gemma-2-2b-it/`)
  is read-only — do not regenerate; the cache is what Phase 5
  trained against.
- `canonical_archs.json` — source of truth for the universe; only
  edit when adding a NEW arch (which Phase 7 doesn't plan to).

Everything written by X stays under
`experiments/phase7_unification/results/`; everything written by Y
stays under `experiments/phase7_unification/case_studies/`.

---

### Coordination protocol with the H200 pod (when it arrives)

- The H200 pod will run a **fourth** agent (call it X-H200) on the
  same `han-phase7-unification` branch — same workstream as X but
  tackling the H200_required cells from `paper_archs.json`.
- X-H200 picks up by reading `paper_archs.json` and
  `results_manifest.json`, training the cells whose
  `training_pod` field = `H200_required`, pushing ckpts to HF, and
  re-running `build_results_manifest.py` so the other agents see
  the new coverage.
- No conflict: A40 (X), 5090 (Z), and H200 (X-H200) cells are
  disjoint by `training_pod` plus prefix (X writes canonical archs;
  Z writes `hill_*`). Y operates only on existing ckpts. The
  methodology (b=4096, paper-wide constants) is identical across
  all four agents, so the union is paper-grade apples-to-apples.
