---
author: Han
date: 2026-04-28
tags:
  - design
  - in-progress
---

## Phase 7 brief — paper-scope unification

> Supersedes the original 2026-04-25 brief (which was written for a
> 49-arch H200 setup with the Agent A/B/C three-way split). The
> original is preserved in git history. Read this version first;
> consult the original only for historical context.

### TL;DR

- **Two paper artefacts** (down from many):
  (i) probe-AUC leaderboard over **12 architectures**;
  (ii) **T-sweep** with two families (barebones TXCDR + a hill-climbed TXC).
- **Two subject models**: `google/gemma-2-2b` (base, L12) and
  `google/gemma-2-2b-it` (IT, L13). Both at the same `k_win = 500`
  paper-wide convention, plus k_win-100 sparse variants of MLC family.
- **Two A40 agents on Phase 7 right now**: Agent X (this) ships the
  probe-AUC leaderboard + T-sweep; Agent Y picks up the Agent-C case
  studies (HH-RLHF + AxBench-style steering). H200 returns in ~3 days
  for cells that don't fit on A40.
- **Single source of truth across phases**: see [§ Source-of-truth
  files](#source-of-truth-files). `results_manifest.json` records
  what's been trained and probed across phases 5/5B/7; `paper_archs.json`
  records which cells we want for the paper. The leaderboard and
  T-sweep are filtered subsets of the manifest.

### Why this phase exists, restated

Same goals as the original Phase 7 brief:

1. Move off `gemma-2-2b-it` to `gemma-2-2b` (base) for direct
   comparability with T-SAE and TFA; **but also keep IT as a second
   subject model** so the paper has cross-model generalisation
   evidence, not just a single-model story.
2. Match `k_win` across all families (Phase 5's MLC was at k_win=100
   while everything else was at k_win=500 — apples to oranges).
3. Use a single, methodologically clean probing aggregation
   (`mean_pool_S32` with per-example `first_real` masking, FLIP on
   the two cross-token tasks).

Plus three new constraints since the original brief:

4. **A40 + 46 GB RAM cap** (vs the original H200 + 188 GB plan).
   Multi-layer MLC cache (70.8 GB) doesn't fit in either GPU or
   pod RAM — those archs defer to H200. See
   `2026-04-28-a40-feasibility.md`.
5. **No apples-to-oranges on training constants**. `batch_size`,
   `lr`, `max_steps`, etc. are paper-wide (`paper_archs.json:training_constants`).
   If a cell can't fit on A40 at b=4096, do not lower batch_size —
   defer to H200.
6. **Reduce arch sprawl**. Down from 49-arch canonical to **12
   leaderboard archs + ~32 T-sweep cells**.

### What's locked in

#### Subject models + layers (both at k_win=500)

| | layer | label | source |
|---|---|---|---|
| `google/gemma-2-2b`     | L12 (0-indexed) | base | matches T-SAE / TFA convention |
| `google/gemma-2-2b-it`  | L13 (0-indexed) | IT   | Phase 5's existing layer; staying for IT-side reuse |

#### Twelve leaderboard archs

See `paper_archs.json:leaderboard_archs` for the full table with
`paper_id`, `arch_id`, `k_win`, `T`, `src_class`, and per-cell
`training_pod` ∈ {`A40_ok`, `H200_required`}. Summary:

```
tfa, tsae_k20, tsae_k500            (3 single-token / SAE-baseline)
mlc, mlc_sparse,                    (2 MLC family at k_win=500 / 100)
ag_mlc_08, ag_mlc_08_sparse         (2 agentic MLC at k_win=500 / 100)
txc_t5, txc_t16                     (2 barebones TXCDR anchor points)
good_txc_p5                         (Phase 5 best — phase5b_subseq_h8)
good_txc_p7_k20                     (Phase 7 best at k_feat=20 — txc_bare_antidead_t5)
good_txc_p7_k5                      (Phase 7 best at k_feat=5 — h8_bare_multidist_t8)
```

Eight A40-feasible, four H200-required (the four MLC family cells —
multi-layer cache exceeds A40 VRAM and pod RAM).

#### T-sweeps

| | family | T values | training_pod_per_T |
|---|---|---|---|
| barebones | `txcdr_t<T>` | T ∈ {3..32}, 16 cells | A40: 14 cells (T≤24); H200: T ∈ {28, 32} |
| hill-climbed | `phase57_partB_h8_bare_multidistance_t<T>` | T ∈ {3..32}, 16 cells | A40: 11 cells (T≤16); H200: T ∈ {18..32} |

T_max=32 is the cap — Subseq T_max=64 is permanently A40-infeasible
and not in the paper scope.

The barebones sweep is **already complete on base side at sd42 + sd1**
(courtesy of the prior H200 work — preserved). The hill-climbed
sweep has 7/16 cells trained at sd42; 9 more to go.

#### Methodology

- Probing: **`phase7_S32_first_real_meanpool`** with FLIP on
  `winogrande_correct_completion` + `wsc_coreference`. Reported at
  `k_feat ∈ {5, 20}`. Anything probed under the old Phase 5/5B
  methodology (S=20, no first_real, distinct `last_position` /
  `mean_pool` / `full_window` aggregations) is historical reference,
  not paper-grade.
- Training: `batch_size=4096, lr=3e-4, max_steps=25_000,
  preload_seqs=24_000, plateau_threshold=0.02, min_steps=3_000`. See
  `paper_archs.json:training_constants` — these are paper-wide.

#### Seeds

`42 primary, 1 secondary, 2 bonus.` Not chasing 3-seed σ on every cell
the way the original Phase 7 plan did — that was wasteful.

### Two A40 agents

Two simultaneous A40 pods with disjoint workstreams:

| | Agent X (this) | Agent Y |
|---|---|---|
| pod | A40, 46 GB VRAM, 46 GB RAM cap, 900 GB | same A40 spec |
| branch | `han-phase7-unification` | `han-phase7-agent-c` (cherry-picks from X's branch as needed) |
| deliverables | leaderboard artefact (i) + T-sweep artefact (ii) | case-studies — HH-RLHF + AxBench-style steering, redoing Agent C's prior pass |
| canonical brief | this file + `plan.md` + `paper_archs.json` | this file + `agent_y_brief.md` |
| key blocker | none currently — work is pure A40 fill-ins + H200 prep | reproduce T-SAE paper's case-study results before extending |

The H200 returning in ~3 days picks up the cells marked
`H200_required` in `paper_archs.json` (the 4 MLC leaderboard cells +
the H8 T≥18 hill-climbed sweep tail + barebones T=28, T=32).

### Source-of-truth files {#source-of-truth-files}

Three files together describe the entire phase. Anything not derivable
from these is a bug.

| file | role |
|---|---|
| `experiments/phase7_unification/canonical_archs.json` | Universe of all arch designs. Spec for any architecture, not paper-restricted. |
| `experiments/phase7_unification/paper_archs.json` | Paper-relevant subset of the universe. Includes `leaderboard_archs`, `tsweep_barebones`, `tsweep_hillclimbed`, paper-wide `training_constants`, per-cell `training_pod` assignments, and `probing_methodology` spec. |
| `experiments/phase7_unification/results/results_manifest.json` | Coverage map: what's actually been trained (ckpts on HF) and probed (under which methodology) across phases 5, 5B, 7. Cell key = `(subject_model, arch_id, seed, k_win)`. |

Plus one query helper that combines the three:

- `experiments/phase7_unification/query_paper_coverage.py` — derived
  view; prints leaderboard cells and T-sweep cells with mean AUC,
  training_pod tags, and IT methodology status. Re-run any time after
  a probing pass to refresh the view.

Plus the rebuilders that keep them up-to-date:

- `experiments/phase7_unification/build_results_manifest.py` —
  rebuilds `results_manifest.json` from the live data sources
  (training_index.jsonl, probing_results.jsonl, HF ckpt listings).
- `paper_archs.json` is hand-edited (it's the SPEC).
- `canonical_archs.json` is the original Phase 7 universe — only edit
  when adding a new arch.

### Onboarding pointers

A fresh agent picking up Phase 7 cold should, in order:

1. Read this brief.
2. Skim `plan.md` for the pre-registered hypotheses + workstream split.
3. Read your specific agent brief (`agent_y_brief.md` if you're Y).
4. `cat experiments/phase7_unification/paper_archs.json | jq` to see
   the spec.
5. `.venv/bin/python -m experiments.phase7_unification.query_paper_coverage`
   to see what's done and what's missing.
6. Check `2026-04-28-a40-feasibility.md` for hardware constraints.
7. Check `2026-04-27-HANDOVER.md` and the dated `2026-04-2*` logs for
   incident-specific context (S=32 cache rebuild, canonical trim
   emergency, etc.).
8. Begin work on the cells your agent owns.

### What this phase will NOT do

- Train new architectures beyond the 12 leaderboard + ~32 T-sweep cells
  in `paper_archs.json`.
- Modify the probing methodology — `mean_pool_S32 + first_real` is
  pinned. Old-methodology rows in the manifest stay as historical
  reference and do not graduate to the paper without re-probing.
- Change `batch_size` per-cell or per-arch. If A40 can't fit, defer to
  H200.
- Re-run the H200-completed work (existing barebones T-sweep) just to
  match A40 numerics — cross-pod cuDNN drift is a documented caveat,
  not a regression.
- Reproduce Phase 6's full autointerp / Pareto pipeline on the new
  paper archs. Agent Y's case studies cover the qualitative
  evaluation; the original 4-Pareto plan is out of scope.
- Extend Subseq to T_max > 32 (permanently A40-infeasible; original
  H200-only plan).
