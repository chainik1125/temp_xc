---
author: Han
date: 2026-04-24
tags:
  - reference
  - in-progress
---

## Master experiment index

One-stop pointer to every trained arch, every results file, and every
summary doc across Phases 5 + 6 + 6.1 + 6.2. Updated whenever a new
run lands. Corresponds to the **state of `han-phase6` HEAD `3f9ab2c`** (post Phase 6.2 + post-compact-priorities doc).

### Branch-sharing protocol (Phase 5 agent on `han` + Phase 6 agent on `han-phase6`)

This file is **designed to be cherry-picked between branches** so
both agents can maintain a single source-of-truth experiment log
without stepping on each other.

**To cherry-pick onto `han`** (or any other branch that doesn't have
the index yet) — cherry-pick the 4 doc-only commits in order. All 4
touch ONLY `docs/han/` (verified `git diff-tree`), so they will not
conflict with Phase 5 work:

```bash
git checkout han
git cherry-pick be47c34  # initial EXPERIMENT_INDEX + T-sweep handover
git cherry-pick 3f9ab2c  # POST_COMPACT_PRIORITIES v1
git cherry-pick 36c4af4  # EXPERIMENT_INDEX adds branch-sharing + C2 note
git cherry-pick 1e052e6  # POST_COMPACT_PRIORITIES v2 (concrete plans + C2 recovery)
# If another agent already added rows to the same section, resolve
# trivially (append both) — conflicts should be rare since each
# agent edits disjoint sections (Phase 5 agent → §1 Phase 5 archs,
# §2 Phase 5 matrix rows, §5 Phase 5 summary pointer; Phase 6 agent
# → §1 Phase 6 archs, §2 Phase 6 rows, §4-5 Phase 6 figures).
```

**To merge back** when both branches have independently extended the
index: three-way merge will usually auto-resolve (different rows in
the same table) — manually union if git can't figure it out.

**Editing rules for concurrent use:**

- Keep row ordering within each table stable (alphabetical or phase
  order). Don't re-sort — it makes merges nasty.
- Each agent appends to the BOTTOM of relevant tables. Don't insert
  in the middle.
- Each agent updates the "HEAD" hash in the opening paragraph to
  their own branch's HEAD when they commit. The merge back picks
  whichever is newer.
- The `§8 Open experiments + follow-ups` section is shared; use
  absolute phase IDs (e.g. "Phase 5.8 foo") so ownership is clear.

### 1. Arch directory (name → ckpts + results + summary)

Seed-count column: `42` / `42+1+2` means ckpts available at those
seeds. All ckpts live at
`experiments/phase5_downstream_utility/results/ckpts/{name}__seed{N}.pt`
and mirror on HF `han1823123123/txcdr/ckpts/`.

#### TXC family (window-based encoder, T=5 unless noted)

Columns: `random /32 (var)` = `semantic_count` top-32 by per-token variance;
`random /32 (pdvar)` = `semantic_count_pdvar` top-32 by passage-discriminative
variance — see [Phase 6.3 pdvar results](research_logs/phase6_2_autoresearch/2026-04-24-pdvar-results.md).

| name | seeds | recipe | last_pos AUC | mean_pool AUC | random /32 (var) | random /32 (pdvar) | summary |
|---|---|---|---|---|---|---|---|
| `agentic_txc_02` | 42+1+2 | matryoshka + multi-scale contrastive, TopK | 0.7749 ± 0.004 (3s) | 0.7987 ± 0.002 (3s) | 0.0 ± 0.0 (3s) | 6.0 (1 seed) | [Phase 5 §5.7](research_logs/phase5_downstream_utility/summary.md#5.7) |
| `agentic_txc_02_batchtopk` (Cycle F) | 42+1+2 | `agentic_txc_02` + BatchTopK | 0.7593 ± 0.003 (3s) | 0.7826 ± 0.003 (3s) | 0.0 ± 0.0 (3s) | **12.3 ± 1.33 (3s)** | [Phase 6.1 §9.5](research_logs/phase6_qualitative_latents/summary.md#9.5) |
| `agentic_txc_09_auxk` (Cycle A) | 42 | `agentic_txc_02` + AuxK only | 0.7657 | 0.7973 | 0 | — | [Phase 6.1 §9.5](research_logs/phase6_qualitative_latents/summary.md#9.5) |
| `agentic_txc_10_bare` (Track 2) | 42+1+2 | bare TXC + full anti-dead stack, TopK | 0.7788 ± 0.003 (3s) | 0.8014 ± 0.002 (3s) | 3.3 ± 1.33 (3s) | 7.3 ± 1.86 (3s) | [Phase 6.1 §9.5](research_logs/phase6_qualitative_latents/summary.md#9.5) |
| `agentic_txc_11_stack` (Cycle H) | 42 | Cycle F + AuxK | 0.7620 | 0.7851 | 0 | — | [Phase 6.1 §9.5](research_logs/phase6_qualitative_latents/summary.md#9.5) |
| `agentic_txc_12_bare_batchtopk` (2×2 cell) | 42+1+2 | Track 2 + BatchTopK | 0.7771 ± 0.005 (3s) | 0.7956 ± 0.005 (3s) | 1.7 ± 0.33 (3s) | 5.7 ± 1.20 (3s) | [Phase 6.1 §9.5](research_logs/phase6_qualitative_latents/summary.md#9.5) |
| `phase62_c1_track2_matryoshka` | 42 | Track 2 + matryoshka H/L | 0.7841 | 0.8042 | 3 | — | [Phase 6.2 summary](research_logs/phase6_2_autoresearch/summary.md) |
| `phase62_c2_track2_contrastive` | 42+1+2 | Track 2 + single-scale InfoNCE (α=1) | 0.7825 (42); seed 2 probing pending | 0.8010 (42); seed 2 probing pending | 3.3 ± 0.33 (3s: 4/3/3) | — | [Phase 6.2 summary](research_logs/phase6_2_autoresearch/summary.md) |
| `phase62_c3_track2_matryoshka_contrastive` | 42 | Track 2 + matryoshka + contrastive | 0.7834 | 0.7972 | 2 | — | [Phase 6.2 summary](research_logs/phase6_2_autoresearch/summary.md) |
| `phase62_c5_track2_longer` | 42 | Track 2 with min_steps=10000 | 0.7758 | 0.7967 | 4 | — | [Phase 6.2 summary](research_logs/phase6_2_autoresearch/summary.md) |
| `phase62_c6_bare_batchtopk_longer` | 42 | 2×2 cell with min_steps=10000 | 0.7709 | 0.7888 | 0 | — | [Phase 6.2 summary](research_logs/phase6_2_autoresearch/summary.md) |
| `phase63_track2_t3` | 42+1+2 | Track 2 recipe at T=3 | 0.7687 ± 0.003 (3s) | 0.7959 ± 0.005 (3s) | 2.7 ± 1.45 (3s) | 16.7 ± 1.45 (3s) | [Phase 6.3 T-sweep](research_logs/phase6_2_autoresearch/2026-04-24-t-sweep-results.md) |
| `phase63_track2_t10` | 42+1+2 | Track 2 recipe at T=10 | 0.7906 ± 0.002 (3s) | 0.8016 ± 0.005 (3s) | 7.7 ± 1.76 (3s) | 9.0 ± 1.15 (3s) | [Phase 6.3 T-sweep](research_logs/phase6_2_autoresearch/2026-04-24-t-sweep-results.md) |
| `phase63_track2_t20` | 42+1+2 | Track 2 recipe at T=20 | 0.7731 ± 0.006 (3s) | 0.7768 ± 0.006 (3s) | **19.0 ± 3.00 (3s)** | 13.7 ± 0.33 (3s) | [Phase 6.3 T-sweep](research_logs/phase6_2_autoresearch/2026-04-24-t-sweep-results.md) |

#### MLC family

| name | seeds | recipe | last_pos AUC | mean_pool AUC | random /32 |
|---|---|---|---|---|---|
| `agentic_mlc_08` | 42+1+2 | 5-layer crosscoder + multi-scale contrastive | 0.8046 ± 0.001 (3s) | 0.7851 ± 0.007 (3s) | 2 |

#### T-SAE baselines

| name | seeds | recipe | last_pos AUC | mean_pool AUC | random /32 |
|---|---|---|---|---|---|
| `tsae_paper` | 42+1+2 | literal port of Ye et al. 2025 | 0.6848 ± 0.004 (3s) | 0.7246 ± 0.007 (3s) | 13.7 ± 1.33 (3s) (pdvar: 23.7 ± 0.67) |
| `tsae_ours` | 42 | naive pre-port sketch | 0.7253 | 0.7488 | 3 |
| `tfa_big` | 42 | Temporal Feature Analysis (Rajamanoharan et al.) | — | — | 0 |

### 2. Metrics — seeds × concats coverage matrix

| arch | A s42 | A s1 | A s2 | B s42 | B s1 | B s2 | rand s42 | rand s1 | rand s2 | probe s42 lp | probe s42 mp | probe s1 lp | probe s1 mp | probe s2 lp | probe s2 mp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `agentic_txc_02` | ✓ | — | — | ✓ | — | — | ✓ | — | — | ✓* | ✓* | ✓* | ✓* | ✓* | ✓* |
| `agentic_txc_02_batchtopk` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `agentic_txc_09_auxk` | ✓ | — | — | ✓ | — | — | ✓ | — | — | ✓ | ✓ | — | — | — | — |
| `agentic_txc_10_bare` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `agentic_txc_11_stack` | ✓ | — | — | ✓ | — | — | ✓ | — | — | ✓ | ✓ | — | — | — | — |
| `agentic_txc_12_bare_batchtopk` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `phase62_c1` | ✓ | — | — | ✓ | — | — | ✓ | — | — | ✓ | ✓ | — | — | — | — |
| `phase62_c2` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | probe pending | probe pending |
| `phase62_c3` | ✓ | — | — | ✓ | — | — | ✓ | — | — | ✓ | ✓ | — | — | — | — |
| `phase62_c5` | ✓ | — | — | ✓ | — | — | ✓ | — | — | ✓ | ✓ | — | — | — | — |
| `phase62_c6` | ✓ | — | — | ✓ | — | — | ✓ | — | — | ✓ | ✓ | — | — | — | — |
| `phase63_track2_t3` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `phase63_track2_t10` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `phase63_track2_t20` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `agentic_mlc_08` | ✓ | — | — | ✓ | — | — | ✓ | — | — | ✓* | ✓* | ✓* | ✓* | ✓* | ✓* |
| `tsae_paper` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `tsae_ours` | ✓ | — | — | ✓ | — | — | ✓ | — | — | ✓ | ✓ | — | — | — | — |
| `tfa_big` | ✓ | — | — | ✓ | — | — | ✓ | — | — | — | — | — | — | — | — |

`✓` = run present; `✓*` = from Phase 5.7 run (reference 3-seed probing);
`—` = not run yet; `(in flight)` = training at time of writing.

### 3. Canonical result files

- **Training logs** (per run): `experiments/phase5_downstream_utility/results/training_logs/{run_id}.json`
- **Training index**: `experiments/phase5_downstream_utility/results/training_index.jsonl`
- **Probing results** (all archs, aggregations, tasks): `experiments/phase5_downstream_utility/results/probing_results.jsonl`
- **Probe cache**: `experiments/phase5_downstream_utility/results/probe_cache/<task>/acts_{anchor,mlc,mlc_tail}.npz` (66 GB; on HF)
- **z_cache**: `experiments/phase6_qualitative_latents/z_cache/{concat}/{arch}__seed{N}__z.npy`
- **Concat corpora**: `experiments/phase6_qualitative_latents/concat_corpora/{concat}.json`
- **Autointerp labels** (per cell): `experiments/phase6_qualitative_latents/results/autointerp/{arch}__seed{N}__concat{A|B|random}__labels.json`
- **Phase 6.2 per-candidate aggregate**: `experiments/phase6_2_autoresearch/results/phase62_results.jsonl`

### 4. Paper-ready figures

| figure | file | what it shows |
|---|---|---|
| Pareto trade-off (main, var) | `experiments/phase6_qualitative_latents/results/phase61_pareto_robust.png` | 2-panel: full probing-vs-qualitative plane (5 primary archs, 3-seed error bars) + TXC-cluster zoom (Phase 6.2 C1-C6); y-axis = top-32 by per-token variance (x/32 var) |
| Pareto trade-off (main, pdvar) | `experiments/phase6_qualitative_latents/results/phase63_pareto_pdvar.png` | Same layout but y-axis = top-32 by passage-discriminative variance (x/32 pdvar). Cycle F re-ranks to best TXC (12.3 vs Track 2 7.3). See [Phase 6.3 pdvar results](research_logs/phase6_2_autoresearch/2026-04-24-pdvar-results.md). |
| T-sweep line plot | `experiments/phase6_qualitative_latents/results/phase63_t_sweep.png` | Side-by-side AUC vs T + qualitative (var + pdvar) vs T for Track 2 recipe at T ∈ {3, 5, 10, 20}, with T-SAE reference lines. Probing peaks at T=10 with anti-dead stack. |
| Top-N sweep | `experiments/phase6_qualitative_latents/results/phase63_topN_sweep.png` | Cumulative SEMANTIC count vs top-N cutoff (32/64/128/256) for 3 archs on concat_random seed 42. T-SAE 95 vs TXC ~20 at N=256 — gap is structural. |
| Pareto trade-off (paper-style probe) | `experiments/phase6_qualitative_latents/results/phase63_pareto_paper_probe.png` | Pareto with y-axis = k=5 multinomial probe accuracy on passage ID across A/B/random (T-SAE §4.2 style). TXC family DOMINATES this metric — the probing-style qualitative axis is not where T-SAE wins. |
| Rigorous metric headline | `experiments/phase6_qualitative_latents/results/phase61_rigorous_headline.png` | Horizontal bar chart: per-arch SEMANTIC count on concat A/B/random |
| Pareto (last_pos) | `experiments/phase6_qualitative_latents/results/phase61_pareto_tradeoff.png` | Simpler 1-panel scatter, last_pos AUC |
| Pareto (mean_pool) | `experiments/phase6_qualitative_latents/results/phase61_pareto_tradeoff_mean_pool.png` | Simpler 1-panel scatter, mean_pool AUC |
| Phase 5 headline bars | `experiments/phase5_downstream_utility/results/plots/headline_bar_k5_{last_position,mean_pool}_auc_full.png` | 25-arch AUC leaderboard |
| Phase 5 T-sweep | `experiments/phase5_downstream_utility/results/plots/txcdr_t_sweep_auc.png` | AUC vs T for vanilla TXCDR |
| Error-overlap heatmap | `experiments/phase5_downstream_utility/results/plots/error_overlap_jaccard_k5_last_position.png` | MLC × TXCDR complementarity |

All figures have `.thumb.png` siblings (≤48 dpi, agent-readable).

### 5. Summary docs

| phase | doc | status |
|---|---|---|
| **★ Phase 6 final summary** | [research_logs/phase6_qualitative_latents/2026-04-25-final-summary.md](research_logs/phase6_qualitative_latents/2026-04-25-final-summary.md) | **canonical** — embeds 3 Pareto plots + qual-metric explanations |
| Phase 5 | [research_logs/phase5_downstream_utility/summary.md](research_logs/phase5_downstream_utility/summary.md) | complete |
| Phase 6 (qualitative baseline) | [research_logs/phase6_qualitative_latents/summary.md](research_logs/phase6_qualitative_latents/summary.md) §1-§9 | complete (long-form history) |
| Phase 6.1 (rigorous metric + agentic cycles) | [research_logs/phase6_qualitative_latents/summary.md](research_logs/phase6_qualitative_latents/summary.md) §9.5 | complete |
| Phase 6.1 handover (pre-Phase-6.2) | [research_logs/phase6_qualitative_latents/2026-04-23-handover-post-compact.md](research_logs/phase6_qualitative_latents/2026-04-23-handover-post-compact.md) | complete |
| Phase 6.2 autoresearch | [research_logs/phase6_2_autoresearch/summary.md](research_logs/phase6_2_autoresearch/summary.md) | complete |
| Phase 6.2 brief | [research_logs/phase6_2_autoresearch/brief.md](research_logs/phase6_2_autoresearch/brief.md) | complete |
| Phase 6.3 T-sweep handover | [research_logs/phase6_2_autoresearch/2026-04-24-handover-t-sweep.md](research_logs/phase6_2_autoresearch/2026-04-24-handover-t-sweep.md) | in-progress (training at T=3 running) |
| Phase 6.3 T-sweep results | [research_logs/phase6_2_autoresearch/2026-04-24-t-sweep-results.md](research_logs/phase6_2_autoresearch/2026-04-24-t-sweep-results.md) | draft (training complete, encode/autointerp/probe running) |
| Phase 6.3 pdvar results (Priority 2a) | [research_logs/phase6_2_autoresearch/2026-04-24-pdvar-results.md](research_logs/phase6_2_autoresearch/2026-04-24-pdvar-results.md) | **complete** — softens structural-gap claim |
| Phase 6.3 top-N sweep (Priority 2b) | [research_logs/phase6_2_autoresearch/2026-04-24-topN-sweep-results.md](research_logs/phase6_2_autoresearch/2026-04-24-topN-sweep-results.md) | **complete** — gap is structural, not top-32 artefact (T-SAE 95 vs TXC ~20 at top-256) |

### 6. Key scripts

- **Training**: `experiments/phase5_downstream_utility/train_primary_archs.py --archs {name} --seeds 42 [--min-steps N]`
- **Training (tsae_paper)**: `experiments/phase6_qualitative_latents/train_tsae_paper.py --seed 42`
- **Encoding**: `experiments/phase6_qualitative_latents/encode_archs.py --archs {name} --sets A B random --seed 42`
- **Autointerp**: `experiments/phase6_qualitative_latents/run_autointerp.py --archs {name} --seeds 42 --concats A B random`
- **Probing**: `experiments/phase5_downstream_utility/probing/run_probing.py --aggregation {last_position|mean_pool} --run-ids {name}__seed42 --skip-baselines`
- **Aggregate table**: `experiments/phase6_qualitative_latents/assemble_phase61_table.py`
- **Figures**: `plot_rigorous_metric_headline.py`, `plot_pareto_tradeoff.py`, `plot_pareto_robust.py`
- **Phase 6.2 loop**: `experiments/phase6_2_autoresearch/run_phase62_loop.sh`
- **Phase 6.2 single cycle**: `experiments/phase6_2_autoresearch/run_phase62_cycle.sh {CID}`
- **HF sync**: `scripts/hf_sync.py --go` (idempotent, manifest-indexed)
- **Env bootstrap**: `source /workspace/temp_xc/.envrc`

### 7. Environment quick-check

```bash
# On any pod restart, verify:
cd /workspace/temp_xc
source .envrc
echo "${HF_HOME:0:20}"    # should be /workspace/hf_cache
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
git status -sb            # should show han-phase6 clean-ish
```

If .venv/bin/python missing: `uv sync`. If uv missing:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

### 8. Open experiments + follow-ups

| ID | what | status | pointer |
|---|---|---|---|
| **Phase 6.3 T-sweep** | Track 2 at T ∈ {3, 10, 20} — tests user's "larger T trades probing for qualitative" hypothesis | **complete (3 seeds)**. T=20 Pareto-dominates T-SAE on (probing AUC, var-ranked qualitative): 0.7768 mp AUC + 19.0/32 var sem vs T-SAE's 0.7246 + 13.7/32. | [handover-t-sweep](research_logs/phase6_2_autoresearch/2026-04-24-handover-t-sweep.md), [results](research_logs/phase6_2_autoresearch/2026-04-24-t-sweep-results.md) |
| **Phase 6.3 Priority 2a (pdvar)** | Replace var-ranking with passage-discriminative variance; tests whether x/32 metric is systematically unfair to TXC | **complete** — softens structural-gap claim (Cycle F re-ranks to best TXC; gap narrows from 10.4 → 16.3 labels on absolute scale) | [pdvar-results](research_logs/phase6_2_autoresearch/2026-04-24-pdvar-results.md) |
| Phase 6.3 Priority 2b (top-N sweep) | Extend top-N beyond 32 to see if TXC catches up at larger N | **complete** — gap is structural (T-SAE 95/256, TXC ~20/256 at seed 42 concat_random) | [topN-sweep results](research_logs/phase6_2_autoresearch/2026-04-24-topN-sweep-results.md) |
| Phase 6.3 Priority 2c (distinct concepts) | Dedup labels via Sonnet clustering ("N distinct concepts"); faithful to user's phrasing | deferred | [POST_COMPACT_PRIORITIES §2c](research_logs/POST_COMPACT_PRIORITIES.md) |
| Gemma base L12 control | probe tsae_paper on base model + L12 to see if IT L13 is the confound | deferred | mentioned in §9.5 |
| MLC-side anti-dead stack | apply Track 2's recipe to the MLC encoder (currently only tested on TXC) | deferred | — |
| C2 seed 2 evaluation | autointerp complete (sem 20/20/3 for A/B/random); probing running in batch | autointerp done, probing in flight | [C2 seed 2 labels](../../experiments/phase6_qualitative_latents/results/autointerp/phase62_c2_track2_contrastive__seed2__concatrandom__labels.json) |

### 9. Protocol for keeping this index updated

- When a new ckpt lands: add a row to §1 (or update seed counts).
- When a new result drops (probing/autointerp): tick the matrix in §2.
- When a new figure is made: add to §4.
- When a new summary doc is written: add to §5.
- Re-run `scripts/hf_sync.py --go` after any of the above.

### 10. HF repos

- **Model ckpts**: `han1823123123/txcdr` — every `__seed{N}.pt` file
  pushed here, mirrors `experiments/phase5_downstream_utility/results/ckpts/`.
- **Data artefacts**: `han1823123123/txcdr-data` — mirrors
  `experiments/phase6_qualitative_latents/{concat_corpora,z_cache,results/autointerp}/`,
  plus `experiments/phase5_downstream_utility/results/{probing_results,training_index}.jsonl`,
  plus paper figures under `results/`.
- Sync script: `scripts/hf_sync.py` (SHA1-indexed, skips unchanged).
