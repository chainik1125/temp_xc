# Reproducing the paper figures

This guide gives **two paths** for every paper figure:

1. **Fast** — re-render the plot from the bundled cached artefacts
   (leaderboard rows, run-dirs, precomputed UMAP arrays, etc.). Seconds
   to a few minutes per figure. No GPU required for the renderer
   itself.
2. **From scratch** — re-train / re-evaluate the underlying cells, then
   re-run the renderer. Hours of GPU time per cell. Required only if
   you want to verify the cached numbers, not if you want to regenerate
   the plots.

Both paths use the same canonical artefact layout:

- `purified/results/leaderboard.jsonl` — append-only metrics for every
  evaluated cell, schema-validated by `temp_bench.schemas.LeaderboardRow`.
- `purified/checkpoints/<train_key>/` — trained model weights +
  `config.json` per cell.
- `purified/results/runs/<eval_key>/` — per-cell run-dirs (Wang
  outputs, judge transcripts, phase-1 unsteered traces, etc.).
- `purified/experiments/c4_qualitative/umap_data/` — pre-computed UMAP
  coords / HDBSCAN labels / lexical-cluster summary for the C4 figure.
- `purified/results/case_studies/hh_rlhf/` — per-arch
  `top_features.json` for the RLHF figure.
- `purified/experiments/c1_noisy_filler/denoising_probe_results.json`
  — Setup B numbers consumed by `c2_paper_renderer`.

## Prerequisites

- A Python env with the repo's `pyproject.toml` synced
  (`uv sync` from inside `purified/`).
- `cd purified` before running any renderer or experiment script.
  Defaults resolve from the in-repo layout via
  `Path(__file__).resolve().parent.parent`.
- For "from scratch" paths, set `TEMP_BENCH_HF_ORG` to the
  HuggingFace organisation that hosts the precomputed activation
  caches and trained checkpoints (the public anonymised release will
  point this at an anonymous org); without it, the experiments will
  re-compute everything from raw inputs.
- `TQDM_DISABLE=1` is recommended for any Python invocation to keep
  log output stable.

---

## Synthetic comparison: C1 + C2

### `synthetic_paper_renderer.py` — C1 TopK sweep + C2 gAUC

**To regen this plot from cached data:**

```bash
cd purified
.venv/bin/python -m scripts.synthetic_paper_renderer
# → writes PNG/PDF figures into purified/figs/synthetic/
```

Reads `purified/results/leaderboard.jsonl`, filters component ∈ {c1, c2}.

**To regen the data from scratch (slow):**

```bash
cd purified
TQDM_DISABLE=1 .venv/bin/python -m experiments.c1_synthetic_topk.run --seed 42
TQDM_DISABLE=1 .venv/bin/python -m experiments.c1_synthetic_topk.run --seed 1
TQDM_DISABLE=1 .venv/bin/python -m experiments.c2_synthetic_coupled.run --seed 42
TQDM_DISABLE=1 .venv/bin/python -m experiments.c2_synthetic_coupled.run --seed 1
.venv/bin/python -m scripts.synthetic_paper_renderer
```

**Hardware:** 1× consumer GPU (RTX 5090 or similar). **Runtime:**
~30 min per (arch, seed) cell, ~6 hr total for the locked sweep.
**Data deps:** activation cache auto-built at
`results/act_cache/<key>/` on first run.

### `c2_paper_renderer.py` — C2 noisy-filler (Setup B + Setup D)

**To regen this plot from cached data:**

```bash
cd purified
.venv/bin/python -m scripts.c2_paper_renderer
# → writes c2_setup_b_singlelatent, c2_setup_d_scatter_clean,
#   c2_synth_global_headline (PNG + PDF) into purified/figs/c2/
```

Reads `experiments/c1_noisy_filler/denoising_probe_results.json`
(Setup B) and `experiments/c2_hierarchical/setup_d_leaderboard.jsonl`
(Setup D, np=10 snapshot of the c2 leaderboard slice with
`datasource=toy_coupled_noisy_K10_M20_d256_pB05_np10`). Falls back to
`results/leaderboard.jsonl` if the snapshot is absent.

**To regen the data from scratch (slow):**

```bash
cd purified
# Setup B — denoising probes on the noisy-filler datasource.
TQDM_DISABLE=1 .venv/bin/python -m experiments.c1_noisy_filler.denoising_probes
# Setup D — coupled-features sweep, np10 datasource, all archs × seeds.
TQDM_DISABLE=1 .venv/bin/python -m experiments.c2_synthetic_coupled.run \
    --datasource toy_coupled_noisy_K10_M20_d256_pB05_np10 --seed 42
TQDM_DISABLE=1 .venv/bin/python -m experiments.c2_synthetic_coupled.run \
    --datasource toy_coupled_noisy_K10_M20_d256_pB05_np10 --seed 1
.venv/bin/python -m scripts.c2_paper_renderer
```

**Hardware:** 1× consumer GPU. **Runtime:** ~45 min per (arch, seed)
cell on Setup D; Setup B probes a few seconds per cell after
checkpoints exist.

---

## Sparse probing: C3

### `c3_paper_renderer.py` — SAEBench-36 sparse probing

**To regen this plot from cached data:**

```bash
cd purified
.venv/bin/python -m scripts.c3_paper_renderer
# → writes c3_sparse_probing_curves_gemma_it,
#   c3_sparse_probing_auc_of_auc_gemma_it,
#   c3_per_task_heatmap (PNG + PDF) into purified/figs/c3/
```

Reads `results/leaderboard.jsonl` (component=c3 rows, IT-only
datasource) plus `checkpoints/<train_key>/config.json` to disambiguate
TXC-base T-sweep variants.

**To regen the data from scratch (slow):**

```bash
cd purified
# 38 SAEBench tasks × 8 archs × 3 seeds × 8 k_feats values.
for SEED in 1 2 42; do
  TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing.run --seed $SEED
done
.venv/bin/python -m scripts.c3_paper_renderer
```

**Hardware:** 1× H100 (24 GB peak). **Runtime:** ~3 hr per seed for
the full SAEBench-38 panel (locked-archs sweep). **Data deps:**
`gemma_2_2b_it_l13_fineweb_24k128` activation cache (auto-built or
fetched from `${TEMP_BENCH_HF_ORG}/temp-bench-data` if set).

---

## Qualitative latents: C4

### `umap_txc_paper_renderer.py` — TXC autointerp UMAP

**To regen this plot from cached data:**

```bash
cd purified
.venv/bin/python -m scripts.umap_txc_paper_renderer
# → writes c3_umap_txc (PNG + PDF) into purified/figs/c4/
```

Reads precomputed
`experiments/c4_qualitative/umap_data/{coords,labels}.npy` and
`summary.json` (5,033 features, 15 lexical clusters).

**To regen the data from scratch (very slow, GPU + LLM-judged):**

The full UMAP pipeline requires:

1. Train the C4 dictionary
   (`experiments.c4_qualitative.run --seed 42`).
2. Compute per-feature top-activating sequences across the
   evaluation corpus.
3. Send each feature's top-activating sequences to an external
   LLM-judge (Anthropic Haiku 4.5) for an autointerp lexical label.
4. Embed each feature's autointerp text with
   `sentence-transformers/all-MiniLM-L6-v2`.
5. UMAP-reduce the embedding to 2-D, then HDBSCAN-cluster.

The full pipeline lives in the upstream branch tag
`case-qualitative` (autointerp + UMAP scripts plus LLM-judge keys).
Re-running it costs ~$30 in API calls and ~12 hr of GPU + CPU time;
the precomputed numpy arrays bundled in this repo are the canonical
inputs for the paper figure.

**Hardware:** 1× H100 (training) + CPU (UMAP/HDBSCAN). **Runtime:**
~12 hr end-to-end. **Note:** sentence-transformers, umap-learn, and
hdbscan are NOT in `pyproject.toml`'s default deps — install with
`uv pip install sentence-transformers umap-learn hdbscan` first.

---

## RLHF case study: C5

### `rlhf_paper_renderer.py` — HH-RLHF top-feature decomposition

**To regen this plot from cached data:**

```bash
cd purified
.venv/bin/python -m scripts.rlhf_paper_renderer
# → writes rlhf_summary.png + rlhf_scatter.png into purified/figs/rlhf/
```

Reads per-arch `top_features.json` from
`results/case_studies/hh_rlhf/<arch>/`.

**To regen the data from scratch (slow):**

```bash
cd purified
TQDM_DISABLE=1 .venv/bin/python -m experiments.c5_steering.run --seed 42
.venv/bin/python -m scripts.rlhf_paper_renderer
```

**Hardware:** 1× A40 (RLHF case study) + CPU for top-feature ranking.
**Runtime:** ~4 hr for the canonical sweep (4 archs × HH-RLHF
harmless-base subset). **Data deps:** the C5 dictionary checkpoints
and the first 1,000 (chosen, rejected) pairs of `Anthropic/hh-rlhf`.

---

## Emergent misalignment: C6

### `c6_paper_renderer.py` — alignment-delta + detection PR-AUC

**To regen this plot from cached data:**

```bash
cd purified
.venv/bin/python -m scripts.c6_paper_renderer
# → writes c6_em_alignment_delta_7bmed,
#   c6_em_detection_prauc_7bmed (PNG + PDF) into purified/figs/c6/
```

Reads `results/leaderboard.jsonl` (component=c6, both
`eval_protocol_version=2.0.0` for steering and `=3.0.0` for
detection) plus per-cell
`results/runs/c6_<train_key>/wang_full.json` for the full Wang
stage-4 frontier.

**To regen the data from scratch (slow):**

```bash
cd purified
# Full-Wang steering protocol — 4 stages per cell (rank → screen →
# strength → 27-α frontier), paired seeds {1, 42}.
for SEED in 1 42; do
  TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em.run \
      --datasource qwen_2_5_7b_instruct_medical_l15_resid_post --seed $SEED
done
# Detection PR-AUC at S=16 (sparse-probe ablation).
for SEED in 1 42; do
  TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em_detection.run \
      --datasource qwen_2_5_7b_instruct_medical_l15_resid_post --seed $SEED
done
.venv/bin/python -m scripts.c6_paper_renderer
```

**Hardware:** 1× H100 (steering) and 1× H100 (detection — sparse-probe
LR fits across ~1,700 stage-4 rollouts per cell). **Runtime:** ~3 hr
per seed for steering, ~30 min per seed for detection.
**External dep:** Anthropic Claude Haiku 4.5 API key (judge of
coherence + alignment); per-rollout transcripts persist to
`judge_outputs.jsonl` for post-hoc κ validation.

---

## Backtracking: C7 (Ward Stage B)

### `c7_paper_renderer.py` — Δgc + PR-AUC across architectures

**To regen this plot from cached data:**

```bash
cd purified
.venv/bin/python -m scripts.c7_paper_renderer
# → writes c7_pr_auc_S8_bar, c7_pr_auc_vs_S, c7_roc_auc_S8_bar,
#   c7_roc_auc_vs_S (PNG + PDF) into purified/figs/c7/
```

Reads `results/leaderboard.jsonl` (component=c7),
`checkpoints/<train_key>/config.json`, and
`results/runs/<eval_key>/judge_outputs.jsonl` (used for the
per-question Δgc bootstrap when present).

`--unified` mode also pulls additional rows via `git show
case-backtracking:results/leaderboard.jsonl` if you want
sprint-vs-extended overlay plots.

**To regen the data from scratch (slow):**

```bash
cd purified
# Llama-3.1-8B BASE L10 Ward Stage B: 7-arch sweep × paired seeds.
for SEED in 1 42; do
  TQDM_DISABLE=1 .venv/bin/python -m experiments.c7_backtracking.run \
      --seed $SEED
done
.venv/bin/python -m scripts.c7_paper_renderer
```

**Hardware:** 1× A40 (24 GB peak). **Runtime:** ~6 hr per seed (full
S∈{1,2,4,8,16,32} sweep × 7 archs). **External dep:** the Ward Stage
B reasoning trace dataset (mirrored on the public-anonymous HF org
when `TEMP_BENCH_HF_ORG` is set).

### `c7_tex_snippets.py` — C7 LaTeX table macros

```bash
cd purified
.venv/bin/python -m scripts.c7_tex_snippets
# → writes c7_pr_auc_table.tex, c7_headline_table.tex,
#   c7_results_macros.tex into purified/figs/c7/
```

Reads the same C7 inputs as the renderer above. Used by the LaTeX
paper to embed C7 numbers + tables without hand-typing.

---

## Cross-component summary: rose plot

### `global_rose_renderer.py` — six-axis rose summary

**To regen this plot from cached data:**

```bash
cd purified
.venv/bin/python -m scripts.global_rose_renderer
# → writes global_rose_summary.{png,pdf} into purified/figs/
```

Reads the precomputed `notes/global_rose_summary_data.json` sidecar,
which carries one normalised score per architecture per case study
(Denoising, Coupling, Sparse Probing, Backtracking, RLHF, EM). Each
axis is min-max normalised across the five target architectures.

**To regen the sidecar from scratch:** rerun every component
renderer above (so each `experiments/cN_*/results.json` is current),
then recompute the headline metric per axis. The sidecar schema is
`{case_studies: [{axis, metric, raw}], archs, normalised}` — see the
shipped JSON for the canonical format.

**Hardware:** trivial (~1 s, plotting only).

---

## Outputs at a glance

| Script | Output dir (default) | Headline figure(s) | From-scratch hardware / runtime |
|---|---|---|---|
| `synthetic_paper_renderer` | `figs/synthetic/` | C1 TopK sweep, C2 gAUC | 1× consumer GPU, ~6 hr |
| `c2_paper_renderer` | `figs/c2/` | Setup B singlelatent, Setup D scatter, headline | 1× consumer GPU, ~3 hr |
| `c3_paper_renderer` | `figs/c3/` | SAEBench-36 probing AUC, per-task heatmap | 1× H100, ~9 hr (3 seeds) |
| `umap_txc_paper_renderer` | `figs/c4/` | qualitative-latent UMAP | 1× H100 + CPU, ~12 hr (full pipeline) |
| `rlhf_paper_renderer` | `figs/rlhf/` | HH-RLHF feature decomposition | 1× A40, ~4 hr |
| `c6_paper_renderer` | `figs/c6/` | EM alignment delta, detection PR-AUC | 1× H100 ×2, ~7 hr (2 seeds) |
| `c7_paper_renderer` | `figs/c7/` | Ward Stage B Δgc, PR-AUC | 1× A40, ~12 hr (2 seeds) |
| `c7_tex_snippets` | `figs/c7/` | LaTeX-ready tables | (consumes c7 cached data) |
| `global_rose_renderer` | `figs/` | 6-axis rose summary across all components | trivial (~1 s, consumes per-component results.json) |

## Caveats

- The renderers do not retrain models or re-run case studies — they
  only re-plot from cached metrics + judge outputs. If a cell is
  missing from `leaderboard.jsonl`, run its
  `experiments/cN_*/run.py` first.
- `results/leaderboard.jsonl` and `checkpoints/manifest.jsonl` are
  append-only; the renderers compute "latest per cell" via
  `(component, arch, seed, eval_protocol_version)` deduplication.
- Some C6 `wang_full.json` files and C7 `judge_outputs.jsonl` files
  are large; they are the binding inputs for the case-study figures.
- `figs/` is not gitignored; renderer outputs end up tracked when you
  commit. Add `figs/` to `.gitignore` locally if you want to iterate
  without that.
- Cell-level cache keys are deterministic from
  `(component, arch, seed, datasource, training_cfg, eval_cfg)`; the
  same (re-)run on the same inputs writes to the same train/eval keys
  and is a no-op if already cached. See `temp_bench/config.py` for the
  hashing rules.
