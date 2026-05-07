# Reproducing the paper figures

Each renderer in `purified/scripts/` re-creates a paper figure (or set
of figures) from the canonical artefacts in this repository:

- `purified/results/leaderboard.jsonl` — append-only metrics for every
  evaluated cell.
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
- `cd purified` before running any renderer; defaults resolve from the
  in-repo layout via `Path(__file__).resolve().parent.parent`.
- Each renderer creates its output directory if missing.

If you also want to reproduce the underlying cells (not just re-render
plots from cached metrics), run the relevant
`experiments/cN_*/run.py` first; that populates `leaderboard.jsonl`
and `checkpoints/` deterministically via the cache-key contract.

## Renderers

### `synthetic_paper_renderer.py` — C1 + C2 synthetic (TopK sweep, coupled features)

```bash
cd purified
.venv/bin/python -m scripts.synthetic_paper_renderer
```

- Reads `purified/results/leaderboard.jsonl`, filters component=c1 / c2.
- Writes PNG/PDF figures to `purified/figs/synthetic/`.
- Paper section: synthetic comparison (C1 NMSE/AUC sweep + C2 gAUC).

### `c2_paper_renderer.py` — C2 noisy-filler

```bash
cd purified
.venv/bin/python -m scripts.c2_paper_renderer
```

- Reads `experiments/c1_noisy_filler/denoising_probe_results.json`
  (Setup B) and `results/leaderboard.jsonl` (component=c2 rows).
- Writes figures to `purified/figs/c2/`.
- Paper section: C2 — coupled-features denoising probe.

### `c3_paper_renderer.py` — C3 sparse probing (SAEBench+CT)

```bash
cd purified
.venv/bin/python -m scripts.c3_paper_renderer
```

- Reads `results/leaderboard.jsonl` and `checkpoints/<train_key>/config.json`
  (used to disambiguate TXC-base T-sweep variants).
- Writes figures to `purified/figs/c3/` (per-task probing AUC, T-sweep,
  k-sweep; tables in adjacent `.tex` snippets where applicable).
- Paper section: C3 — sparse probing.

### `umap_txc_paper_renderer.py` — C4 qualitative latents UMAP

```bash
cd purified
.venv/bin/python -m scripts.umap_txc_paper_renderer
```

- Reads `experiments/c4_qualitative/umap_data/{coords,labels}.npy` and
  `summary.json`.
- Writes `c3_umap_txc.png` (paper-styled scatter) into `purified/figs/c4/`.
- Paper section: C4 — qualitative latents.
- Note: heavy deps (sentence-transformers / umap-learn / hdbscan) ran
  upstream; only the precomputed numpy arrays + summary json are
  required at render time.

### `rlhf_paper_renderer.py` — C5 RLHF case study

```bash
cd purified
.venv/bin/python -m scripts.rlhf_paper_renderer
```

- Reads `results/case_studies/hh_rlhf/<arch>/top_features.json`
  (per-arch HH-RLHF feature attribution outputs).
- Writes `rlhf_summary.png` and `rlhf_scatter.png` to `purified/figs/rlhf/`.
- Paper section: HH-RLHF case study.

### `c6_paper_renderer.py` — C6 emergent misalignment

```bash
cd purified
.venv/bin/python -m scripts.c6_paper_renderer
```

- Reads `results/leaderboard.jsonl` (component=c6 rows) and
  `results/runs/c6_<train_key>/wang_full.json` for each cell's full
  Wang procedure output (also accepts the alternate
  `wang_<train_key>.json` flat layout).
- Writes alignment-delta plots to `purified/figs/c6/`.
- Paper section: C6 — emergent misalignment.

### `c7_paper_renderer.py` — C7 backtracking (Ward Stage B)

```bash
cd purified
.venv/bin/python -m scripts.c7_paper_renderer
```

- Reads `results/leaderboard.jsonl` (component=c7 rows),
  `checkpoints/<train_key>/config.json`, and
  `results/runs/<eval_key>/judge_outputs.jsonl` (when present, used
  for the per-question Δgc bootstrap).
- Writes figures + the per-component results markdown into
  `purified/figs/c7/`.
- Paper section: C7 — Ward Stage B backtracking case study.
- `--unified` mode also pulls additional rows from `origin/final` via
  `git show` if you want sprint-vs-extended overlay plots.

### `c7_tex_snippets.py` — C7 LaTeX table macros

```bash
cd purified
.venv/bin/python -m scripts.c7_tex_snippets
```

- Reads the same C7 inputs as the renderer above.
- Writes `c7_pr_auc_table.tex`, `c7_headline_table.tex`, and
  `c7_results_macros.tex` to `purified/figs/c7/`.
- Used by the LaTeX paper to embed C7 numbers + tables without
  hand-typing.

## Outputs at a glance

| Script | Output dir (default) | Headline figure(s) |
|---|---|---|
| `synthetic_paper_renderer` | `purified/figs/synthetic/` | C1 TopK sweep, C2 gAUC |
| `c2_paper_renderer` | `purified/figs/c2/` | Setup A, Setup B (noisy filler) |
| `c3_paper_renderer` | `purified/figs/c3/` | SAEBench+CT probing AUC |
| `umap_txc_paper_renderer` | `purified/figs/c4/` | qualitative-latent UMAP |
| `rlhf_paper_renderer` | `purified/figs/rlhf/` | HH-RLHF feature scatter |
| `c6_paper_renderer` | `purified/figs/c6/` | EM alignment delta |
| `c7_paper_renderer` | `purified/figs/c7/` | Ward Stage B Δgc, PR-AUC |
| `c7_tex_snippets` | `purified/figs/c7/` | LaTeX-ready tables |

## Caveats

- Some C6 `wang_full.json` files and C7 `judge_outputs.jsonl` files are
  large; they are the binding inputs for the case-study figures.
- The renderers do not retrain models or re-run case studies — they
  only re-plot from cached metrics + judge outputs. If a cell is
  missing from `leaderboard.jsonl`, run its
  `experiments/cN_*/run.py` first.
- `results/leaderboard.jsonl` and `checkpoints/manifest.jsonl` are
  append-only; the renderers compute "latest per cell" via
  `(component, arch, seed, eval_protocol_version)` deduplication.
- `figs/` is not gitignored; renderer outputs end up tracked when you
  commit. Add `figs/` to `.gitignore` locally if you want to iterate
  without that.
