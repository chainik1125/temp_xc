# TempBench

A benchmark suite for **temporal sparse-dictionary architectures** on
language-model activations. TempBench packages seven evaluations —
two synthetic, five real-LM — and a single, locked training pipeline
so that every architecture is compared at matched hookpoint, matched
sparsity, and matched activation cache.

The accompanying paper introduces two **temporal crosscoder** (TXC)
architectures — TXC-base and TXC-pro — and benchmarks them against
TopK SAE, T-SAE, TFA, Stacked SAE, MLC, and SAE-arditi baselines.

## What's in here

```
TempBench/
├── src/temp_bench/         # the library — architectures, training,
│                           # evaluation, runner, cache, schemas
├── experiments/c1_..c7_*/  # one dir per benchmark component (run.py
│                           # + analysis.py, both invoked via the
│                           # framework runner)
├── configs/                # locked_archs.yaml + datasources.yaml —
│                           # single source of truth for hparams
├── scripts/                # paper-figure renderers (c2, c3, c6,
│                           # rlhf, umap, global rose, etc.)
├── results/                # leaderboard.jsonl + per-cell run-dirs +
│                           # case-study judge transcripts
├── checkpoints/            # trained model weights (manifest.jsonl
│                           # + train-keyed sub-dirs)
├── data/                   # synthetic data generators
├── docs/components/        # per-component writeups (c1…c7.md)
├── docs/paper/             # framework, architecture, training and
│                           # compute-resources reference docs
├── figs/                   # rendered paper figures
├── tests/                  # framework + cache-key + schema smoke tests
├── REPRODUCE_FIGURES.md    # one-stop reviewer guide
└── PROTOCOL.md             # framework discipline (cache contract,
                            # version-bump rules, two-TXC commitment)
```

## The seven components

| C | Subject | Headline metric |
|---|---|---|
| C1 | Synthetic TopK sweep on toy Markov features | NMSE / single-latent AUC |
| C2 | Synthetic coupled features + noisy emissions | $g$AUC at $n_{\mathrm{parents}}=10$ |
| C3 | Sparse probing on Gemma-2-2B-IT layer 13 (38 SAEBench tasks) | $\overline{\mathrm{AUC}}$ over $\log_2 k_{\mathrm{feats}}$ |
| C4 | Qualitative latents on Gemma-2-2B-IT (UMAP + autointerp) | per-cluster Pareto |
| C5 | RLHF steering on HH-RLHF (Gemma-2-2B-IT) | peak success grade @ coh ≥ 1.75 |
| C6 | Emergent misalignment on Qwen-2.5-7B-Instruct + bad-medical-advice LoRA | Δalign at coh ≥ 70, detection PR-AUC |
| C7 | Backtracking on Llama-3.1-8B-BASE layer 10 (Ward Stage B) | peak Δgc, PR-AUC at $S{=}8$ |

A six-axis rose summary across all components is in
`figs/global_rose_summary.pdf`.

## Quick start

```bash
# 1. Clone, set up venv, install package
git clone <this-repo>
cd <this-repo>
uv sync                     # pins dependencies from uv.lock

# 2. Sanity-check the install (cache-key, schema, runner contract)
bash scripts/smoke_test.sh

# 3. Re-render every paper figure from cached data (~1 minute total)
.venv/bin/python -m scripts.c2_paper_renderer
.venv/bin/python -m scripts.c3_paper_renderer
.venv/bin/python -m scripts.c6_paper_renderer
.venv/bin/python -m scripts.umap_txc_paper_renderer
.venv/bin/python -m scripts.rlhf_paper_renderer
.venv/bin/python -m scripts.global_rose_renderer
# → figures land in figs/, matching the paper's \includegraphics paths
```

For each paper figure, `REPRODUCE_FIGURES.md` documents both the
**fast path** (regen the plot from the cached `results/leaderboard.jsonl`
slice that ships in this repo) and the **from-scratch path** (re-train
checkpoints from raw activations and rebuild the leaderboard).

## Running an experiment from scratch

Every experiment goes through `runner.run_cell` so that training and
evaluation are deterministic, cache-keyed by
`(arch, datasource, training_cfg, seed)`, and append-only to a single
leaderboard. To re-run a component from raw activations:

```bash
TQDM_DISABLE=1 .venv/bin/python -m experiments.c1_synthetic_topk.run
TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing.run
# ...
```

Hardware budgets per component are tabulated in
`docs/paper/compute_resources.md` (range: ~1 H100-hr for C1, ~135
H100-hr for the C2 coupled-features sweep).

## Architectures

The two locked TXC architectures and the five baselines share a single
training pipeline. Hyperparameters are not hand-tuned per component —
the only free knobs are sparsity ($k_{\mathrm{pos}}$) and dictionary
width ($d_{\mathrm{SAE}}$). Full spec in `docs/paper/architecture.md`.

| arch | family | locked window | sparsity | extras |
|---|---|---|---|---|
| TopK SAE | per-token | — | TopK at $k_{\mathrm{pos}}$ | — |
| Stacked SAE | per-token × T | T = 5 | $k_{\mathrm{win}} = T \cdot k_{\mathrm{pos}}$ | T independent SAEs |
| T-SAE | window | T = 5 | Matryoshka BatchTopK | temporal contrastive, AuxK |
| TFA | window | T = 5 | $k_{\mathrm{pos}}$ | 4-head attention prior |
| MLC | per-token, multi-layer | L = 5 | TopK at $k_{\mathrm{pos}}$ | layer-axis crosscoder |
| **TXC-base** | window | T = 5 | TopK at $k_{\mathrm{win}}$ | anti-dead stack |
| **TXC-pro** | window | $T_{\max}{=}10$, $t_{\mathrm{sample}}{=}5$ | TopK + matryoshka H8 | multi-distance InfoNCE |

## Adding an architecture

1. Drop a class subclassing `temp_bench.architectures.base.TempBenchArch`
   into `src/temp_bench/architectures/<name>.py`.
2. Register it in `configs/locked_archs.yaml` (class path,
   `arch_version`, hparams, optional per-component overrides).
3. Re-run any component's `experiments/cN_*/run.py`. Only your new
   architecture's cells will compute; everything else is cached.

## License

See `LICENSE`.

## Citation

If you use TempBench, please cite the accompanying paper.
