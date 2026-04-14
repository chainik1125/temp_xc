# Sprint Pipeline — temporal crosscoders on real LMs

This doc explains the end-to-end pipeline for the NeurIPS/ICML exploration sprint: training SAE / temporal crosscoder / TFA variants on cached activations from real language models (DeepSeek-R1-Distill-Llama-8B, Llama-3.1-8B, Gemma 2 2B) and comparing them on temporal-feature metrics.

If you're a teammate who just pulled `aniket` and wants to run an experiment, **read sections 1–3, skim section 4, jump to section 5**.

## 1. Goal

Demonstrate that the temporal crosscoder discovers **qualitatively new feature types** — multi-position structure, reasoning traces, backtracking — that standard SAEs miss. Not "lower NMSE"; *different and useful features*.

Core comparison axis per dataset:

1. Standard SAE (`topk_sae`) — single-token baseline
2. Stacked SAE (`stacked_sae`) — T independent per-position SAEs, window baseline
3. **Temporal crosscoder (`crosscoder` / TXCDRv2)** — shared latent across a T-token window
4. TFA (`tfa`, `tfa_pos`) — Han's port, in progress

Each comparison has to run **both with shuffled and unshuffled temporal order**. If an "advantage" survives shuffling, it was never temporal (this is the TFA "free dense channel" confound).

## 2. Architecture: model-agnostic by design

Everything model-specific flows from **one file**: `src/bench/model_registry.py`. No hardcoded `d_model`, layer index, HF path, or tokenizer anywhere else in the sprint code.

```python
from src.bench.model_registry import get_model_config
cfg = get_model_config("deepseek-r1-distill-llama-8b")
cfg.d_model         # 4096
cfg.n_layers        # 32
cfg.default_layer_indices  # (12, 24)
cfg.is_thinking_model      # True — produces <think> traces
cfg.dtype           # "bfloat16"
cfg.architecture_family    # "llama" — drives hook paths
```

**Adding a new subject model is a 10-line registry edit.** Append a new `ModelConfig(...)` entry, run `bash scripts/download_models.sh`, done. No call-site changes.

### Registered models

| key | HF path | d_model | layers | thinking | dtype |
|---|---|---|---|---|---|
| `deepseek-r1-distill-llama-8b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 4096 | 32 | ✅ | bf16 |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B` | 4096 | 32 | ❌ | bf16 |
| `gemma-2-2b` | `google/gemma-2-2b` | 2304 | 26 | ❌ | fp16 |
| `gemma-2-2b-it` | `google/gemma-2-2b-it` | 2304 | 26 | ❌ | fp16 |

## 3. The three stages

### Stage 1 — cache activations from a subject model

```bash
python -m temporal_crosscoders.NLP.cache_activations \
    --model deepseek-r1-distill-llama-8b \
    --dataset gsm8k \
    --mode generate \
    --num-sequences 1000 \
    --gen_max_new_tokens 1024 \
    --layer_indices 12 24
```

Writes to `data/cached_activations/<model>/<dataset>/`:

- `<layer_key>.npy` — (N, T, d_model) float32 tensors
- `trace_lengths.npy` — real completion length per sequence (for masking)
- `trace_tokens.jsonl` — token ids of the generated traces (for autointerp)
- `layer_specs.json` — metadata sidecar

Two modes:

- **`forward`** — batch forward pass on tokenized text. Use for FineWeb, coding corpora.
- **`generate`** — autoregressive decoding with hooks capturing the last-token activation of each step. Use for reasoning traces on thinking models (GSM8K, MATH500).

Supported datasets out of the box: `fineweb`, `coding`, `gsm8k`, `math500`, `custom`.

### Stage 2 — train architectures on the cached activations

```bash
python -m src.bench.sweep \
    --dataset-type cached_activations \
    --model-name deepseek-r1-distill-llama-8b \
    --cached-dataset gsm8k \
    --cached-layer-key resid_L12 \
    --models topk_sae stacked_sae crosscoder \
    --k 50 --T 5 --steps 10000 \
    --results-dir results/nlp/run1
```

Flags:

- `--dataset-type markov` (default) = toy-data baseline sweep, as before
- `--dataset-type cached_activations` = real-LM mode; skips rho loop, no ground-truth AUC
- `--shuffle-within-sequence` = temporal control baseline. **Run every real experiment twice: once with, once without.**
- `--expansion-factor` (default 8) = `d_sae = d_model × this`

`run_cached_sweep()` in `src/bench/sweep.py` builds one `DataPipeline` via `build_pipeline()` and iterates (seed × k × arch). Writes one JSON per `(dataset × layer × shuffle)` combination to `results/nlp/<run-dir>/`.

### Stage 3 — aggregate + post

```bash
python scripts/aggregate_results.py --root results/nlp --out reports/day1-gsm8k
```

Emits:

- `report.md` — markdown comparison table + embedded plot links
- `nmse_l0.png` — Pareto frontier of reconstruction vs sparsity per arch
- `max_cos_hist.png` — mean max-cosine feature overlap distribution
- `temporal_mi.png` — mean temporal mutual information by lag, per arch
- `span_hist.png` — mean activation span distribution per arch

Every PNG is Slack-pasteable. The markdown table is copy-paste friendly as a code fence.

## 4. Temporal metrics (the sprint-specific evaluation)

On real-LM data (`eval_data.dim() == 3`), `evaluate_model()` in `src/bench/eval.py` automatically computes three temporal metrics via `src/shared/temporal_metrics.py`:

1. **`temporal_mi(acts, lags=(1,2,4,8))`** — binarizes feature activations and computes MI between `f_t` and `f_{t+k}` for each lag. Returns mean MI per lag and fraction of features with MI above threshold. *A feature that can predict its own future at lag > 0 is carrying temporal information.*

2. **`activation_span_stats(acts)`** — per-feature contiguous-burst statistics. Mean span, number of bursts per sequence, duty cycle, fraction of features that only fire on single tokens. *SAE-like features have `mean_span ≈ 1`, temporal features should show `mean_span > 1`.*

3. **`cluster_features(decoder_directions)`** — UMAP + HDBSCAN (KMeans fallback) over decoder directions. Produces 2D coordinates and cluster assignments. *Feeds the Dmitry unsupervised→supervised pipeline: cluster the learned features, auto-interp centroids, compare cluster structure across architectures.*

The metrics only fire if the architecture exposes an `encode()` method returning `(B, T, F)`. If yours doesn't, the fields come back `None` silently — not an error, but the aggregator plots will be empty for that arch. **If you add a new architecture, expose `encode()`.**

## 5. Running an experiment (cheat sheet)

### On Trillium (primary compute)

All commands live under `scripts/trillium_*`. Each wraps the underlying Python with the right `srun` / `sbatch` invocation.

```bash
# One-time setup
bash scripts/trillium_setup.sh           # install deps, create venv
nano ~/.txc_secrets.env                   # HF_TOKEN, WANDB_API_KEY, ANTHROPIC_API_KEY
bash scripts/trillium_setup.sh           # re-run, smoke test
bash scripts/trillium_download_models.sh # pull registry models + datasets
bash scripts/trillium_bootstrap.sh       # all of the above + verify GPU + smoke cache

# Cache activations from a subject model
MODE=smoke bash scripts/trillium_cache_reasoning.sh                               # 100 seqs, interactive
MODE=full bash scripts/trillium_cache_reasoning.sh                                # 1000 seqs, sbatch
MODE=full MODEL=gemma-2-2b DATASET=fineweb bash scripts/trillium_cache_reasoning.sh

# Run a real-LM architecture sweep
bash scripts/trillium_sweep_nlp.sh                                                # defaults
SHUFFLE=1 bash scripts/trillium_sweep_nlp.sh                                      # temporal control
MODEL=gemma-2-2b DATASET=fineweb LAYER=resid_L13 bash scripts/trillium_sweep_nlp.sh
ARCHS="topk_sae stacked_sae crosscoder tfa" bash scripts/trillium_sweep_nlp.sh    # add TFA

# Aggregate + report
bash scripts/trillium_aggregate.sh results/nlp reports/day1-gsm8k

# Spot-check
bash scripts/trillium_spotcheck_acts.sh deepseek-r1-distill-llama-8b gsm8k resid_L12
bash scripts/trillium_sanity_pipeline.sh
```

Env vars all scripts respect: `MODEL`, `DATASET`, `LAYER`, `MODE`, `SHUFFLE`, `ARCHS`, `K`, `T`, `STEPS`, `WALL`. Override on the command line — never edit the scripts.

### On your laptop

```bash
# Pull reports, per-run JSONs, and SLURM logs from Trillium
bash scripts/fetch_from_trillium.sh                    # all three
bash scripts/fetch_from_trillium.sh reports            # just reports, for Slack
bash scripts/fetch_from_trillium.sh --dry-run          # preview

# Re-aggregate locally from pulled JSONs
python scripts/aggregate_results.py --root results/nlp --out reports/local
```

**Cached activations (~30 GB per run) and model weights (~40 GB) stay pinned on Trillium.** Do not rsync them locally.

### On other compute (RunPod, your own GPU)

The `scripts/trillium_*` wrappers are Trillium-specific (SLURM, `$SCRATCH`, Compute Canada wheel pins). Everything else is portable:

```bash
pip install -r requirements.txt   # roll your own requirements — no trillium_sprint_requirements.txt
export HF_TOKEN=...
bash scripts/download_models.sh
python scripts/verify_gpu_fit.py --model deepseek-r1-distill-llama-8b
python -m temporal_crosscoders.NLP.cache_activations --model ... --dataset ...
python -m src.bench.sweep --dataset-type cached_activations --model-name ... ...
python scripts/aggregate_results.py --root results/nlp --out reports/latest
```

## 6. Adding things

### Add a new subject model

1. Append a `ModelConfig(...)` entry to `src/bench/model_registry.py::MODEL_REGISTRY`.
2. Add `HF_PATH` to `scripts/download_models.sh::MODELS`.
3. Run `bash scripts/trillium_download_models.sh` (or its portable equivalent).
4. Accept the license on HuggingFace for that repo if gated.
5. `python scripts/verify_gpu_fit.py --model <new-key>` to confirm it loads.

That's it. Every downstream script now accepts `--model <new-key>` or `MODEL=<new-key>`.

### Add a new dataset

1. Add a branch to `_load_text_stream()` in `temporal_crosscoders/NLP/cache_activations.py`. Set `stream=False` for small curated datasets (Trillium compute nodes are offline).
2. If it's a small curated dataset, pre-cache it in `scripts/download_models.sh::datasets` section.
3. Run `cache_activations` with `--dataset <new-key>`.

### Add a new architecture

1. Create `src/bench/architectures/<name>.py` implementing `ArchSpec`. Must expose:
    - `create(d_in, d_sae, k, device)` → model
    - `train(model, gen_fn, ...)` → training loop
    - `eval_forward(model, x)` → `EvalOutput(sum_se, sum_signal, sum_l0, n_tokens)`
    - `decoder_directions(model, pos=None)` → `(d_in, d_sae)` tensor
    - **`encode(x)`** method (needed for temporal metrics — without it, temporal_mi/span/cluster come back None)
    - `data_format` in `{"flat", "seq", "window"}`
    - `gen_key` matching `data_format` (e.g. `"window_5"`)
2. Register it in `src/bench/architectures/__init__.py::get_default_models`.
3. Run with `ARCHS="... <name>" bash scripts/trillium_sweep_nlp.sh`.

### Add a new metric

1. Implement in `src/shared/temporal_metrics.py` (keep `sae_lens`-importing code out of this module).
2. Wire into `evaluate_model()` in `src/bench/eval.py` — add a field to `EvalResult` and populate it in the real-LM branch.
3. Add a plotting function to `scripts/aggregate_results.py`.

## 7. Layout

```
src/
  bench/
    model_registry.py          # ModelConfig + MODEL_REGISTRY (add models here)
    config.py                  # DataConfig with dataset_type / cached_* / shuffle
    data.py                    # build_pipeline() dispatch; cached-activation loader
    eval.py                    # EvalResult + evaluate_model() with temporal metrics
    sweep.py                   # run_sweep (toy) + run_cached_sweep (real LM)
    architectures/
      base.py                  # ArchSpec interface
      topk_sae.py              # single-token SAE baseline
      stacked_sae.py           # T independent per-position SAEs
      crosscoder.py            # TXCDRv2 (temporal crosscoder — the one we care about)
      tfa.py  _tfa_module.py   # Han's TFA port
  shared/
    metrics.py                 # legacy SAE metrics (imports sae_lens)
    temporal_metrics.py        # temporal_mi, activation_span_stats, cluster_features
temporal_crosscoders/
  NLP/
    cache_activations.py       # --model / --dataset / --mode {forward,generate}
    config.py                  # model-agnostic NLP defaults
    fast_models.py             # FastTemporalCrosscoder / FastStackedSAE (d_in-parameterized)
    autointerp.py              # TopKFinder + Claude-Haiku explainer
scripts/
  download_models.sh           # pull all registry models + small datasets
  verify_gpu_fit.py            # memory sanity check for any registry model
  cache_reasoning_traces.py    # wrapper for GSM8K/MATH500 in generate mode
  aggregate_results.py         # Pareto / MI / span plots + markdown report
  fetch_from_trillium.sh       # (laptop) pull reports + results JSON + SLURM logs
  trillium_*                   # Trillium-specific wrappers (SLURM, Compute Canada)
data/
  cached_activations/          # .gitignored — Stage 1 outputs, ~30 GB/run
results/nlp/                   # .gitignored — Stage 2 outputs, KB/run
reports/                       # .gitignored — Stage 3 outputs, few MB/run
exploration.md                 # .gitignored — live sprint planning doc (Aniket's)
```

## 8. Sprint reporting cadence

Target: **2–3 Slack reports per week**, each one = 1 PNG + 1 paragraph + 1 open question. Generated from `bash scripts/trillium_aggregate.sh` output. Fetch locally with `bash scripts/fetch_from_trillium.sh reports` and paste.

The one-line north star: *does the temporal crosscoder's MI-per-lag curve sit measurably above the SAE baseline on unshuffled data, and does it collapse to the SAE baseline under shuffling?* If yes, the paper's thesis holds. If no, we need a different cut.

## 9. Known gaps / things we're waiting on

- **TFA NLP port (Han)** — `src/bench/architectures/tfa.py` works on toy Markov data but hasn't been validated on real-LM activations. Don't add it to the default `ARCHS` until confirmed.
- **Feature steering harness** — Han is owning this. Coordinate before touching `src/bench/steering.py` (doesn't exist yet).
- **`encode()` on every ArchSpec** — some architectures may not expose an encode method, in which case temporal metrics silently return None. If you notice empty plots in the temporal MI / span / cluster sections of a report, that's why.
- **Large-dataset login-node prefetching** — `fineweb` and `coding` are streamed from HuggingFace and Trillium compute nodes have no outbound internet. We'd need a login-node prefetcher to cache a slice before a compute job can read them. Not a blocker yet, GSM8K / MATH500 are small enough to cache fully.
- **`src/bench/data.py` eval_hidden load** — currently `.float()` copies the entire cached tensor into RAM up front. For a 1000×1024×4096 float32 tensor that's 16 GB — fine for now, but if we scale the cache this'll need to become truly lazy (mmap tensor passed straight to model).
