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

All commands below assume you have a single GPU box (RunPod, local workstation, whatever your group uses) with CUDA set up. The pipeline is deliberately portable — nothing below depends on a specific cluster or scheduler.

### One-time setup

```bash
# 1. Clone and install
git clone <repo-url>
cd temp_xc
git checkout aniket
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets huggingface_hub accelerate safetensors sentencepiece \
            numpy scipy einops tqdm pyyaml \
            wandb matplotlib seaborn plotly pandas \
            scikit-learn umap-learn hdbscan numba \
            anthropic pydantic
pip install --no-deps -e .

# 2. HuggingFace + wandb tokens
export HF_TOKEN="hf_..."
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="..."
# Accept licenses on HuggingFace for: DeepSeek-R1-Distill-Llama-8B, Llama-3.1-8B,
# google/gemma-2-2b, google/gemma-2-2b-it — gated models will 401 otherwise.

# 3. Download model weights + small datasets (~45 GB)
bash scripts/download_models.sh

# 4. GPU sanity check
python scripts/verify_gpu_fit.py --model deepseek-r1-distill-llama-8b
# Expect peak ≈ 16–18 GB on an 80 GB H100, comfortable headroom on A100 40 GB.
```

### Stage 1 — cache activations

```bash
# Reasoning traces from a thinking model (DeepSeek on GSM8K)
python scripts/cache_reasoning_traces.py \
    --model deepseek-r1-distill-llama-8b \
    --dataset gsm8k --num-sequences 1000 \
    --gen_max_new_tokens 1024 --layer_indices 12 24

# Forward pass on non-reasoning data (Gemma on FineWeb, coding corpora)
python -m temporal_crosscoders.NLP.cache_activations \
    --model gemma-2-2b --dataset fineweb --mode forward \
    --num-sequences 24000 --seq-length 32 --layer_indices 13 25
```

Output lands at `data/cached_activations/<model>/<dataset>/`. Safe to delete between runs; re-running the same command is idempotent (existing layers are skipped).

### Stage 2 — run an architecture sweep

```bash
# Default comparison: topk_sae + stacked_sae + crosscoder (TXCDRv2)
python -m src.bench.sweep \
    --dataset-type cached_activations \
    --model-name deepseek-r1-distill-llama-8b \
    --cached-dataset gsm8k --cached-layer-key resid_L12 \
    --models topk_sae stacked_sae crosscoder \
    --k 50 --T 5 --steps 10000 \
    --results-dir results/nlp/day1-gsm8k

# Shuffled control — re-run everything with temporal order destroyed.
# Every real experiment runs twice. This is non-negotiable for any claim
# about "temporal" features (see the TFA free-dense-channel confound).
python -m src.bench.sweep \
    --dataset-type cached_activations \
    --model-name deepseek-r1-distill-llama-8b \
    --cached-dataset gsm8k --cached-layer-key resid_L12 \
    --models topk_sae stacked_sae crosscoder \
    --k 50 --T 5 --steps 10000 \
    --shuffle-within-sequence \
    --results-dir results/nlp/day1-gsm8k-shuffled
```

### Stage 3 — aggregate + post to Slack

```bash
python scripts/aggregate_results.py --root results/nlp --out reports/day1-gsm8k
# Emits report.md + nmse_l0.png + max_cos_hist.png + temporal_mi.png + span_hist.png
# Drag the whole reports/day1-gsm8k/ dir into Slack — PNGs render inline.
```

### Working with cached data across sessions

- `data/cached_activations/` can be tens of GB per run — keep it on local/scratch disk, never commit it (already gitignored).
- `results/nlp/` is KB per run — fine to sync anywhere.
- `reports/` is what you paste into Slack.

### Note on the `scripts/runpod_*` scripts

Those are Aniket's RunPod-specific wrappers for the common flows (setup, verify GPU, cache activations, sweep, autointerp scan/explain/fmap, backfill metrics, delphi labeling). Portable by design — they just run the same Python commands as sections 1–3 of this cheat sheet, with `uv`-managed env activation and sensible defaults. Read one if you want a template for your own pod.

## 6. Adding things

### Add a new subject model

1. Append a `ModelConfig(...)` entry to `src/bench/model_registry.py::MODEL_REGISTRY`.
2. Add the HF path to `scripts/download_models.sh::MODELS`.
3. `bash scripts/download_models.sh`
4. Accept the license on HuggingFace for that repo if gated.
5. `python scripts/verify_gpu_fit.py --model <new-key>` to confirm it loads.

That's it. Every downstream script now accepts `--model <new-key>`.

### Add a new dataset

1. Add a branch to `_load_text_stream()` in `temporal_crosscoders/NLP/cache_activations.py`. Prefer `stream=False` for small curated datasets — it makes runs robust in offline / disconnected environments.
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
3. Run with `--models ... <name>` (or the env var on your own runner script).

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
  download_models.sh           # pull all registry models + small datasets (portable)
  verify_gpu_fit.py            # memory sanity check for any registry model
  cache_reasoning_traces.py    # wrapper for GSM8K/MATH500 in generate mode
  aggregate_results.py         # Pareto / MI / span plots + markdown report
  runpod_*                     # RunPod-specific wrappers (portable, optional)
data/
  cached_activations/          # .gitignored — Stage 1 outputs, ~30 GB/run
results/nlp/                   # .gitignored — Stage 2 outputs, KB/run
reports/                       # .gitignored — Stage 3 outputs, few MB/run
```

## 8. Sprint reporting cadence

Target: **2–3 Slack reports per week**, each = 1 PNG + 1 paragraph + 1 open question. Generated from `python scripts/aggregate_results.py` output. Drag the report dir into Slack; PNGs render inline.

The one-line north star: *does the temporal crosscoder's MI-per-lag curve sit measurably above the SAE baseline on unshuffled data, and does it collapse to the SAE baseline under shuffling?* If yes, the paper's thesis holds. If no, we need a different cut.

## 9. Known gaps / things we're waiting on

- **TFA NLP port (Han)** — `src/bench/architectures/tfa.py` works on toy Markov data but hasn't been validated on real-LM activations. Don't add it to the default `--models` list until confirmed.
- **Feature steering harness** — Han is owning this. Coordinate before touching `src/bench/steering.py` (doesn't exist yet).
- **`encode()` on every ArchSpec** — some architectures may not expose an encode method, in which case temporal metrics silently return None. If you notice empty plots in the temporal MI / span / cluster sections of a report, that's why.
- **Large-dataset streaming in offline contexts** — `fineweb` and `coding` are streamed from HuggingFace. If your compute box doesn't have outbound network, you'll need to pre-fetch a slice on a machine that does, or switch `stream=False` and take the full-download hit.
- **`src/bench/data.py` eval_hidden load** — currently `.float()` copies the entire cached tensor into RAM up front. For a 1000×1024×4096 float32 tensor that's 16 GB — fine for now, but if the cache scales up this'll need to become truly lazy (mmap tensor passed straight to model).
