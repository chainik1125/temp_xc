#!/bin/bash
# runpod_delphi_label.sh — SOTA auto-interp via EleutherAI delphi
# (Paulo et al., ICLR 2025). Labels features on a trained checkpoint
# using delphi's DefaultExplainer + DetectionScorer + FuzzingScorer,
# with Claude Haiku 4.5 as the explainer.
#
# Adapter: temporal_crosscoders/NLP/delphi_adapter.py
#   - precomputed_to_safetensors(): walks our .npy activation caches,
#     runs the arch-specific encode(), emits sharded safetensors in
#     delphi's expected format. No Gemma/DeepSeek forward re-runs.
#   - AnthropicClient: delphi.Client-compatible wrapper over the
#     anthropic SDK with retry/semaphore.
#
# Usage:
#   bash scripts/runpod_delphi_label.sh \
#     --checkpoint results/nlp/step1-unshuffled/ckpts/topk_sae__gemma-2-2b__fineweb__resid_L13__k100__seed42.pt \
#     --arch topk_sae \
#     --subject-model gemma-2-2b \
#     --cached-dataset fineweb \
#     --layer-key resid_L13 \
#     --label step1-unshuffled__topk_sae \
#     --output-dir reports/step1-gemma-replication/delphi \
#     --max-features 10             # smoke test — remove for full run
#
# Smoke-test cost: ~$0.05 at Haiku for 10 features.
# Full 5000-feature cost: ~$20 at Haiku per checkpoint.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "FAIL: ANTHROPIC_API_KEY not set. Edit .env and re-source runpod_activate.sh."
    exit 1
fi

# ─── defaults ─────────────────────────────────────────────────────────────
CHECKPOINT=""
ARCH=""
SUBJECT_MODEL=""
CACHED_DATASET=""
LAYER_KEY=""
LABEL=""
OUTPUT_DIR="reports/delphi"
K=100
T=5
EXPANSION_FACTOR=8
MAX_FEATURES=""
N_SPLITS=5
EXPLAIN_MODEL="claude-haiku-4-5-20251001"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --arch) ARCH="$2"; shift 2;;
        --subject-model) SUBJECT_MODEL="$2"; shift 2;;
        --cached-dataset) CACHED_DATASET="$2"; shift 2;;
        --layer-key) LAYER_KEY="$2"; shift 2;;
        --label) LABEL="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --k) K="$2"; shift 2;;
        --T) T="$2"; shift 2;;
        --expansion-factor) EXPANSION_FACTOR="$2"; shift 2;;
        --max-features) MAX_FEATURES="$2"; shift 2;;
        --n-splits) N_SPLITS="$2"; shift 2;;
        --explain-model) EXPLAIN_MODEL="$2"; shift 2;;
        *) echo "Unknown flag: $1"; exit 2;;
    esac
done

for req in CHECKPOINT ARCH SUBJECT_MODEL CACHED_DATASET LAYER_KEY LABEL; do
    if [ -z "${!req}" ]; then
        echo "FAIL: --${req,,} is required" | tr '_' '-'
        exit 2
    fi
done

if [ ! -f "$CHECKPOINT" ]; then
    echo "FAIL: checkpoint not found: $CHECKPOINT"
    exit 1
fi

SAFETENSORS_DIR="$OUTPUT_DIR/$LABEL/safetensors"
RESULTS_DIR="$OUTPUT_DIR/$LABEL/results"
mkdir -p "$SAFETENSORS_DIR" "$RESULTS_DIR"

echo "=== delphi auto-interp ==="
echo "  label:          $LABEL"
echo "  checkpoint:     $CHECKPOINT"
echo "  arch:           $ARCH"
echo "  subject model:  $SUBJECT_MODEL"
echo "  cached dataset: $CACHED_DATASET"
echo "  layer key:      $LAYER_KEY"
echo "  explain model:  $EXPLAIN_MODEL"
echo "  safetensors →   $SAFETENSORS_DIR"
echo "  results →       $RESULTS_DIR"
echo "  max_features:   ${MAX_FEATURES:-<all>}"
echo ""

# ─── 1. adapter: precomputed acts → safetensors ──────────────────────────
echo ">> [1/2] precomputed_to_safetensors"
python - <<PY
from temporal_crosscoders.NLP.delphi_adapter import precomputed_to_safetensors

precomputed_to_safetensors(
    checkpoint="$CHECKPOINT",
    arch="$ARCH",
    subject_model="$SUBJECT_MODEL",
    cached_dataset="$CACHED_DATASET",
    layer_key="$LAYER_KEY",
    k=$K,
    T=$T,
    expansion_factor=$EXPANSION_FACTOR,
    out_dir="$SAFETENSORS_DIR",
    n_splits=$N_SPLITS,
    device="cuda",
)
PY

# ─── 2. delphi Pipeline: explainer + scorers ─────────────────────────────
echo ""
echo ">> [2/2] delphi Pipeline (explainer + detection + fuzzing)"
python - <<PY
import asyncio
from pathlib import Path

from delphi.latents import LatentDataset
from delphi.explainers import DefaultExplainer
from delphi.scorers import DetectionScorer, FuzzingScorer
from delphi.pipeline import Pipeline
from delphi.config import SamplerConfig, ConstructorConfig

from temporal_crosscoders.NLP.delphi_adapter import AnthropicClient

MAX_FEATURES = ${MAX_FEATURES:-None}

client = AnthropicClient(model="$EXPLAIN_MODEL", max_concurrent=5)

dataset = LatentDataset(
    raw_dir=Path("$SAFETENSORS_DIR"),
    sampler_cfg=SamplerConfig(
        train_type="quantiles",
        n_examples_train=40,
        n_quantiles=10,
    ),
    constructor_cfg=ConstructorConfig(
        min_examples=50,
        example_ctx_len=32,
        n_non_activating=20,
        non_activating_source="random",
    ),
)

if MAX_FEATURES is not None:
    dataset = dataset.sample_latents(MAX_FEATURES)  # smoke-test subset

explainer = DefaultExplainer(client=client, threshold=0.3)
detection = DetectionScorer(client=client, n_examples_shown=5)
fuzzing = FuzzingScorer(client=client, n_examples_shown=5)

pipe = Pipeline(
    loader=dataset,
    explainer=explainer,
    scorers=[detection, fuzzing],
    output_dir=Path("$RESULTS_DIR"),
)

asyncio.run(pipe.run())

print()
print(f"done. {client.n_calls} API calls, {client.n_errors} errors.")
print(f"results: $RESULTS_DIR")
PY

echo ""
echo "=== delphi done ==="
ls -1 "$RESULTS_DIR" | head -20
