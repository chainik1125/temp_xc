#!/bin/bash
# download_models.sh — Pre-fetch all registry models into the local HF cache.
#
# Download-only, no GPU needed. Reads $HF_HOME if set, otherwise defaults to
# the HuggingFace default (~/.cache/huggingface).
#
#   bash scripts/download_models.sh
#
# For gated models (Llama, Gemma) you need:
#   1. HF account with license accepted for each repo
#   2. HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) exported in your environment
#
# Adding a model: append to MODELS below AND to src/bench/model_registry.py.

set -euo pipefail

: "${HF_HOME:=$HOME/.cache/huggingface}"
export HF_HOME
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "WARNING: no HF_TOKEN in env — gated models (Llama/Gemma) will 401."
fi

MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   # priority
    "meta-llama/Llama-3.1-8B"                    # base for Venhoff hybrid
    "google/gemma-2-2b"                          # Andre's existing pipeline
    "google/gemma-2-2b-it"                       # instruction-tuned variant
)

echo "=== Downloading ${#MODELS[@]} models to $HF_HOME ==="

for repo in "${MODELS[@]}"; do
    echo ""
    echo ">> $repo"
    python - <<PY
import os
from huggingface_hub import snapshot_download
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
path = snapshot_download(
    repo_id="${repo}",
    token=token,
    allow_patterns=["*.json", "*.model", "*.safetensors", "tokenizer*", "*.txt"],
)
print(f"   cached at {path}")
PY
done

echo ""
echo "=== All models cached ==="
echo "Memory check: verify DeepSeek-R1-Distill-Llama-8B fits in fp16 on target GPU."
echo "Run inside a GPU allocation:"
echo "  python scripts/verify_gpu_fit.py --model deepseek-r1-distill-llama-8b"
