#!/bin/bash
# runpod_setup.sh — One-time pod bootstrap.
#
# Assumes a fresh RunPod PyTorch template (PyTorch 2.4 + CUDA 12.1 or
# similar). Installs uv, syncs project deps, creates a .env template
# for API keys.
#
# Run from the repo root:
#   cd /workspace/temp_xc
#   bash scripts/runpod_setup.sh
#
# Then fill in .env, then `source scripts/runpod_activate.sh`.

set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /workspace/temp_xc)"

echo ">> Installing uv..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo ">> uv sync (resolves pyproject.toml + lockfile)..."
uv sync

echo ">> Creating .env template (you'll need to fill it in)..."
if [ ! -f .env ]; then
    cat > .env <<'ENV'
# RunPod environment variables — sourced by scripts/runpod_activate.sh.
# Fill in your real values, then `source scripts/runpod_activate.sh`.

# Anthropic API for Claude Haiku autointerp explanations
export ANTHROPIC_API_KEY=""

# HuggingFace for downloading subject models + streaming datasets
export HF_TOKEN=""
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Weights & Biases — experiment tracking + checkpoint artifact registry
# When WANDB_API_KEY is set, sweep.py logs per-step train loss/L0,
# checkpoints as W&B Artifacts, and final eval NMSE/L0 to the project
# below. Leave WANDB_ENTITY blank for a personal project, or set it to
# your team's entity name (e.g. "spar-temp-xc") for team-visible runs.
export WANDB_API_KEY=""
export WANDB_PROJECT="temporal-crosscoders"
export WANDB_ENTITY=""
export WANDB_GROUP=""

# HF cache lives on the persistent volume so it survives pod restarts
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
ENV
    echo "   Created .env template — edit it with your keys."
else
    echo "   .env already exists, leaving it."
fi

mkdir -p "/workspace/.cache/huggingface" data/cached_activations reports results/nlp logs

echo ""
echo "=== runpod_setup done ==="
echo "Next:"
echo "  1. nano .env   # fill in API keys"
echo "  2. source scripts/runpod_activate.sh"
echo "  3. bash scripts/runpod_verify_gpu.sh gemma-2-2b"
