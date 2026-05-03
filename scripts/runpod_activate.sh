#!/bin/bash
# runpod_activate.sh — Source this in every new shell.
#
#   source scripts/runpod_activate.sh
#
# Loads .env, activates the uv-managed venv, sets sane defaults for
# offline-friendly tooling. Idempotent.

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo /workspace/temp_xc)"
cd "$REPO_ROOT"

# Load API keys + HF cache config from .env if present
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# uv writes the venv to .venv by default
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export PATH="$HOME/.local/bin:$PATH"

# Put the repo root on Python's import path so `from src.bench...` and
# `from temporal_crosscoders.NLP...` work regardless of cwd or how the
# script is invoked. Trillium's env did this implicitly via a venv
# activate.d hook; RunPod doesn't get that for free.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# tqdm progress bars work better when stderr isn't buffered
export PYTHONUNBUFFERED=1

# Don't go offline — RunPod has internet, unlike Trillium
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE 2>/dev/null || true

echo "(runpod-txc) repo: $REPO_ROOT"
[ -n "${ANTHROPIC_API_KEY:-}" ] && echo "  ANTHROPIC_API_KEY: set"   || echo "  ANTHROPIC_API_KEY: MISSING"
[ -n "${HF_TOKEN:-}" ]          && echo "  HF_TOKEN:          set"   || echo "  HF_TOKEN:          MISSING"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
