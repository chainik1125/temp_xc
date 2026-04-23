#!/bin/bash
# runpod_scrappy_bootstrap.sh — one-shot setup for a FRESH single-GPU pod
# to run the venhoff_scrappy autoresearch loop.
#
# Strictly smaller sibling of scripts/runpod_venhoff_bootstrap.sh — same
# repo + venv + vendor setup, but with all the lessons-learned from the
# 2026-04-22/23 paper-budget run baked in:
#   - HF_HOME + TORCHINDUCTOR_CACHE_DIR pointed at /workspace BEFORE any
#     download (prevents 32 GB of container-disk HF cache growth).
#   - `hf auth login` (modern cmd; huggingface-cli is deprecated).
#   - Deletes ~/.cache/uv after sync (reclaims ~15 GB container disk).
#   - Runs the vendor-patch smoke test so drift fails fast.
#   - Pre-downloads the 2 × 8B models so the first cycle isn't a ~10 min
#     wait for model shards.
#
# Pod sizing (see docs/aniket/experiments/venhoff_scrappy/plan.md):
#   - Container disk: 30 GB (20 GB triggers the uv cache trap)
#   - Volume: 100 GB (30 GB for HF cache + repo + vendor + results)
#   - GPU: any single 48GB+ GPU. Recommended order:
#       1. RTX PRO 6000 (96 GB, $1.61/hr, Medium avail)
#       2. A100 SXM (80 GB, $1.22/hr, Medium)
#       3. A40 (48 GB, $0.44/hr, if it returns)
#     Do NOT use 24 GB cards (A5000, 3090, 4090) — both 8B models + KV
#     cache won't fit without 8-bit quantization, which is exactly what
#     we've patched out.
#
# Usage (from anywhere on a fresh pod):
#   curl -sL https://raw.githubusercontent.com/chainik1125/temp_xc/aniket/scripts/runpod_scrappy_bootstrap.sh | bash
# — or, if the repo is already cloned:
#   bash scripts/runpod_scrappy_bootstrap.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/chainik1125/temp_xc.git}"
REPO_BRANCH="${REPO_BRANCH:-aniket}"
REPO_DIR="${REPO_DIR:-/workspace/spar-temporal-crosscoders}"
VENHOFF_URL="${VENHOFF_URL:-https://github.com/cvenhoff/thinking-llms-interp.git}"

# ── Cache locations on the 100GB volume (not the 30GB container disk!)
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.cache/torchinductor}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$UV_CACHE_DIR"

# Persist cache exports for future shells.
cat > /etc/profile.d/scrappy_caches.sh <<EOF
export HF_HOME=$HF_HOME
export TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR
export UV_CACHE_DIR=$UV_CACHE_DIR
export PATH=\$HOME/.local/bin:\$PATH
EOF
echo ">> wrote /etc/profile.d/scrappy_caches.sh — caches pinned to /workspace"

echo "=== venhoff_scrappy pod bootstrap ==="
echo "  repo:     $REPO_URL ($REPO_BRANCH)"
echo "  location: $REPO_DIR"
echo "  HF_HOME:  $HF_HOME"
echo "  UV cache: $UV_CACHE_DIR"
echo ""

# ── 1. Clone or update the main repo.
if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo ">> cloning temp_xc into $REPO_DIR"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
    echo ">> $REPO_DIR already a git repo — fetching + fast-forwarding $REPO_BRANCH"
    cd "$REPO_DIR"
    git fetch origin "$REPO_BRANCH"
    git checkout "$REPO_BRANCH" 2>/dev/null || git checkout -b "$REPO_BRANCH" "origin/$REPO_BRANCH"
    git pull --ff-only origin "$REPO_BRANCH" || true
fi
cd "$REPO_DIR"

# ── 2. Install uv.
if ! command -v uv >/dev/null 2>&1; then
    echo ">> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ── 3. Main venv (Python 3.11).
if [[ ! -d .venv ]]; then
    echo ">> creating main 3.11 venv"
    uv venv --python 3.11 .venv
fi
echo ">> uv sync"
source .venv/bin/activate
uv lock --upgrade 2>&1 | tail -3 || true
uv sync
deactivate

# Install pyyaml for run_cycle.py + autoresearch_summarise.py.
.venv/bin/pip install --quiet pyyaml || true

# ── 4. Vendor repo.
mkdir -p vendor
if [[ ! -d vendor/thinking-llms-interp/.git ]]; then
    echo ">> cloning Venhoff vendor repo"
    git clone --depth 1 "$VENHOFF_URL" vendor/thinking-llms-interp
fi

# ── 5. Venhoff's own 3.12 venv.
if [[ ! -d vendor/thinking-llms-interp/.venv ]]; then
    echo ">> creating Venhoff 3.12 venv"
    cd vendor/thinking-llms-interp
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r <(echo "
    torch
    transformers
    accelerate
    datasets
    nnsight
    tqdm
    sentencepiece
    protobuf
    einops
    huggingface_hub
    openai
    anthropic
    ") 2>&1 | tail -5
    cd "$REPO_DIR"
fi

# ── 6. Container-disk reclaim — uv cache can balloon to ~15 GB after
#      sync, and on a 20 GB container disk that eats the whole budget.
#      With UV_CACHE_DIR on /workspace this is defensive but cheap.
if [[ -d "$HOME/.cache/uv" ]]; then
    echo ">> clearing legacy ~/.cache/uv (redundant with UV_CACHE_DIR=$UV_CACHE_DIR)"
    rm -rf "$HOME/.cache/uv"
fi

# ── 7. Apply + verify vendor patches. Fails loud if upstream drifted.
echo ">> verifying vendor patches"
.venv/bin/python - <<'PY'
from pathlib import Path
from src.bench.venhoff.vendor_patches import (
    ensure_hybrid_judge_patched, ensure_steering_patched,
)
root = Path("vendor/thinking-llms-interp")
ensure_hybrid_judge_patched(root)
ensure_steering_patched(root)
print("[ok] all vendor patches applied/confirmed")
PY

# ── 8. .env template.
if [[ ! -f .env ]]; then
    cat > .env <<'EOF'
# Fill these in manually after bootstrap.
HF_TOKEN=
ANTHROPIC_API_KEY=
# Optional, only if using OpenRouter as a fallback:
OPENROUTER_API_KEY=
# Git identity for automatic per-cycle commits (already defaulted in
# run_autoresearch.sh but can override):
GIT_USER_NAME=aniket
GIT_USER_EMAIL=aniketdeshh@gmail.com
GIT_BRANCH=aniket
EOF
    echo ">> .env template written — fill in HF_TOKEN + ANTHROPIC_API_KEY"
fi

# ── 9. Disk check.
echo ""
echo "=== disk sanity ==="
df -h / | tail -1 | awk '{print "container disk:", $3, "/", $2, "("$5" used)"}'
df -h /workspace | tail -1 | awk '{print "volume /workspace:", $3, "/", $2, "("$5" used)"}'

echo ""
echo "=== bootstrap done ==="
echo ""
echo "Next steps — all single-line bash, no python or heredocs required:"
echo "  1. nano /workspace/spar-temporal-crosscoders/.env     # fill in HF_TOKEN + ANTHROPIC_API_KEY"
echo "  2. cd /workspace/spar-temporal-crosscoders && source .env"
echo "  3. hf auth login --token \$HF_TOKEN"
echo "  4. tmux new -s scrappy"
echo "  5. bash scripts/prefetch_scrappy_models.sh            # ~10 min, primes HF cache"
echo "  6. bash scripts/smoketest_judge.sh                    # ~2s, checks judge API is live"
echo "  7. bash experiments/venhoff_scrappy/run_autoresearch.sh baseline_sae"
echo ""
