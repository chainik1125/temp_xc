#!/bin/bash
# runpod_venhoff_bootstrap.sh — one-shot setup for a FRESH pod to run
# the Venhoff reasoning-eval paper-budget pipeline.
#
# Does everything that doesn't need secrets:
#   1. Clones the temp_xc repo into /workspace/spar-temporal-crosscoders
#   2. Installs uv + creates the main 3.11 venv + syncs deps
#   3. Clones Venhoff's `cvenhoff/thinking-llms-interp` into vendor/
#   4. Creates Venhoff's own 3.12 venv + installs their deps
#      (their steering-vector optimizer imports their own package tree)
#   5. Drops a .env template with every key the pipeline reads
#   6. Prepares cache dirs on the persistent volume
#
# After this script finishes, manually:
#   nano /workspace/spar-temporal-crosscoders/.env   # fill in secrets
#   cd /workspace/spar-temporal-crosscoders
#   source .env
#   huggingface-cli login --token "$HF_TOKEN"       # one-time auth
#   bash scripts/runpod_venhoff_paper_run.sh        # kick off the run
#
# Usage (from anywhere on a fresh pod):
#   curl -sL https://raw.githubusercontent.com/chainik1125/temp_xc/aniket/scripts/runpod_venhoff_bootstrap.sh | bash
# — or, if the repo is already cloned:
#   bash scripts/runpod_venhoff_bootstrap.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/chainik1125/temp_xc.git}"
REPO_BRANCH="${REPO_BRANCH:-aniket}"
REPO_DIR="${REPO_DIR:-/workspace/spar-temporal-crosscoders}"
VENHOFF_URL="${VENHOFF_URL:-https://github.com/cvenhoff/thinking-llms-interp.git}"

echo "=== Venhoff paper-budget pod bootstrap ==="
echo "  repo:     $REPO_URL ($REPO_BRANCH)"
echo "  location: $REPO_DIR"
echo "  venhoff:  $VENHOFF_URL"
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

# ── 2. Install uv (fast resolver/venv manager). Idempotent.
if ! command -v uv >/dev/null 2>&1; then
    echo ">> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ── 3. Main repo venv (Python 3.11 — what pyproject.toml requires).
if [[ ! -d .venv ]]; then
    echo ">> creating main 3.11 venv"
    uv venv --python 3.11 .venv
fi
echo ">> uv sync (resolving pyproject.toml + lockfile)"
source .venv/bin/activate
# Regenerate lock if the committed one predates the simplexity-extra removal
# (old lockfiles still reference the extra → unsatisfiable on 3.11).
uv lock --upgrade 2>&1 | tail -3 || true
uv sync
deactivate

# ── 4. Clone Venhoff's vendor repo (needed for steering-vector training
#      AND to reuse their shipped pre-trained SVs for the SAE arch).
mkdir -p vendor
if [[ ! -d vendor/thinking-llms-interp/.git ]]; then
    echo ">> cloning Venhoff's thinking-llms-interp into vendor/"
    git clone --depth 1 "$VENHOFF_URL" vendor/thinking-llms-interp
else
    echo ">> vendor/thinking-llms-interp already present — pulling latest"
    (cd vendor/thinking-llms-interp && git pull --ff-only origin main 2>&1 | tail -3 || true)
fi

# Sanity-check: the 16 pre-trained Llama-8B steering vectors should be checked
# into their repo. If not, something's wrong with the clone (LFS? submodule?).
SV_DIR="vendor/thinking-llms-interp/train-vectors/results/vars/optimized_vectors"
SV_COUNT=$(ls "$SV_DIR"/llama-3.1-8b_bias.pt "$SV_DIR"/llama-3.1-8b_idx{0..14}.pt 2>/dev/null | wc -l)
if [[ "$SV_COUNT" -eq 16 ]]; then
    echo ">> found 16/16 pre-trained Llama-8B steering vectors — SAE Phase 2 will skip"
else
    echo ">> WARNING: only $SV_COUNT/16 pre-trained vectors found at $SV_DIR"
    echo "   Check the Venhoff repo state; the SAE arch will try to re-train missing ones."
fi

# ── 5. Venhoff's Python 3.12 venv (their pyproject requires >=3.12,<3.13).
#      Needed by optimize_steering_vectors.py + hybrid_token.py subprocesses.
if ! command -v python3.12 >/dev/null 2>&1; then
    # uv can provision 3.12 automatically via --python 3.12.
    true
fi
if [[ ! -d vendor/thinking-llms-interp/.venv ]]; then
    echo ">> creating Venhoff's 3.12 venv at vendor/thinking-llms-interp/.venv"
    (cd vendor/thinking-llms-interp && uv venv --python 3.12 .venv)
fi
echo ">> installing Venhoff's deps (~400 MB, takes a few minutes)"
(cd vendor/thinking-llms-interp && source .venv/bin/activate && uv pip install -e . && deactivate) \
    || {
        echo "   NOTE: uv pip install -e . failed — falling back to pip install -r requirements.txt if present"
        if [[ -f vendor/thinking-llms-interp/requirements.txt ]]; then
            (cd vendor/thinking-llms-interp && source .venv/bin/activate \
                && uv pip install -r requirements.txt && deactivate)
        fi
    }

# ── 6. Drop .env template (user fills in real values afterward).
if [[ ! -f .env ]]; then
    echo ">> writing .env template"
    cat > .env <<'ENV'
# Venhoff reasoning-eval environment variables.
# Fill in your real values; NEVER commit this file.
# After editing, source with: set -a; source .env; set +a

# ── Required for the paper run ─────────────────────────────────────
# HuggingFace — downloads Llama-3.1-8B + DeepSeek-R1-Distill-Llama-8B.
# Both require an accepted license on the respective HF model pages.
export HF_TOKEN=""
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Anthropic API — used by the Haiku 4.5 judge for taxonomy scoring
# (Phase 0 sentence labeling + score/bridge stages).
export ANTHROPIC_API_KEY=""

# OpenAI API — used by the GPT-4o judge-drift bridge check. Skip with
# SKIP_BRIDGE=1 if you don't want to pay for the 100-sentence bridge run.
export OPENAI_API_KEY=""

# ── Optional ───────────────────────────────────────────────────────
# OpenRouter — only matters if you swap the judge to an OR-routed model.
export OPENROUTER_API_KEY=""

# Weights & Biases — experiment tracking. Leave WANDB_API_KEY empty to
# disable W&B logging entirely.
export WANDB_API_KEY=""
export WANDB_PROJECT="temporal-crosscoders"
export WANDB_ENTITY=""
export WANDB_GROUP=""

# ── Cache locations (persistent volume) ────────────────────────────
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1  # faster model downloads
ENV
    echo "   .env created — EDIT THIS NEXT: nano .env"
else
    echo ">> .env already exists — not overwriting"
fi

# ── 7. Cache dirs on the persistent volume.
mkdir -p /workspace/.cache/huggingface \
         "$REPO_DIR/logs" \
         "$REPO_DIR/results/venhoff_eval" \
         "$REPO_DIR/data/cached_activations"

# ── 8. Quick health report.
echo ""
echo "=== bootstrap complete ==="
echo "repo:     $REPO_DIR"
echo "main venv: $REPO_DIR/.venv  ($("$REPO_DIR/.venv/bin/python" --version 2>&1))"
echo "venhoff venv: $REPO_DIR/vendor/thinking-llms-interp/.venv  ($("$REPO_DIR/vendor/thinking-llms-interp/.venv/bin/python" --version 2>&1))"
echo "pre-trained SVs: $SV_COUNT/16 present at $SV_DIR"
echo "GPUs detected: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""
echo "Next steps (manual):"
echo "  cd $REPO_DIR"
echo "  nano .env                                     # paste HF_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY"
echo "  set -a; source .env; set +a"
echo "  huggingface-cli login --token \"\$HF_TOKEN\"     # verify HF access"
echo "  source .venv/bin/activate"
echo "  bash scripts/runpod_venhoff_paper_run.sh      # kick off the run"
