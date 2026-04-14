#!/bin/bash
# trillium_setup.sh — One-time environment setup on Trillium for the
# NeurIPS/ICML exploration sprint.
#
# Run on the Trillium LOGIN NODE (not inside a compute allocation):
#   ssh aniketrd@trillium-gpu.scinet.utoronto.ca
#   bash $SCRATCH/temp_xc/scripts/trillium_setup.sh
#
# Idempotent: safe to rerun. Creates venv at $HOME/envs/txc, clones repo
# to $SCRATCH/temp_xc, points HF + wandb caches at $SCRATCH so $HOME quota
# doesn't blow up on the first Gemma 2 2B download (~5 GB).

set -euo pipefail

echo "=== temporal-crosscoders sprint setup on Trillium ==="

# ---------------------------------------------------------------- modules ---
# Compute Canada ships a "dummy" pyarrow wheel that intentionally errors out
# so users load the real thing from the `arrow` module. Load it here so the
# datasets package finds pyarrow via $PYTHONPATH during the install below.
module load StdEnv/2023 gcc python/3.11 cuda/12.2 scipy-stack/2024a arrow

# ---------------------------------------------------------------- repo -----
cd "$SCRATCH"
if [ ! -d "temp_xc" ]; then
    git clone https://github.com/chainik1125/temp_xc.git
    cd temp_xc
    git checkout aniket
else
    cd temp_xc
    git fetch origin
    git checkout aniket
    git pull origin aniket
fi
REPO_DIR="$SCRATCH/temp_xc"

# ---------------------------------------------------------------- venv -----
ENV_DIR="$HOME/envs/txc"
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating venv at $ENV_DIR ..."
    python3 -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------- torch ----
# Install torch FIRST with matching CUDA 12.x wheel, then --no-deps everything
# else so pip can't pull in a conflicting torch.
pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.3,<2.5"

# ---------------------------------------------------------------- deps -----
pip install --no-deps -r "$REPO_DIR/scripts/trillium_sprint_requirements.txt"

# Resolve any secondary deps the --no-deps pass missed. Run without --no-deps
# but with torch already pinned; `--upgrade-strategy only-if-needed` prevents
# surprise upgrades. Do NOT list pyarrow here — on Compute Canada it's served
# by the `arrow` module (loaded above), not pip.
pip install --upgrade-strategy only-if-needed \
    filelock packaging regex requests \
    fsspec aiohttp multidict yarl \
    jinja2 sympy networkx \
    click psutil gitpython sentry-sdk setproctitle \
    pillow kiwisolver cycler fonttools pyparsing \
    pynndescent llvmlite tbb joblib threadpoolctl \
    httpx httpcore h11 anyio sniffio \
    dill "multiprocess<0.70.20" xxhash \
    "cython<3"

# typing-extensions from CC's ipykernel module is 4.12.2 but anthropic wants
# >=4.14. Force-install a new one into the venv so it shadows the module path.
pip install --upgrade "typing_extensions>=4.14"

# pydantic for wandb; anthropic SDK extras.
pip install "pydantic>=2.0,<3" distro docstring-parser jiter

# Editable install of the project itself.
pip install --no-deps -e "$REPO_DIR"

# ---------------------------------------------------------------- caches ---
# Redirect HF + wandb caches off $HOME (2 GB quota) onto $SCRATCH.
CACHE_DIR="$SCRATCH/.cache"
mkdir -p "$CACHE_DIR/huggingface" "$CACHE_DIR/wandb" "$CACHE_DIR/torch"

ACTIVATE_ADDON="$ENV_DIR/bin/activate.d/txc_env.sh"
mkdir -p "$(dirname "$ACTIVATE_ADDON")"
cat > "$ACTIVATE_ADDON" <<EOF
# Auto-sourced by trillium_activate.sh
export HF_HOME="$CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface/transformers"
export HF_DATASETS_CACHE="$CACHE_DIR/huggingface/datasets"
export WANDB_DIR="$CACHE_DIR/wandb"
export WANDB_CACHE_DIR="$CACHE_DIR/wandb"
export TORCH_HOME="$CACHE_DIR/torch"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
EOF

# ---------------------------------------------------------------- secrets --
# Tokens live in $HOME/.txc_secrets.env (gitignored, sourced by sweep scripts).
SECRETS="$HOME/.txc_secrets.env"
if [ ! -f "$SECRETS" ]; then
    cat > "$SECRETS" <<'EOF'
# Fill in and `chmod 600 ~/.txc_secrets.env`
export HF_TOKEN=""
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY=""
export ANTHROPIC_API_KEY=""
EOF
    chmod 600 "$SECRETS"
    echo "Created $SECRETS — fill in your tokens and rerun this script to verify."
fi

# ---------------------------------------------------------------- smoke ----
echo ""
echo "=== Smoke test ==="
python - <<'PY'
import importlib, sys
mods = ["torch", "numpy", "scipy", "einops", "transformers",
        "datasets", "huggingface_hub",
        "wandb", "matplotlib", "anthropic", "pyarrow",
        "sklearn", "umap", "hdbscan", "sentencepiece"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok  {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
        sys.exit(1)
import torch
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
PY

echo ""
echo "=== Setup complete ==="
echo "Repo:   $REPO_DIR"
echo "Env:    $ENV_DIR"
echo "Cache:  $CACHE_DIR"
echo ""
echo "To activate in future sessions:"
echo "  source $REPO_DIR/scripts/trillium_activate.sh"
echo ""
echo "Next steps:"
echo "  1. Fill tokens in ~/.txc_secrets.env (HF_TOKEN, WANDB_API_KEY, ANTHROPIC_API_KEY)"
echo "  2. Accept model licenses on HF: DeepSeek-R1-Distill-Llama-8B, Llama-3.1-8B, Gemma 2"
echo "  3. bash $REPO_DIR/scripts/download_models.sh          # ~40 GB total"
echo "  4. salloc + python scripts/verify_gpu_fit.py --model deepseek-r1-distill-llama-8b"
