#!/bin/bash
# trillium_setup.sh — One-time environment setup on Trillium.
# Run this interactively on the login node.
set -euo pipefail

echo "=== Setting up temporal crosscoders env on Trillium ==="

module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a

# Clone repo to scratch
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

# Create venv
ENV_DIR="$HOME/envs/txc"
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating venv at $ENV_DIR ..."
    python3 -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

# Install only what bench needs — skip sae-lens/transformer-lens
pip install --upgrade pip
pip install torch numpy scipy matplotlib tqdm pytest

# Install project without pulling all deps (sae-lens breaks on Trillium)
pip install --no-deps -e .

echo ""
echo "=== Setup complete ==="
echo "Repo: $SCRATCH/temp_xc"
echo "Env:  $ENV_DIR"
echo "Next: bash scripts/trillium_sweep.sh"
