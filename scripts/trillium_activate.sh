#!/bin/bash
# Source this at the top of every Trillium session / job script:
#   source $SCRATCH/temp_xc/scripts/trillium_activate.sh

module load StdEnv/2023 gcc python/3.11 cuda/12.2 scipy-stack/2024a arrow
source "$HOME/envs/txc/bin/activate"
source "$HOME/envs/txc/bin/activate.d/txc_env.sh"
if [ -f "$HOME/.txc_secrets.env" ]; then
    source "$HOME/.txc_secrets.env"
fi
export REPO_DIR="$SCRATCH/temp_xc"
# Numba (used by UMAP) tries to write JIT cache next to the source .py file,
# which fails on compute nodes where /home/ is read-only. Point it at $SCRATCH.
export NUMBA_CACHE_DIR="$SCRATCH/.cache/numba"
mkdir -p "$NUMBA_CACHE_DIR" 2>/dev/null || true
cd "$REPO_DIR"
