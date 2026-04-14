#!/bin/bash
# trillium_download_models.sh — Pull latest aniket branch, run the shared
# download_models.sh script. No GPU, safe on the login node.
#
#   bash scripts/trillium_download_models.sh
set -euo pipefail

source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"
git pull origin aniket
bash scripts/download_models.sh
echo ""
echo "Models cached under $HF_HOME"
du -sh "$HF_HOME" 2>/dev/null || true
