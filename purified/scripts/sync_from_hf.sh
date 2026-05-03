#!/usr/bin/env bash
# sync_from_hf.sh — pull latest checkpoints + caches from HF.
#
# Required on EPHEMERAL pods (4× A40) at session start: /workspace is
# wiped on pod stop, so any cached state must come from the HF model
# and dataset repos. Idempotent — files already present locally are
# not redownloaded.
#
# On PERSISTENT pods (2× H100, H200, local) this script is a fast
# no-op (everything already on disk). Safe to run anyway.
#
# Usage:
#   bash scripts/sync_from_hf.sh                  # full sync
#   bash scripts/sync_from_hf.sh --models-only    # skip data
#   bash scripts/sync_from_hf.sh --data-only      # skip models

set -eu

cd "$(dirname "$0")/.."   # purified/

MODELS_REPO="han1823123123/temp-bench-models"
DATA_REPO="han1823123123/temp-bench-data"

CKPT_DIR="$(pwd)/checkpoints"
ACT_CACHE_DIR="$(pwd)/results/act_cache"

mode="all"
case "${1:-}" in
    --models-only) mode="models" ;;
    --data-only)   mode="data"   ;;
    "")            mode="all"    ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
esac

echo "[sync_from_hf] mode: $mode"
echo "[sync_from_hf] TEMP_BENCH_POD_MODE=${TEMP_BENCH_POD_MODE:-unset}"

# Resolve HF token via the unified chain (tokens.py): env → .tokens/ → ~/.cache.
HF_TOKEN_RESOLVED="$(.venv/bin/python -c '
from temp_bench.utils.tokens import get_token
t = get_token("hf")
print(t or "")
' 2>/dev/null)"
if [ -z "$HF_TOKEN_RESOLVED" ]; then
    echo "[sync_from_hf] no HF token found — run scripts/bootstrap_{local,runpod}.sh first" >&2
    exit 1
fi
export HF_TOKEN="$HF_TOKEN_RESOLVED"

mkdir -p "$CKPT_DIR" "$ACT_CACHE_DIR"

if [ "$mode" = "all" ] || [ "$mode" = "models" ]; then
    echo "[sync_from_hf] pulling $MODELS_REPO …"
    .venv/bin/huggingface-cli download "$MODELS_REPO" \
        --repo-type model \
        --local-dir "$CKPT_DIR" \
        --local-dir-use-symlinks False \
        2>&1 | tail -5 || echo "  (no models yet — that's fine)"
fi

if [ "$mode" = "all" ] || [ "$mode" = "data" ]; then
    echo "[sync_from_hf] pulling $DATA_REPO act_cache …"
    .venv/bin/huggingface-cli download "$DATA_REPO" \
        --repo-type dataset \
        --include "act_cache/**" \
        --local-dir "$ACT_CACHE_DIR/.." \
        --local-dir-use-symlinks False \
        2>&1 | tail -5 || echo "  (no act_cache yet — that's fine)"
fi

echo "[sync_from_hf] done"
