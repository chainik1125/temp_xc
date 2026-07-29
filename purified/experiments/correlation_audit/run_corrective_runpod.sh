#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/workspace/txc-neurips-aniket
readonly PURIFIED="$REPO/purified"
readonly ENV="$REPO/.venv-e0-extract"
readonly CACHE_DIR="$REPO/data/e0/full"
readonly RESULT_DIR="$PURIFIED/results/neurips_theory/e0_corrective"
readonly LOG_DIR="$PURIFIED/logs/neurips_theory"
readonly LOG_PATH="$LOG_DIR/e0_corrective.log"
readonly EXIT_PATH="$LOG_DIR/e0_corrective.exit"

mkdir -p "$RESULT_DIR" "$LOG_DIR" /tmp/e0-corrective-mpl
printf 'running\n' > "$EXIT_PATH"
exec > >(tee -a "$LOG_PATH") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$EXIT_PATH"; printf "E0 corrective exit=%s at %s\n" "$status" "$(date -u +%FT%TZ)"; exit "$status"' EXIT

printf 'E0 corrective start: %s\n' "$(date -u +%FT%TZ)"
printf '%s\n' 'estimand: cached article-prefix reconstruction with cross-block lag pairs; block context/position resets and discarded remainders remain a stated limitation'
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

cd "$PURIFIED"
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/e0-corrective-mpl \
  PYTHONPATH=. "$ENV/bin/python" -u -m experiments.correlation_audit.corrective \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$RESULT_DIR" \
  --layers 6 8 \
  --projection-dim 64 \
  --fit-tokens 60000 \
  --fit-blocks 1000 \
  --max-lag 48 \
  --bootstrap 200 \
  --bootstrap-batch 20 \
  --batch-articles 16 \
  --spectrum-frequencies 65 \
  --device cuda:0 \
  --seed 0
