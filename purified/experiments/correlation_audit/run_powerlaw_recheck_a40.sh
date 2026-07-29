#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
if [[ "$(git -C "$ROOT" branch --show-current)" != "neurips-aniket" ]]; then
  echo "refusing to continue: current branch must be neurips-aniket" >&2
  exit 2
fi

FUTURELENS_ROOT="${FUTURELENS_ROOT:-/workspace/tensor-network-futurelens-rerun}"
PYTHON="${CORRELATION_PYTHON:-/workspace/venvs/tn-futurelens-rerun/bin/python}"
RUN_ROOT="${CORRELATION_RUN_ROOT:-/workspace/correlation-rerun-20260729}"
CACHE="$FUTURELENS_ROOT/data/cache/gpt2/wikitext103"
LOG_ROOT="$RUN_ROOT/logs"
ROBUST_ROOT="$RUN_ROOT/robustness"
CORRECTIVE_ROOT="$RUN_ROOT/corrective"
EXPECTED_SOURCE_REV="77a8e70ada0511ca696b83048e90547dd37db428"

mkdir -p "$LOG_ROOT" "$ROBUST_ROOT" "$CORRECTIVE_ROOT"
if [[ ! -x "$PYTHON" ]]; then
  echo "missing Python environment: $PYTHON" >&2
  exit 2
fi
if [[ "$(git -C "$FUTURELENS_ROOT" rev-parse HEAD)" != "$EXPECTED_SOURCE_REV" ]]; then
  echo "FutureLens source revision mismatch" >&2
  exit 2
fi

cd "$FUTURELENS_ROOT"
if [[ -s "$CACHE/tokens.pt" ]]; then
  printf 'reusing existing non-empty token cache: %s\n' "$CACHE/tokens.pt" \
    >"$LOG_ROOT/build.log"
else
  "$PYTHON" scripts/cache_residuals.py \
    --build-only \
    --num-sequences 8000 \
    --layers 6 8 \
    >"$LOG_ROOT/build.log" 2>&1
fi

CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/cache_residuals.py \
  --device cuda:0 \
  --start 0 \
  --end 4000 \
  --num-sequences 8000 \
  --layers 6 8 \
  >"$LOG_ROOT/cache-gpu1.log" 2>&1 &
cache_one=$!

CUDA_VISIBLE_DEVICES=2 "$PYTHON" scripts/cache_residuals.py \
  --device cuda:0 \
  --start 4000 \
  --end 8000 \
  --num-sequences 8000 \
  --layers 6 8 \
  >"$LOG_ROOT/cache-gpu2.log" 2>&1 &
cache_two=$!

wait "$cache_one"
wait "$cache_two"

shard_count="$(find "$CACHE" -maxdepth 1 -type f -name 'shard_*.pt' | wc -l | tr -d ' ')"
if [[ "$shard_count" != "8" ]]; then
  echo "expected 8 activation shards, found $shard_count" >&2
  exit 3
fi

"$PYTHON" scripts/exp16_powerlaw.py \
  --layers 6 8 \
  --blocks 1 2 4 8 \
  --p 64 \
  --max-delta 48 \
  --n-persist 8 \
  >"$LOG_ROOT/futurelens-exp16.log" 2>&1

cp \
  "$FUTURELENS_ROOT/results/runs/gpt2_exp16_powerlaw/powerlaw_vs_exp.json" \
  "$RUN_ROOT/futurelens-powerlaw-vs-exp.json"

cd "$ROOT/purified"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. MPLBACKEND=Agg "$PYTHON" \
  -m experiments.correlation_audit.robustness \
  --cache-dir "$CACHE" \
  --layer 6 \
  --output-dir "$ROBUST_ROOT" \
  --projection-dim 64 \
  --fit-tokens 60000 \
  --fit-documents 1000 \
  --max-lag 48 \
  --persistent-rank 8 \
  --psd-bootstrap 200 \
  --device cuda:0 \
  >"$LOG_ROOT/robust-layer6.log" 2>&1 &
robust_one=$!

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. MPLBACKEND=Agg "$PYTHON" \
  -m experiments.correlation_audit.robustness \
  --cache-dir "$CACHE" \
  --layer 8 \
  --output-dir "$ROBUST_ROOT" \
  --projection-dim 64 \
  --fit-tokens 60000 \
  --fit-documents 1000 \
  --max-lag 48 \
  --persistent-rank 8 \
  --psd-bootstrap 200 \
  --device cuda:0 \
  >"$LOG_ROOT/robust-layer8.log" 2>&1 &
robust_two=$!

wait "$robust_one"
wait "$robust_two"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. MPLBACKEND=Agg "$PYTHON" \
  -m experiments.correlation_audit.corrective \
  --cache-dir "$CACHE" \
  --output-dir "$CORRECTIVE_ROOT" \
  --layers 6 \
  --projection-dim 64 \
  --fit-tokens 60000 \
  --fit-blocks 1000 \
  --max-lag 48 \
  --bootstrap 200 \
  --device cuda:0 \
  >"$LOG_ROOT/corrective-layer6.log" 2>&1 &
corrective_one=$!

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. MPLBACKEND=Agg "$PYTHON" \
  -m experiments.correlation_audit.corrective \
  --cache-dir "$CACHE" \
  --output-dir "$CORRECTIVE_ROOT" \
  --layers 8 \
  --projection-dim 64 \
  --fit-tokens 60000 \
  --fit-blocks 1000 \
  --max-lag 48 \
  --bootstrap 200 \
  --device cuda:0 \
  >"$LOG_ROOT/corrective-layer8.log" 2>&1 &
corrective_two=$!

wait "$corrective_one"
wait "$corrective_two"

PYTHONPATH=. "$PYTHON" \
  -m experiments.correlation_audit.summarize_corrective \
  --output-dir "$CORRECTIVE_ROOT" \
  --layers 6 8 \
  >"$LOG_ROOT/corrective-summary.log" 2>&1

printf 'complete\n' >"$RUN_ROOT/COMPLETE"
