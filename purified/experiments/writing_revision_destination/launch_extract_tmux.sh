#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly SESSION="${KLICKE_EXTRACT_SESSION:-klicke-deletion-l10-extract}"
readonly GPU="${KLICKE_EXTRACT_GPU:-0}"
readonly PYTHON_BIN="${KLICKE_PYTHON_BIN:-python}"
readonly COHORT="${KLICKE_TOKEN_COHORT:-$ROOT/purified/results/neurips_rebuttal/writing_revision_destination/token_cohort.parquet}"
readonly MANIFEST="${KLICKE_TOKEN_MANIFEST:-$ROOT/purified/results/neurips_rebuttal/writing_revision_destination/token_audit.json}"
readonly OUTPUT="${KLICKE_ACTIVATION_CACHE:-$ROOT/purified/results/neurips_rebuttal/writing_revision_destination/activation_cache}"
readonly LOG="${KLICKE_EXTRACT_LOG:-$ROOT/purified/logs/writing_revision_destination/extract.log}"
readonly REVISION="${KLICKE_MODEL_REVISION:-}"
readonly BATCH_SIZE="${KLICKE_EXTRACT_BATCH_SIZE:-1}"
readonly SHARD_SIZE="${KLICKE_EXTRACT_SHARD_SIZE:-256}"
readonly ATTENTION="${KLICKE_EXTRACT_ATTENTION:-sdpa}"

if [[ "$(git -C "$ROOT" branch --show-current)" != "neurips-aniket" ]]; then
  echo "refusing extraction outside branch neurips-aniket" >&2
  exit 1
fi
if [[ ! -f "$COHORT" || ! -f "$MANIFEST" ]]; then
  echo "missing token cohort or manifest" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$(dirname "$LOG")" "$OUTPUT"

command=(
  env "CUDA_VISIBLE_DEVICES=$GPU"
  "$PYTHON_BIN"
  -m purified.experiments.writing_revision_destination.extract_activations
  --cohort "$COHORT"
  --cohort-manifest "$MANIFEST"
  --output-dir "$OUTPUT"
  --device cuda:0
  --batch-size "$BATCH_SIZE"
  --shard-size "$SHARD_SIZE"
  --attention "$ATTENTION"
)
if [[ -n "$REVISION" ]]; then
  command+=(--revision "$REVISION")
fi
printf -v quoted_command "%q " "${command[@]}"
printf -v quoted_root "%q" "$ROOT"
printf -v quoted_log "%q" "$LOG"
tmux new-session -d -s "$SESSION" bash -lc \
  "set -o pipefail; cd $quoted_root && $quoted_command 2>&1 | tee $quoted_log"

printf "launched %s on physical GPU %s; log %s\n" \
  "$SESSION" "$GPU" "$LOG"
