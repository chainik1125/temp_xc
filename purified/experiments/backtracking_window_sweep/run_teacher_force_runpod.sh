#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly PYTHON="${TXC_RUNPOD_PYTHON:-$ROOT/.venv-e0-extract/bin/python}"
readonly PHASE="${BACKTRACKING_TEACHER_PHASE:-extract}"
readonly DEVICE_ID="${BACKTRACKING_TEACHER_CUDA_DEVICE:-0}"
readonly PUBLIC_ROOT="${BACKTRACKING_TEACHER_PUBLIC_ROOT:-$ROOT/purified/artifacts/ward-stage-b-cache}"
readonly C7_ROOT="${BACKTRACKING_TEACHER_C7_ROOT:-$ROOT/purified/artifacts/c7}"
readonly PROMPTS="${BACKTRACKING_TEACHER_PROMPTS:-$PUBLIC_ROOT/stageA_prompts.json}"
readonly LABELS="${BACKTRACKING_TEACHER_LABELS:-$PUBLIC_ROOT/stageA_sentence_labels.json}"
readonly OFFICIAL="${BACKTRACKING_TEACHER_OFFICIAL:-$C7_ROOT/sentence_acts_L10.npz}"
readonly TRACES="${BACKTRACKING_TEACHER_TRACES:?set BACKTRACKING_TEACHER_TRACES to the explicitly supplied traces.json}"
readonly TRACES_SHA256="${BACKTRACKING_TEACHER_TRACES_SHA256:?set the pinned traces.json SHA-256}"
readonly SOURCE_PATH="${BACKTRACKING_TEACHER_SOURCE_PATH:?set the repository-relative traces.json source path}"
readonly SOURCE_COMMIT="${BACKTRACKING_TEACHER_SOURCE_COMMIT:?set the 40-hex source commit}"
readonly OUTPUT_DIR="${BACKTRACKING_TEACHER_OUTPUT_DIR:-$C7_ROOT/teacher_force_t16}"
readonly ARTIFACT="${BACKTRACKING_TEACHER_ARTIFACT:-$C7_ROOT/sentence_acts_L10_T16.npz}"
readonly MANIFEST="${BACKTRACKING_TEACHER_MANIFEST:-$C7_ROOT/sentence_acts_L10_T16.manifest.json}"
readonly LOG_ROOT="${BACKTRACKING_TEACHER_LOG_ROOT:-$ROOT/purified/logs/backtracking_teacher_force}"
readonly SHARD_INDEX="${BACKTRACKING_TEACHER_SHARD_INDEX:-0}"
readonly LOG="${BACKTRACKING_TEACHER_LOG_FILE:-$LOG_ROOT/${PHASE}_shard${SHARD_INDEX}.log}"
readonly EXIT_FILE="${BACKTRACKING_TEACHER_EXIT_FILE:-$LOG_ROOT/${PHASE}_shard${SHARD_INDEX}.exit}"

if [[ "$PHASE" != "preflight" && "$PHASE" != "extract" && "$PHASE" != "assemble" ]]; then
  echo "BACKTRACKING_TEACHER_PHASE must be preflight, extract, or assemble" >&2
  exit 2
fi
if [[ "$(git -C "$ROOT" branch --show-current)" != "neurips-aniket" ]]; then
  echo "refusing to run: current branch must be neurips-aniket" >&2
  exit 2
fi
for input in "$PROMPTS" "$LABELS" "$TRACES" "$OFFICIAL"; do
  if [[ ! -f "$input" ]]; then
    echo "missing required local input: $input" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON" ]]; then
  echo "missing executable Python environment: $PYTHON" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG") 2>&1
write_exit() {
  local status=$?
  printf "%s\n" "$status" >"$EXIT_FILE"
}
trap write_exit EXIT

export PYTHONPATH="$ROOT/purified/src:$ROOT/purified"
common=(
  --prompts "$PROMPTS"
  --labels "$LABELS"
  --traces "$TRACES"
  --official "$OFFICIAL"
  --traces-sha256 "$TRACES_SHA256"
  --source-path "$SOURCE_PATH"
  --source-commit "$SOURCE_COMMIT"
)
module="experiments.backtracking_window_sweep.extract_wide_teacher_force"

if [[ "$PHASE" == "preflight" ]]; then
  "$PYTHON" -m "$module" preflight "${common[@]}"
elif [[ "$PHASE" == "assemble" ]]; then
  "$PYTHON" -m "$module" assemble "${common[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --artifact "$ARTIFACT" \
    --manifest "$MANIFEST"
else
  export CUDA_VISIBLE_DEVICES="$DEVICE_ID"
  args=(
    extract
    "${common[@]}"
    --output-dir "$OUTPUT_DIR"
    --device cuda:0
    --num-shards "${BACKTRACKING_TEACHER_NUM_SHARDS:-1}"
    --shard-index "$SHARD_INDEX"
  )
  if [[ -n "${BACKTRACKING_TEACHER_MAX_TRACES:-}" ]]; then
    args+=(--max-traces "$BACKTRACKING_TEACHER_MAX_TRACES")
  fi
  "$PYTHON" -m "$module" "${args[@]}"
fi
