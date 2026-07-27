#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly PYTHON="${KLICKE_PYTHON_BIN:-$ROOT/purified/.venv/bin/python}"
readonly GPU="${KLICKE_FROZEN_GPU:-3}"
readonly COHORT="${KLICKE_FROZEN_TOKEN_COHORT:-$ROOT/purified/results/neurips_rebuttal/writing_revision_destination/token_cohort.parquet}"
readonly MANIFEST="${KLICKE_FROZEN_TOKEN_MANIFEST:-$ROOT/purified/results/neurips_rebuttal/writing_revision_destination/token_audit.json}"
readonly ACTIVATION_CACHE="${KLICKE_FROZEN_ACTIVATION_CACHE:-$ROOT/purified/results/neurips_rebuttal/writing_revision_destination/activation_cache_singleton_v1}"
readonly CHECKPOINT_ROOT="${KLICKE_FROZEN_CHECKPOINT_ROOT:-$ROOT/purified/checkpoints}"
readonly OUTPUT_ROOT="${KLICKE_FROZEN_OUTPUT_ROOT:-$ROOT/purified/results/neurips_rebuttal/writing_revision_destination/frozen_dictionary_t5_v1}"
readonly CODE_DIR="${KLICKE_FROZEN_CODE_DIR:-$OUTPUT_ROOT/codes}"
readonly LOG_ROOT="${KLICKE_FROZEN_LOG_ROOT:-$ROOT/purified/logs/writing_revision_destination}"
readonly LOG="${KLICKE_FROZEN_LOG:-$LOG_ROOT/frozen_dictionary_t5.log}"
readonly EXIT_FILE="${KLICKE_FROZEN_EXIT_FILE:-$LOG_ROOT/frozen_dictionary_t5.exit}"
readonly DOWNLOAD="${KLICKE_FROZEN_DOWNLOAD_CHECKPOINTS:-1}"
readonly MIN_FREE_KB="${KLICKE_FROZEN_MIN_FREE_KB:-6291456}"

if [[ "$(git -C "$ROOT" branch --show-current)" != "neurips-aniket" ]]; then
  echo "refusing to run: current branch must be neurips-aniket" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "missing Python interpreter: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$COHORT" || ! -f "$MANIFEST" ]]; then
  echo "missing exact deletion token cohort or manifest" >&2
  exit 1
fi
for required in request.json runtime.json complete.json; do
  if [[ ! -f "$ACTIVATION_CACHE/$required" ]]; then
    echo "missing deletion activation-cache file: $ACTIVATION_CACHE/$required" >&2
    exit 1
  fi
done
available_kb="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
if [[ -z "$available_kb" || "$available_kb" -lt "$MIN_FREE_KB" ]]; then
  echo "refusing to run with less than $MIN_FREE_KB KiB free at $ROOT" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$CODE_DIR" "$CHECKPOINT_ROOT"
exec > >(tee -a "$LOG") 2>&1
write_exit() {
  local status=$?
  printf "%s\n" "$status" >"$EXIT_FILE"
}
trap write_exit EXIT

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/purified/src:$ROOT/purified"
args=(
  --cohort "$COHORT"
  --cohort-manifest "$MANIFEST"
  --activation-cache "$ACTIVATION_CACHE"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --code-dir "$CODE_DIR"
  --output-dir "$OUTPUT_ROOT"
  --device cuda:0
  --batch-size "${KLICKE_FROZEN_BATCH_SIZE:-32}"
  --folds "${KLICKE_FROZEN_FOLDS:-5}"
  --s-grid "${KLICKE_FROZEN_S_GRID:-8,16,32,64,128}"
  --primary-budget "${KLICKE_FROZEN_PRIMARY_BUDGET:-32}"
  --c-value "${KLICKE_FROZEN_C_VALUE:-1.0}"
  --bootstrap-draws "${KLICKE_FROZEN_BOOTSTRAP_DRAWS:-2000}"
  --seed "${KLICKE_FROZEN_SEED:-20260726}"
  --gate-margin "${KLICKE_FROZEN_GATE_MARGIN:-0.02}"
)
if [[ "$DOWNLOAD" == "1" ]]; then
  args+=(--download-checkpoints)
elif [[ "$DOWNLOAD" != "0" ]]; then
  echo "KLICKE_FROZEN_DOWNLOAD_CHECKPOINTS must be 0 or 1" >&2
  exit 2
fi

cd "$ROOT/purified"
"$PYTHON" -m experiments.writing_revision_destination.frozen_dictionary \
  "${args[@]}"
