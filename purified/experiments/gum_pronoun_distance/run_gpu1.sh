#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly PYTHON="${GUM_PRONOUN_PYTHON_BIN:-$ROOT/purified/.venv/bin/python}"
readonly PHYSICAL_GPU="${GUM_PRONOUN_GPU:-1}"
readonly RESULT_ROOT="${GUM_PRONOUN_RESULT_ROOT:-$ROOT/purified/results/neurips_rebuttal/gum_pronoun_distance}"
readonly GUM_ROOT="${GUM_PRONOUN_SOURCE_ROOT:-$RESULT_ROOT/source/gum}"
readonly TOKENIZER_ROOT="${GUM_PRONOUN_TOKENIZER_ROOT:-$RESULT_ROOT/source/tokenizer}"
readonly COHORT="${GUM_PRONOUN_COHORT:-$RESULT_ROOT/cohort.parquet}"
readonly MANIFEST="${GUM_PRONOUN_MANIFEST:-$RESULT_ROOT/cohort_manifest.json}"
readonly ACTIVATION_CACHE="${GUM_PRONOUN_ACTIVATION_CACHE:-$RESULT_ROOT/activation_cache}"
readonly CHECKPOINT_ROOT="${GUM_PRONOUN_CHECKPOINT_ROOT:-$ROOT/purified/checkpoints}"
readonly CODE_DIR="${GUM_PRONOUN_CODE_DIR:-$RESULT_ROOT/codes}"
readonly OUTPUT_DIR="${GUM_PRONOUN_OUTPUT_DIR:-$RESULT_ROOT/frozen_t5}"
readonly LOG_ROOT="${GUM_PRONOUN_LOG_ROOT:-$ROOT/purified/logs/gum_pronoun_distance}"
readonly LOG="${GUM_PRONOUN_LOG:-$LOG_ROOT/gpu1.log}"
readonly EXIT_FILE="${GUM_PRONOUN_EXIT_FILE:-$LOG_ROOT/gpu1.exit}"
readonly HF_HOME_ROOT="${GUM_PRONOUN_HF_HOME:-/workspace/.cache/huggingface}"
readonly MIN_FREE_KB="${GUM_PRONOUN_MIN_FREE_KB:-27262976}"
readonly GUM_REVISION="22fdf87f9c71c96bcc771461d06e689b1f90020d"
readonly TOKENIZER_REVISION="1f47e50cdbe801ad8a5174156ec3a0655108fb9f"

mkdir -p "$LOG_ROOT"
write_exit() {
  local status=$?
  printf "%s\n" "$status" >"$EXIT_FILE"
}
trap write_exit EXIT

if [[ "$(git -C "$ROOT" branch --show-current)" != "neurips-aniket" ]]; then
  echo "refusing to run: current branch must be neurips-aniket" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "missing Python interpreter: $PYTHON" >&2
  exit 1
fi
available_kb="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
if [[ -z "$available_kb" || "$available_kb" -lt "$MIN_FREE_KB" ]]; then
  echo "refusing to run with less than $MIN_FREE_KB KiB free at $ROOT" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT/source" "$TOKENIZER_ROOT" "$CHECKPOINT_ROOT" "$OUTPUT_DIR"
exec > >(tee -a "$LOG") 2>&1
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export HF_HOME="$HF_HOME_ROOT"
export PYTHONPATH="$ROOT/purified/src:$ROOT/purified"
export MPLCONFIGDIR="${GUM_PRONOUN_MPLCONFIGDIR:-/tmp/gum-pronoun-mpl}"

if [[ ! -d "$GUM_ROOT/.git" ]]; then
  git clone \
    --filter=blob:none \
    --no-checkout \
    --depth 1 \
    --branch V12.1.0 \
    https://github.com/amir-zeldes/gum.git \
    "$GUM_ROOT"
  git -C "$GUM_ROOT" sparse-checkout init --no-cone
  git -C "$GUM_ROOT" sparse-checkout set --no-cone \
    "/coref/gum/tsv/" \
    "/dep/" \
    "/splits.md" \
    "/LICENSE.md" \
    "/README.md"
  git -C "$GUM_ROOT" checkout --detach "$GUM_REVISION"
fi
if [[ "$(git -C "$GUM_ROOT" rev-parse HEAD)" != "$GUM_REVISION" ]]; then
  echo "refusing to use a non-pinned GUM checkout" >&2
  exit 1
fi

"$PYTHON" -c \
  "from huggingface_hub import snapshot_download; snapshot_download(
      'NousResearch/Meta-Llama-3.1-8B',
      revision='$TOKENIZER_REVISION',
      allow_patterns=[
          'config.json',
          'special_tokens_map.json',
          'tokenizer.json',
          'tokenizer_config.json',
      ],
      local_dir='$TOKENIZER_ROOT',
  )"

cd "$ROOT/purified"
"$PYTHON" -m experiments.gum_pronoun_distance.cohort \
  --gum-root "$GUM_ROOT" \
  --tokenizer-path "$TOKENIZER_ROOT" \
  --cohort "$COHORT" \
  --manifest "$MANIFEST"

"$PYTHON" -m experiments.gum_pronoun_distance.extract_activations \
  --cohort "$COHORT" \
  --manifest "$MANIFEST" \
  --output-dir "$ACTIVATION_CACHE" \
  --device cuda:0 \
  --shard-size "${GUM_PRONOUN_SHARD_SIZE:-256}" \
  --attention "${GUM_PRONOUN_ATTENTION:-sdpa}"

"$PYTHON" -m experiments.gum_pronoun_distance.evaluate_frozen \
  --cohort "$COHORT" \
  --manifest "$MANIFEST" \
  --activation-cache "$ACTIVATION_CACHE" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --code-dir "$CODE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --download-checkpoints \
  --device cuda:0 \
  --batch-size "${GUM_PRONOUN_ENCODER_BATCH_SIZE:-32}" \
  --budgets "${GUM_PRONOUN_BUDGETS:-8,16,32,64,128}" \
  --primary-budget "${GUM_PRONOUN_PRIMARY_BUDGET:-32}" \
  --folds "${GUM_PRONOUN_FOLDS:-5}" \
  --bootstrap-draws "${GUM_PRONOUN_BOOTSTRAP_DRAWS:-2000}" \
  --seed "${GUM_PRONOUN_SEED:-20260726}" \
  --gate-margin "${GUM_PRONOUN_GATE_MARGIN:-0.02}"
