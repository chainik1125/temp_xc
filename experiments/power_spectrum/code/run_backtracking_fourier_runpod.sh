#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly PYTHON="${TXC_RUNPOD_PYTHON:-$ROOT/.venv-e0-extract/bin/python}"
readonly PHASE="${BACKTRACKING_FOURIER_PHASE:-plan}"
readonly C7="$ROOT/purified/artifacts/c7"
readonly LOG_ROOT="$ROOT/purified/logs/backtracking_fourier_matched"
readonly LOG="$LOG_ROOT/${PHASE}.log"
readonly EXIT_FILE="$LOG_ROOT/${PHASE}.exit"

if [[ "$PHASE" != "plan" && "$PHASE" != "memory-smoke" \
      && "$PHASE" != "train" && "$PHASE" != "eval" && "$PHASE" != "all" ]]; then
  echo "invalid BACKTRACKING_FOURIER_PHASE=$PHASE" >&2
  exit 2
fi
if ! git -C "$ROOT" merge-base --is-ancestor d9c7fc7b2 HEAD; then
  echo "reference checkout does not contain Aniket's d9c7fc7b2 protocol" >&2
  exit 2
fi
if [[ ! -f "$ROOT/purified/experiments/power_spectrum/code/run_backtracking_fourier.py" ]]; then
  echo "Fourier experiment was not staged under the reference package" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
write_exit() {
  local status=$?
  printf "%s\n" "$status" >"$EXIT_FILE"
}
trap write_exit EXIT
exec > >(tee -a "$LOG") 2>&1

export CUDA_VISIBLE_DEVICES="${BACKTRACKING_FOURIER_CUDA_DEVICE:-0}"
export PYTHONPATH="$ROOT/purified/src:$ROOT/purified"

args=(
  --phase "$PHASE"
  --artifact "${BACKTRACKING_FOURIER_ARTIFACT:-$C7/sentence_acts_L10_T16.npz}"
  --artifact-manifest "${BACKTRACKING_FOURIER_MANIFEST:-$C7/sentence_acts_L10_T16.manifest.json}"
  --reference-artifact "${BACKTRACKING_FOURIER_REFERENCE:-$C7/sentence_acts_L10.npz}"
  --activation-cache "${BACKTRACKING_ACTIVATION_CACHE:-$ROOT/purified/artifacts/hf_temp_bench_data/act_cache/fb2a74be884e512a/resid_post_L10.npy}"
  --output-root "${BACKTRACKING_FOURIER_RESULT_ROOT:-$ROOT/purified/results/neurips_rebuttal/backtracking_fourier_matched/reviewer-five-point-v1}"
  --checkpoint-root "${BACKTRACKING_FOURIER_CHECKPOINT_ROOT:-$ROOT/checkpoints/backtracking_fourier_matched/reviewer-five-point-v1}"
  --device cuda:0
  --steps "${BACKTRACKING_FOURIER_STEPS:-20000}"
  --batch-size "${BACKTRACKING_FOURIER_BATCH_SIZE:-1024}"
  --checkpoint-every "${BACKTRACKING_FOURIER_CHECKPOINT_EVERY:-1000}"
  --encode-batch-size "${BACKTRACKING_FOURIER_ENCODE_BATCH_SIZE:-32}"
)
if [[ -n "${BACKTRACKING_FOURIER_WINDOWS:-}" ]]; then
  args+=(--windows "$BACKTRACKING_FOURIER_WINDOWS")
fi
if [[ -n "${BACKTRACKING_FOURIER_SEEDS:-}" ]]; then
  args+=(--seeds "$BACKTRACKING_FOURIER_SEEDS")
fi
if [[ -n "${BACKTRACKING_FOURIER_MAX_CELLS:-}" ]]; then
  args+=(--max-cells "$BACKTRACKING_FOURIER_MAX_CELLS")
fi
if [[ "${BACKTRACKING_FOURIER_KEEP_OPTIMIZER_STATE:-0}" != "1" ]]; then
  args+=(--cleanup-optimizer-state)
fi

"$PYTHON" -m experiments.power_spectrum.code.run_backtracking_fourier "${args[@]}"
