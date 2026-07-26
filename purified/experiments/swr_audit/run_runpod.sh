#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-smoke}
PROJECT_ROOT=${PROJECT_ROOT:-/workspace/txc-neurips-aniket/purified}
PYTHON=${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}
ARTIFACT=${ARTIFACT:-${PROJECT_ROOT}/artifacts/c7/sentence_acts_L10.npz}
RESULT_ROOT=${RESULT_ROOT:-${PROJECT_ROOT}/results/neurips_theory/e1}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}
export PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

mkdir -p "${RESULT_ROOT}" "${RESULT_ROOT}/codes"
cd "${PROJECT_ROOT}"

case "${MODE}" in
  smoke)
    "${PYTHON}" -m experiments.swr_audit.matched_filter \
      --artifact "${ARTIFACT}" \
      --output "${RESULT_ROOT}/matched_filter_smoke.json" \
      --windows 1,6 --normalizations raw --pca-dim 16 --folds 3 \
      --bootstrap-repeats 50 --max-rows 1000
    "${PYTHON}" -m experiments.swr_audit.run \
      --artifact "${ARTIFACT}" \
      --output "${RESULT_ROOT}/swr_smoke.jsonl" \
      --windows 1,6 --normalizations raw --pca-dim 16 --rank 8 --folds 3 \
      --epochs 8 --max-rows 1000 --device cuda
    ;;
  matched)
    "${PYTHON}" -m experiments.swr_audit.matched_filter \
      --artifact "${ARTIFACT}" \
      --output "${RESULT_ROOT}/matched_filter.json" \
      --windows 1,3,5,6 --normalizations raw --pca-dim 32 --folds 5 \
      --bootstrap-repeats 2000
    "${PYTHON}" -m experiments.swr_audit.report \
      --input "${RESULT_ROOT}/matched_filter.json" \
      --figure "${RESULT_ROOT}/matched_filter.png" \
      --markdown "${RESULT_ROOT}/matched_filter_table.md"
    ;;
  swr-primary)
    "${PYTHON}" -m experiments.swr_audit.run \
      --artifact "${ARTIFACT}" \
      --output "${RESULT_ROOT}/swr_primary.jsonl" \
      --windows 6 --normalizations raw --pca-dim 32 --rank 20 --folds 5 \
      --epochs 80 --device cuda
    "${PYTHON}" -m experiments.swr_audit.aggregate \
      --input "${RESULT_ROOT}/swr_primary.jsonl" \
      --output "${RESULT_ROOT}/swr_primary_summary.json"
    ;;
  swr-sensitivity)
    "${PYTHON}" -m experiments.swr_audit.run \
      --artifact "${ARTIFACT}" \
      --output "${RESULT_ROOT}/swr_sensitivity.jsonl" \
      --windows 1,3,5,6 --normalizations raw,token_rms \
      --pca-dim 32 --rank 20 --folds 5 --epochs 80 --device cuda
    "${PYTHON}" -m experiments.swr_audit.aggregate \
      --input "${RESULT_ROOT}/swr_sensitivity.jsonl" \
      --output "${RESULT_ROOT}/swr_sensitivity_summary.json"
    ;;
  dictionary-txc)
    "${PYTHON}" -m experiments.swr_audit.dictionary \
      --artifact "${ARTIFACT}" --checkpoint-root "${CHECKPOINT_ROOT}" \
      --train-key 08fe3af07682fab4 \
      --checkpoint-sha256 ed2ecf4670f889fd97e82c53a949f963c36a292f05944cd678f943a87f1f9cb1 \
      --output "${RESULT_ROOT}/dictionary_txc.jsonl" \
      --code-dir "${RESULT_ROOT}/codes" --device cuda
    ;;
  dictionary-topk)
    "${PYTHON}" -m experiments.swr_audit.dictionary \
      --artifact "${ARTIFACT}" --checkpoint-root "${CHECKPOINT_ROOT}" \
      --train-key f437e623fabc37ec \
      --checkpoint-sha256 2efc15ad39603bcb554730c900e0c4b69035ac5be07d925aa489adbd15533d40 \
      --output "${RESULT_ROOT}/dictionary_topk.jsonl" \
      --code-dir "${RESULT_ROOT}/codes" --device cuda
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
