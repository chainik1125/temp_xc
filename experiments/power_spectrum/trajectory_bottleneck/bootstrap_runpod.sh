#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${TRAJECTORY_RUN_ROOT:-/workspace/trajectory_bottleneck_c7}"
REPO_ROOT="${TRAJECTORY_REPO_ROOT:-/workspace/txc-neurips-aniket}"
SOURCE_ROOT="${REPO_ROOT}/purified"
REFERENCE_BRANCH="${TRAJECTORY_REFERENCE_BRANCH:-neurips-aniket}"
EXPERIMENT_BRANCH="${TRAJECTORY_EXPERIMENT_BRANCH:-codex/spectral-screen-overnight-20260729}"
REVISION="6ef9b1debf863dedcef9555cad3a4903fb9e8c43"
HF_REPO="han1823123123/temp-bench-data"

mkdir -p "${RUN_ROOT}/logs"
exec > >(tee -a "${RUN_ROOT}/logs/bootstrap.log") 2>&1
date +%s > "${RUN_ROOT}/billing_started_epoch"
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_ROOT}/bootstrap_started_utc"

stop_on_failure() {
  status=$?
  if (( status != 0 )); then
    printf '%s\n' "${status}" > "${RUN_ROOT}/bootstrap_exit_code"
    if [[ -x "${RUN_ROOT}/stop_pod.sh" ]]; then
      "${RUN_ROOT}/stop_pod.sh" || true
    fi
  fi
}
trap stop_on_failure EXIT

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  git clone --branch "${REFERENCE_BRANCH}" --single-branch \
    https://github.com/chainik1125/temp_xc.git "${REPO_ROOT}"
fi
cd "${REPO_ROOT}"
git fetch origin \
  "${EXPERIMENT_BRANCH}:refs/remotes/origin/${EXPERIMENT_BRANCH}"
git archive "origin/${EXPERIMENT_BRANCH}" \
  experiments/power_spectrum/trajectory_bottleneck \
  | tar -x -C "${SOURCE_ROOT}"

python -m pip install --quiet \
  "numpy>=2.1" "scipy>=1.14" "scikit-learn>=1.5" \
  "safetensors>=0.4" "matplotlib>=3.9" \
  "huggingface-hub>=0.26" "hf-xet>=1.1"

export PYTHONPATH="${SOURCE_ROOT}/src:${SOURCE_ROOT}"
export HF_XET_HIGH_PERFORMANCE=1
python -m experiments.backtracking_assets \
  --destination "${SOURCE_ROOT}/artifacts/hf_temp_bench_data" \
  --revision "${REVISION}" \
  --training-cache-only
hf download "${HF_REPO}" \
  c7_backtracking/stage_a/sentence_acts_L10.npz \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${SOURCE_ROOT}/artifacts/hf_temp_bench_data"

mkdir -p "${SOURCE_ROOT}/artifacts/c7"
ln -sfn \
  "${SOURCE_ROOT}/artifacts/hf_temp_bench_data/c7_backtracking/stage_a/sentence_acts_L10.npz" \
  "${SOURCE_ROOT}/artifacts/c7/sentence_acts_L10.npz"

LOCAL_CACHE="/dev/shm/trajectory_resid_post_L10.npy"
cp \
  "${SOURCE_ROOT}/artifacts/hf_temp_bench_data/act_cache/fb2a74be884e512a/resid_post_L10.npy" \
  "${LOCAL_CACHE}"
echo "dc34dfb117f77abddef4b4396d0d00afc707c39876d0ee36015de1e7b8406914  ${LOCAL_CACHE}" \
  | sha256sum --check

python -m experiments.power_spectrum.trajectory_bottleneck.smoke
python -m experiments.power_spectrum.trajectory_bottleneck.run \
  --activation-cache "${LOCAL_CACHE}" \
  --artifact "${SOURCE_ROOT}/artifacts/c7/sentence_acts_L10.npz" \
  --checkpoint-root "${RUN_ROOT}/checkpoints" \
  --output-root "${RUN_ROOT}/results" \
  --dry-run

date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_ROOT}/bootstrap_complete_utc"
trap - EXIT
setsid nohup env \
  TRAJECTORY_RUN_ROOT="${RUN_ROOT}" \
  TRAJECTORY_SOURCE_ROOT="${SOURCE_ROOT}" \
  TRAJECTORY_ACTIVATION_CACHE="${LOCAL_CACHE}" \
  TRAJECTORY_MAX_SECONDS=28800 \
  "${SOURCE_ROOT}/experiments/power_spectrum/trajectory_bottleneck/supervisor.sh" \
  > "${RUN_ROOT}/logs/supervisor_launcher.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "${RUN_ROOT}/supervisor_launcher.pid"
