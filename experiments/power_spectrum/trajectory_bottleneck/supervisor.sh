#!/usr/bin/env bash
set -uo pipefail

# Connection-independent restart loop for the RunPod worker. The surrounding
# launcher runs this file under setsid/nohup on the persistent /workspace
# volume. GNU timeout enforces the overnight compute budget.

RUN_ROOT="${TRAJECTORY_RUN_ROOT:-/workspace/trajectory_bottleneck_c7}"
SOURCE_ROOT="${TRAJECTORY_SOURCE_ROOT:-/workspace/txc-neurips-aniket/purified}"
ACTIVATION_CACHE="${TRAJECTORY_ACTIVATION_CACHE:-${SOURCE_ROOT}/artifacts/hf_temp_bench_data/act_cache/fb2a74be884e512a/resid_post_L10.npy}"
ARTIFACT="${TRAJECTORY_ARTIFACT:-${SOURCE_ROOT}/artifacts/c7/sentence_acts_L10.npz}"
MAX_SECONDS="${TRAJECTORY_MAX_SECONDS:-28800}"
MAX_RESTARTS="${TRAJECTORY_MAX_RESTARTS:-12}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results"
exec 9>"${RUN_ROOT}/supervisor.lock"
if ! flock -n 9; then
  echo "A trajectory-bottleneck supervisor already owns ${RUN_ROOT}."
  exit 0
fi

if [[ -s "${RUN_ROOT}/billing_started_epoch" ]]; then
  started_epoch="$(<"${RUN_ROOT}/billing_started_epoch")"
else
  started_epoch="$(date +%s)"
fi
deadline_epoch="$((started_epoch + MAX_SECONDS))"
printf '%s\n' "${started_epoch}" > "${RUN_ROOT}/started_epoch"
printf '%s\n' "${deadline_epoch}" > "${RUN_ROOT}/deadline_epoch"
printf '%s\n' "$$" > "${RUN_ROOT}/supervisor.pid"

attempt=0
exit_code=1
while (( attempt < MAX_RESTARTS )); do
  now="$(date +%s)"
  remaining="$((deadline_epoch - now))"
  if (( remaining <= 60 )); then
    exit_code=124
    break
  fi
  attempt="$((attempt + 1))"
  printf '%s\n' "${attempt}" > "${RUN_ROOT}/attempt"
  log="${RUN_ROOT}/logs/attempt_$(printf '%02d' "${attempt}")_$(date -u +%Y%m%dT%H%M%SZ).log"
  cd "${SOURCE_ROOT}"
  set +e
  timeout --signal=TERM --kill-after=60 "${remaining}" \
    python -u -m experiments.power_spectrum.trajectory_bottleneck.run \
      --activation-cache "${ACTIVATION_CACHE}" \
      --artifact "${ARTIFACT}" \
      --checkpoint-root "${RUN_ROOT}/checkpoints" \
      --output-root "${RUN_ROOT}/results" \
      --device cuda:0 \
      --base-steps 300000 \
      --adapter-steps 300000 \
      --batch-size 1024 \
      --adapter-microbatch 512 \
      --checkpoint-every 5000 \
      --ranks 0,256 \
      2>&1 | tee -a "${log}"
  pipeline_status=("${PIPESTATUS[@]}")
  exit_code="${pipeline_status[0]}"
  set -e
  printf '%s\n' "${exit_code}" > "${RUN_ROOT}/last_exit_code"
  if (( exit_code == 0 )); then
    touch "${RUN_ROOT}/COMPLETE"
    break
  fi
  if (( exit_code == 124 || exit_code == 137 || exit_code == 143 )); then
    break
  fi
  sleep 15
done

date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_ROOT}/supervisor_finished_utc"
printf '%s\n' "${exit_code}" > "${RUN_ROOT}/supervisor_exit_code"

# The stop helper and its credential live only on the remote machine. It is
# invoked after results/checkpoints are flushed, preventing idle GPU billing.
if [[ -x "${RUN_ROOT}/stop_pod.sh" ]]; then
  "${RUN_ROOT}/stop_pod.sh" || true
fi
exit "${exit_code}"
