#!/usr/bin/env bash
set -uo pipefail

run_root=/workspace/steering_baselines_20k
canonical_root="$run_root/temp-bench"
workspace="$run_root/results/fresh_25mag_seed42"
python_bin="$run_root/venv/bin/python"
runner="$run_root/run_baselines.py"
pod_id=k804r98a0a6w6b
timeout_seconds=14400

mkdir -p "$workspace"
date +%s > "$workspace/billing_started_epoch"

export PYTHONPATH="$canonical_root/src"
export TEMP_BENCH_ROOT="$canonical_root"
export HF_HOME="$run_root/hf"
export HF_TOKEN="$(cat /workspace/.tokens/hf_token)"
export TOKENIZERS_PARALLELISM=false

heartbeat_pid=""

write_heartbeat() {
  while true; do
    now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"ts":"%s","supervisor_pid":%d}\n' "$now" "$$" \
      > "$workspace/heartbeat.json.tmp"
    mv "$workspace/heartbeat.json.tmp" "$workspace/heartbeat.json"
    sleep 60
  done
}

stop_pod() {
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
  fi
  sync
  if [[ -s /workspace/.tokens/runpod_api_key ]]; then
    runpod_key="$(cat /workspace/.tokens/runpod_api_key)"
    curl -fsS -X POST \
      -H "Authorization: Bearer $runpod_key" \
      "https://rest.runpod.io/v1/pods/$pod_id/stop" \
      > "$workspace/stop_response.json" 2> "$workspace/stop_error.log" || true
  fi
}
trap stop_pod EXIT

write_heartbeat &
heartbeat_pid=$!

common_args=(
  --temp-bench-root "$canonical_root"
  --workspace "$workspace"
  --checkpoints-root "$run_root/checkpoints"
  --sentence-acts /workspace/txc-neurips-aniket/purified/artifacts/hf_temp_bench_data/c7_backtracking/stage_a/sentence_acts_L10.npz
  --gen-batch-size 8
)

set +e
timeout --signal=TERM "$timeout_seconds" bash -c '
  set -e
  "$0" -u "$1" --arm topk_sae "${@:2}"
  "$0" -u "$1" --arm txc_base "${@:2}"
' "$python_bin" "$runner" "${common_args[@]}"
status=$?
set -e

finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"exit_code":%d,"finished":"%s","timeout_seconds":%d}\n' \
  "$status" "$finished" "$timeout_seconds" > "$workspace/supervisor_status.json.tmp"
mv "$workspace/supervisor_status.json.tmp" "$workspace/supervisor_status.json"
exit "$status"

