#!/usr/bin/env bash
set -euo pipefail

repo_dir="${1:-/workspace/txc-decision-repo}"
result_dir="${2:-/workspace/txc_decision_sprint/results}"
log_dir="${3:-/workspace/txc_decision_sprint/logs}"
state_dir="${4:-/workspace/txc_decision_sprint/temp_bench_state}"
mkdir -p "$result_dir" "$log_dir" "$state_dir"
ln -sfn "$repo_dir/configs" "$state_dir/configs"

finish() {
  code=$?
  set +e
  trap - EXIT
  python - "$result_dir" "$code" <<'PY'
import json
import pathlib
import sys
import time

root = pathlib.Path(sys.argv[1])
payload = {
    "exit_code": int(sys.argv[2]),
    "unix_time": time.time(),
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
temporary = root / "supervisor_exit.json.tmp"
temporary.write_text(json.dumps(payload, indent=2))
temporary.replace(root / "supervisor_exit.json")
PY
  if [[ -x /workspace/txc_decision_sprint/stop_pod.sh ]]; then
    /workspace/txc_decision_sprint/stop_pod.sh >>"$log_dir/stop.log" 2>&1 || true
  fi
  exit "$code"
}
trap finish EXIT

cd "$repo_dir"
export TEMP_BENCH_ROOT="$state_dir"
export PYTHONPATH="$repo_dir/src:$repo_dir"
export HF_HOME="/workspace/huggingface"
export TRANSFORMERS_CACHE="/workspace/huggingface"
export TOKENIZERS_PARALLELISM=false

timeout --signal=TERM --kill-after=10m 1h \
  python -m experiments.explorations.task_hunt.dialevel.cache_acts gpt2 \
  2>&1 | tee "$log_dir/cache.log"

timeout --signal=TERM --kill-after=10m 405m \
  python -m experiments.power_spectrum.decision_sprint.run \
  --output "$result_dir" 2>&1 | tee "$log_dir/run.log"

python -m experiments.power_spectrum.decision_sprint.analyze \
  --raw "$result_dir/raw_results.json" \
  --output "$result_dir" 2>&1 | tee "$log_dir/analyze.log"
