#!/usr/bin/env bash
# Wait for in-flight three-arch + HMM sweeps to finish, then run the
# regular_sae_kT add-on for both grids in parallel, then merge the JSONs.
#
# Designed to run unattended in its own tmux session ("queue") while the
# user sleeps. Polls the result JSONs every 60 seconds; doesn't rely on
# tmux/process introspection so it's robust to disconnects.
#
# Usage:
#     cd /temp_xc
#     git pull                       # pick up this script
#     tmux new -s queue
#     bash scripts/queue_kt_sweeps.sh 2>&1 | tee queue.log
#     # Ctrl-b d to detach, sleep
#
# When you reconnect: `tmux attach -t queue` to see the log, then push
# results and stop the pod.

set -uo pipefail   # not -e — we want to continue even if one step fails

cd /temp_xc

EXPECTED_THREE_ARCH=72
EXPECTED_HMM=72

THREE_RESULTS="results/three_arch_sweep/sweep_results.json"
HMM_RESULTS="results/hmm_denoising/sweep_results.json"

count() {
    local path=$1
    if [[ -f "$path" ]]; then
        jq 'length' "$path" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

wait_for_completion() {
    local path=$1
    local expected=$2
    local label=$3
    echo "[queue] waiting for $label to reach $expected rows..."
    local n=$(count "$path")
    while [ "$n" -lt "$expected" ]; do
        sleep 60
        n=$(count "$path")
    done
    echo "[queue] $label complete: $n rows at $(date)"
}

echo "[queue] starting at $(date)"
echo "[queue] current: three-arch=$(count $THREE_RESULTS)/$EXPECTED_THREE_ARCH  hmm=$(count $HMM_RESULTS)/$EXPECTED_HMM"

wait_for_completion "$THREE_RESULTS" "$EXPECTED_THREE_ARCH" "three-arch sweep"
wait_for_completion "$HMM_RESULTS"   "$EXPECTED_HMM"        "HMM sweep"

echo "[queue] both base sweeps done. pulling latest code..."
git pull --ff-only || echo "[queue] git pull failed (non-fatal)"

echo "[queue] launching regular_sae_kT for both grids in parallel at $(date)"
uv run python scripts/run_three_arch_sweep.py \
    --models regular_sae_kT \
    --output-dir results/regular_sae_kT \
    > kt_three_arch.log 2>&1 &
PID_THREE=$!
echo "[queue] three-arch kt PID=$PID_THREE"

uv run python scripts/run_hmm_denoising_sweep.py \
    --models regular_sae_kT \
    --output-dir results/regular_sae_kT_hmm \
    > kt_hmm.log 2>&1 &
PID_HMM=$!
echo "[queue] hmm kt PID=$PID_HMM"

wait $PID_THREE
echo "[queue] three-arch kt finished at $(date) (exit=$?)"
wait $PID_HMM
echo "[queue] hmm kt finished at $(date) (exit=$?)"

echo "[queue] merging JSONs..."
uv run python scripts/merge_results.py \
    "$THREE_RESULTS" \
    results/regular_sae_kT/sweep_results.json \
    --output results/three_arch_sweep_4arch/sweep_results.json

uv run python scripts/merge_results.py \
    "$HMM_RESULTS" \
    results/regular_sae_kT_hmm/sweep_results.json \
    --output results/hmm_denoising_4arch/sweep_results.json

# Drop a sentinel file so a quick `ls` tells you it's done.
touch QUEUE_DONE
echo "[queue] ALL DONE at $(date)"
echo "[queue] merged JSONs:"
echo "  results/three_arch_sweep_4arch/sweep_results.json"
echo "  results/hmm_denoising_4arch/sweep_results.json"
echo "[queue] sentinel file: /temp_xc/QUEUE_DONE"
