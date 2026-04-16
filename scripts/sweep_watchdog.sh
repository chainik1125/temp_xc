#!/bin/bash
# Watchdog for run_nlp_sweep_16h.sh: auto-resumes if the sweep crashes
# (OOM, transient failure). Each individual run saves JSON + checkpoint
# incrementally to /workspace/temp_xc/results/nlp_sweep/, and the sweep
# module skips completed run_keys on restart — so re-running is safe.
#
# Usage:
#   cd /workspace/temp_xc
#   TQDM_DISABLE=1 nohup bash scripts/sweep_watchdog.sh > logs/nlp_sweep.log 2>&1 &

set -u

MAX_RETRIES=10
RETRY_DELAY=30
REPO=/workspace/temp_xc
SCRIPT="$REPO/scripts/run_nlp_sweep_16h.sh"

cd "$REPO"

i=0
while :; do
    echo ""
    echo "[watchdog] $(date) — attempt $((i+1))/$MAX_RETRIES"
    echo "[watchdog] completed runs so far: $(find results/nlp_sweep -name 'results_*.json' -exec sh -c 'python3 -c "import json,sys; print(len(json.load(open(sys.argv[1])))) if __import__(\"os\").path.getsize(sys.argv[1]) else 0" "$1" 2>/dev/null' _ {} \; 2>/dev/null | awk '{s+=$1} END {print s+0}')"
    echo ""

    bash "$SCRIPT"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo ""
        echo "[watchdog] $(date) — sweep finished successfully"
        exit 0
    fi
    i=$((i+1))
    echo ""
    echo "[watchdog] $(date) — sweep exited with code $rc"

    if [ $i -ge $MAX_RETRIES ]; then
        echo "[watchdog] max retries ($MAX_RETRIES) reached; giving up"
        exit 1
    fi

    echo "[watchdog] resuming in ${RETRY_DELAY}s (attempt $((i+1))/$MAX_RETRIES)"
    sleep $RETRY_DELAY
done
