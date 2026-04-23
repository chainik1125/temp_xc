#!/bin/bash
# Runs after run_training.py has been launched. Waits for all three checkpoints
# to exist, then chains the four eval passes. Writes logs/watcher.log so
# progress is visible to any check-in script. Drops a sentinel
# ``results/pipeline_complete.flag`` once everything finishes.
#
# Idempotent: each eval checks for its own ``.done`` flag before running, so
# re-invocations after partial progress resume from where they left off.

set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs results plots checkpoints
export PYTHONPATH=../separation_scaling/vendor/src
VENV=/root/temp_xc_cb/.venv/bin/python

log() {
    echo "$(date -u +%FT%TZ) $*" >> logs/watcher.log
}

log "watcher started (pid=$$)"

# ---- Wait for training completion: all 3 checkpoints present ----
MAX_POLLS=720   # 6h at 30s interval
POLL=0
while [[ $POLL -lt $MAX_POLLS ]]; do
    if [[ -f checkpoints/topk_sae.pt && -f checkpoints/txc_t5.pt && -f checkpoints/mlc_l5.pt ]]; then
        log "all 3 checkpoints present — proceeding to evals"
        break
    fi
    POLL=$((POLL + 1))
    sleep 30
done

if [[ ! -f checkpoints/mlc_l5.pt ]]; then
    log "TIMEOUT waiting for all checkpoints after $MAX_POLLS polls"
    exit 1
fi

# ---- Sequential evaluation passes ----
run_eval() {
    local script="$1"
    local flag="results/${script%.py}.done"
    shift
    local extra_args=("$@")

    if [[ -f "$flag" ]]; then
        log "skipping $script (already done)"
        return 0
    fi

    log "starting $script ${extra_args[*]}"
    if "$VENV" "$script" --config config.yaml --device cuda "${extra_args[@]}" \
        > "logs/${script%.py}.log" 2>&1; then
        touch "$flag"
        log "completed $script"
    else
        log "FAILED $script (exit $?)"
        # keep going — other passes may still work
    fi
}

run_eval run_eval_coarse.py --with-lm
run_eval run_eval_phase1_spread.py
run_eval run_eval_phase2_state.py
run_eval run_eval_phase3_kl.py

# ---- Completion sentinel ----
echo "$(date -u +%FT%TZ)" > results/pipeline_complete.flag
log "PIPELINE COMPLETE"
