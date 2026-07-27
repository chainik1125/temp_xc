#!/bin/bash
# ACTMIX RLHF seed-extension orchestrator (CARD § 7 A1).
# Phase A: s1_T8 (frac 0.52) parallel with s2_T{1,2,5} (frac 0.34)
#          — combined peak ~52 GB on GPU 2.
# Phase B: ext_c serial uncapped (T16 never co-resides with T8).
# Usage: nohup bash experiments/explorations/actmix_rlhf/run_ext.sh <pin> &
set -u
cd "$(git rev-parse --show-toplevel)"
source scripts/set_agent_env.sh runpod-2
PIN="$1"
PY=.venv/bin/python
LOGD=/workspace/logs

TEMP_BENCH_GPU_FRACTION=0.52 $PY -m experiments.explorations.actmix_rlhf.run_cells \
    --lane ext_a --pin "$PIN" > "$LOGD/actmix_rlhf_lane_ext_a.log" 2>&1 &
A=$!
TEMP_BENCH_GPU_FRACTION=0.34 $PY -m experiments.explorations.actmix_rlhf.run_cells \
    --lane ext_b --pin "$PIN" > "$LOGD/actmix_rlhf_lane_ext_b.log" 2>&1 &
B=$!
wait "$A" "$B"
echo "[run_ext] phase A drained ($(date -u +%H:%M:%SZ)); starting ext_c"

$PY -m experiments.explorations.actmix_rlhf.run_cells \
    --lane ext_c --pin "$PIN" > "$LOGD/actmix_rlhf_lane_ext_c.log" 2>&1
echo "[run_ext] all lanes drained ($(date -u +%H:%M:%SZ))"
