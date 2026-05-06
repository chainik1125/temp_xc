#!/usr/bin/env bash
# Wait for txc_pro|1024 (GPU 3) to finish its 300K training + canonical
# eval, then auto-launch TFA on GPU 3 at the same protocol. Once TFA
# lands its canonical leaderboard row, the live extended-mags and
# optimal-mag chains pick it up without further intervention.
#
# Per Dmitry's request 2026-05-06: "we also need to add TFA". This
# script chains TFA *after* txc_pro|1024 because they share GPU 3.
#
# Usage:
#   set -a; source /workspace/aniket/temp_xc/.env; set +a
#   cd /workspace/aniket/temp_xc-final/purified
#   nohup bash scripts/c7_tfa_after_txc_pro.sh \
#       > logs/c7_tfa_after_txc_pro.log 2>&1 &
#   echo $! > logs/c7_tfa_after_txc_pro.pid

set -u
cd /workspace/aniket/temp_xc-final/purified

GPU=3
TXC_PID=10552                         # txc_pro|1024 training PID
ARCH=tfa
SEED=42
N_STEPS=300000
BATCH_SIZE=1024
LOGFILE="logs/c7_300k_${ARCH}_bs${BATCH_SIZE}.log"

log() { printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*"; }

log "watching txc_pro|1024 pid=${TXC_PID} on GPU ${GPU}"
while kill -0 "${TXC_PID}" 2>/dev/null; do
    sleep 60
done
log "txc_pro|1024 EXITED — training + canonical eval done. Launching TFA."

CUDA_VISIBLE_DEVICES=${GPU} \
AGENT_NAME=agent_back_300k \
TEMP_BENCH_POD_MODE=persistent \
TQDM_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup .venv/bin/python -m experiments.c7_backtracking.run \
        --archs "${ARCH}" --seeds ${SEED} \
        --n-steps ${N_STEPS} --batch-size ${BATCH_SIZE} \
        --probe-every 100 \
        > "${LOGFILE}" 2>&1 &
TFA_PID=$!
echo "${TFA_PID}" > "logs/c7_300k_${ARCH}_bs${BATCH_SIZE}.pid"
log "TFA pid=${TFA_PID} → ${LOGFILE}"

while kill -0 "${TFA_PID}" 2>/dev/null; do
    sleep 60
done
log "TFA EXITED — training + canonical eval done. Extended-mags and optimal-mag chains will pick it up automatically."
