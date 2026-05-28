#!/usr/bin/env bash
# Re-run cells that previously FAILed in the mini-sweep.
# Uses cached checkpoints where they exist, so retries are cheap.
set -e
cd "$(dirname "$0")/.."

export TEMP_BENCH_ALLOW_DIRTY=1
export TQDM_DISABLE=1
PY=.venv/bin/python
SEED=${SEED:-1}
N_STEPS=${N_STEPS:-10000}
BATCH=${BATCH:-1024}

LOG_OUT="logs/synth_retry_seed${SEED}.log"
: > "$LOG_OUT"

# (arch, datasource, k_pos) hardcoded from the first sweep's FAIL list.
CELLS=(
    "txc_pro:toy_coupled_K10_M20_d256:1"
    "txc_pro:toy_coupled_K10_M20_d256:2"
    "txc_pro:toy_coupled_K10_M20_d256:5"
    "txc_pro:toy_coupled_K10_M20_d256:10"
    "txc_pro:toy_coupled_K10_M20_d256:20"
    "tfa:toy_coupled_K10_M20_d256:10"
    "tfa:toy_coupled_K10_M20_d256:20"
)

for cell in "${CELLS[@]}"; do
    arch=$(cut -d: -f1 <<<"$cell")
    ds=$(cut -d: -f2 <<<"$cell")
    kp=$(cut -d: -f3 <<<"$cell")
    printf "[retry] %-10s %-32s k=%-2s ... " "$arch" "$ds" "$kp" | tee -a "$LOG_OUT"
    t0=$SECONDS
    out=$($PY run.py synthetic \
            --arch "$arch" --seed "$SEED" \
            --datasource "$ds" \
            --k-pos $kp \
            --n-steps $N_STEPS --batch-size $BATCH 2>&1 || echo "FAILED")
    dt=$((SECONDS - t0))
    if echo "$out" | grep -qE "FAILED|Traceback"; then
        echo "FAIL ($dt s)" | tee -a "$LOG_OUT"
        echo "$out" | tail -4 >> "$LOG_OUT"
    else
        e=$(echo "$out" | grep -oP "^\s*eauc\s+=\s+\K[\d.]+")
        g=$(echo "$out" | grep -oP "^\s*gauc\s+=\s+\K[\d.]+")
        n=$(echo "$out" | grep -oP "^\s*nmse\s+=\s+\K[\d.]+")
        echo "eAUC=$e gAUC=$g NMSE=$n ($dt s)" | tee -a "$LOG_OUT"
    fi
done

echo "[retry] done. results: $LOG_OUT"
