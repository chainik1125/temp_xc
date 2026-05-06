#!/usr/bin/env bash
# Part 3 figure: gAUC vs HMM complexity (n_parents) sweep.
# PARALLEL launcher: spawns (model x seed) shards concurrently, each
# running its full (n_parents x T x k) grid for one model+seed pair.
#
# Tests whether TXCDR's global-recovery advantage over a token-local SAE
# grows with HMM complexity (the per-token coupling-inversion problem
# becomes more ill-posed as n_parents grows).
#
# Usage on runpod:
#     cd /temp_xc && git pull
#     tmux new -s complexity
#     bash scripts/run_coupled_complexity_runpod.sh 2>&1 | tee complexity.log
#
# Output: results/coupled_complexity/sweep_results.json (merged).

set -uo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

MODELS=(regular_sae regular_sae_kT txcdr)
SEEDS=(42 43 44)
N_PARENTS=(1 2 3 5 7 10)
T_VALUES=(2 5 10)
K_VALUES=(1 3)
N_STEPS=${N_STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-256}

OUT_DIR="results/coupled_complexity"
SHARD_DIR="$OUT_DIR/shards"
SENTINEL="$REPO_ROOT/COMPLEXITY_DONE"

mkdir -p "$SHARD_DIR"
rm -f "$SENTINEL"

echo "[complexity] starting at $(date)"
echo "[complexity] models: ${MODELS[*]}"
echo "[complexity] seeds:  ${SEEDS[*]}"
echo "[complexity] n_parents: ${N_PARENTS[*]}"
echo "[complexity] T: ${T_VALUES[*]}, k: ${K_VALUES[*]}"
echo "[complexity] $((${#MODELS[@]} * ${#SEEDS[@]})) parallel shards"

declare -a PIDS LABELS LOGS
for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        label="${model}_seed${seed}"
        shard_out="$SHARD_DIR/$label"
        log_path="$SHARD_DIR/$label.log"
        echo "[complexity] spawn $label -> $shard_out"
        uv run python scripts/run_coupled_complexity_sweep.py \
            --models "$model" \
            --n-seeds 1 \
            --seed "$seed" \
            --n-parents "${N_PARENTS[@]}" \
            --T "${T_VALUES[@]}" \
            --k "${K_VALUES[@]}" \
            --steps "$N_STEPS" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$shard_out" \
            > "$log_path" 2>&1 &
        PIDS+=("$!")
        LABELS+=("$label")
        LOGS+=("$log_path")
    done
done

echo "[complexity] all ${#PIDS[@]} shards launched at $(date), waiting..."

FAIL_COUNT=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    label=${LABELS[$i]}
    log=${LOGS[$i]}
    if wait "$pid"; then
        echo "[complexity] OK   $label (pid=$pid)"
    else
        rc=$?
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "[complexity] FAIL $label (pid=$pid, exit=$rc) -- tail of $log:"
        tail -n 20 "$log" | sed 's/^/    /'
    fi
done

echo "[complexity] all shards finished at $(date) ($FAIL_COUNT failures)"

SHARD_JSONS=()
for label in "${LABELS[@]}"; do
    p="$SHARD_DIR/$label/sweep_results.json"
    if [ -s "$p" ]; then
        SHARD_JSONS+=("$p")
    else
        echo "[complexity] WARN: missing or empty $p; excluding from merge"
    fi
done

if [ ${#SHARD_JSONS[@]} -eq 0 ]; then
    echo "[complexity] no shard JSONs to merge; aborting"
    exit 1
fi

echo "[complexity] merging ${#SHARD_JSONS[@]} shard JSONs..."
uv run python scripts/merge_results.py \
    "${SHARD_JSONS[@]}" \
    --dedupe \
    --output "$OUT_DIR/sweep_results.json"

if [ $FAIL_COUNT -eq 0 ]; then
    touch "$SENTINEL"
    echo "[complexity] DONE — sentinel: $SENTINEL"
else
    echo "[complexity] DONE WITH FAILURES — $FAIL_COUNT shard(s) failed."
fi

echo "[complexity] next: render the figure locally with"
echo "    uv run python scripts/plot_fig_complexity.py \\"
echo "        --input $OUT_DIR/sweep_results.json \\"
echo "        --output-dir docs/bill/results/coupled_complexity"
