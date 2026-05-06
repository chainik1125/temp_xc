#!/usr/bin/env bash
# Paper figure: HMM denoising bench, 4 archs, 3 seeds, full T x k grid.
# PARALLEL launcher: spawns 12 (model x seed) jobs concurrently on a
# single GPU. Each cell uses a tiny fraction of GPU memory + compute, so
# concurrency speedup is near-linear up to ~8-16 processes on any modern
# pod. Net: an overnight run becomes ~30-90 minutes wall-clock.
#
# Fills two gaps in the midterm bench:
#   1. Adds regular_sae_kT (framing-B per-token-budget baseline).
#   2. Multi-seed (3 seeds) so fig9 can show error bars.
#
# Usage on runpod:
#     cd /temp_xc && git pull
#     tmux new -s paperfig
#     bash scripts/run_hmm_paperfig_runpod.sh 2>&1 | tee paperfig.log
#     # Ctrl-b d to detach. When done, $REPO_ROOT/PAPERFIG_DONE exists.
#
# Output: results/hmm_paperfig/sweep_results.json (merged).
# Per-shard logs and partial JSONs in results/hmm_paperfig/shards/.

set -uo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

MODELS=(regular_sae regular_sae_kT stacked_sae txcdr)
SEEDS=(42 43 44)
N_STEPS=${N_STEPS:-65000}
BATCH_SIZE=${BATCH_SIZE:-64}

OUT_DIR="results/hmm_paperfig"
SHARD_DIR="$OUT_DIR/shards"
SENTINEL="$REPO_ROOT/PAPERFIG_DONE"

mkdir -p "$SHARD_DIR"

# Drop a stale sentinel if present.
rm -f "$SENTINEL"

echo "[paperfig] starting at $(date)"
echo "[paperfig] repo: $REPO_ROOT"
echo "[paperfig] models: ${MODELS[*]}"
echo "[paperfig] seeds:  ${SEEDS[*]}"
echo "[paperfig] $((${#MODELS[@]} * ${#SEEDS[@]})) parallel shards"
echo "[paperfig] output: $OUT_DIR"

# Track shard PIDs and identifiers so we can report each one's exit code.
declare -a PIDS LABELS LOGS
for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        label="${model}_seed${seed}"
        shard_out="$SHARD_DIR/$label"
        log_path="$SHARD_DIR/$label.log"
        echo "[paperfig] spawn $label -> $shard_out"
        uv run python scripts/run_hmm_denoising_sweep.py \
            --models "$model" \
            --n-seeds 1 \
            --seed "$seed" \
            --steps "$N_STEPS" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$shard_out" \
            > "$log_path" 2>&1 &
        PIDS+=("$!")
        LABELS+=("$label")
        LOGS+=("$log_path")
    done
done

echo "[paperfig] all ${#PIDS[@]} shards launched at $(date), waiting..."

# Wait on each PID and report status; keep going if some fail so partial
# results are still merged.
FAIL_COUNT=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    label=${LABELS[$i]}
    log=${LOGS[$i]}
    if wait "$pid"; then
        echo "[paperfig] OK   $label (pid=$pid)"
    else
        rc=$?
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "[paperfig] FAIL $label (pid=$pid, exit=$rc) -- tail of $log:"
        tail -n 20 "$log" | sed 's/^/    /'
    fi
done

echo "[paperfig] all shards finished at $(date) ($FAIL_COUNT failures)"

# Merge whatever shards completed (skip empty/missing JSONs gracefully).
SHARD_JSONS=()
for label in "${LABELS[@]}"; do
    p="$SHARD_DIR/$label/sweep_results.json"
    if [ -s "$p" ]; then
        SHARD_JSONS+=("$p")
    else
        echo "[paperfig] WARN: missing or empty $p; excluding from merge"
    fi
done

if [ ${#SHARD_JSONS[@]} -eq 0 ]; then
    echo "[paperfig] no shard JSONs to merge; aborting"
    exit 1
fi

echo "[paperfig] merging ${#SHARD_JSONS[@]} shard JSONs..."
uv run python scripts/merge_results.py \
    "${SHARD_JSONS[@]}" \
    --dedupe \
    --output "$OUT_DIR/sweep_results.json"

if [ $FAIL_COUNT -eq 0 ]; then
    touch "$SENTINEL"
    echo "[paperfig] DONE — sentinel: $SENTINEL"
else
    echo "[paperfig] DONE WITH FAILURES — $FAIL_COUNT shard(s) failed."
    echo "[paperfig] partial merged result: $OUT_DIR/sweep_results.json"
fi

echo "[paperfig] next: render the figure locally with"
echo "    uv run python scripts/plot_fig9_denoising_vs_T.py \\"
echo "        --input $OUT_DIR/sweep_results.json \\"
echo "        --output-dir docs/bill/results/hmm_paperfig"
