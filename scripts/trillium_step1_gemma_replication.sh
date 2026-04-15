#!/bin/bash
# trillium_step1_gemma_replication.sh — Step 1 of the feature-map sprint
# experiment. Single bash invocation runs the whole chain:
#
#   cache FineWeb activations on Gemma 2 2B  (~1-2 h)
#      -> unshuffled architecture sweep    (depends on cache)
#      -> shuffled   architecture sweep    (depends on cache)
#      -> feature_map on both checkpoints  (inline at end of each sweep)
#
# Replicates Andre's nlp_feature_map.md setup (k=100, T=5, mid_res = layer 13)
# with the shuffled temporal control he didn't run. The intended A/B: does
# TXCDR's two-megacluster structure survive temporal shuffling?
#
# Outputs:
#   results/nlp/step1-unshuffled/  (sweep JSONs + ckpts)
#   results/nlp/step1-shuffled/
#   reports/step1-gemma-replication/feature_map_step1-unshuffled.png / .html
#   reports/step1-gemma-replication/feature_map_step1-shuffled.png / .html
#
# Usage:
#   bash scripts/trillium_step1_gemma_replication.sh
#
# Env overrides:
#   LAYER=resid_L25  to target the late layer instead of mid
#   STEPS=20000      longer training
#   SKIP_CACHE=1     reuse whatever cache is currently on disk

set -euo pipefail

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm reports

MODEL="gemma-2-2b"
DATASET="fineweb"
LAYER="${LAYER:-resid_L13}"           # Andre's mid_res for Gemma 2 2B (26 layers)
K="${K:-100}"
T="${T:-5}"
STEPS="${STEPS:-10000}"
NUM_SEQS="${NUM_SEQS:-24000}"
SEQ_LEN="${SEQ_LEN:-32}"
SKIP_CACHE="${SKIP_CACHE:-0}"

REPORT_DIR="reports/step1-gemma-replication"

# ─── 1. cache job ──────────────────────────────────────────────────────────
if [ "$SKIP_CACHE" = "1" ]; then
    echo ">> SKIP_CACHE=1, not re-caching"
    DEP=""
else
    CACHE_JOB=$(mktemp "logs/slurm/step1-cache-XXXX.sh")
    cat > "$CACHE_JOB" <<EOF
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --job-name=step1-cache
#SBATCH --output=logs/slurm/step1-cache_%j.out
#SBATCH --error=logs/slurm/step1-cache_%j.err
set -euo pipefail
source "\$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0
python -m temporal_crosscoders.NLP.cache_activations \\
    --model $MODEL --dataset $DATASET --mode forward \\
    --num-sequences $NUM_SEQS --seq-length $SEQ_LEN \\
    --layer_indices 13 25 --components resid
EOF
    CACHE_ID=$(sbatch --parsable "$CACHE_JOB")
    echo ">> cache job submitted: $CACHE_ID"
    DEP="--dependency=afterok:$CACHE_ID"
fi

# ─── 2. sweep + feature_map combined jobs ──────────────────────────────────
submit_sweep() {
    local TAG="$1"          # "unshuffled" | "shuffled"
    local SHUF_FLAG="$2"    # "" | "--shuffle-within-sequence"
    local SHUF_SUFFIX="$3"  # "" | "_shuffled"

    local RESULTS="results/nlp/step1-${TAG}"
    local JOB
    JOB=$(mktemp "logs/slurm/step1-sweep-${TAG}-XXXX.sh")
    cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --job-name=step1-${TAG}
#SBATCH --output=logs/slurm/step1-sweep-${TAG}_%j.out
#SBATCH --error=logs/slurm/step1-sweep-${TAG}_%j.err
set -euo pipefail
source "\$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0

echo "=== step1 ${TAG} sweep: $MODEL / $DATASET / $LAYER / k=$K T=$T ==="

python -m src.bench.sweep \\
    --dataset-type cached_activations \\
    --model-name $MODEL \\
    --cached-dataset $DATASET \\
    --cached-layer-key $LAYER \\
    --models topk_sae stacked_sae crosscoder \\
    --k $K --T $T --steps $STEPS \\
    --results-dir $RESULTS \\
    $SHUF_FLAG

# Feature map on the crosscoder checkpoint, inline, CPU-only (no LLM labels).
CKPT="$RESULTS/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k${K}__seed42${SHUF_SUFFIX}.pt"
if [ -f "\$CKPT" ]; then
    echo "=== feature_map on \$CKPT ==="
    mkdir -p $REPORT_DIR
    python -m temporal_crosscoders.NLP.feature_map \\
        --checkpoint "\$CKPT" \\
        --model crosscoder --subject-model $MODEL \\
        --k $K --T $T \\
        --include-unlabeled --skip-llm-labels \\
        --label "step1-${TAG}" \\
        --output-dir $REPORT_DIR
else
    echo "WARN: expected checkpoint not found: \$CKPT"
    ls -la $RESULTS/ckpts/ || true
fi
EOF
    if [ -n "$DEP" ]; then
        sbatch --parsable $DEP "$JOB"
    else
        sbatch --parsable "$JOB"
    fi
}

UN_ID=$(submit_sweep "unshuffled" "" "")
SHUF_ID=$(submit_sweep "shuffled" "--shuffle-within-sequence" "_shuffled")

echo ""
echo "=== step1 chain submitted ==="
echo "  cache:      ${CACHE_ID:-<skipped>}"
echo "  unshuffled: $UN_ID"
echo "  shuffled:   $SHUF_ID"
echo ""
echo "Watch: squeue -u \$USER"
echo "Tail:  tail -f logs/slurm/step1-*.out"
echo ""
echo "When all three finish, fetch results on your laptop:"
echo "  bash scripts/fetch_from_trillium.sh reports"
echo "Then open:"
echo "  reports/step1-gemma-replication/feature_map_step1-unshuffled.png"
echo "  reports/step1-gemma-replication/feature_map_step1-shuffled.png"
