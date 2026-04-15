#!/bin/bash
# trillium_step2_deepseek_reasoning.sh — Step 2 of the feature-map sprint
# experiment. Single bash invocation runs the full chain on DeepSeek-R1-
# Distill-Llama-8B + GSM8K reasoning traces:
#
#   cache 1000 reasoning traces (generate mode)  (~4-6 h)
#      -> unshuffled architecture sweep          (depends on cache)
#      -> shuffled   architecture sweep          (depends on cache)
#      -> feature_map on both checkpoints        (inline, CPU)
#
# Designed to pair with Step 1. After both steps land you have a 4-panel
# figure: {Gemma, DeepSeek} x {unshuffled, shuffled} TXCDR decoder clustering.
#
# Outputs:
#   results/nlp/step2-unshuffled/  (sweep JSONs + ckpts)
#   results/nlp/step2-shuffled/
#   reports/step2-deepseek-reasoning/feature_map_step2-unshuffled.png / .html
#   reports/step2-deepseek-reasoning/feature_map_step2-shuffled.png / .html
#
# Usage:
#   bash scripts/trillium_step2_deepseek_reasoning.sh
#
# Env overrides:
#   LAYER=resid_L24      late layer instead of 12
#   STEPS=20000          longer training
#   NUM_SEQS=500         smaller cache for a faster turnaround
#   SKIP_CACHE=1         reuse whatever cache is currently on disk

set -euo pipefail

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm reports

MODEL="deepseek-r1-distill-llama-8b"
DATASET="gsm8k"
LAYER="${LAYER:-resid_L12}"           # ~37% depth for DeepSeek 32 layers
K="${K:-100}"
T="${T:-5}"
STEPS="${STEPS:-10000}"
NUM_SEQS="${NUM_SEQS:-1000}"
GEN_MAX_NEW="${GEN_MAX_NEW:-1024}"
SKIP_CACHE="${SKIP_CACHE:-0}"

REPORT_DIR="reports/step2-deepseek-reasoning"

# ─── 1. cache job (autoregressive generate on reasoning traces) ────────────
if [ "$SKIP_CACHE" = "1" ]; then
    echo ">> SKIP_CACHE=1, not re-caching"
    DEP=""
else
    CACHE_JOB=$(mktemp "logs/slurm/step2-cache-XXXX.sh")
    cat > "$CACHE_JOB" <<EOF
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=step2-cache
#SBATCH --output=logs/slurm/step2-cache_%j.out
#SBATCH --error=logs/slurm/step2-cache_%j.err
set -euo pipefail
source "\$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0
python scripts/cache_reasoning_traces.py \\
    --model $MODEL --dataset $DATASET \\
    --num-sequences $NUM_SEQS --gen_max_new_tokens $GEN_MAX_NEW \\
    --layer_indices 12 24
EOF
    CACHE_ID=$(sbatch --parsable "$CACHE_JOB")
    echo ">> cache job submitted: $CACHE_ID"
    DEP="--dependency=afterok:$CACHE_ID"
fi

# ─── 2. sweep + feature_map combined jobs ──────────────────────────────────
submit_sweep() {
    local TAG="$1"
    local SHUF_FLAG="$2"
    local SHUF_SUFFIX="$3"

    local RESULTS="results/nlp/step2-${TAG}"
    local JOB
    JOB=$(mktemp "logs/slurm/step2-sweep-${TAG}-XXXX.sh")
    cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=step2-${TAG}
#SBATCH --output=logs/slurm/step2-sweep-${TAG}_%j.out
#SBATCH --error=logs/slurm/step2-sweep-${TAG}_%j.err
set -euo pipefail
source "\$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0

echo "=== step2 ${TAG} sweep: $MODEL / $DATASET / $LAYER / k=$K T=$T ==="

python -m src.bench.sweep \\
    --dataset-type cached_activations \\
    --model-name $MODEL \\
    --cached-dataset $DATASET \\
    --cached-layer-key $LAYER \\
    --models topk_sae stacked_sae crosscoder \\
    --k $K --T $T --steps $STEPS \\
    --results-dir $RESULTS \\
    $SHUF_FLAG

CKPT="$RESULTS/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k${K}__seed42${SHUF_SUFFIX}.pt"
if [ -f "\$CKPT" ]; then
    echo "=== feature_map on \$CKPT ==="
    mkdir -p $REPORT_DIR
    python -m temporal_crosscoders.NLP.feature_map \\
        --checkpoint "\$CKPT" \\
        --model crosscoder --subject-model $MODEL \\
        --k $K --T $T \\
        --include-unlabeled --skip-llm-labels \\
        --label "step2-${TAG}" \\
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
echo "=== step2 chain submitted ==="
echo "  cache:      ${CACHE_ID:-<skipped>}"
echo "  unshuffled: $UN_ID"
echo "  shuffled:   $SHUF_ID"
echo ""
echo "Watch: squeue -u \$USER"
echo "Tail:  tail -f logs/slurm/step2-*.out"
