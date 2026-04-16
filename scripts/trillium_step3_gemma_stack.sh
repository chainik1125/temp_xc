#!/bin/bash
# trillium_step3_gemma_stack.sh — Step 3 of the feature-map sprint:
# same-model, same-inference-mode, code-modality cell that rules out H3
# (model/mode confound) in the 2×3 experimental matrix.
#
#   cache Stack-Python activations on Gemma 2 2B   (~1-2 h)
#     -> unshuffled architecture sweep              (depends on cache)
#     -> shuffled   architecture sweep              (depends on cache)
#     -> feature_map on crosscoder ckpt             (inline at end of each)
#
# Matches step1 hyperparameters (k=100, T=5, 10k steps, layer 13) with
# the only variable being the dataset — critical for within-model data
# ablation. seq_len is bumped to 256 because Python structure spans
# function bodies (20–100 lines), not web-text sentences.
#
# Outputs:
#   results/nlp/step3-unshuffled/  (sweep JSONs + ckpts)
#   results/nlp/step3-shuffled/
#   reports/step3-gemma-stack/feature_map_step3-unshuffled.png / .html
#   reports/step3-gemma-stack/feature_map_step3-shuffled.png / .html
#
# Pre-registration: docs/aniket/sprint_coding_dataset_plan.md
#
# Usage:
#   bash scripts/trillium_step3_gemma_stack.sh
#
# Env overrides:
#   SEQ_LEN=512      longer code context (cache cost scales linearly)
#   NUM_SEQS=6000    more sequences
#   STEPS=20000      longer training
#   SKIP_CACHE=1     reuse whatever cache is currently on disk
#   SKIP_PREFETCH=1  skip login-node pre-fetch (Stack streams from HF)

set -euo pipefail

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm reports data/prefetched

# ─── 0. prefetch Stack-Python on the login node (compute has no network) ─
# The Stack v2 requires gated HuggingFace access + token. Prefetch script
# streams a slice to data/prefetched/stack-python_<N>.jsonl for the compute
# node to read.
if [ "${SKIP_PREFETCH:-0}" != "1" ]; then
    bash scripts/prefetch_text_dataset.sh stack-python "${NUM_SEQS:-3000}"
fi

MODEL="gemma-2-2b"
DATASET="stack-python"
LAYER="${LAYER:-resid_L13}"
K="${K:-100}"
T="${T:-5}"
STEPS="${STEPS:-10000}"
NUM_SEQS="${NUM_SEQS:-3000}"
SEQ_LEN="${SEQ_LEN:-256}"
SKIP_CACHE="${SKIP_CACHE:-0}"

REPORT_DIR="reports/step3-gemma-stack"

# ─── 1. cache job ─────────────────────────────────────────────────────────
if [ "$SKIP_CACHE" = "1" ]; then
    echo ">> SKIP_CACHE=1, not re-caching"
    DEP=""
else
    CACHE_JOB=$(mktemp "logs/slurm/step3-cache-XXXX.sh")
    cat > "$CACHE_JOB" <<EOF
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --job-name=step3-cache
#SBATCH --output=logs/slurm/step3-cache_%j.out
#SBATCH --error=logs/slurm/step3-cache_%j.err
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

# ─── 2. sweep + feature_map combined jobs ─────────────────────────────────
submit_sweep() {
    local TAG="$1"          # "unshuffled" | "shuffled"
    local SHUF_FLAG="$2"    # "" | "--shuffle-within-sequence"
    local SHUF_SUFFIX="$3"  # "" | "_shuffled"

    local RESULTS="results/nlp/step3-${TAG}"
    local JOB
    JOB=$(mktemp "logs/slurm/step3-sweep-${TAG}-XXXX.sh")
    cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --job-name=step3-${TAG}
#SBATCH --output=logs/slurm/step3-sweep-${TAG}_%j.out
#SBATCH --error=logs/slurm/step3-sweep-${TAG}_%j.err
set -euo pipefail
source "\$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0

echo "=== step3 ${TAG} sweep: $MODEL / $DATASET / $LAYER / k=$K T=$T ==="

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
        --label "step3-${TAG}" \\
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
echo "=== step3 chain submitted ==="
echo "  cache:      ${CACHE_ID:-<skipped>}"
echo "  unshuffled: $UN_ID"
echo "  shuffled:   $SHUF_ID"
echo ""
echo "Watch: squeue -u \$USER"
echo "Tail:  tail -f logs/slurm/step3-*.out"
echo ""
echo "After finish, backfill the three pre-registered metrics:"
echo "  bash scripts/trillium_backfill_metrics.sh   # also covers step1, step2"
echo ""
echo "Fetch results on your laptop:"
echo "  bash scripts/fetch_from_trillium.sh reports"
echo "Open:"
echo "  reports/step3-gemma-stack/feature_map_step3-unshuffled.png"
echo "  reports/step3-gemma-stack/feature_map_step3-shuffled.png"
