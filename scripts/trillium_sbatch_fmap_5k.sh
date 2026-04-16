#!/bin/bash
# trillium_sbatch_fmap_5k.sh — Sbatched feature_map pass that reads ~5000
# autointerp feat_*.json files per checkpoint for hover-text labels.
# Produces labeled interactive HTML and cluster-ID PNG outputs.
#
# Usage (after scan_5k + explain_5k have run):
#   bash scripts/trillium_sbatch_fmap_5k.sh

set -euo pipefail

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm

JOB=$(mktemp "logs/slurm/fmap-5k-XXXX.sh")
cat > "$JOB" <<'EOF'
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=txc-fmap-5k
#SBATCH --output=logs/slurm/fmap-5k_%j.out
#SBATCH --error=logs/slurm/fmap-5k_%j.err
set -euo pipefail
source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0

run_fmap() {
    local TAG="$1"
    local MODEL="$2"
    local DATASET="$3"
    local LAYER="$4"
    local SHUF_SUFFIX="$5"
    local REPORT_DIR="$6"

    local CKPT="results/nlp/$TAG/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k100__seed42${SHUF_SUFFIX}.pt"
    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP $TAG — ckpt not found: $CKPT"
        return 0
    fi

    echo ""
    echo ">> [$TAG] feature_map (5k labels)"
    python -m temporal_crosscoders.NLP.feature_map \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model "$MODEL" \
        --k 100 --T 5 \
        --label "$TAG" \
        --output-dir "$REPORT_DIR" \
        --include-unlabeled --skip-llm-labels
}

run_fmap step1-unshuffled gemma-2-2b                     fineweb  resid_L13 "" \
    reports/step1-gemma-replication
run_fmap step1-shuffled   gemma-2-2b                     fineweb  resid_L13 "_shuffled" \
    reports/step1-gemma-replication
run_fmap step2-unshuffled deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "" \
    reports/step2-deepseek-reasoning
run_fmap step2-shuffled   deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "_shuffled" \
    reports/step2-deepseek-reasoning

echo ""
echo "=== fmap 5k done ==="
ls -lh reports/step1-gemma-replication/feature_map_*.png \
       reports/step2-deepseek-reasoning/feature_map_*.png 2>/dev/null || true
EOF

JOB_ID=$(sbatch --parsable "$JOB")
echo ""
echo "=== fmap 5k sbatch submitted ==="
echo "  job:   $JOB_ID"
echo "  watch: squeue -u \$USER"
echo ""
echo "When done, on your laptop:"
echo "  bash scripts/fetch_from_trillium.sh reports"
