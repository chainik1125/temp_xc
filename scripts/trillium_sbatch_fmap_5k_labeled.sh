#!/bin/bash
# trillium_sbatch_fmap_5k_labeled.sh — Final Trillium fmap pass with
# gemma-2-2b-it cluster summaries in the legend. Same as
# trillium_sbatch_fmap_5k.sh but WITHOUT --skip-llm-labels, so each
# cluster gets a one-sentence thematic label instead of "Cluster 0..19".
#
# Per-feature hover text still comes from the 5k autointerp JSONs.
# Runs on the compute node (has a GPU to host gemma-2-2b-it).
#
# Intended use:
#   - One-off before retiring Trillium. Overwrites the old unlabeled
#     PNGs + HTMLs in reports/step{1,2}-*/ with labeled versions.
#   - Pull locally after with: bash scripts/fetch_trillium_results.sh
#
# Usage:
#   bash scripts/trillium_sbatch_fmap_5k_labeled.sh

set -euo pipefail

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm

JOB=$(mktemp "logs/slurm/fmap-5k-labeled-XXXX.sh")
cat > "$JOB" <<'EOF'
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --job-name=txc-fmap-5k-labeled
#SBATCH --output=logs/slurm/fmap-5k-labeled_%j.out
#SBATCH --error=logs/slurm/fmap-5k-labeled_%j.err
set -euo pipefail
source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0

run_fmap_labeled() {
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
    echo ">> [$TAG] feature_map (5k labels + gemma-2-2b-it cluster summaries)"
    python -m temporal_crosscoders.NLP.feature_map \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model "$MODEL" \
        --k 100 --T 5 \
        --label "$TAG" \
        --output-dir "$REPORT_DIR" \
        --include-unlabeled \
        --explain-model google/gemma-2-2b-it \
        --explain-device cuda:0
}

run_fmap_labeled step1-unshuffled gemma-2-2b                     fineweb  resid_L13 "" \
    reports/step1-gemma-replication
run_fmap_labeled step1-shuffled   gemma-2-2b                     fineweb  resid_L13 "_shuffled" \
    reports/step1-gemma-replication
run_fmap_labeled step2-unshuffled deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "" \
    reports/step2-deepseek-reasoning
run_fmap_labeled step2-shuffled   deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "_shuffled" \
    reports/step2-deepseek-reasoning

echo ""
echo "=== fmap 5k labeled done ==="
ls -lh reports/step1-gemma-replication/feature_map_*.png \
       reports/step2-deepseek-reasoning/feature_map_*.png 2>/dev/null || true
EOF

JOB_ID=$(sbatch --parsable "$JOB")
echo ""
echo "=== fmap 5k labeled sbatch submitted ==="
echo "  job:   $JOB_ID"
echo "  watch: squeue -u \$USER"
echo "  tail:  tail -f logs/slurm/fmap-5k-labeled_${JOB_ID}.out"
echo ""
echo "Time: ~15-25 min once it leaves the queue."
echo ""
echo "When done, from your laptop (overwrites the unlabeled plots locally):"
echo "  bash scripts/fetch_trillium_results.sh"
echo ""
echo "This is the final Trillium job for the sprint. Retire Trillium after."
