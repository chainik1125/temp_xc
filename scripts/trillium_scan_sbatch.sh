#!/bin/bash
# trillium_scan_sbatch.sh — Run the autointerp SCAN phase on all 4 crosscoder
# checkpoints inside a single sbatch job (compute node, GPU, no network).
# Produces feat_*.json files with top_texts/top_activations populated but
# empty explanation fields — the login-node explain phase fills those in
# via Claude afterward.
#
# Usage:
#   bash scripts/trillium_scan_sbatch.sh
#
# Then, once squeue is empty:
#   bash scripts/trillium_explain.sh
#   bash scripts/trillium_sbatch_fmap_labeled.sh

set -euo pipefail

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm \
         reports/step1-gemma-replication reports/step2-deepseek-reasoning

JOB=$(mktemp "logs/slurm/scan-sbatch-XXXX.sh")
cat > "$JOB" <<'EOF'
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=txc-scan
#SBATCH --output=logs/slurm/scan-sbatch_%j.out
#SBATCH --error=logs/slurm/scan-sbatch_%j.err
set -euo pipefail
source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0

run_scan() {
    local TAG="$1"           # step1-unshuffled / step1-shuffled / step2-unshuffled / step2-shuffled
    local MODEL="$2"         # gemma-2-2b / deepseek-r1-distill-llama-8b
    local DATASET="$3"       # fineweb / gsm8k
    local LAYER="$4"         # resid_L13 / resid_L12
    local SHUF_SUFFIX="$5"   # "" or "_shuffled"
    local REPORT_DIR="$6"    # reports subdir

    local CKPT="results/nlp/$TAG/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k100__seed42${SHUF_SUFFIX}.pt"
    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP $TAG — ckpt not found: $CKPT"
        return 0
    fi

    echo ""
    echo ">> [$TAG] autointerp scan phase"
    python -m temporal_crosscoders.NLP.autointerp \
        --phase scan \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model "$MODEL" \
        --cached-dataset "$DATASET" --layer-key "$LAYER" \
        --k 100 --T 5 \
        --label "$TAG" \
        --output-dir "$REPORT_DIR" \
        --top-features 30 \
        --scan-device cuda
}

run_scan step1-unshuffled gemma-2-2b                     fineweb  resid_L13 "" \
    reports/step1-gemma-replication
run_scan step1-shuffled   gemma-2-2b                     fineweb  resid_L13 "_shuffled" \
    reports/step1-gemma-replication
run_scan step2-unshuffled deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "" \
    reports/step2-deepseek-reasoning
run_scan step2-shuffled   deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "_shuffled" \
    reports/step2-deepseek-reasoning

echo ""
echo "=== scan phase done ==="
find reports -name "feat_*.json" | wc -l
EOF

JOB_ID=$(sbatch --parsable "$JOB")
echo ""
echo "=== scan sbatch submitted ==="
echo "  job:   $JOB_ID"
echo "  watch: squeue -u \$USER"
echo "  tail:  tail -f logs/slurm/scan-sbatch_${JOB_ID}.out"
echo ""
echo "When the job finishes, run on the login node:"
echo "  bash scripts/trillium_explain.sh"
echo ""
echo "Close your laptop freely. Scan is ~5-20 min on a single compute GPU."
