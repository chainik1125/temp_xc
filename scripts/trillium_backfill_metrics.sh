#!/bin/bash
# trillium_backfill_metrics.sh — Backfill the three pre-registered sprint
# metrics (silhouette, cluster_entropy, mean_auto_mi across lags) on the
# four existing step1/step2 sweeps — three architectures each (TopKSAE,
# Stacked T=5, TXCDRv2). This converts the current qualitative
# "noticeably more diffuse" language into numbers, apples-to-apples with
# any future step3 (Gemma+Stack) cell.
#
# Metric definitions and rationale: docs/aniket/sprint_coding_dataset_plan.md
# Script the compute step wraps:        scripts/compute_temporal_metrics.py
#
# Usage:
#   bash scripts/trillium_backfill_metrics.sh

set -euo pipefail

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm \
         reports/step1-gemma-replication reports/step2-deepseek-reasoning

JOB=$(mktemp "logs/slurm/backfill-metrics-XXXX.sh")
cat > "$JOB" <<'EOF'
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --job-name=txc-backfill-metrics
#SBATCH --output=logs/slurm/backfill-metrics_%j.out
#SBATCH --error=logs/slurm/backfill-metrics_%j.err
set -euo pipefail
source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0

run_metrics() {
    local TAG="$1"
    local ARCH="$2"
    local MODEL="$3"
    local DATASET="$4"
    local LAYER="$5"
    local SHUF_SUFFIX="$6"
    local REPORT_DIR="$7"

    local CKPT="results/nlp/${TAG}/ckpts/${ARCH}__${MODEL}__${DATASET}__${LAYER}__k100__seed42${SHUF_SUFFIX}.pt"
    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP ${TAG}/${ARCH} — ckpt not found: $CKPT"
        return 0
    fi

    local LABEL="${TAG}__${ARCH}"
    echo ""
    echo ">> [${LABEL}] compute_temporal_metrics"
    python scripts/compute_temporal_metrics.py \
        --checkpoint "$CKPT" \
        --arch "$ARCH" \
        --subject-model "$MODEL" \
        --cached-dataset "$DATASET" \
        --layer-key "$LAYER" \
        --k 100 --T 5 \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --device cuda
}

# ─── step1: Gemma 2B + FineWeb ─────────────────────────────────────────
for ARCH in topk_sae stacked_sae crosscoder; do
    run_metrics step1-unshuffled "$ARCH" gemma-2-2b fineweb resid_L13 "" \
        reports/step1-gemma-replication
    run_metrics step1-shuffled   "$ARCH" gemma-2-2b fineweb resid_L13 "_shuffled" \
        reports/step1-gemma-replication
done

# ─── step2: DeepSeek-R1-Distill-8B + GSM8K ─────────────────────────────
for ARCH in topk_sae stacked_sae crosscoder; do
    run_metrics step2-unshuffled "$ARCH" deepseek-r1-distill-llama-8b gsm8k resid_L12 "" \
        reports/step2-deepseek-reasoning
    run_metrics step2-shuffled   "$ARCH" deepseek-r1-distill-llama-8b gsm8k resid_L12 "_shuffled" \
        reports/step2-deepseek-reasoning
done

echo ""
echo "=== backfill done ==="
echo ""
echo "Step 1 metrics:"
ls -1 reports/step1-gemma-replication/metrics_*.json 2>/dev/null | sed 's|.*metrics_|  |' || true
echo ""
echo "Step 2 metrics:"
ls -1 reports/step2-deepseek-reasoning/metrics_*.json 2>/dev/null | sed 's|.*metrics_|  |' || true
EOF

JOB_ID=$(sbatch --parsable "$JOB")
echo ""
echo "=== backfill-metrics sbatch submitted ==="
echo "  job:   $JOB_ID"
echo "  watch: squeue -u \$USER"
echo "  tail:  tail -f logs/slurm/backfill-metrics_${JOB_ID}.out"
echo ""
echo "Produces 12 JSONs (3 archs × 4 shuffle-conditions):"
echo "  reports/step1-gemma-replication/metrics_step1-{un,}shuffled__{topk_sae,stacked_sae,crosscoder}.json"
echo "  reports/step2-deepseek-reasoning/metrics_step2-{un,}shuffled__{topk_sae,stacked_sae,crosscoder}.json"
echo ""
echo "Pre-registered scalars in each JSON:"
echo "  .decoder_geometry.silhouette"
echo "  .decoder_geometry.cluster_entropy"
echo "  .auto_mi.mean_auto_mi_scalar"
