#!/bin/bash
# trillium_sbatch_fmap_sae_control.sh — Run feature_map on the TopKSAE
# checkpoints from step2 (DeepSeek + GSM8K) as a critical control.
#
# If the SAE baseline ALSO shows the isolated right-side island that
# TXCDR showed in step2-unshuffled, then the structure is a property
# of the data/layer, not TXCDR's temporal inductive bias. If the SAE
# shows a diffuse cloud while TXCDR shows structure, the claim holds.
#
# This should run BEFORE spending $80 on 5k autointerp — it gates
# the entire "TXCDR finds temporal structure" claim.
#
# Usage:
#   bash scripts/trillium_sbatch_fmap_sae_control.sh

set -euo pipefail

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm reports/sae-control-deepseek

JOB=$(mktemp "logs/slurm/fmap-sae-control-XXXX.sh")
cat > "$JOB" <<'EOF'
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=txc-fmap-sae
#SBATCH --output=logs/slurm/fmap-sae-control_%j.out
#SBATCH --error=logs/slurm/fmap-sae-control_%j.err
set -euo pipefail
source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0

run_fmap() {
    local TAG="$1"
    local ARCH="$2"
    local MODEL="$3"
    local DATASET="$4"
    local LAYER="$5"
    local SHUF_SUFFIX="$6"
    local REPORT_DIR="$7"
    local LABEL="$8"

    local CKPT="results/nlp/$TAG/ckpts/${ARCH}__${MODEL}__${DATASET}__${LAYER}__k100__seed42${SHUF_SUFFIX}.pt"
    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP $LABEL — ckpt not found: $CKPT"
        return 0
    fi

    echo ""
    echo ">> [$LABEL] feature_map ($ARCH)"
    python -m temporal_crosscoders.NLP.feature_map \
        --checkpoint "$CKPT" \
        --model "$ARCH" --subject-model "$MODEL" \
        --k 100 --T 5 \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --include-unlabeled --skip-llm-labels
}

# TopKSAE on DeepSeek, both shuffle conditions
run_fmap step2-unshuffled topk_sae deepseek-r1-distill-llama-8b gsm8k resid_L12 "" \
    reports/sae-control-deepseek sae-deepseek-unshuffled
run_fmap step2-shuffled   topk_sae deepseek-r1-distill-llama-8b gsm8k resid_L12 "_shuffled" \
    reports/sae-control-deepseek sae-deepseek-shuffled

# TXCDR on DeepSeek for side-by-side (reuses existing, just different output dir)
run_fmap step2-unshuffled crosscoder deepseek-r1-distill-llama-8b gsm8k resid_L12 "" \
    reports/sae-control-deepseek txcdr-deepseek-unshuffled
run_fmap step2-shuffled   crosscoder deepseek-r1-distill-llama-8b gsm8k resid_L12 "_shuffled" \
    reports/sae-control-deepseek txcdr-deepseek-shuffled

echo ""
echo "=== SAE control done ==="
ls -lh reports/sae-control-deepseek/*.png 2>/dev/null || true
EOF

JOB_ID=$(sbatch --parsable "$JOB")
echo ""
echo "=== SAE control sbatch submitted ==="
echo "  job:   $JOB_ID"
echo "  watch: squeue -u \$USER"
echo ""
echo "Produces 4 PNGs in reports/sae-control-deepseek/:"
echo "  sae-deepseek-unshuffled.png   — SAE baseline, natural order"
echo "  sae-deepseek-shuffled.png     — SAE baseline, shuffled"
echo "  txcdr-deepseek-unshuffled.png — TXCDR (same as step2, for side-by-side)"
echo "  txcdr-deepseek-shuffled.png   — TXCDR shuffled"
echo ""
echo "If the SAE plots look like a diffuse cloud while TXCDR shows"
echo "the isolated island → claim holds. Proceed with 5k autointerp."
echo "If SAE also shows the island → claim collapses. Don't spend $80."
