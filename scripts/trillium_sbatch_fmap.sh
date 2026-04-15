#!/bin/bash
# trillium_sbatch_fmap.sh — Run feature_map on all 4 crosscoder checkpoints
# in a single sbatch job on a compute node. Pure geometry (PCA->UMAP->KMeans
# over decoder directions), no autointerp, no Claude, no network needed.
#
# This is the "just give me the PNGs, I'll close my laptop" path:
#   bash scripts/trillium_sbatch_fmap.sh
#
# Outputs (after the job runs):
#   reports/step1-gemma-replication/feature_map_step1-unshuffled.{png,html}
#   reports/step1-gemma-replication/feature_map_step1-shuffled.{png,html}
#   reports/step2-deepseek-reasoning/feature_map_step2-unshuffled.{png,html}
#   reports/step2-deepseek-reasoning/feature_map_step2-shuffled.{png,html}
#
# No labels on the PNGs — that requires Claude API calls which don't work on
# compute nodes (no outbound internet). If you want labels later, run
# scripts/trillium_step{1,2}_finalize.sh as a follow-up on the login node
# (it'll reuse these same checkpoints + add autointerp + relabeled HTMLs).

set -euo pipefail

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm reports/step1-gemma-replication reports/step2-deepseek-reasoning

JOB=$(mktemp "logs/slurm/sbatch-fmap-XXXX.sh")
cat > "$JOB" <<'EOF'
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=txc-fmap
#SBATCH --output=logs/slurm/sbatch-fmap_%j.out
#SBATCH --error=logs/slurm/sbatch-fmap_%j.err
set -euo pipefail
source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0

run_fmap() {
    local TAG="$1"           # e.g. step1-unshuffled
    local MODEL="$2"         # e.g. gemma-2-2b
    local DATASET="$3"       # e.g. fineweb
    local LAYER="$4"         # e.g. resid_L13
    local SHUF_SUFFIX="$5"   # "" or "_shuffled"
    local REPORT_DIR="$6"    # reports subdir

    local CKPT="results/nlp/$TAG/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k100__seed42${SHUF_SUFFIX}.pt"
    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP $TAG — checkpoint not found: $CKPT"
        ls -la "results/nlp/$TAG/ckpts/" 2>/dev/null || true
        return 0
    fi

    echo ""
    echo ">> [$TAG] feature_map"
    mkdir -p "$REPORT_DIR"
    python -m temporal_crosscoders.NLP.feature_map \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model "$MODEL" \
        --k 100 --T 5 \
        --include-unlabeled --skip-llm-labels \
        --label "$TAG" \
        --output-dir "$REPORT_DIR"
}

# Step 1 — Gemma + FineWeb
run_fmap step1-unshuffled gemma-2-2b fineweb resid_L13 "" \
    reports/step1-gemma-replication
run_fmap step1-shuffled   gemma-2-2b fineweb resid_L13 "_shuffled" \
    reports/step1-gemma-replication

# Step 2 — DeepSeek + GSM8K
run_fmap step2-unshuffled deepseek-r1-distill-llama-8b gsm8k resid_L12 "" \
    reports/step2-deepseek-reasoning
run_fmap step2-shuffled   deepseek-r1-distill-llama-8b gsm8k resid_L12 "_shuffled" \
    reports/step2-deepseek-reasoning

echo ""
echo "=== fmap done ==="
ls -lh reports/step1-gemma-replication/ reports/step2-deepseek-reasoning/
EOF

JOB_ID=$(sbatch --parsable "$JOB")
echo ""
echo "=== sbatch submitted ==="
echo "  job:     $JOB_ID"
echo "  watch:   squeue -u \$USER"
echo "  tail:    tail -f logs/slurm/sbatch-fmap_${JOB_ID}.out"
echo ""
echo "Wall time should be <5 minutes. When it's done:"
echo "  ls -lh reports/step1-gemma-replication/"
echo "  ls -lh reports/step2-deepseek-reasoning/"
echo ""
echo "Close your laptop freely. The sbatch job survives."
