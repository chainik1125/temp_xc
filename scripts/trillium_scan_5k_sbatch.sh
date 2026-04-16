#!/bin/bash
# trillium_scan_5k_sbatch.sh — Run autointerp SCAN phase with --top-features
# 5000 on all 4 crosscoder checkpoints. Sbatched on a compute node (GPU, no
# network). Produces feat_*.json files with top_texts/top_activations
# populated but empty explanations — the login-node explain phase fills those
# in via Claude afterward.
#
# This is the "scale up from 30 to 5000 features" version of
# trillium_scan_sbatch.sh. Expect ~30-60 min total wall time for all 4
# checkpoints (TopKFinder scans more features, batched forward passes
# are GPU-bound).
#
# Usage:
#   bash scripts/trillium_scan_5k_sbatch.sh
#
# Then, once squeue is empty:
#   bash scripts/trillium_explain_5k.sh           # ~$20 on Claude Haiku
#   bash scripts/trillium_sbatch_fmap_5k.sh        # labeled PNGs

set -euo pipefail

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm \
         reports/step1-gemma-replication reports/step2-deepseek-reasoning

TOP_FEATURES="${TOP_FEATURES:-5000}"

JOB=$(mktemp "logs/slurm/scan-5k-sbatch-XXXX.sh")
cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=txc-scan-5k
#SBATCH --output=logs/slurm/scan-5k-sbatch_%j.out
#SBATCH --error=logs/slurm/scan-5k-sbatch_%j.err
set -euo pipefail
source "\$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0

run_scan() {
    local TAG="\$1"
    local MODEL="\$2"
    local DATASET="\$3"
    local LAYER="\$4"
    local SHUF_SUFFIX="\$5"
    local REPORT_DIR="\$6"

    local CKPT="results/nlp/\$TAG/ckpts/crosscoder__\${MODEL}__\${DATASET}__\${LAYER}__k100__seed42\${SHUF_SUFFIX}.pt"
    if [ ! -f "\$CKPT" ]; then
        echo ">> SKIP \$TAG — ckpt not found: \$CKPT"
        return 0
    fi

    echo ""
    echo ">> [\$TAG] autointerp scan phase (top_features=$TOP_FEATURES)"
    python -m temporal_crosscoders.NLP.autointerp \\
        --phase scan \\
        --checkpoint "\$CKPT" \\
        --model crosscoder --subject-model "\$MODEL" \\
        --cached-dataset "\$DATASET" --layer-key "\$LAYER" \\
        --k 100 --T 5 \\
        --label "\$TAG" \\
        --output-dir "\$REPORT_DIR" \\
        --top-features $TOP_FEATURES \\
        --scan-device cuda
}

run_scan step1-unshuffled gemma-2-2b                     fineweb  resid_L13 "" \\
    reports/step1-gemma-replication
run_scan step1-shuffled   gemma-2-2b                     fineweb  resid_L13 "_shuffled" \\
    reports/step1-gemma-replication
run_scan step2-unshuffled deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "" \\
    reports/step2-deepseek-reasoning
run_scan step2-shuffled   deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "_shuffled" \\
    reports/step2-deepseek-reasoning

echo ""
echo "=== scan 5k done ==="
for d in reports/step1-gemma-replication/autointerp/step1-* reports/step2-deepseek-reasoning/autointerp/step2-*; do
    echo "  \$(basename \$d): \$(find \$d -name 'feat_*.json' | wc -l) features"
done
EOF

JOB_ID=$(sbatch --parsable "$JOB")
echo ""
echo "=== scan 5k sbatch submitted ==="
echo "  job:   $JOB_ID"
echo "  top_features: $TOP_FEATURES"
echo "  watch: squeue -u \$USER"
echo "  tail:  tail -f logs/slurm/scan-5k-sbatch_${JOB_ID}.out"
echo ""
echo "When done, run on the login node:"
echo "  bash scripts/trillium_explain_5k.sh"
