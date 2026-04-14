#!/bin/bash
# trillium_sweep_nlp.sh — Submit a real-LM architecture sweep over cached
# activations (not the toy Markov sweep in trillium_sweep.sh).
#
#   bash scripts/trillium_sweep_nlp.sh                                          # defaults
#   MODEL=gemma-2-2b DATASET=fineweb LAYER=resid_L13 bash scripts/trillium_sweep_nlp.sh
#   SHUFFLE=1 bash scripts/trillium_sweep_nlp.sh                                # temporal control run
set -euo pipefail

MODEL="${MODEL:-deepseek-r1-distill-llama-8b}"
DATASET="${DATASET:-gsm8k}"
LAYER="${LAYER:-resid_L12}"
ARCHS="${ARCHS:-topk_sae stacked_sae crosscoder}"
K="${K:-50}"
T="${T:-5}"
STEPS="${STEPS:-10000}"
SHUFFLE="${SHUFFLE:-0}"
WALL="${WALL:-08:00:00}"

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm

TAG="$MODEL-$DATASET-$LAYER-k$K-T$T"
[ "$SHUFFLE" = "1" ] && TAG="${TAG}-shuffled"

JOB=$(mktemp "logs/slurm/${TAG}-XXXX.sh")
cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=$WALL
#SBATCH --job-name=txc-$TAG
#SBATCH --output=logs/slurm/${TAG}_%j.out
#SBATCH --error=logs/slurm/${TAG}_%j.err
set -euo pipefail
source "\$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0

SHUF_FLAG=""
[ "$SHUFFLE" = "1" ] && SHUF_FLAG="--shuffle-within-sequence"

python -m src.bench.sweep \\
    --dataset-type cached_activations \\
    --model-name $MODEL \\
    --cached-dataset $DATASET \\
    --cached-layer-key $LAYER \\
    --models $ARCHS \\
    --k $K --T $T --steps $STEPS \\
    --results-dir results/nlp/$TAG \\
    \$SHUF_FLAG
EOF

sbatch "$JOB"
echo "Submitted: $TAG"
echo "Watch: squeue -u \$USER"
echo "Tail:  tail -f logs/slurm/${TAG}_*.out"
echo ""
echo "NOTE: if src.bench.sweep doesn't accept these flags yet, the CLI"
echo "      plumbing still needs wiring (see exploration.md gaps)."
