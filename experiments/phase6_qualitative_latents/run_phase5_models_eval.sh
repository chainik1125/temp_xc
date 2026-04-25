#!/usr/bin/env bash
# Run encode + autointerp + pdvar + passage-probe for a set of Phase 5 archs
# at their available seeds.
#
# Usage:  bash experiments/phase6_qualitative_latents/run_phase5_models_eval.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source .envrc 2>/dev/null || true

# (arch, seeds) pairs
declare -A TARGETS=(
    ["mlc"]="42 1 2"
    ["txcdr_t5"]="42 1 2"
    ["mlc_contrastive_alpha100"]="42"
    ["agentic_mlc_08_batchtopk"]="42"
)

for ARCH in mlc txcdr_t5 mlc_contrastive_alpha100 agentic_mlc_08_batchtopk; do
    SEEDS="${TARGETS[$ARCH]}"
    for SEED in $SEEDS; do
        echo "[$(date +%H:%M:%S)] === encode $ARCH seed=$SEED ==="
        TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
            experiments/phase6_qualitative_latents/encode_archs.py \
            --archs "$ARCH" --sets A B random --seed "$SEED" \
            >> logs/phase63/phase5_models_eval.log 2>&1
    done
    # autointerp + pdvar + passage probe — call per arch (handles all seeds)
    for SEED in $SEEDS; do
        echo "[$(date +%H:%M:%S)] === autointerp $ARCH seed=$SEED ==="
        TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
            experiments/phase6_qualitative_latents/run_autointerp.py \
            --archs "$ARCH" --seeds "$SEED" --concats A B random \
            >> logs/phase63/phase5_models_eval.log 2>&1
    done
    echo "[$(date +%H:%M:%S)] === pdvar $ARCH ==="
    TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
        experiments/phase6_qualitative_latents/run_autointerp_pdvar.py \
        --archs "$ARCH" --seeds $SEEDS --concats A B random \
        >> logs/phase63/phase5_models_eval.log 2>&1
    echo "[$(date +%H:%M:%S)] === passage-probe $ARCH ==="
    TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
        experiments/phase6_qualitative_latents/run_passage_probe.py \
        --archs "$ARCH" --seeds $SEEDS --concats A B random \
        >> logs/phase63/phase5_models_eval.log 2>&1
    echo "[$(date +%H:%M:%S)] === ${ARCH}: done ==="
done

echo "[$(date +%H:%M:%S)] all 4 Phase 5 archs done"
