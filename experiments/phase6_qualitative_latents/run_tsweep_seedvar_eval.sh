#!/usr/bin/env bash
# Phase 6.3 seedvar eval: wait for ALL T-sweep seed 1/2 ckpts + training
# processes to finish, then encode + autointerp + probe + pdvar +
# regenerate Pareto figures serially.
#
# Waits for the training process to exit before doing any GPU work, to
# avoid OOM from parallel training + encoding.
#
# Usage:  bash experiments/phase6_qualitative_latents/run_tsweep_seedvar_eval.sh

set -euo pipefail
cd "$(dirname "$0")/../.."
source .envrc 2>/dev/null || true

CKPT_DIR=experiments/phase5_downstream_utility/results/ckpts

echo "[$(date +%H:%M:%S)] waiting for all 6 T-sweep seed {1,2} ckpts"
for SEED in 1 2; do
    for T in 3 10 20; do
        CKPT="${CKPT_DIR}/phase63_track2_t${T}__seed${SEED}.pt"
        while [ ! -f "$CKPT" ]; do
            sleep 60
        done
    done
done

echo "[$(date +%H:%M:%S)] all ckpts present, waiting for training process to exit"
while pgrep -f "train_primary_archs.py.*phase63_track2" > /dev/null; do
    sleep 30
done

# One more 30s safety delay to let GPU mem flush.
sleep 30

echo "[$(date +%H:%M:%S)] training done, GPU free — starting eval batch"

for SEED in 1 2; do
    for T in 3 10 20; do
        ARCH="phase63_track2_t${T}"
        echo "[$(date +%H:%M:%S)] === eval ${ARCH} seed=${SEED} ==="

        TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
            experiments/phase6_qualitative_latents/encode_archs.py \
            --archs "$ARCH" --sets A B random --seed "$SEED" \
            >> logs/phase63/seedvar_eval.log 2>&1

        TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
            experiments/phase6_qualitative_latents/run_autointerp.py \
            --archs "$ARCH" --seeds "$SEED" --concats A B random \
            >> logs/phase63/seedvar_eval.log 2>&1

        TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
            experiments/phase6_qualitative_latents/run_autointerp_pdvar.py \
            --archs "$ARCH" --seeds "$SEED" --concats A B random \
            >> logs/phase63/seedvar_eval.log 2>&1

        for AGG in last_position mean_pool; do
            TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
                experiments/phase5_downstream_utility/probing/run_probing.py \
                --aggregation "$AGG" \
                --run-ids "${ARCH}__seed${SEED}" --skip-baselines \
                >> logs/phase63/seedvar_eval.log 2>&1
        done

        echo "[$(date +%H:%M:%S)] ${ARCH} seed=${SEED}: eval complete"
    done
done

echo "[$(date +%H:%M:%S)] regenerating Pareto figures + T-sweep line plot"

TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
    experiments/phase6_qualitative_latents/plot_pareto_robust.py \
    --agg mean_pool --metric semantic_count \
    --out experiments/phase6_qualitative_latents/results/phase61_pareto_robust.png \
    >> logs/phase63/seedvar_eval.log 2>&1

TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
    experiments/phase6_qualitative_latents/plot_pareto_robust.py \
    --agg mean_pool --metric semantic_count_pdvar \
    --out experiments/phase6_qualitative_latents/results/phase63_pareto_pdvar.png \
    >> logs/phase63/seedvar_eval.log 2>&1

TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
    experiments/phase6_qualitative_latents/plot_t_sweep.py \
    >> logs/phase63/seedvar_eval.log 2>&1

.venv/bin/python scripts/hf_sync.py --go >> logs/phase63/seedvar_eval.log 2>&1

echo "[$(date +%H:%M:%S)] seedvar eval pipeline complete"
