#!/usr/bin/env bash
# Launch the seed=1 batch on Agent C's H100. Run from repo root.
# Auto-detaches via nohup; check progress in logs/seed1_h100_batch.log.
set -e

cd /workspace/temp_xc

mkdir -p logs

# Sanity: the activation cache must exist for all 5 layers.
for L in 10 11 12 13 14; do
    f=data/cached_activations/gemma-2-2b/fineweb/resid_L${L}.npy
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f — run src.data.nlp.cache_activations first." >&2
        exit 2
    fi
    sz=$(stat -c %s "$f")
    if [ "$sz" -lt 14100000000 ]; then
        echo "ERROR: $f is only $sz bytes (<14.1 GB); cache may be incomplete." >&2
        exit 2
    fi
done

ARCHS=experiments/phase7_unification/case_studies/seed1_h100_archs.txt
if [ ! -f "$ARCHS" ]; then
    echo "ERROR: missing arch list at $ARCHS" >&2
    exit 2
fi

export HF_HOME=/workspace/hf_cache
export HF_TOKEN=$(cat /workspace/.tokens/hf_token)
export TQDM_DISABLE=1
export PHASE7_REPO=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG=logs/seed1_h100_batch.log
echo "starting seed=1 batch at $(date -Is)" > "$LOG"

nohup /workspace/temp_xc/.venv/bin/python -u \
    -m experiments.phase7_unification.train_phase7 \
    --canonical --seed 1 --max_steps 8000 \
    --archs "$ARCHS" \
    >> "$LOG" 2>&1 &
PID=$!
disown
echo "launched pid=$PID; log=$LOG"
echo "$PID" > logs/seed1_h100_batch.pid
