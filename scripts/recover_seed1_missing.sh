#!/usr/bin/env bash
# Recover the 6 non-MLC archs that hit the OOM cascade in the seed=1 batch.
# Each arch trains in a SEPARATE Python process so per-arch GPU leak
# (Adam state, gen-fn closures, etc.) never compounds.
#
# Apples-to-apples preserved:
#   - PRELOAD_SEQS=24_000 (canonical)
#   - --max_steps 8000 (matches seed=42 + the rest of the seed=1 batch)
#   - same TrainCfg defaults from _train_utils.py (lr, batch, plateau)
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is allocator-only;
#     doesn't change numerics, just reduces fragmentation that contributed
#     to the original cascade.
#
# MLC family (mlc, mlc_contrastive_alpha100_batchtopk, agentic_mlc_08) is
# DELIBERATELY EXCLUDED. Their preload_multilayer (5*14.1 GB = 70 GB on H100)
# leaves only ~9 GB headroom — too tight to guarantee a fair fit. Per
# user's "if it can't be made fair, SKIP it" rule, those go to Agent A's
# H200 across all seeds.

set -e
cd /workspace/temp_xc
mkdir -p logs

export HF_HOME=/workspace/hf_cache
export HF_TOKEN=$(cat /workspace/.tokens/hf_token)
export TQDM_DISABLE=1
export PHASE7_REPO=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=logs/seed1_recovery.log
echo "starting seed=1 recovery at $(date -Is)" > "$LOG"
echo "missing archs to recover (6): high-T TXCDR + 2 anchor cells" >> "$LOG"

# Order matters: do the txcdr T-sweep first (smallest to largest T to fail
# fast on memory issues), then the 2 anchor cells. Each gets a fresh
# Python interpreter via separate `--arch` invocation.
ARCHS=(
    txcdr_t20
    txcdr_t24
    txcdr_t28
    txcdr_t32
    txcdr_t20_kpos100
    phase57_partB_h8_bare_multidistance_t20_kpos100
)

for arch in "${ARCHS[@]}"; do
    echo "" >> "$LOG"
    echo "=== $arch === at $(date -Is)" >> "$LOG"
    /workspace/temp_xc/.venv/bin/python -u \
        -m experiments.phase7_unification.train_phase7 \
        --arch "$arch" --seed 1 --max_steps 8000 \
        >> "$LOG" 2>&1 \
        && echo "  $arch: OK at $(date -Is)" >> "$LOG" \
        || echo "  $arch: FAILED at $(date -Is)" >> "$LOG"
done

echo "" >> "$LOG"
echo "seed=1 recovery done at $(date -Is)" >> "$LOG"
