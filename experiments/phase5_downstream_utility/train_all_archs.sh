#!/usr/bin/env bash
# Sequential training driver for Phase 5 local run.
# Skips archs whose ckpt already exists; continues on per-arch failure.

set -u
REPO="${PHASE5_REPO:-/home/elysium/temp_xc}"
cd "$REPO"

export PHASE5_REPO="$REPO"
export PYTHONPATH="$REPO"
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

CKPTS_DIR="$REPO/experiments/phase5_downstream_utility/results/ckpts"
LOG_DIR="$REPO/experiments/phase5_downstream_utility/results/logs"
mkdir -p "$LOG_DIR"

# Order: small / mid-size first (can run alongside probe cache), big archs last.
ARCHS=(
    "topk_sae"
    "matryoshka_t5"
    "txcdr_shared_dec_t5"
    "txcdr_shared_enc_t5"
    "txcdr_tied_t5"
    "txcdr_pos_t5"
    "txcdr_causal_t5"
    "txcdr_block_sparse_t5"
    "txcdr_lowrank_dec_t5"
    "txcdr_rank_k_dec_t5"
    "temporal_contrastive"
    "tfa_small"
    "tfa_pos_small"
    "txcdr_t5"
    "stacked_t5"
    # Big-VRAM batch below — wait for probe cache to free Gemma's 19 GB
    # before starting. The driver does not enforce that; launch this
    # script AFTER probe cache finishes, or monitor free VRAM separately.
    "mlc"
    "time_layer_crosscoder_t5"
    "txcdr_t20"
    "stacked_t20"
)

for arch in "${ARCHS[@]}"; do
    ckpt="$CKPTS_DIR/${arch}__seed42.pt"
    if [ -f "$ckpt" ]; then
        echo "[$(date +%H:%M:%S)] $arch: ckpt exists, skip"
        continue
    fi
    echo "[$(date +%H:%M:%S)] $arch: training..."
    .venv/bin/python experiments/phase5_downstream_utility/train_primary_archs.py \
        --seeds 42 --max-steps 25000 --archs "$arch" \
        >"$LOG_DIR/train_${arch}.log" 2>&1
    ec=$?
    if [ $ec -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] $arch: OK"
    else
        echo "[$(date +%H:%M:%S)] $arch: FAILED (exit $ec) — see $LOG_DIR/train_${arch}.log"
    fi
done

echo "[$(date +%H:%M:%S)] DONE. Checkpoints in $CKPTS_DIR:"
ls -la "$CKPTS_DIR"
