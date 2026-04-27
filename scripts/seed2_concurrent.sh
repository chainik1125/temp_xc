#!/usr/bin/env bash
# Seed=2 training, runs CONCURRENTLY with the seed=1 probing chain.
# Strategy:
#   1. Wait for the probing chain to reach stage 3 (run_probing_phase7 —
#      per-task SAE inference, ~15 GB GPU). Stage 1 (Gemma-loaded
#      build_probe_cache) and stage 2 (rebuild_S32, no GPU) are skipped.
#   2. Then sequentially train each seed=2 arch in its OWN Python process
#      via train_phase7 --arch <id> mode. Per-arch process = clean GPU
#      between archs (avoids the cascade-leak we saw in the seed=1 batch).
#   3. Order: smallest-T first → fail-fast on memory issues. If a large-T
#      arch OOMs, the smaller ones will already be done.
#
# VRAM budget on H100 80 GB:
#   probing per-task:   ~15 GB
#   training anchor:    ~14 GB (PRELOAD_SEQS=24_000 fp16)
#   training arch:      5-30 GB (model + Adam + activations, T-dependent)
#   total peak:         34-59 GB → fits with margin for most archs.
#   txcdr_t32 worst:    ~73 GB → squeeze but feasible.
#
# Skips:
#   - MLC family (mlc, mlc_contrastive_alpha100_batchtopk, agentic_mlc_08)
#     — multilayer preload (70 GB) doesn't fit alongside probing on H100;
#     deferred to Agent A's H200 across all seeds (apples-to-apples).
#   - H200-only Group 6 (phase5b_subseq_h8_T32_s5 / T64_s5).
#
# OOM handling: train_phase7 --arch returns nonzero on OOM, the script
# logs it as FAILED and moves to the next arch. Recovery via a similar
# per-process pass after probing finishes.

set -e
cd /workspace/temp_xc
mkdir -p logs

LOG=logs/seed2_concurrent.log
echo "seed=2 concurrent waiter started at $(date -Is)" > "$LOG"

# 1. Wait for probing chain stage 3 to start.
until grep -q "=== 3. run_probing_phase7" \
    /workspace/temp_xc/logs/probing_seed1_chain.log 2>/dev/null; do
    sleep 60
done
echo "probing stage 3 detected at $(date -Is); waiting 90s for first-task stability" >> "$LOG"
sleep 90

export HF_HOME=/workspace/hf_cache
export HF_TOKEN=$(cat /workspace/.tokens/hf_token)
export TQDM_DISABLE=1
export PHASE7_REPO=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "starting seed=2 concurrent training at $(date -Is)" >> "$LOG"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> "$LOG"

# Order: smallest-T → largest-T to fail-fast on memory.
# 35 archs total (= seed=1 done set; no MLC, no H200-only).
ARCHS=(
    topk_sae                                                   # T=1, tiny
    tsae_paper_k500
    tsae_paper_k20
    txcdr_t3
    txcdr_t4
    txcdr_t5
    txcdr_t6
    txcdr_t7
    txcdr_t8
    txcdr_t9
    txcdr_t10
    phase57_partB_h8_bare_multidistance_t3
    phase57_partB_h8_bare_multidistance_t4
    phase57_partB_h8_bare_multidistance_t5
    phase57_partB_h8_bare_multidistance_t6
    phase57_partB_h8_bare_multidistance_t7
    phase57_partB_h8_bare_multidistance_t8
    phase57_partB_h8_bare_multidistance_t9
    txc_bare_antidead_t5
    txc_bare_antidead_t10
    agentic_txc_02
    phase5b_subseq_track2
    phase5b_subseq_h8
    txc_bare_antidead_t20
    txcdr_t12
    txcdr_t14
    txcdr_t16
    txcdr_t18
    tfa_big                                                    # full-sequence; risky
    txcdr_t20
    txcdr_t24
    txcdr_t28
    txcdr_t32
    txcdr_t20_kpos100
    phase57_partB_h8_bare_multidistance_t20_kpos100            # heaviest, last
)

for arch in "${ARCHS[@]}"; do
    echo "" >> "$LOG"
    echo "=== $arch === at $(date -Is)" >> "$LOG"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 >> "$LOG"
    /workspace/temp_xc/.venv/bin/python -u \
        -m experiments.phase7_unification.train_phase7 \
        --arch "$arch" --seed 2 --max_steps 8000 \
        >> "$LOG" 2>&1 \
        && echo "  $arch: OK at $(date -Is)" >> "$LOG" \
        || echo "  $arch: FAILED at $(date -Is) (likely OOM; recovery later)" >> "$LOG"
done

echo "" >> "$LOG"
echo "seed=2 concurrent training done at $(date -Is)" >> "$LOG"
nvidia-smi --query-gpu=memory.used --format=csv >> "$LOG"
