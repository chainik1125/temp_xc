#!/bin/bash
# runpod_venhoff_paper_run.sh — one-shot launcher for the full
# paper-budget Venhoff reasoning-eval on the Llama-3.1-8B × DeepSeek-R1-
# Distill-Llama-8B cell.
#
# Wraps runpod_venhoff_launch.sh with the canonical publication-run
# settings baked in:
#   - MODE=hybrid                  → MATH500 Gap Recovery (post-2026-04-20 pivot)
#   - ARCHES="sae tempxc mlc"      → full three-way comparison
#   - STEER_TOP_K=15               → all clusters trained (match paper)
#   - SteeringConfig paper budget  → max_iters=50, n_training=2048, minibatch=6
#                                    (already the default in steering.py)
#   - NUM_GPUS_STEERING / HYBRID   → auto-detected from nvidia-smi
#   - nohup + timestamped log      → survives ssh drop, re-tailable any time
#
# Usage (from the pod, inside the repo root):
#   bash scripts/runpod_venhoff_paper_run.sh
#
# After launch, the script prints the PID and the log path, then tails
# the log. Ctrl-C exits the tail; the run keeps going. Re-tail with:
#   tail -f logs/venhoff_paper_*.log
#
# Override anything via env vars:
#   STEER_TOP_K=5 bash scripts/runpod_venhoff_paper_run.sh    # quicker
#   ARCHES="tempxc" bash scripts/runpod_venhoff_paper_run.sh  # single-arch
#   HYBRID_N_TASKS=100 bash scripts/runpod_venhoff_paper_run.sh  # small Phase 3

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Auto-detect GPU count — falls back to 1 if nvidia-smi is missing.
N_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
if [[ "$N_GPUS" -lt 1 ]]; then N_GPUS=1; fi

mkdir -p logs
LOG="logs/venhoff_paper_$(date +%Y%m%d_%H%M).log"

echo "=== Venhoff paper-budget run ==="
echo "  GPUs detected:   $N_GPUS"
echo "  log:             $LOG"
echo "  arches:          ${ARCHES:-sae tempxc mlc}"
echo "  cluster top-k:   ${STEER_TOP_K:-15}  (all clusters = 15)"
echo "  steering budget: max_iters=50, n_training=2048, minibatch=6  (Venhoff App C.1)"
echo ""

# Sanity-check: the SAE arch should resume from Venhoff's shipped vectors.
# Count how many of the 16 expected files (bias + idx0..14) are present.
VENHOFF_SV_DIR="vendor/thinking-llms-interp/train-vectors/results/vars/optimized_vectors"
SV_PRESENT=$(ls "$VENHOFF_SV_DIR"/llama-3.1-8b_bias.pt \
               "$VENHOFF_SV_DIR"/llama-3.1-8b_idx{0..14}.pt 2>/dev/null | wc -l)
if [[ "$SV_PRESENT" -eq 16 ]]; then
    echo "  SAE Phase 2:     SKIPPED (all 16 Venhoff vectors found — will resume)"
elif [[ "$SV_PRESENT" -gt 0 ]]; then
    echo "  SAE Phase 2:     PARTIAL ($SV_PRESENT/16 Venhoff vectors found; the rest will train)"
else
    echo "  SAE Phase 2:     WILL TRAIN (no Venhoff vectors found at $VENHOFF_SV_DIR)"
fi
echo ""

# Launch under nohup so ssh disconnects don't kill the job.
nohup bash -c "
  MODE=hybrid \
  ARCHES=\"${ARCHES:-sae tempxc mlc}\" \
  STEER_TOP_K=${STEER_TOP_K:-15} \
  NUM_GPUS_STEERING=${NUM_GPUS_STEERING:-$N_GPUS} \
  NUM_GPUS_HYBRID=${NUM_GPUS_HYBRID:-$N_GPUS} \
  HYBRID_N_TASKS=${HYBRID_N_TASKS:-500} \
  bash scripts/runpod_venhoff_launch.sh
" > "$LOG" 2>&1 &

PID=$!
echo "launched PID=$PID"
echo "tailing $LOG (ctrl-C to detach; run continues in background)"
echo ""
sleep 2
tail -f "$LOG"
