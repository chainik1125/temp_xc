#!/usr/bin/env bash
# Waits for the Phase 6.1 full-chain pipeline to complete, then
# launches the Phase 6.2 autoresearch loop. Intended to be run
# in background:
#
#   bash experiments/phase6_2_autoresearch/launch_after_phase61.sh \
#     > logs/phase62_launcher.log 2>&1 &

set -euo pipefail
cd /workspace/temp_xc
source /workspace/temp_xc/.envrc

CHAIN_LOG="logs/phase61_full_chain.log"
DONE_MARKER="FULL PIPELINE DONE"

echo "[launcher] $(date -u) waiting for Phase 6.1 pipeline done marker"
while true; do
  if grep -q "$DONE_MARKER" "$CHAIN_LOG" 2>/dev/null; then
    echo "[launcher] $(date -u) Phase 6.1 complete"
    break
  fi
  # Abort if chain process is gone AND no done marker → pipeline
  # failed. Don't silently proceed to Phase 6.2 on a broken baseline.
  if ! pgrep -f "run_phase61_full_chain\|run_phase61_post_train\|run_phase61_triangle_train" \
       >/dev/null; then
    if ! grep -q "$DONE_MARKER" "$CHAIN_LOG" 2>/dev/null; then
      echo "[launcher] Phase 6.1 chain gone without DONE marker — aborting"
      exit 1
    fi
  fi
  sleep 120
done

echo "[launcher] $(date -u) launching Phase 6.2 autoresearch loop"
bash experiments/phase6_2_autoresearch/run_phase62_loop.sh \
  > logs/phase62_loop.log 2>&1
echo "[launcher] $(date -u) Phase 6.2 loop finished"
