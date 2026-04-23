#!/usr/bin/env bash
# Waits for the triangle-training process to finish, then chains
# the post-training pipeline + autointerp for every new cell.
#
# Usage: this script is launched once training is in progress. It
# polls for the triangle-training bash ancestor to exit, then runs:
#   1. run_phase61_post_train.sh (encoding all new seed × concat cells,
#      plus probing Cycle F + 2x2 cell)
#   2. run_autointerp.py over every new (arch × seed × concat) cell
#   3. assemble_phase61_table.py — generates the headline tables
#
# TRAIN_PID env var or a sentinel-file check identifies "training is done".

set -euo pipefail
cd /workspace/temp_xc
source /workspace/temp_xc/.envrc
mkdir -p logs

TRAIN_LOG="logs/phase61_triangle_train.log"
POST_LOG="logs/phase61_post_train.log"
AUTO_LOG="logs/phase61_autointerp_new_cells.log"
ASM_LOG="logs/phase61_assemble.log"

wait_for_training() {
  echo "[chain] waiting for triangle training to complete..."
  while true; do
    if ! pgrep -f "run_phase61_triangle_train" >/dev/null \
         && ! pgrep -f "train_primary_archs.*batchtopk" >/dev/null \
         && ! pgrep -f "train_primary_archs.*tsae_paper" >/dev/null \
         && ! pgrep -f "train_tsae_paper.py" >/dev/null; then
      # One last check: the log must contain the final "DONE" marker.
      if grep -q "PHASE 6.1 TRAINING DONE" "$TRAIN_LOG" 2>/dev/null; then
        echo "[chain] training DONE marker found"
        return 0
      fi
      # If not, assume it failed. Return failure.
      echo "[chain] training processes absent BUT no DONE marker — assuming failure"
      echo "        grep for errors:"
      grep -iE "Error|Traceback|OOM|Killed" "$TRAIN_LOG" | tail -20
      return 1
    fi
    sleep 60
  done
}

wait_for_training

echo "[chain] === STAGE 2: post_train (encode + probe) ==="
bash experiments/phase6_qualitative_latents/run_phase61_post_train.sh \
  > "$POST_LOG" 2>&1

echo "[chain] === STAGE 3: autointerp on new cells ==="
# All cells: 10 archs × {seed=42 concat=random} (seed=42 existing on A,B)
#          + 3 triangle archs × seeds {1, 2} × 3 concats {A, B, random}
#          + 2x2 cell (agentic_txc_12_bare_batchtopk) seed=42 × {A, B}
# The script is idempotent: it skips cells whose z_cache is missing.

# a) seed=42 concat=random for all archs (z caches just backfilled)
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/run_autointerp.py \
  --archs agentic_txc_02 agentic_txc_02_batchtopk agentic_txc_09_auxk \
          agentic_txc_10_bare agentic_txc_11_stack \
          agentic_txc_12_bare_batchtopk \
          agentic_mlc_08 tsae_paper tsae_ours tfa_big \
  --seeds 42 --concats random >> "$AUTO_LOG" 2>&1

# b) 2x2 cell seed=42 on concat_A, concat_B (new arch, new cells)
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/run_autointerp.py \
  --archs agentic_txc_12_bare_batchtopk \
  --seeds 42 --concats A B >> "$AUTO_LOG" 2>&1

# c) Triangle seeds {1, 2} across all 3 concats
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/run_autointerp.py \
  --archs agentic_txc_02_batchtopk agentic_txc_12_bare_batchtopk tsae_paper \
  --seeds 1 2 --concats A B random >> "$AUTO_LOG" 2>&1

echo "[chain] === STAGE 4: assemble table ==="
.venv/bin/python experiments/phase6_qualitative_latents/assemble_phase61_table.py \
  > "$ASM_LOG" 2>&1

echo "[chain] === STAGE 5: Track 2 seed-variance (TXC-family winner) ==="
# Track 2 is the TXC-family qualitative winner at seed=42. Get 3-seed
# variance on its probing + qualitative numbers before Phase 6.2.
bash experiments/phase6_qualitative_latents/run_track2_seedvar.sh \
  > logs/phase61_track2_seedvar.log 2>&1

echo "[chain] === STAGE 6: regenerate paper figures + tables ==="
.venv/bin/python experiments/phase6_qualitative_latents/assemble_phase61_table.py \
  >> "$ASM_LOG" 2>&1
.venv/bin/python experiments/phase6_qualitative_latents/plot_rigorous_metric_headline.py \
  >> "$ASM_LOG" 2>&1
.venv/bin/python experiments/phase6_qualitative_latents/plot_pareto_tradeoff.py \
  --agg last_position >> "$ASM_LOG" 2>&1
.venv/bin/python experiments/phase6_qualitative_latents/plot_pareto_tradeoff.py \
  --agg mean_pool --out experiments/phase6_qualitative_latents/results/phase61_pareto_tradeoff_mean_pool.png \
  >> "$ASM_LOG" 2>&1

echo "[chain] === STAGE 7: launch Phase 6.2 autoresearch loop ==="
# C3 (highest-prior) runs first. Covers C3, C1, C2, C5, C6 back to back.
bash experiments/phase6_2_autoresearch/run_phase62_loop.sh \
  > logs/phase62_loop.log 2>&1

echo "[chain] === FULL PIPELINE DONE: $(date -u) ==="
