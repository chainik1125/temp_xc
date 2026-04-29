#!/usr/bin/env bash
# Re-run sweep + plot only, after train has already produced checkpoints.
# Same arch list and sweep params as run_full.sh.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results outputs/data outputs/plots
LOG=results/sweep_plot.log
: > "$LOG"

run() {
  local label="$1"; shift
  echo "=== $label ===" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "  $label OK" | tee -a "$LOG"
  else
    echo "  $label FAILED (exit $?)" | tee -a "$LOG"
    exit 1
  fi
}

export TQDM_DISABLE=1
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

ARCHS=(
  sae_l0_ln1 sae_l0_pre sae_l0_mid sae_l0_post sae_l1_ln1
  tsae_l0_ln1 tsae_l0_pre tsae_l0_mid tsae_l0_post tsae_l1_ln1
  txc_l0_ln1 txc_l0_pre txc_l0_mid txc_l0_post txc_l1_ln1
)

run sweep \
  uv run python run_ablation_sweep.py \
    --top_k 100 --stage2_keep 10 \
    --alphas 0.25 0.5 1.0 1.5 2.0 \
    --delta_util 0.05 \
    --device cuda \
    --archs "${ARCHS[@]}"

run plot \
  uv run python plot_pareto.py

echo "=== SWEEP+PLOT DONE ===" | tee -a "$LOG"
