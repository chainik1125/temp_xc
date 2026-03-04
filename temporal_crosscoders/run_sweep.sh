#!/usr/bin/env bash
# run_sweep.sh — Launch the sweep in tmux with GPU parallelism.
#
# Usage:
#   ./run_sweep.sh              # full sweep (sequential, 1 GPU)
#   ./run_sweep.sh --parallel   # split across tmux panes by dataset
#   ./run_sweep.sh --test       # quick 1000-step test
#
# Prerequisites:
#   pip install -r requirements.txt
#   export WANDB_API_KEY=...    (optional, or WANDB_MODE=disabled)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STEPS="${STEPS:-1000000}"
WANDB_MODE="${WANDB_MODE:-online}"
LOG_DIR="${LOG_DIR:-logs}"
VIZ_DIR="${VIZ_DIR:-viz_outputs}"

export WANDB_MODE LOG_DIR VIZ_DIR

# ─── Parse args ──────────────────────────────────────────────────────────────────
PARALLEL=0
TEST=0
for arg in "$@"; do
    case $arg in
        --parallel) PARALLEL=1 ;;
        --test)     TEST=1; STEPS=1000 ;;
    esac
done

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  TEMPORAL CROSSCODER SWEEP                                ║"
echo "║  Steps: $STEPS                                     "
echo "║  wandb: $WANDB_MODE                                "
echo "║  Logs:  $LOG_DIR                                   "
echo "╚════════════════════════════════════════════════════════════╝"

mkdir -p "$LOG_DIR" "$VIZ_DIR"

if [ "$PARALLEL" -eq 1 ]; then
    # ─── Parallel mode: one tmux pane per dataset ────────────────────────────
    SESSION="txcdr_sweep"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" -n main

    # Pane 0: iid
    tmux send-keys -t "$SESSION:main" \
        "cd $SCRIPT_DIR && python sweep.py --dataset iid --steps $STEPS 2>&1 | tee ${LOG_DIR}/iid_stdout.log" C-m

    # Pane 1: markov
    tmux split-window -h -t "$SESSION:main"
    tmux send-keys -t "$SESSION:main.1" \
        "cd $SCRIPT_DIR && python sweep.py --dataset markov --steps $STEPS 2>&1 | tee ${LOG_DIR}/markov_stdout.log" C-m

    # Balance panes
    tmux select-layout -t "$SESSION:main" even-horizontal

    echo ""
    echo "  Launched in tmux session: $SESSION"
    echo "  Attach with:  tmux attach -t $SESSION"
    echo ""
    echo "  When both panes finish, run:  python viz.py"
    echo ""

    # Attach interactively
    tmux attach -t "$SESSION"
else
    # ─── Sequential mode ─────────────────────────────────────────────────────
    echo ""
    echo "Running sequential sweep..."
    python sweep.py --steps "$STEPS" 2>&1 | tee "${LOG_DIR}/sweep_stdout.log"

    echo ""
    echo "Generating visualizations..."
    python viz.py --log-dir "$LOG_DIR" --viz-dir "$VIZ_DIR"

    echo ""
    echo "Done. Plots in ${VIZ_DIR}/"
fi
