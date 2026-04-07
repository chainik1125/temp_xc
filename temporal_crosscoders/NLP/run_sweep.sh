#!/bin/bash
# run_sweep.sh — tmux-friendly launcher for the NLP temporal crosscoder pipeline.
#
# Usage:
#   ./run_sweep.sh                    # full pipeline: cache → sweep → viz
#   ./run_sweep.sh --cache-only       # just cache activations
#   ./run_sweep.sh --sweep-only       # just run the sweep (activations must exist)
#   ./run_sweep.sh --viz-only         # just generate visualizations
#   ./run_sweep.sh --quick            # quick test with 1000 steps
#
# Environment variables:
#   WANDB_MODE=online|offline|disabled
#   WANDB_ENTITY=your-wandb-entity
#   NLP_CACHE_DIR=/path/to/cache
#   STEPS=50000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
STEPS="${STEPS:-50000}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

CACHE_ONLY=false
SWEEP_ONLY=false
VIZ_ONLY=false
AUTOINTERP=false
QUICK=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-only)  CACHE_ONLY=true; shift ;;
        --sweep-only)  SWEEP_ONLY=true; shift ;;
        --viz-only)    VIZ_ONLY=true; shift ;;
        --autointerp)  AUTOINTERP=true; shift ;;
        --quick)       QUICK=true; STEPS=1000; shift ;;
        --steps)       STEPS="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  NLP Temporal Crosscoder Pipeline"
echo "  Steps: $STEPS"
echo "  WANDB_MODE: $WANDB_MODE"
echo "  Working dir: $SCRIPT_DIR"
echo "========================================"

# Step 1: Cache activations
if [[ "$SWEEP_ONLY" == false && "$VIZ_ONLY" == false ]]; then
    echo ""
    echo "── Step 1: Caching Gemma 2 2B activations ──"
    if [[ "$QUICK" == true ]]; then
        python cache_activations.py --num-chains 100 --batch-size 4
    else
        python cache_activations.py
    fi

    if [[ "$CACHE_ONLY" == true ]]; then
        echo "Cache complete. Exiting (--cache-only)."
        exit 0
    fi
fi

# Step 2: Run sweep
if [[ "$VIZ_ONLY" == false ]]; then
    echo ""
    echo "── Step 2: Running sweep ($STEPS steps per job) ──"
    python sweep.py --steps "$STEPS"
fi

# Step 3: Visualize
echo ""
echo "── Step 3: Generating visualizations ──"
python viz.py --fit-rho

# Step 4: Autointerp (optional)
if [[ "$AUTOINTERP" == true ]]; then
    echo ""
    echo "── Step 4: Running autointerp ──"
    python autointerp.py
fi

echo ""
echo "========================================"
echo "  Pipeline complete!"
echo "  Logs: NLP/logs/"
echo "  Plots: NLP/viz_outputs/"
echo "  Checkpoints: NLP/checkpoints/"
echo "========================================"
