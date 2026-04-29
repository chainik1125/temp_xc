#!/bin/bash
# run_all.sh — Stage A end-to-end on a fresh A40 pod.
#
# Each step is idempotent (resume by default; pass --force to a step to
# rebuild). Total wall clock target: ~3-6 hours, dominated by the
# steering eval (4 source × 5 magnitudes × 2 targets × 20 prompts × 1500
# tokens ≈ 800 generations).
#
# Prereqs on the pod:
#   - HF_TOKEN, OPENAI_API_KEY, ANTHROPIC_API_KEY exported (or in .env)
#   - pip install -e . (project)  +  uv sync
#   - GPU visible (single A40 is fine)
#
# Usage:
#   bash experiments/ward_backtracking/run_all.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP=experiments/ward_backtracking
LOG=logs/ward_backtracking
mkdir -p "$LOG"

# Source .env if present so HF_TOKEN / OPENAI_API_KEY / ANTHROPIC_API_KEY
# pick up automatically. Match the pattern used by other experiments.
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi
: "${HF_TOKEN:?HF_TOKEN must be set (env or .env)}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (env or .env)}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set (env or .env)}"

# Belt-and-suspenders for fragmented allocator on A40.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Ward 2025 backtracking replication — Stage A ==="
echo "  config: $EXP/config.yaml"
echo "  log dir: $LOG"
echo ""

step() {
    local name="$1"; shift
    echo ">>> [$name] $*"
    "$@" 2>&1 | tee -a "$LOG/${name}.log"
    echo "<<< [$name] done"
    echo ""
}

# 1. Seed prompts (Anthropic — Sonnet 4.6, ~1 minute)
step seed_prompts python -m experiments.ward_backtracking.seed_prompts

# 2. Generate 300 distill traces (vLLM, ~30 minutes on A40)
step generate_traces python -m experiments.ward_backtracking.generate_traces

# 3. GPT-4o sentence labelling (~5-10 minutes wall, ~300 calls)
step label_sentences python -m experiments.ward_backtracking.label_sentences

# 4. Per-offset activations for BOTH base and reasoning (~20 min each)
step collect_offsets python -m experiments.ward_backtracking.collect_offsets

# 5. DoM vectors (instant — pure numpy)
step derive_dom python -m experiments.ward_backtracking.derive_dom

# 6. Steering eval — the long step (~2-3 h)
step steer_eval python -m experiments.ward_backtracking.steer_eval

# 7. Plot
step plot python -m experiments.ward_backtracking.plot

# 8. LLM-judge × keyword-judge F1 validation (cheap, ~$2)
step validate python -m experiments.ward_backtracking.validate

echo "=== Stage A done ==="
echo "Plots:    results/ward_backtracking/plots/"
echo "Results:  results/ward_backtracking/steering_results.json"
echo ""
echo "Next: write summary.md with the Fig 3 reproduction and decide on Stage B."
