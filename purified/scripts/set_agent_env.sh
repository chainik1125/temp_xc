#!/usr/bin/env bash
# set_agent_env.sh — pin one agent to one GPU on a shared pod.
#
# Usage:  source scripts/set_agent_env.sh <agent_name>
#
# The mapping (agent → GPU index) duplicates the table in
# purified/agents/README.md. Both must update together when the roster
# changes. The README is documentation; this script is operational.
#
# After sourcing, the agent's CUDA-using process sees exactly one GPU.
# `nvidia-smi` still shows all GPUs (it ignores CUDA_VISIBLE_DEVICES);
# `python -c "import torch; print(torch.cuda.device_count())"` shows 1.
#
# WHY: when two agents share a pod (e.g. agent_nlp + agent_em on the
# 2× H100 pod), they would otherwise both default to cuda:0 and collide.
# CUDA_VISIBLE_DEVICES=N constrains a process to GPU N only.

if [ -z "${1:-}" ]; then
    echo "Usage: source $(basename "$0") <agent_name>" >&2
    return 1 2>/dev/null || exit 1
fi

agent="$1"

case "$agent" in
    # ── 2× H100 pod ─────────────────────────────────────────────────
    agent_nlp)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_nlp
        ;;
    agent_em)
        export CUDA_VISIBLE_DEVICES=1
        export AGENT_NAME=agent_em
        ;;

    # ── 3× A40 pod ──────────────────────────────────────────────────
    agent_steer)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_steer
        ;;
    agent_back)
        export CUDA_VISIBLE_DEVICES=1
        export AGENT_NAME=agent_back
        ;;
    agent_synth)
        export CUDA_VISIBLE_DEVICES=2
        export AGENT_NAME=agent_synth
        ;;

    # ── Single-GPU pods ─────────────────────────────────────────────
    agent_em_h200)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_em_h200
        ;;
    agent_paper)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_paper
        ;;

    *)
        echo "unknown agent: $agent" >&2
        echo "known: agent_paper, agent_nlp, agent_em, agent_em_h200, agent_steer, agent_back, agent_synth" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

# Double-checks
echo "[set_agent_env] AGENT_NAME=$AGENT_NAME"
echo "[set_agent_env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Confirm only one GPU is visible (if torch is available in the env)
if command -v python >/dev/null 2>&1; then
    n=$(python -c "
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print('?')
" 2>/dev/null || echo "?")
    echo "[set_agent_env] torch.cuda.device_count() = $n"
fi
