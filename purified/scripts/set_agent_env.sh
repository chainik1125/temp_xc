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

# All paper work happens from inside purified/. Refuse to source from
# anywhere else — paths in the framework, the .venv location, and
# `git add -A` safety all depend on this convention.
if [ "$(basename "$PWD")" != "purified" ] && [ "${TEMP_BENCH_ALLOW_ANY_CWD:-}" != "1" ]; then
    echo "[set_agent_env] error: cd into purified/ first." >&2
    echo "  current cwd: $PWD" >&2
    echo "  try:         cd \$(git rev-parse --show-toplevel)/purified && source scripts/set_agent_env.sh $1" >&2
    return 1 2>/dev/null || exit 1
fi

agent="$1"

case "$agent" in
    # ── 2× H100 pod (1 TB persistent /workspace) ────────────────────
    agent_nlp)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_nlp
        export TEMP_BENCH_POD_MODE=persistent
        ;;
    agent_em)
        export CUDA_VISIBLE_DEVICES=1
        export AGENT_NAME=agent_em
        export TEMP_BENCH_POD_MODE=persistent
        ;;

    # ── 4× A40 pod (ephemeral storage — auto-push to HF) ────────────
    agent_steer)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_steer
        export TEMP_BENCH_POD_MODE=ephemeral
        ;;
    agent_back)
        export CUDA_VISIBLE_DEVICES=1
        export AGENT_NAME=agent_back
        export TEMP_BENCH_POD_MODE=ephemeral
        ;;
    # ── Spare A40 GPU slots (no named agent — used by lead agents for
    #    launching parallel cell processes; AGENT_NAME inherits from
    #    the parent shell so leaderboard rows are still attributable.) ─
    a40_helper_gpu2)
        export CUDA_VISIBLE_DEVICES=2
        export AGENT_NAME="${AGENT_NAME:-a40_helper_gpu2}"
        export TEMP_BENCH_POD_MODE=ephemeral
        ;;
    a40_helper_gpu3)
        export CUDA_VISIBLE_DEVICES=3
        export AGENT_NAME="${AGENT_NAME:-a40_helper_gpu3}"
        export TEMP_BENCH_POD_MODE=ephemeral
        ;;

    # ── Single-GPU pods ─────────────────────────────────────────────
    agent_em_h200)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_em_h200
        export TEMP_BENCH_POD_MODE=persistent
        ;;
    agent_paper)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_paper
        export TEMP_BENCH_POD_MODE=persistent
        ;;

    *)
        echo "unknown agent: $agent" >&2
        echo "known: agent_paper, agent_nlp, agent_em, agent_em_h200, agent_steer, agent_back, a40_helper_gpu2, a40_helper_gpu3" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

# Double-checks
echo "[set_agent_env] AGENT_NAME=$AGENT_NAME"
echo "[set_agent_env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[set_agent_env] TEMP_BENCH_POD_MODE=$TEMP_BENCH_POD_MODE"

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
