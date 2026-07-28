#!/usr/bin/env bash
# set_agent_env.sh — pin one agent to one GPU on a shared pod.
#
# Usage:  source scripts/set_agent_env.sh <agent_name>
#
# The mapping (agent → GPU index) duplicates the table in
# agents/README.md. Both must update together when the roster
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

# All work happens from the repo root. Refuse to source from anywhere
# else — paths in the framework, the .venv location, and `git add -A`
# safety all depend on this convention.
if { [ ! -f pyproject.toml ] || [ ! -d src/temp_bench ]; } && [ "${TEMP_BENCH_ALLOW_ANY_CWD:-}" != "1" ]; then
    echo "[set_agent_env] error: cd into the repo root first." >&2
    echo "  current cwd: $PWD" >&2
    echo "  try:         cd \$(git rev-parse --show-toplevel) && source scripts/set_agent_env.sh $1" >&2
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

    # ── Interim 6× A40 pod (2026-07-25 force majeure; EPHEMERAL disk,
    #    three agents in per-agent clones under /workspace/agents/<id>/
    #    — see briefings/a40-bootstrap.md; push after every batch) ────
    runpod-d)
        export CUDA_VISIBLE_DEVICES=0,1,2
        export AGENT_NAME=runpod-d
        export TEMP_BENCH_POD_MODE=ephemeral
        export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
        ;;
    runpod-e)
        export CUDA_VISIBLE_DEVICES=3,4,5
        export AGENT_NAME=runpod-e
        export TEMP_BENCH_POD_MODE=ephemeral
        export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
        ;;
    runpod-b)
        # CPU-ONLY by design: empty CUDA_VISIBLE_DEVICES hides every GPU
        # so a stray torch call cannot collide with the panel agents.
        export CUDA_VISIBLE_DEVICES=""
        export AGENT_NAME=runpod-b
        export TEMP_BENCH_POD_MODE=ephemeral
        export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
        ;;

    # ── Pod A: 2× H100 pod (2026-07-28; runpod-a GPU 0 + runpod-b GPU 1
    #    in per-agent clones /workspace/agents/<id>/temp_xc) ───────────
    #
    #    ⚑ CPU QUOTA, not core count. `nproc` reports 224 (host) but
    #    /sys/fs/cgroup/cpu.max = 4760000/100000 = **47.6 cores**, and
    #    sched_getaffinity is unmasked at 224 — so the cgroup THROTTLES
    #    rather than masks, and torch's auto-sizing is blind to it
    #    (torch.get_num_threads() = 112 by default = 2.4x oversubscribed).
    #    Every lane launched here MUST bound its own threads. When N
    #    lanes share the pod, override downward: OMP_NUM_THREADS
    #    ~ floor(47.6 / N). Measured by runpod-b 12:46 (0bed01849):
    #    naive 2-lane co-tenancy at defaults is 0.75x ONE lane;
    #    thread-partitioned it is 1.59x.
    runpod-a)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=runpod-a
        export TEMP_BENCH_POD_MODE=ephemeral
        export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
        ;;

    # ── Local Mac autoresearch agents (Modal for GPU — no CUDA
    #    pinning; clones ~/research/projects/agents/<id>/) ─
    mac-a)
        export AGENT_NAME=mac-a
        export TEMP_BENCH_POD_MODE=persistent
        ;;
    mac-b)
        export AGENT_NAME=mac-b
        export TEMP_BENCH_POD_MODE=persistent
        ;;
    mac-c)
        export AGENT_NAME=mac-c
        export TEMP_BENCH_POD_MODE=persistent
        ;;

    # ── ACTMIX shared 3× H100 pod (2026-07-26 evening; 84 CPU /
    #    564 GB RAM / 2 TB persistent volume; TWO agents in per-agent
    #    clones /workspace/agents/<id>/temp_xc — see
    #    briefings/actmix-pod-bootstrap.md) ────────────────────────────
    runpod-1)
        # sparse-probing ablations — the bigger grid gets 2 GPUs
        export CUDA_VISIBLE_DEVICES=0,1
        export AGENT_NAME=runpod-1
        export TEMP_BENCH_POD_MODE=persistent
        export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
        ;;
    runpod-2)
        # EM ablations — 1 GPU
        export CUDA_VISIBLE_DEVICES=2
        export AGENT_NAME=runpod-2
        export TEMP_BENCH_POD_MODE=persistent
        export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
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
        echo "known: agent_paper, agent_nlp, agent_em, agent_em_h200, agent_steer, agent_back, a40_helper_gpu2, a40_helper_gpu3, runpod-d, runpod-e, runpod-b (interim A40 pod), mac-a, mac-b (local Mac)" >&2
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
