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

# Per-agent clone-path check on shared pods. If the agent's expected
# clone path doesn't match the cwd, warn loudly — two agents sharing
# /workspace/temp_xc/.git/ collide on index.lock during pull-rebase.
# Han runs `add_agent_clone.sh <agent>` to provision the second clone
# before spawning the agent.
case "$agent" in
    agent_em)      expected_root="/workspace/temp_xc_em" ;;
    agent_steer)   expected_root="/workspace/temp_xc_steer" ;;
    agent_em_h200) expected_root="/workspace/temp_xc" ;;
    agent_em_100k) expected_root="/workspace/temp_xc" ;;
    agent_steer_100k) expected_root="/workspace/temp_xc" ;;
    agent_filler) expected_root="/workspace/temp_xc" ;;
    agent_synth)  expected_root="/workspace/temp_xc" ;;
    agent_hammer) expected_root="/workspace/temp_xc" ;;
    agent_pro)    expected_root="/workspace/temp_xc" ;;
    agent_nlp|agent_back) expected_root="/workspace/temp_xc" ;;
    agent_paper)   expected_root="" ;;   # local; no expected path
    *)             expected_root="" ;;
esac
if [ -n "$expected_root" ] && [ -d /workspace ]; then
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"
    if [ -n "$repo_root" ] && [ "$repo_root" != "$expected_root" ]; then
        echo "[set_agent_env] WARNING: $agent expected clone at $expected_root" >&2
        echo "  but cwd resolves to $repo_root. On a shared pod this means you" >&2
        echo "  are about to share .git/ with another agent — they will collide" >&2
        echo "  on index.lock during pull-rebase." >&2
        echo "  Fix: ask Han to run" >&2
        echo "    bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh $agent" >&2
        echo "  then start over from $expected_root/purified/." >&2
    fi
fi

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
    # ── Single-GPU pods, 100K-iter copies (1× H100 each, ephemeral,
    #    240 GB system RAM, 1 TB /workspace). Each agent is a literal
    #    copy of agent_em / agent_steer respectively but with
    #    n_steps=100_000 instead of the canonical short schedule.
    #    See decisions.md § 13.
    agent_em_100k)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_em_100k
        export TEMP_BENCH_POD_MODE=ephemeral
        ;;
    agent_steer_100k)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_steer_100k
        export TEMP_BENCH_POD_MODE=ephemeral
        ;;
    # ── 8× A40 multi-GPU filler pod (401 GB RAM, 76 CPU, 1 TB ephemeral).
    #    One agent identity ("agent_filler") that runs cells in parallel
    #    via subprocesses pinned to GPUs 0..7 each. The agent's own python
    #    process is pinned to GPU 0 by default; spawn parallel cells with
    #    `bash scripts/run_on_gpu.sh <0..7> -- <cmd>`.
    agent_filler)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_filler
        export TEMP_BENCH_POD_MODE=ephemeral
        ;;
    # ── 8× H100 multi-GPU synthetic-investigation pod (Han 2026-05-06 PM —
    #    upgraded from 8× 5090). 640 GB GPU mem, 1.8 TB system RAM, 224 CPUs.
    #    Same parallel-launch pattern as agent_filler: process pinned to
    #    GPU 0 by default; spawn parallel cells via
    #    `bash scripts/run_on_gpu.sh <0..7> -- <cmd>`. Mission focus:
    #    "Global vs Local" narrative — show TXC dictionaries skew toward
    #    GLOBAL features and per-token SAE dictionaries skew toward LOCAL
    #    features. Includes Dmitry-bench reproductions for honest caveats.
    #    See agents/agent_synth/briefing.md for the full mission.
    agent_synth)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_synth
        export TEMP_BENCH_POD_MODE=ephemeral
        ;;
    # ── 8× RTX PRO 6000 baseline-backfill pod (Han 2026-05-06T23:30 — added
    #    to fill missing tsae_paper / topk_sae cells in C2 Setup A + Setup B).
    #    Blackwell-gen consumer pro card, 96 GB VRAM each. Same parallel-launch
    #    pattern as agent_filler / agent_synth: process pinned to GPU 0 by
    #    default; spawn parallel cells via `bash scripts/run_on_gpu.sh
    #    <0..7> -- <cmd>`. ~108 cells, ~30 min wall.
    #    See agents/agent_hammer/briefing.md for the full mission.
    agent_hammer)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_hammer
        export TEMP_BENCH_POD_MODE=ephemeral
        ;;
    # ── 7× RTX 5090 probing-protocol exploration pod (Han 2026-05-06T23:18 —
    #    spawned to rapidly identify the best probing recipe for TXC archs).
    #    Blackwell-gen consumer card, 32 GB VRAM each = 224 GB total;
    #    989 GB system RAM, 224 CPUs. Same parallel-launch pattern as
    #    agent_filler / agent_synth / agent_hammer: process pinned to
    #    GPU 0 by default; spawn parallel cells via
    #    `bash scripts/run_on_gpu.sh <0..6> -- <cmd>`.
    #    Mission focus: 9-variant pooling/stride sweep on txc_base T=5
    #    first, then validate across 3 seeds + extend to TXC family if a
    #    winner emerges. See agents/agent_pro/briefing.md.
    agent_pro)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_pro
        export TEMP_BENCH_POD_MODE=ephemeral
        ;;
    agent_paper)
        export CUDA_VISIBLE_DEVICES=0
        export AGENT_NAME=agent_paper
        export TEMP_BENCH_POD_MODE=persistent
        ;;

    *)
        echo "unknown agent: $agent" >&2
        echo "known: agent_paper, agent_nlp, agent_em, agent_em_h200, agent_em_100k, agent_steer, agent_steer_100k, agent_back, agent_filler, agent_synth, agent_hammer, agent_pro, a40_helper_gpu2, a40_helper_gpu3" >&2
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
