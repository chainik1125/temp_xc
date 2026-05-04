#!/usr/bin/env bash
# run_on_gpu.sh — pin a subprocess to a specific GPU.
#
# Usage:
#     bash scripts/run_on_gpu.sh <gpu_idx> -- <command...>
#
# Examples:
#     bash scripts/run_on_gpu.sh 0 -- python -m experiments.c6_em.run --seeds 1
#     bash scripts/run_on_gpu.sh 2 -- .venv/bin/python script.py
#
# Sets ``CUDA_VISIBLE_DEVICES=<gpu_idx>`` in the subprocess env and
# execs the command. The current shell's env is unchanged.
#
# Convention reminder (PROTOCOL.md § 12 *GPU sharing*):
# - Your primary is your default. Use this wrapper to launch on a
#   peer's GPU only when:
#   1. Peer's briefing's "Current state" says they're idle, AND
#   2. ``nvidia-smi`` confirms the GPU is free (memory < ~1 GB used).
# - Update YOUR briefing's "Current state" with "borrowing GPU N until
#   ETA HH:MM UTC" before kicking off any long borrowed-GPU work.

set -eu

if [ "${1:-}" = "" ] || [ "${2:-}" != "--" ]; then
    echo "Usage: $(basename "$0") <gpu_idx> -- <command...>" >&2
    echo "  e.g. $(basename "$0") 0 -- python -m experiments.c6_em.run" >&2
    exit 1
fi

GPU_IDX="$1"
shift 2  # drop gpu_idx + the "--" separator

case "$GPU_IDX" in
    [0-9]|[0-9][0-9]) ;;
    *) echo "[run_on_gpu] gpu_idx must be a non-negative integer; got '$GPU_IDX'" >&2; exit 1 ;;
esac

# Quick safety: warn (not block) if the GPU appears occupied.
if command -v nvidia-smi >/dev/null 2>&1; then
    used_mb=$(nvidia-smi --id="$GPU_IDX" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "0")
    if [ -n "$used_mb" ] && [ "$used_mb" -gt 1024 ]; then
        echo "[run_on_gpu] WARNING: GPU $GPU_IDX has ${used_mb} MB used. Consider checking" >&2
        echo "  'nvidia-smi' and the peer agent's briefing before proceeding." >&2
        echo "  Continuing in 3 s — Ctrl-C to abort." >&2
        sleep 3
    fi
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDX"
echo "[run_on_gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES exec: $*" >&2
exec "$@"
