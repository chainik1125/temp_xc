#!/usr/bin/env bash
# C4 — qualitative-latents component runner wrapper.
#
# Usage:
#   ./experiments/c4_qualitative/run.sh                          # full sweep
#   ./experiments/c4_qualitative/run.sh --smoke                  # 1-cell, n=8
#   ./experiments/c4_qualitative/run.sh --archs tsae_paper       # subset
#
# Pre-conditions:
#   - cwd is purified/
#   - `bash scripts/agent_smoke_test.sh` passes
#   - C3 has trained the same (arch, seed, training_cfg) → checkpoint
#     SHARED via runner.run_cell auto-skip
#   - data/concat_corpora/{concat_A,concat_B,concat_random}.json present
#   - ANTHROPIC_API_KEY env or /workspace/.tokens/anthropic_key
set -euo pipefail
cd "$(dirname "$0")/../.."  # purified/

export TQDM_DISABLE=1

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f /workspace/.tokens/anthropic_key ]; then
    export ANTHROPIC_API_KEY="$(cat /workspace/.tokens/anthropic_key)"
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERR: ANTHROPIC_API_KEY not set and /workspace/.tokens/anthropic_key not found." >&2
    exit 1
fi

exec .venv/bin/python -u -m experiments.c4_qualitative.run "$@"
