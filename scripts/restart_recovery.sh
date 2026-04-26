#!/usr/bin/env bash
# Phase 7 RunPod restart-recovery — one-command bootstrap.
#
# Run this immediately after a RunPod stop/start that wipes /home/appuser
# but leaves /workspace intact. Idempotent — safe to re-run.
#
# Usage (from any working dir):
#
#     bash /workspace/temp_xc/scripts/restart_recovery.sh
#
# Restores:
#   1. uv binary at ~/.local/bin/uv (re-installs if /home was wiped)
#   2. ~/.bashrc env vars: HF_HOME, UV_LINK_MODE, PATH
#   3. ~/.claude → /workspace/claude_home symlink (via bootstrap_claude.sh
#      if /workspace/claude_home exists — preserves agent conversation state)
#   4. Repo .venv at /workspace/temp_xc/.venv via `uv sync`
#      Also nukes a half-broken venv first (the MooseFS "lib not empty"
#      pattern that breaks if a prior `uv sync` was interrupted).
#
# What this script does NOT restore:
#   - In-flight long-running jobs (cache builds, training loops). Use the
#     Phase 7 drivers' resumable-by-default behaviour: re-run them and
#     they pick up where they left off (per-task .npz files, per-layer
#     .progress.json files, training_index.jsonl appends).

set -eu

REPO=/workspace/temp_xc
TOKENS_DIR=/workspace/.tokens
HF_HOME=/workspace/hf_cache

echo "=== Phase 7 restart recovery ==="

# 1. uv
if ! [ -x "$HOME/.local/bin/uv" ]; then
    echo "[uv] re-installing (no binary at ~/.local/bin/uv)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
else
    echo "[uv] already present: $(~/.local/bin/uv --version)"
fi
export PATH="$HOME/.local/bin:$PATH"

# 2. ~/.bashrc env vars
for line in \
    "export HF_HOME=$HF_HOME" \
    "export UV_LINK_MODE=copy" \
    "export PATH=\$HOME/.local/bin:\$PATH"; do
    if ! grep -qF "$line" "$HOME/.bashrc" 2>/dev/null; then
        echo "$line" >> "$HOME/.bashrc"
        echo "[bashrc] added: $line"
    fi
done
mkdir -p "$HF_HOME"
export HF_HOME UV_LINK_MODE=copy

# 3. ~/.claude symlink (Claude Code conversation state)
if [ -d /workspace/claude_home ] && [ -x "$REPO/scripts/bootstrap_claude.sh" ]; then
    bash "$REPO/scripts/bootstrap_claude.sh"
elif [ -d /workspace/claude_home ]; then
    # Manual fallback if bootstrap script is absent.
    if [ ! -L "$HOME/.claude" ]; then
        rm -rf "$HOME/.claude" 2>/dev/null || true
        ln -s /workspace/claude_home "$HOME/.claude"
        echo "[claude] symlinked $HOME/.claude -> /workspace/claude_home"
    fi
else
    echo "[claude] no /workspace/claude_home (no agent state to restore)"
fi

# 4. venv
cd "$REPO"
VENV_PYTHON="$REPO/.venv/bin/python"
if [ -d "$REPO/.venv" ] && ! [ -f "$VENV_PYTHON" ]; then
    echo "[venv] half-broken (.venv exists but no python binary) — nuking"
    rm -rf "$REPO/.venv"
fi
echo "[uv] sync (this can take a few minutes if torch needs reinstalling)..."
~/.local/bin/uv sync 2>&1 | tail -5
echo "[uv] second sync (idempotency check)..."
~/.local/bin/uv sync 2>&1 | tail -3

# 5. Verify
echo "=== Verify ==="
"$VENV_PYTHON" -c "
import torch, numpy as np
print(f'  python: $($VENV_PYTHON --version)')
print(f'  numpy:  {np.__version__}')
print(f'  torch:  {torch.__version__}, cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:    {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)')
"
echo "  HF_HOME=$HF_HOME  UV_LINK_MODE=$UV_LINK_MODE"

echo "=== Done. Resume work via 'cd $REPO && source ~/.bashrc' ==="
