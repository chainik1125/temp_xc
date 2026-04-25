#!/usr/bin/env bash
# One-shot setup for a fresh runpod pod (or any other GPU box).
#
# Usage on the pod:
#     curl -fsSL https://raw.githubusercontent.com/chainik1125/temp_xc/bill-three-arch-bench/scripts/bootstrap.sh | bash
# or, if the repo is already cloned:
#     bash scripts/bootstrap.sh
#
# What it does:
#   1. Installs uv if not present.
#   2. Clones temp_xc and checks out bill-three-arch-bench (skipped if already
#      inside that repo).
#   3. Runs `uv sync` to install pinned deps (torch + cuda wheels).
#   4. Prints GPU info to confirm CUDA is visible.
#
# It does NOT launch the sweeps automatically — start those manually inside
# tmux so a dropped SSH connection doesn't kill the run.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/chainik1125/temp_xc.git}"
BRANCH="${BRANCH:-bill-three-arch-bench}"
TARGET_DIR="${TARGET_DIR:-temp_xc}"

# 1. Install uv if missing.
if ! command -v uv >/dev/null 2>&1; then
    echo "[bootstrap] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv puts its binary in $HOME/.local/bin (or $HOME/.cargo/bin on older
    # installers). Make sure both are on PATH for the rest of this script.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "[bootstrap] uv: $(command -v uv) ($(uv --version))"

# Persist uv on PATH for future shells in this pod (tmux, ssh, web-terminal
# reopens) so future shells don't get "uv: command not found".
PERSIST_LINE='export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"'
if [[ -f "$HOME/.bashrc" ]] && ! grep -qF "$PERSIST_LINE" "$HOME/.bashrc"; then
    echo "$PERSIST_LINE" >> "$HOME/.bashrc"
    echo "[bootstrap] appended PATH export to ~/.bashrc"
fi

# 2. Get the repo + branch into place.
if [[ -d .git ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[bootstrap] already inside a git repo at $(pwd) — fetching + checking out $BRANCH"
    git fetch origin
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH" || true
else
    if [[ ! -d "$TARGET_DIR" ]]; then
        echo "[bootstrap] cloning $REPO_URL into $TARGET_DIR"
        git clone "$REPO_URL" "$TARGET_DIR"
    fi
    cd "$TARGET_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH" || true
fi
echo "[bootstrap] HEAD: $(git rev-parse --short HEAD)  ($(git rev-parse --abbrev-ref HEAD))"

# 3. Sync deps.
echo "[bootstrap] uv sync"
uv sync

# 4. GPU sanity check.
echo "[bootstrap] GPU sanity check:"
uv run python -c "
import torch
if torch.cuda.is_available():
    print(f'  cuda available: {torch.cuda.get_device_name(0)}')
    print(f'  vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: cuda not available — sweeps will run on CPU and take forever')
"

cat <<'EOF'

[bootstrap] done. To launch the sweeps:

  source $HOME/.local/bin/env       # only needed in fresh shells / new tmux sessions
  cd temp_xc                        # if not already inside

  apt-get update && apt-get install -y tmux
  tmux new -s sweep
  # inside tmux:
  uv run python scripts/run_three_arch_sweep.py 2>&1 | tee three_arch.log
  # ctrl-b d to detach; tmux attach -t sweep to come back

  # in a second tmux pane / window for parallel run:
  uv run python scripts/run_hmm_denoising_sweep.py 2>&1 | tee hmm.log

When sweeps are done, push results back:

  git config user.email "you@example.com"
  git config user.name  "Your Name"
  git checkout -b results-from-runpod
  git add results/
  git commit -m "Sweep results"
  git push -u origin results-from-runpod
EOF
