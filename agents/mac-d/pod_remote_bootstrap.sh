#!/usr/bin/env bash
# mac-d remote bootstrap — runs ON the pod (scp this file up, then run it).
# Prereq: tokens already scp'd from the mac to /workspace/.tokens/
#   {gh_token, hf_token, hf_token_datasets}  — NO modal, NO anthropic keys
#   (mac-d STATUS rule; bootstrap_runpod.sh skips absent tokens cleanly).
#
#   bash pod_remote_bootstrap.sh <PIN_SHA>
#
# Clones the (public) repo, detaches at the card's pin, seeds env, runs the
# repo's canonical bootstrap (uv, HF_HOME, MooseFS link-mode), validates.
# Substrate sync (card-specific cache builders) and the actual lane launch
# are NOT here — they come verbatim from the frozen card.
set -euo pipefail
PIN="${1:?usage: pod_remote_bootstrap.sh <pin-sha>}"

export UV_LINK_MODE=copy
export HF_HOME=/workspace/hf_cache

cd /workspace
[ -d temp_xc ] || git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc
git fetch origin arxiv
git checkout --detach "$PIN"
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo "PIN MISMATCH: $(git rev-parse HEAD) != $PIN"; exit 1; }

echo "--- tokens present on pod (values never printed):"
ls -l /workspace/.tokens/ || { echo "tokens missing — scp them first"; exit 1; }
chmod 600 /workspace/.tokens/* 2>/dev/null || true

# canonical bootstrap: loads existing token files, configures gh/hf, uv sync.
# stdin closed => any absent token (anthropic) is skipped, never prompted.
bash scripts/bootstrap_runpod.sh </dev/null

export AGENT_NAME=mac-d
.venv/bin/python run.py validate
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "BOOTSTRAP-DONE pin=$PIN agent=$AGENT_NAME"
