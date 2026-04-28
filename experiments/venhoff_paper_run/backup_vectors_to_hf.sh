#!/bin/bash
# backup_vectors_to_hf.sh — Push the Phase-2 steering vectors to a
# HuggingFace repo so we don't lose ~3-4h of training per arch if the
# pod's volume goes away.
#
# What gets uploaded:
#   vendor/thinking-llms-interp/train-vectors/results/vars/optimized_vectors/
#       *_tempxc.pt   ← TempXC fresh-trained vectors (16 idxs)
#       *_mlc.pt      ← MLC fresh-trained vectors (16 idxs)
#       *.pt          ← Venhoff's shipped SAE vectors (16 idxs, bare)
#       *.json        ← per-vector metadata sidecars
#
# Prerequisites on the pod:
#   - HF_TOKEN exported in env (already in /workspace/.env)
#   - huggingface_hub installed (.venv has it)
#   - Repo created beforehand (private recommended); see HF_REPO env var
#
# Usage on the pod, from repo root:
#   HF_REPO=aniket-desh/temp_xc-venhoff-vectors \
#       bash experiments/venhoff_paper_run/backup_vectors_to_hf.sh
#
# To restore on a fresh pod:
#   huggingface-cli download aniket-desh/temp_xc-venhoff-vectors \
#       --local-dir vendor/thinking-llms-interp/train-vectors/results/vars/optimized_vectors \
#       --repo-type model

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

HF_REPO="${HF_REPO:?must set HF_REPO=org/name (e.g. aniket-desh/temp_xc-venhoff-vectors)}"
REPO_TYPE="${REPO_TYPE:-model}"

VECTOR_DIR="${VECTOR_DIR:-vendor/thinking-llms-interp/train-vectors/results/vars/optimized_vectors}"
if [[ ! -d "$VECTOR_DIR" ]]; then
    echo "FAIL: vector dir $VECTOR_DIR not found"
    exit 2
fi

# Source HF_TOKEN if present in .env.
if [[ -f .env ]] && [[ -z "${HF_TOKEN:-}" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
: "${HF_TOKEN:?HF_TOKEN must be set (env or .env)}"

echo "=== HF backup ==="
echo "  repo:       $HF_REPO ($REPO_TYPE)"
echo "  vector_dir: $VECTOR_DIR"
n_files=$(find "$VECTOR_DIR" -maxdepth 1 -type f \( -name "*.pt" -o -name "*.json" \) | wc -l | tr -d ' ')
total_mb=$(du -sm "$VECTOR_DIR" 2>/dev/null | awk '{print $1}')
echo "  files:      $n_files   (~${total_mb} MB total)"
echo ""

python - <<PY
import os, sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

repo = os.environ["HF_REPO"]
repo_type = os.environ.get("REPO_TYPE", "model")
token = os.environ["HF_TOKEN"]
src = Path(os.environ["VECTOR_DIR"]).resolve()

api = HfApi(token=token)
try:
    create_repo(repo, repo_type=repo_type, private=True, exist_ok=True, token=token)
    print(f"[ok] repo ready: {repo}")
except Exception as e:
    print(f"[warn] create_repo: {e}", file=sys.stderr)

print(f"[info] uploading folder {src} → {repo}")
api.upload_folder(
    folder_path=str(src),
    repo_id=repo,
    repo_type=repo_type,
    commit_message=f"Phase-2 steering vectors snapshot ({sum(1 for _ in src.glob('*.pt'))} .pt files)",
    allow_patterns=["*.pt", "*.json"],
)
print(f"[ok] upload complete")
print(f"     https://huggingface.co/{repo}")
PY
