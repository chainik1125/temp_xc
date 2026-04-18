#!/bin/bash
# runpod_push_results.sh — Commit + push SAEBench eval results from
# the pod to the `aniket-runpod` branch, so the laptop can `git pull`
# them down without scp/rsync/ssh-banner gymnastics.
#
# What gets committed:
#   results/saebench/results/*.jsonl       — aggregate probing records
#   results/saebench/results/saebench_json/ — raw per-run SAEBench JSON
#   results/saebench/logs/                  — training logs (FLOPs etc)
#   results/saebench/preflight/             — preflight validation
#
# What is NEVER committed:
#   results/saebench/ckpts/*.pt             — 26 GB of checkpoints
#   results/saebench/saebench_artifacts/    — cached Gemma forwards (~15 GB)
#   data/cached_activations/                — multi-layer training cache
#   wandb/, logs/ (root)                    — unrelated cruft
#
# Usage (on the pod, from repo root):
#   bash scripts/runpod_push_results.sh
#
# Run this after the orchestrator's eval phase finishes. Re-run after
# any additional evals — it only commits what's changed.

set -euo pipefail

BRANCH="aniket-runpod"

cd "$(git rev-parse --show-toplevel)"

echo "=== push SAEBench results to $BRANCH ==="
echo "  repo:     $(pwd)"
echo "  pod host: $(hostname)"
echo ""

# 1. Ensure we're on the results branch (fetch first so checkout works).
git fetch origin "$BRANCH"
current=$(git rev-parse --abbrev-ref HEAD)
if [ "$current" != "$BRANCH" ]; then
    echo ">> switching $current → $BRANCH"
    git checkout "$BRANCH"
else
    echo ">> already on $BRANCH"
fi

# Pull latest in case laptop pushed anything (unlikely but safe).
git pull --ff-only origin "$BRANCH" || true

# 2. Stage only the result artifacts. `.pt` ckpts and saebench_artifacts
#    are deliberately excluded — they're huge and reproducible.
echo ""
echo ">> staging results/saebench/ (excluding ckpts and cached activations)"
paths=(
    results/saebench/results/
    results/saebench/logs/
    results/saebench/preflight/
)
to_add=()
for p in "${paths[@]}"; do
    if [ -e "$p" ]; then
        to_add+=("$p")
    fi
done
if [ "${#to_add[@]}" -eq 0 ]; then
    echo "   no result paths found — nothing to commit"
    exit 0
fi
git add "${to_add[@]}"

# 3. Safety check: never commit anything over 50 MB (avoids accidental
#    ckpt add if someone reshuffles paths).
big_files=$(git diff --cached --name-only | while read -r f; do
    if [ -f "$f" ]; then
        size=$(wc -c < "$f" 2>/dev/null || echo 0)
        if [ "$size" -gt 52428800 ]; then
            echo "$f ($size bytes)"
        fi
    fi
done)
if [ -n "$big_files" ]; then
    echo ""
    echo "FAIL: staged file(s) > 50 MB, refusing to commit:"
    echo "$big_files"
    echo ""
    echo "Unstage with: git reset HEAD <path>"
    exit 2
fi

# 4. Commit + push. Skip gracefully if nothing changed.
if git diff --cached --quiet; then
    echo "   no changes to commit — working tree clean"
    exit 0
fi

stamp=$(date -u +%Y-%m-%dT%H-%M-%SZ)
n_jsonl_lines=0
if ls results/saebench/results/*.jsonl >/dev/null 2>&1; then
    n_jsonl_lines=$(cat results/saebench/results/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
fi

echo ""
echo ">> committing"
git commit -m "SAEBench results snapshot $stamp ($n_jsonl_lines JSONL records)"

echo ""
echo ">> pushing to origin/$BRANCH"
git push origin "$BRANCH"

echo ""
echo "=== done ==="
echo "On your laptop:"
echo "  git fetch origin"
echo "  git checkout $BRANCH"
echo "  # results now in results/saebench/"
