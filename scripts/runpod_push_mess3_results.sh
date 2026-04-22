#!/bin/bash
# runpod_push_mess3_results.sh — Commit + push the Mess3 Mat-TopK-SAE
# ablation results from the pod to the `aniket-runpod` branch, so the
# laptop can `git pull` them down without scp/rsync gymnastics.
#
# What gets committed:
#   experiments/mess3_mat_ablation/results/  — per-cell + combined JSONs
#   experiments/mess3_mat_ablation/plots/    — fig1, fig2 PDFs
#   docs/aniket/experiments/mess3_mat_ablation/plots/  — mirrored PDFs
#
# What is NEVER committed:
#   experiments/mess3_mat_ablation/vendor/   — Dmitry's vendored code (.gitignored)
#   experiments/mess3_mat_ablation/.venv/    — local 3.12 venv
#   logs/                                    — training noise
#
# Usage (on the pod, from repo root):
#   bash scripts/runpod_push_mess3_results.sh
#
# Then on the laptop:
#   git fetch origin && git checkout aniket-runpod
# (or `git checkout aniket-runpod -- experiments/mess3_mat_ablation/`
#  to grab just the results without switching branches)

set -euo pipefail

BRANCH="aniket-runpod"

cd "$(git rev-parse --show-toplevel)"

echo "=== push mess3 ablation results to $BRANCH ==="
echo "  repo:     $(pwd)"
echo "  pod host: $(hostname)"
echo ""

git fetch origin "$BRANCH" || true
current=$(git rev-parse --abbrev-ref HEAD)
if [ "$current" != "$BRANCH" ]; then
    echo ">> switching $current → $BRANCH"
    git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"
else
    echo ">> already on $BRANCH"
fi

git pull --ff-only origin "$BRANCH" || true

echo ""
echo ">> staging results + plots"
paths=(
    experiments/mess3_mat_ablation/results/
    experiments/mess3_mat_ablation/plots/
    docs/aniket/experiments/mess3_mat_ablation/plots/
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

# Safety: refuse to commit anything > 50 MB.
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
    exit 2
fi

if git diff --cached --quiet; then
    echo "   no changes to commit — working tree clean"
    exit 0
fi

stamp=$(date -u +%Y-%m-%dT%H-%M-%SZ)
n_cells=$(find experiments/mess3_mat_ablation/results -name results.json 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo ">> committing"
git commit -m "Mess3 ablation results snapshot $stamp ($n_cells cells)"

echo ""
echo ">> pushing to origin/$BRANCH"
git push origin "$BRANCH"

echo ""
echo "=== done ==="
echo "On your laptop:"
echo "  git fetch origin"
echo "  git checkout $BRANCH -- experiments/mess3_mat_ablation/ docs/aniket/experiments/mess3_mat_ablation/"
echo "  open experiments/mess3_mat_ablation/plots/fig1_gap_recovery_2x2.pdf"
