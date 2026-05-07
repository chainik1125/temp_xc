#!/usr/bin/env bash
# wasteland_refresh.sh — keep cross-branch wasteland reads fresh.
#
# Other-branch contributors (case-em-nanda, case-backtracking) are still
# pushing updates. We never merge their branches into `final` — the paper
# would inherit conflicts and stale snapshots. Instead, agents read
# their files via:
#
#     git show origin/case-em-nanda:docs/legacy/results/em_features/em_nanda_results_paper.md
#     git show origin/case-backtracking:docs/legacy/experiments/ward_backtracking/handoff_neurips_push.md
#
# This script just runs a fast `git fetch` so the `origin/<branch>:path`
# reads always resolve to the latest pushed state.
#
# Idempotent. Safe to run on every agent session. Costs ~5 sec.

set -eu

cd "$(git rev-parse --show-toplevel)"

echo "[wasteland_refresh] fetching all remotes…"
git fetch --all --prune --quiet

# Quick health check — list the head commits of the branches agents read
for ref in origin/case-em-nanda origin/case-backtracking origin/wasteland-canonical origin/main; do
    if git rev-parse "$ref" >/dev/null 2>&1; then
        commit=$(git log -1 --format='%h %s' "$ref" | head -c 100)
        printf "  %-38s %s\n" "$ref" "$commit"
    else
        echo "  $ref  (not found)"
    fi
done

echo "[wasteland_refresh] done"
