#!/usr/bin/env bash
# Stage + commit + push one or more em_repl summary docs to dmitry-em-repl branch.
#
# Usage:
#   bash scripts/auto_push_em_repl_summary.sh "phase 1: filled with results" \
#       docs/dmitry/c6_em/2026-05-07_em_repl/phase1_reproduce.md
#
# Idempotent: if no diff, exits 0 quietly. Always pushes the current branch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 2 ]]; then
    echo "usage: $(basename "$0") <commit-message> <doc-path> [<doc-path> ...]" >&2
    exit 2
fi

MSG=$1; shift
PATHS=("$@")

# Sanity: must be on the em_repl branch
BR=$(git symbolic-ref --short HEAD)
if [[ "$BR" != "dmitry-em-repl" ]]; then
    echo "[auto-push] not on dmitry-em-repl (branch: $BR) — aborting." >&2
    exit 3
fi

git add "${PATHS[@]}"

# Skip commit if nothing actually changed
if git diff --cached --quiet; then
    echo "[auto-push] no changes to commit."
    exit 0
fi

git commit -m "$(cat <<EOF
em_repl: $MSG

Auto-push of summary doc(s): ${PATHS[*]}.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

git push origin dmitry-em-repl
echo "[auto-push] pushed: ${PATHS[*]}"
