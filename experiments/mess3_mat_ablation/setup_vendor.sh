#!/bin/bash
# setup_vendor.sh — pull Dmitry's separation_scaling vendored code into
# this experiment dir via `git archive origin/dmitry`.
#
# Read-only extraction: no worktree, no branch modification. The files
# under ./vendor/ are a byte-for-byte snapshot of Dmitry's vendored
# sae_day pipeline at his pinned commit, dropped in so our runner can
# import them without touching his branch.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMITRY_COMMIT="origin/dmitry"

cd "$REPO_ROOT"

# Ensure Dmitry's branch is fetched.
if ! git rev-parse --verify --quiet "$DMITRY_COMMIT" >/dev/null; then
    echo "[info] fetching $DMITRY_COMMIT"
    git fetch origin dmitry
fi

COMMIT_SHA="$(git rev-parse "$DMITRY_COMMIT")"
echo "[info] vendoring from commit $COMMIT_SHA"

# Clean vendor dir + any stray files that may have landed at $HERE root
# from the previous (broken) version of this script.
rm -rf "$HERE/vendor" "$HERE/src" "$HERE/experiments"
mkdir -p "$HERE/vendor"

# Extract the full vendored subtree into vendor/ (not into $HERE).
# Archive paths look like `experiments/separation_scaling/vendor/<x>/<y>`;
# --strip-components=3 drops that 3-segment prefix so files land at
# `vendor/<x>/<y>` relative to $HERE. Dmitry's vendored layout includes
# both `src/sae_day/` and `experiments/{standard_hmm,transformer_standard_hmm,transformer_nonergodic}/`
# so one extraction covers everything.
git archive "$DMITRY_COMMIT" experiments/separation_scaling/vendor/ \
    | tar -x -C "$HERE/vendor" --strip-components=3

# Pin the commit sha in a marker file for provenance.
cat > "$HERE/vendor/COMMIT_SHA" <<EOF
Vendored from origin/dmitry @ $COMMIT_SHA
Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "[done] vendored into $HERE/vendor/"
ls "$HERE/vendor/"
echo "[info] src/sae_day contents:"
ls "$HERE/vendor/src/sae_day/" 2>/dev/null || echo "  (missing — extraction failed)"
