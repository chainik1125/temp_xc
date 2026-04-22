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

# Clean vendor dir before archiving so stale files don't linger.
rm -rf "$HERE/vendor"
mkdir -p "$HERE/vendor"

# Extract only the vendored subtree (excludes Dmitry's configs/results/plots).
git archive "$DMITRY_COMMIT" experiments/separation_scaling/vendor/ \
    | tar -x -C "$HERE" --strip-components=3

# Also grab his top-level eval helpers (ARCH_CONFIGS, evaluate_representation,
# fit_linear_probe_r2, train_topk, encode_all_sae). These live under
# experiments/standard_hmm/ and experiments/transformer_standard_hmm/ in his
# repo layout.
mkdir -p "$HERE/vendor/experiments"
git archive "$DMITRY_COMMIT" experiments/separation_scaling/vendor/experiments/ \
    | tar -x -C "$HERE" --strip-components=3

# Pin the commit sha in a marker file for provenance.
cat > "$HERE/vendor/COMMIT_SHA" <<EOF
Vendored from origin/dmitry @ $COMMIT_SHA
Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "[done] vendored into $HERE/vendor/"
ls "$HERE/vendor/"
