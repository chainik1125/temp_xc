#!/usr/bin/env bash
set -euo pipefail

# run-checks.sh - Run documentation quality checks
#
# Usage:
#   ./run-checks.sh       # all checks
#   ./run-checks.sh -m    # markdown lint only
#   ./run-checks.sh -t    # tag check only
#   ./run-checks.sh -l    # link check only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_lint() {
    echo "=== Markdown Lint ==="
    npx markdownlint-cli2 "$SCRIPT_DIR/docs/**/*.md" "!$SCRIPT_DIR/docs/.obsidian/**"
    echo ""
}

run_tags() {
    echo "=== Tag Check ==="
    "$SCRIPT_DIR/check-tags.sh"
    echo ""
}

run_links() {
    echo "=== Link Check ==="
    # Use GitHub CLI token if available (for private repo links)
    if [ -z "${GITHUB_TOKEN:-}" ] && command -v gh >/dev/null 2>&1; then
        if gh auth status >/dev/null 2>&1; then
            GITHUB_TOKEN="$(gh auth token 2>/dev/null || true)"
            export GITHUB_TOKEN
        fi
    fi
    if ! command -v lychee >/dev/null 2>&1; then
        echo "lychee not installed. Install with one of:"
        echo "  cargo install lychee"
        echo "  brew install lychee"
        return 1
    fi
    lychee --accept 200,201,202,204,301,302,307,429,403 --timeout 20 "$SCRIPT_DIR/docs/**/*.md" "$SCRIPT_DIR/"*.md
    echo ""
}

# Parse flags
if [[ $# -eq 0 ]]; then
    run_lint
    run_tags
    run_links
    echo "All checks passed."
    exit 0
fi

case "${1:-}" in
    -m) run_lint ;;
    -t) run_tags ;;
    -l) run_links ;;
    *)
        echo "Usage: $0 [-m|-t|-l]"
        echo "  -m  markdown lint only"
        echo "  -t  tag check only"
        echo "  -l  link check only"
        exit 1
        ;;
esac
