#!/usr/bin/env bash
set -euo pipefail

# get-tags.sh - Extract frontmatter tags from a markdown file
# Usage: ./get-tags.sh <file>

if [[ $# -lt 1 || ! -f "$1" ]]; then
    echo "Usage: $0 <file>" >&2
    exit 1
fi

FILE="$1"

# Check if file starts with ---
if ! head -1 "$FILE" | grep -q '^---$'; then
    exit 0
fi

# Extract frontmatter and parse tags
sed -n '1,/^---$/p' "$FILE" | tail -n +2 | sed '$d' | awk '
    /^tags:/ {
        in_tags = 1
        # Handle inline array: tags: [foo, bar]
        if (match($0, /\[.*\]/)) {
            content = substr($0, RSTART+1, RLENGTH-2)
            gsub(/["\047]/, "", content)
            n = split(content, arr, /[[:space:]]*,[[:space:]]*/)
            for (i = 1; i <= n; i++) {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", arr[i])
                if (arr[i] != "") print arr[i]
            }
            in_tags = 0
        }
        next
    }
    in_tags && /^[[:space:]]*-[[:space:]]+/ {
        tag = $0
        sub(/^[[:space:]]*-[[:space:]]+/, "", tag)
        gsub(/["\047]/, "", tag)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", tag)
        if (tag != "") print tag
        next
    }
    in_tags && /^[a-zA-Z]/ { in_tags = 0 }
    in_tags && /^---$/ { in_tags = 0 }
'
