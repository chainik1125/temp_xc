#!/usr/bin/env bash
set -euo pipefail

# check-tags.sh - Validate tags in docs/ markdown files
#
# Checks:
# 1. All markdown files in docs/ have at least one tag
# 2. All tags used are listed in docs/Tags.md (or parent for nested tags)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GET_TAGS="$SCRIPT_DIR/get-tags.sh"
TAGS_FILE="$SCRIPT_DIR/docs/Tags.md"
DOCS_DIR="$SCRIPT_DIR/docs"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ERRORS=0

# Extract valid tags from Tags.md
get_valid_tags() {
    grep -oE '`[a-zA-Z][a-zA-Z0-9_-]*`' "$TAGS_FILE" | tr -d '`' | sort -u
}

# Check if a tag (or its parent for nested tags) is valid
is_valid_tag() {
    local tag="$1"
    local valid_tags="$2"

    if echo "$valid_tags" | grep -qx "$tag"; then
        return 0
    fi

    # For nested tags (parent/child), check if parent is valid
    if [[ "$tag" == */* ]]; then
        local parent="${tag%%/*}"
        if echo "$valid_tags" | grep -qx "$parent"; then
            return 0
        fi
    fi

    return 1
}

echo "Checking tags in documentation files..."
echo ""

VALID_TAGS=$(get_valid_tags)

while IFS= read -r -d '' file; do
    rel_path="${file#$SCRIPT_DIR/}"

    tags=$("$GET_TAGS" "$file" 2>/dev/null || true)

    # Check 1: File must have at least one tag
    if [[ -z "$tags" ]]; then
        echo -e "${RED}✗ No tags found:${NC} $rel_path"
        ((ERRORS++))
        continue
    fi

    # Check 2: All tags must be valid
    invalid_tags=()
    while IFS= read -r tag; do
        [[ -z "$tag" ]] && continue
        if ! is_valid_tag "$tag" "$VALID_TAGS"; then
            invalid_tags+=("$tag")
        fi
    done <<< "$tags"

    if [[ ${#invalid_tags[@]} -gt 0 ]]; then
        echo -e "${RED}✗ Invalid tags in${NC} $rel_path${RED}:${NC} ${invalid_tags[*]}"
        ((ERRORS++))
    fi
done < <(find "$DOCS_DIR" -name '*.md' -type f ! -path '*/templates/*' ! -path '*/.obsidian/*' -print0)

echo ""

if [[ $ERRORS -eq 0 ]]; then
    echo -e "${GREEN}✓ All tag checks passed${NC}"
    exit 0
else
    echo -e "${RED}Found $ERRORS file(s) with tag issues${NC}"
    exit 1
fi
