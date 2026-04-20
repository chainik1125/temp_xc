#!/usr/bin/env bash
# Restore the /home/appuser/.claude → /workspace/claude_home symlink after
# a container rebuild / pod-settings change that wiped container disk.
#
# Run this ONCE after any pod-settings edit, BEFORE launching Claude Code.
# Idempotent — safe to run when the symlink is already in place.
#
# Usage:
#   bash /workspace/temp_xc/scripts/bootstrap_claude.sh
#
# See docs/han/research_logs/phase5_downstream_utility/2026-04-20-overnight-handoff.md
# for context on why this relocation exists.

set -eu

TARGET=/workspace/claude_home
LINK=/home/appuser/.claude

if [ ! -d "$TARGET" ]; then
    echo "FATAL: $TARGET does not exist on pod volume."
    echo "       State has been lost. Recovery options:"
    echo "         1. Restore from /home/appuser/.claude.bak.* if that still exists."
    echo "         2. Start fresh (will lose scheduled wakeups + history)."
    exit 1
fi

# If $LINK exists and is NOT already our symlink, archive it first so we
# don't clobber fresh state that Claude Code may have created post-wipe.
if [ -e "$LINK" ] && [ ! -L "$LINK" ]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    echo "Found a non-symlink at $LINK — archiving to $LINK.fresh.$STAMP"
    mv "$LINK" "$LINK.fresh.$STAMP"
fi

# If $LINK is an already-valid symlink to the right target, nothing to do.
if [ -L "$LINK" ] && [ "$(readlink "$LINK")" = "$TARGET" ]; then
    echo "Symlink already in place: $LINK -> $TARGET"
    exit 0
fi

# Otherwise: clean up any stale symlink and create the right one.
rm -f "$LINK" 2>/dev/null || true
ln -s "$TARGET" "$LINK"
echo "Restored: $LINK -> $(readlink "$LINK")"
echo "Data:     $(du -sh "$TARGET" | cut -f1) on $(df -hT "$TARGET" | tail -1 | awk '{print $2}')"
