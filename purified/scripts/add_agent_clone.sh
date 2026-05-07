#!/usr/bin/env bash
# add_agent_clone.sh — give a second agent on a shared pod its own clone.
#
# ──────────────────────────────────────────────────────────────────────
#  RUN BY THE OPERATOR, NOT BY AN AGENT.
#  Run AFTER bootstrap_runpod.sh has set up /workspace/temp_xc/.
#  This script is non-interactive (no token prompts — tokens are
#  already in /workspace/.tokens/ from bootstrap), but it is part of
#  pod provisioning and lives outside any agent session.
# ──────────────────────────────────────────────────────────────────────
#
# Why a second clone (not a worktree):
#   - Two agents sharing a single .git/ collide on index.lock /
#     refs/heads/final.lock and risk clobbering each other's
#     uncommitted edits during pull-rebase.
#   - `git worktree` won't allow two worktrees on the same branch
#     (`final`), so worktrees would force per-agent local branches.
#   - Disk cost of a second clone is ~5 GB on a 1 TB workspace.
#     HF model cache (/workspace/hf_cache) and tokens
#     (/workspace/.tokens) are SHARED across clones.
#
# Usage:
#     bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh <agent_name>
#
# Examples (the canonical second-agent pairs):
#     # On the 2× H100 pod (agent_nlp at /workspace/temp_xc, agent_em second):
#     bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh agent_em
#     # → creates /workspace/temp_xc_em/
#
#     # On the 4× A40 pod (agent_back at /workspace/temp_xc, agent_steer second):
#     bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh agent_steer
#     # → creates /workspace/temp_xc_steer/
#
# Idempotent: re-running on an existing clone just `git pull`s and
# re-runs `uv sync`. Safe to invoke repeatedly.

set -eu

if [ -z "${1:-}" ]; then
    echo "Usage: $(basename "$0") <agent_name>" >&2
    echo "       e.g.  $(basename "$0") agent_em" >&2
    exit 1
fi

AGENT="$1"
case "$AGENT" in
    agent_em|agent_steer|agent_back|agent_nlp)
        ;;
    *)
        echo "[add_agent_clone] warning: '$AGENT' is not in the canonical" >&2
        echo "  shared-pod agent set (agent_nlp, agent_em, agent_steer, agent_back)." >&2
        echo "  Continuing anyway — but check agents/README.md to confirm." >&2
        ;;
esac

REPO_URL="${REPO_URL:-https://anonymous.4open.science/r/temp-bench.git}"
PARENT_DIR="${PARENT_DIR:-/workspace}"
PRIMARY_CLONE="$PARENT_DIR/temp_xc"
AGENT_CLONE="$PARENT_DIR/temp_xc_${AGENT#agent_}"   # strip "agent_" prefix
BRANCH="${BRANCH:-final}"
TOKENS_DIR="${TOKENS_DIR:-/workspace/.tokens}"
HF_HOME_TARGET="${HF_HOME:-/workspace/hf_cache}"

echo "=== add_agent_clone: $AGENT → $AGENT_CLONE ==="

# ── Sanity: primary clone must exist (bootstrap_runpod.sh ran first)
if [ ! -d "$PRIMARY_CLONE/.git" ]; then
    echo "[add_agent_clone] error: $PRIMARY_CLONE has no .git/" >&2
    echo "  Run bash $PRIMARY_CLONE/purified/scripts/bootstrap_runpod.sh first." >&2
    exit 1
fi

# ── Sanity: tokens must be set up
if [ ! -d "$TOKENS_DIR" ]; then
    echo "[add_agent_clone] error: $TOKENS_DIR missing." >&2
    echo "  Run bash $PRIMARY_CLONE/purified/scripts/bootstrap_runpod.sh first." >&2
    exit 1
fi

# ── Clone (or refresh existing)
if [ -d "$AGENT_CLONE/.git" ]; then
    echo "[$AGENT] clone exists at $AGENT_CLONE — refreshing"
    cd "$AGENT_CLONE"
    git fetch --all --prune
    git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
    git pull --rebase --autostash origin "$BRANCH"
else
    echo "[$AGENT] cloning $REPO_URL into $AGENT_CLONE"
    git clone "$REPO_URL" "$AGENT_CLONE"
    cd "$AGENT_CLONE"
    git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
    git pull --rebase --autostash origin "$BRANCH"
fi

# ── Configure git author identity for this clone (matches bootstrap)
# Each clone has its own .git/config so we set author identity per-clone.
git config user.name "anonymous"
git config user.email "anonymous@example.com"

# ── uv sync from the new purified/
cd "$AGENT_CLONE/purified"
if ! command -v uv >/dev/null 2>&1; then
    echo "[$AGENT] uv not found in PATH; assuming bootstrap_runpod.sh installed it"
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[$AGENT] running uv sync from $AGENT_CLONE/purified"
uv sync

# ── Sanity: tokens + HF cache are reachable (they're under /workspace,
#    not under the clone, so this should always be true — verify anyway)
[ -f "$TOKENS_DIR/hf_token" ] && echo "  ✓ HF token reachable at $TOKENS_DIR/hf_token" || echo "  ✗ HF token MISSING — re-run bootstrap"
[ -d "$HF_HOME_TARGET" ] && echo "  ✓ HF cache reachable at $HF_HOME_TARGET" || echo "  ✗ HF cache MISSING — re-run bootstrap"

echo
echo "=== Done ==="
echo "Agent:     $AGENT"
echo "Clone:     $AGENT_CLONE (branch: $BRANCH)"
echo "purified:  $AGENT_CLONE/purified/"
echo "venv:      $AGENT_CLONE/purified/.venv/"
echo "Shared:    tokens=$TOKENS_DIR, hf_cache=$HF_HOME_TARGET"
echo
echo "Next: when spawning $AGENT, start the agent with cwd = $AGENT_CLONE/purified/"
echo "      cd $AGENT_CLONE/purified && source scripts/set_agent_env.sh $AGENT"
