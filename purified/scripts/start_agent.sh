#!/usr/bin/env bash
# start_agent.sh — single-command launcher for a worker agent on a pod.
#
# Solves the env-inheritance problem: Claude Code's Bash tool calls do
# NOT share shell state, so an agent sourcing set_agent_env.sh as its
# "first action" exports CUDA_VISIBLE_DEVICES / AGENT_NAME / pod mode
# into a one-shot subshell and immediately discards them. The next
# Bash call sees an unpinned GPU, AGENT_NAME=unknown, no pod mode.
#
# This wrapper sources set_agent_env.sh in the parent shell and then
# `exec`s claude, so the claude process and every Bash tool call it
# spawns inherit the env. One command per agent spawn.
#
# Run by Han, not by an agent.
#
# Usage:
#     bash <clone_path>/purified/scripts/start_agent.sh <agent_name> [--fresh]
#
# By default, the wrapper invokes `claude --continue` so a disconnected
# worker resumes its previous session rather than re-reading the briefing
# from scratch. Pass `--fresh` to start a brand-new session (use this on
# the FIRST launch of an agent on a fresh pod — there's no session to
# resume yet, and `--continue` will start a new one anyway, but `--fresh`
# is more explicit).
#
# Canonical commands per pod:
#
#   2× H100 pod
#     bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_nlp --fresh   # first launch
#     bash /workspace/temp_xc_em/purified/scripts/start_agent.sh agent_em --fresh
#     # later, after disconnects:
#     bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_nlp           # resumes
#
#   4× A40 pod
#     bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_back --fresh
#     bash /workspace/temp_xc_steer/purified/scripts/start_agent.sh agent_steer --fresh  # T+~3hr
#
# (You can run from any clone — the wrapper resolves the agent's
# expected clone path independent of where you invoked it.)

set -eu

if [ -z "${1:-}" ]; then
    echo "Usage: $(basename "$0") <agent_name>" >&2
    echo "  e.g.  $(basename "$0") agent_em" >&2
    exit 1
fi

AGENT="$1"
shift
PARENT_DIR="${PARENT_DIR:-/workspace}"

# Default: --continue (resume most recent session). Override with --fresh.
RESUME=1
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --fresh) RESUME=0 ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

case "$AGENT" in
    agent_nlp|agent_back|agent_em_h200)
        CLONE="$PARENT_DIR/temp_xc"
        IS_SECOND_AGENT=0
        ;;
    agent_em)
        CLONE="$PARENT_DIR/temp_xc_em"
        IS_SECOND_AGENT=1
        ;;
    agent_steer)
        CLONE="$PARENT_DIR/temp_xc_steer"
        IS_SECOND_AGENT=1
        ;;
    agent_paper)
        echo "[start_agent] agent_paper runs locally, not on a RunPod pod." >&2
        echo "  Just 'cd' to your local temp_xc/purified/ and run 'claude' directly." >&2
        exit 1
        ;;
    *)
        echo "[start_agent] unknown agent '$AGENT'." >&2
        echo "  Known: agent_nlp, agent_em, agent_back, agent_steer, agent_em_h200" >&2
        exit 1
        ;;
esac

PURIFIED="$CLONE/purified"

# ── Sanity: the clone must exist
if [ ! -d "$PURIFIED" ]; then
    echo "[start_agent] error: $PURIFIED does not exist." >&2
    if [ "$IS_SECOND_AGENT" -eq 1 ]; then
        echo "  $AGENT is a second-on-pod agent and needs its own clone." >&2
        echo "  Run this on the pod, then re-run start_agent.sh:" >&2
        echo
        echo "    bash $PARENT_DIR/temp_xc/purified/scripts/add_agent_clone.sh $AGENT" >&2
    else
        echo "  No primary clone found. Bootstrap the pod first:" >&2
        echo
        echo "    cd $PARENT_DIR && git clone https://github.com/chainik1125/temp_xc.git" >&2
        echo "    cd $PARENT_DIR/temp_xc && git checkout final" >&2
        echo "    bash purified/scripts/bootstrap_runpod.sh" >&2
    fi
    exit 1
fi

# ── Sanity: claude CLI must be installed
if ! command -v claude >/dev/null 2>&1; then
    echo "[start_agent] error: 'claude' not found in PATH." >&2
    echo "  Install Claude Code on the pod and ensure 'claude' is on PATH." >&2
    exit 1
fi

# ── Sanity: tokens directory exists (bootstrap_runpod.sh ran)
if [ ! -d "${TOKENS_DIR:-/workspace/.tokens}" ]; then
    echo "[start_agent] warning: ${TOKENS_DIR:-/workspace/.tokens} not found." >&2
    echo "  bootstrap_runpod.sh probably hasn't run. Continuing anyway." >&2
fi

# ── cd + source env. set_agent_env.sh exports CUDA_VISIBLE_DEVICES,
#    AGENT_NAME, TEMP_BENCH_POD_MODE into THIS shell — then exec
#    claude propagates them into the claude process.
cd "$PURIFIED"

# shellcheck source=/dev/null
source "scripts/set_agent_env.sh" "$AGENT"

if [ "$RESUME" -eq 1 ]; then
    CLAUDE_FLAGS=("--continue")
    SESSION_DESC="resuming most-recent session"
else
    CLAUDE_FLAGS=()
    SESSION_DESC="fresh session (no --continue)"
fi

echo
echo "[start_agent] launching claude in $PURIFIED — $SESSION_DESC"
echo "  AGENT_NAME=${AGENT_NAME:-unset}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  TEMP_BENCH_POD_MODE=${TEMP_BENCH_POD_MODE:-unset}"
echo

exec claude "${CLAUDE_FLAGS[@]}" "${EXTRA_ARGS[@]}"
