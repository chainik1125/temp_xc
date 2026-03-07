#!/bin/bash
# Launch Claude Code sandbox container with GPU access
#
# Usage:
#   ./launch-sandbox.sh              # interactive bash, then run `claude` inside
#   ./launch-sandbox.sh auto "prompt" # headless mode with a task prompt
#
# The container gets:
#   - GPU access (your RTX 5090)
#   - Project files mounted at /workspace
#   - Your Claude OAuth credentials
#   - TQDM disabled, PYTHONPATH set

set -e

IMAGE_NAME="claude-sandbox"
CONTAINER_NAME="claude-ml-sandbox"

# Check image exists
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Image '$IMAGE_NAME' not found. Build it first:"
    echo "  docker build -t $IMAGE_NAME ."
    exit 1
fi

# Kill any existing container with the same name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Run as current host user so mounted files have correct ownership
DOCKER_USER="$(id -u):$(id -g)"

if [ "$1" = "auto" ] && [ -n "$2" ]; then
    # Headless autonomous mode
    PROMPT="$2"
    MAX_TURNS="${3:-200}"
    echo "Starting autonomous Claude session (max $MAX_TURNS turns)..."
    echo "Prompt: ${PROMPT:0:100}..."
    docker run -it --gpus all \
        --name "$CONTAINER_NAME" \
        --user "$DOCKER_USER" \
        -e HOME=/home/sandbox \
        -v "$(pwd)":/workspace \
        -v "$HOME/.claude":/home/sandbox/.claude \
        "$IMAGE_NAME" \
        bash -c "claude --dangerouslySkipPermissions --max-turns $MAX_TURNS -p \"$PROMPT\""
else
    # Interactive mode — drop into bash, run claude manually
    echo "Starting interactive sandbox..."
    echo "Inside the container, run:"
    echo "  claude --dangerouslySkipPermissions"
    docker run -it --gpus all \
        --name "$CONTAINER_NAME" \
        --user "$DOCKER_USER" \
        -e HOME=/home/sandbox \
        -v "$(pwd)":/workspace \
        -v "$HOME/.claude":/home/sandbox/.claude \
        "$IMAGE_NAME" \
        bash
fi
