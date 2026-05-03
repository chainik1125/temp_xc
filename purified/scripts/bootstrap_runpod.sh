#!/usr/bin/env bash
# bootstrap_runpod.sh — set up a fresh RunPod pod for the `final` paper branch.
#
# ──────────────────────────────────────────────────────────────────────
#  RUN BY THE USER (HAN), NOT BY AN AGENT.
#  This script is INTERACTIVE — it prompts for tokens via `read -rs`.
#  An agent CANNOT enter input here; do not call this from an
#  agent session. By the time an agent starts, this script has
#  already been run by Han and `/workspace/.tokens/` is populated.
# ──────────────────────────────────────────────────────────────────────
#
# Adapted from scripts/runpod_phase7_bootstrap.sh (wasteland) — no functional
# dependency on it. This script lives in purified/ and is the only supported
# entry point on RunPod.
#
# Usage (interactive — the typical path):
#     bash /workspace/temp_xc/purified/scripts/bootstrap_runpod.sh
#
# Usage (non-interactive, e.g. CI):
#     GH_TOKEN=ghp_xxx HF_TOKEN=hf_xxx ANTHROPIC_API_KEY=sk-xxx \
#         bash /workspace/temp_xc/purified/scripts/bootstrap_runpod.sh
#
# What this does:
#   1. Stores tokens in /workspace/.tokens/{gh_token,hf_token,anthropic_key}
#      (mode 0600) and validates each.
#   2. Configures `gh`, `huggingface-cli`, and exports ANTHROPIC_API_KEY in
#      ~/.bashrc.
#   3. Sets HF_HOME=/workspace/hf_cache and UV_LINK_MODE=copy in ~/.bashrc
#      (UV_LINK_MODE=copy is REQUIRED on MooseFS — the default hardlink mode
#      silently produces partial installs).
#   4. Clones temp_xc on /workspace if absent, checks out `final`, and runs
#      `uv sync` from purified/.
#
# Idempotent: re-running re-validates existing tokens.

set -eu

REPO_URL="${REPO_URL:-https://github.com/chainik1125/temp_xc.git}"
REPO_DIR="${REPO_DIR:-/workspace/temp_xc}"
BRANCH="${BRANCH:-final}"
TOKENS_DIR="${TOKENS_DIR:-/workspace/.tokens}"
HF_HOME_TARGET="${HF_HOME:-/workspace/hf_cache}"

mkdir -p "$TOKENS_DIR"
chmod 700 "$TOKENS_DIR"

GH_TOKEN_FILE="$TOKENS_DIR/gh_token"
HF_TOKEN_FILE="$TOKENS_DIR/hf_token"
ANTHROPIC_TOKEN_FILE="$TOKENS_DIR/anthropic_key"

prompt_or_load() {
    local name="$1"
    local path="$2"
    local envvar="${3:-}"
    local val=""
    if [ -n "$envvar" ] && [ -n "${!envvar:-}" ]; then
        val="${!envvar}"
        echo "[$name] using value from env \$$envvar"
    elif [ -f "$path" ] && [ -s "$path" ]; then
        val="$(cat "$path")"
        echo "[$name] using existing token at $path"
    else
        echo -n "[$name] paste token (input hidden): "
        read -rs val
        echo
    fi
    if [ -z "$val" ]; then
        echo "[$name] empty token; skipping"
        return 1
    fi
    echo "$val" > "$path"
    chmod 600 "$path"
    return 0
}

persist_export() {
    # idempotent ~/.bashrc append
    local line="$1"
    grep -qF "$line" "$HOME/.bashrc" 2>/dev/null || echo "$line" >> "$HOME/.bashrc"
}

# ─────────────────────────────────────────── GitHub
echo
echo "=== GitHub ==="
if prompt_or_load "GitHub" "$GH_TOKEN_FILE" "GH_TOKEN"; then
    GH=$(cat "$GH_TOKEN_FILE")
    if command -v gh >/dev/null 2>&1; then
        echo "$GH" | gh auth login --with-token 2>/dev/null \
            && echo "  ✓ gh auth ok ($(gh api user --jq .login))" \
            || echo "  ✗ gh auth failed"
    fi
    if curl -fsSL -H "Authorization: token $GH" https://api.github.com/user > /tmp/_gh.json 2>/dev/null; then
        login=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("login","?"))' < /tmp/_gh.json 2>/dev/null || echo "?")
        echo "  ✓ GitHub API: logged in as $login"
    else
        echo "  ✗ GitHub API check failed"
    fi
fi

# ─────────────────────────────────────────── HuggingFace
echo
echo "=== HuggingFace ==="
mkdir -p "$HF_HOME_TARGET"
if prompt_or_load "HuggingFace" "$HF_TOKEN_FILE" "HF_TOKEN"; then
    HF=$(cat "$HF_TOKEN_FILE")
    echo "$HF" > "$HF_HOME_TARGET/token"
    chmod 600 "$HF_HOME_TARGET/token"
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "$HF" | huggingface-cli login --token "$HF" --add-to-git-credential 2>/dev/null \
            && echo "  ✓ huggingface-cli login ok" \
            || echo "  (token-file fallback)"
    fi
    if curl -fsSL -H "Authorization: Bearer $HF" https://huggingface.co/api/whoami-v2 > /tmp/_hf.json 2>/dev/null; then
        name=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("name", d.get("type","?")))' < /tmp/_hf.json 2>/dev/null || echo "?")
        echo "  ✓ HF API: $name"
    fi
fi

# ─────────────────────────────────────────── Anthropic
echo
echo "=== Anthropic ==="
if prompt_or_load "Anthropic" "$ANTHROPIC_TOKEN_FILE" "ANTHROPIC_API_KEY"; then
    ANT=$(cat "$ANTHROPIC_TOKEN_FILE")
    persist_export "export ANTHROPIC_API_KEY=\$(cat $ANTHROPIC_TOKEN_FILE 2>/dev/null)"
    export ANTHROPIC_API_KEY="$ANT"
    if curl -fsSL -X POST https://api.anthropic.com/v1/messages \
        -H "x-api-key: $ANT" \
        -H "anthropic-version: 2023-06-01" \
        -H "Content-Type: application/json" \
        -d '{"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
        > /tmp/_ant.json 2>/dev/null; then
        if grep -q '"type":"message"' /tmp/_ant.json; then
            echo "  ✓ Anthropic API: ok (Haiku 4.5)"
        else
            echo "  ✗ Anthropic API non-message response"
            head -c 200 /tmp/_ant.json
        fi
    fi
fi

# ─────────────────────────────────────────── Persist HF_HOME + UV_LINK_MODE
persist_export "export HF_HOME=$HF_HOME_TARGET"
persist_export "export UV_LINK_MODE=copy"
export HF_HOME="$HF_HOME_TARGET"
export UV_LINK_MODE=copy

# ─────────────────────────────────────────── Clone + checkout + uv sync
echo
echo "=== Repository ==="
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --all --prune
git checkout "$BRANCH" || git checkout -b "$BRANCH" "origin/$BRANCH"
git pull --rebase --autostash origin "$BRANCH"

cd "$REPO_DIR/purified"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv …"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Running uv sync (purified) …"
uv sync

echo
echo "=== Done ==="
echo "Repo:        $REPO_DIR (branch: $BRANCH)"
echo "purified env: $REPO_DIR/purified/.venv"
echo "Tokens:      $TOKENS_DIR"
echo "HF cache:    $HF_HOME_TARGET"
echo
echo "Next: cd $REPO_DIR/purified && bash scripts/agent_smoke_test.sh"
