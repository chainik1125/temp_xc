#!/usr/bin/env bash
# Phase 7 RunPod bootstrap — sets up GitHub, HuggingFace, and Anthropic
# auth tokens on a fresh pod.
#
# Usage (interactive — prompts for each token):
#     bash /workspace/temp_xc/scripts/runpod_phase7_bootstrap.sh
#
# Usage (non-interactive — provide tokens via env vars):
#     GH_TOKEN=ghp_xxx HF_TOKEN=hf_xxx ANTHROPIC_API_KEY=sk-xxx \
#         bash /workspace/temp_xc/scripts/runpod_phase7_bootstrap.sh
#
# What this does:
#   1. Saves each token to /workspace/.tokens/{gh,hf,anthropic}
#      (mode 0600, owned by the current user).
#   2. Configures `gh` CLI auth (if `gh` is installed).
#   3. Configures `huggingface-cli` auth via $HF_HOME (default
#      /workspace/hf_cache).
#   4. Adds ANTHROPIC_API_KEY export to ~/.bashrc.
#   5. Validates each token by hitting the corresponding API.
#
# Idempotent: re-running re-validates existing tokens; only prompts for
# any that are missing AND not provided via env var.

set -eu

TOKENS_DIR=/workspace/.tokens
mkdir -p "$TOKENS_DIR"
chmod 700 "$TOKENS_DIR"

GH_TOKEN_FILE="$TOKENS_DIR/gh_token"
HF_TOKEN_FILE="$TOKENS_DIR/hf_token"
ANTHROPIC_TOKEN_FILE="$TOKENS_DIR/anthropic_key"

prompt_or_load() {
    # $1 = friendly name (e.g. "GitHub")
    # $2 = path to token file
    # $3 = name of env var to check (optional)
    local name="$1"
    local path="$2"
    local envvar="${3:-}"
    local val=""
    # Priority: env var → existing file → prompt.
    if [ -n "$envvar" ] && [ -n "${!envvar:-}" ]; then
        val="${!envvar}"
        echo "[$name] using value from env \$$envvar"
    elif [ -f "$path" ] && [ -s "$path" ]; then
        val="$(cat "$path")"
        echo "[$name] using existing token at $path"
    else
        echo -n "[$name] paste token (input hidden): "
        # silent prompt; input ends at newline
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

# ─────────────────────────────────────────── GitHub
echo
echo "=== GitHub ==="
if prompt_or_load "GitHub" "$GH_TOKEN_FILE" "GH_TOKEN"; then
    GH=$(cat "$GH_TOKEN_FILE")
    if command -v gh >/dev/null 2>&1; then
        # Idempotent: 'gh auth login --with-token' re-runs cleanly even if logged in.
        echo "$GH" | gh auth login --with-token 2>/dev/null \
            && echo "  ✓ gh auth ok ($(gh api user --jq .login))" \
            || echo "  ✗ gh auth failed (token bad?)"
    else
        echo "  (gh CLI not installed — token saved to $GH_TOKEN_FILE for manual use)"
    fi
    # Validate token directly via API regardless of gh.
    if curl -fsSL -H "Authorization: token $GH"  https://api.github.com/user > /tmp/_gh_check.json 2>/dev/null; then
        login=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("login","?"))' < /tmp/_gh_check.json 2>/dev/null || echo "?")
        echo "  ✓ GitHub API check: logged in as $login"
    else
        echo "  ✗ GitHub API check failed"
    fi
fi

# ─────────────────────────────────────────── HuggingFace
echo
echo "=== HuggingFace ==="
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME"
if prompt_or_load "HuggingFace" "$HF_TOKEN_FILE" "HF_TOKEN"; then
    HF=$(cat "$HF_TOKEN_FILE")
    # Configure huggingface_hub. Two paths: write to HF_HOME/token (legacy)
    # and use huggingface-cli login --token if available.
    mkdir -p "$HF_HOME"
    echo "$HF" > "$HF_HOME/token"
    chmod 600 "$HF_HOME/token"
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "$HF" | huggingface-cli login --token "$HF" --add-to-git-credential 2>/dev/null \
            && echo "  ✓ huggingface-cli login ok" \
            || echo "  (huggingface-cli login fallback to token-file)"
    fi
    # Validate via API
    if curl -fsSL -H "Authorization: Bearer $HF" https://huggingface.co/api/whoami-v2 > /tmp/_hf_check.json 2>/dev/null; then
        name=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("name", d.get("type","?")))' < /tmp/_hf_check.json 2>/dev/null || echo "?")
        echo "  ✓ HF API check: $name"
    else
        echo "  ✗ HF API check failed"
    fi
fi

# ─────────────────────────────────────────── Anthropic
echo
echo "=== Anthropic ==="
if prompt_or_load "Anthropic" "$ANTHROPIC_TOKEN_FILE" "ANTHROPIC_API_KEY"; then
    ANT=$(cat "$ANTHROPIC_TOKEN_FILE")
    # Persist ANTHROPIC_API_KEY in bashrc so it loads on future shells.
    grep -qF "export ANTHROPIC_API_KEY=" "$HOME/.bashrc" 2>/dev/null \
        || echo "export ANTHROPIC_API_KEY=\$(cat $ANTHROPIC_TOKEN_FILE 2>/dev/null)" >> "$HOME/.bashrc"
    export ANTHROPIC_API_KEY="$ANT"
    # Validate via API (Messages endpoint with a tiny ping)
    if curl -fsSL -X POST https://api.anthropic.com/v1/messages \
        -H "x-api-key: $ANT" \
        -H "anthropic-version: 2023-06-01" \
        -H "Content-Type: application/json" \
        -d '{"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
        > /tmp/_anthropic_check.json 2>/dev/null; then
        if grep -q '"type":"message"' /tmp/_anthropic_check.json; then
            echo "  ✓ Anthropic API check: ok (Haiku 4.5)"
        else
            echo "  ✗ Anthropic API check returned non-message; check token / model access"
            head -c 200 /tmp/_anthropic_check.json
        fi
    else
        echo "  ✗ Anthropic API check failed (curl)"
    fi
fi

# ─────────────────────────────────────────── Persist HF_HOME for future shells
grep -qF "export HF_HOME=$HF_HOME" "$HOME/.bashrc" 2>/dev/null \
    || echo "export HF_HOME=$HF_HOME" >> "$HOME/.bashrc"

echo
echo "=== Done ==="
echo "Tokens stored at $TOKENS_DIR/{gh_token,hf_token,anthropic_key}"
echo "ANTHROPIC_API_KEY exported in this shell and persisted to ~/.bashrc"
echo "HF_HOME=$HF_HOME (persisted to ~/.bashrc)"
echo
echo "Next steps:"
echo "  - cd /workspace/temp_xc"
echo "  - git checkout han-phase7-unification"
echo "  - uv sync"
echo "  - read docs/han/research_logs/phase7_unification/{brief,plan}.md"
