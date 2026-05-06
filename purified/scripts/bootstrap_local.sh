#!/usr/bin/env bash
# bootstrap_local.sh — populate ~/.tokens/ on the local box.
#
# Mirrors bootstrap_runpod.sh's layout: same filenames, same
# canonical store. The framework's `temp_bench.utils.get_token(kind)`
# resolves the same way on local and RunPod (looks in /workspace/.tokens
# first, then ~/.tokens — see utils/tokens.py).
#
# Usage:
#   bash purified/scripts/bootstrap_local.sh                      # interactive
#   HF_TOKEN=hf_… ANTHROPIC_API_KEY=sk-… GH_TOKEN=ghp_… \
#     bash purified/scripts/bootstrap_local.sh                    # non-interactive
#
# The script tries to seed each token from existing local sources
# before prompting:
#
#   hf_token       <- $HF_TOKEN env, then ~/.cache/huggingface/token
#   anthropic_key  <- $ANTHROPIC_API_KEY env, then a `.env_autointerp`
#                     file under $HOME or repo root (legacy wasteland location)
#   gh_token       <- $GH_TOKEN env, then `gh auth token` (gh CLI), else prompt
#
# Idempotent: re-running re-validates existing tokens.

set -eu

TOKENS_DIR="${TEMP_BENCH_TOKENS_DIR:-$HOME/.tokens}"
mkdir -p "$TOKENS_DIR"
chmod 700 "$TOKENS_DIR"

HF_FILE="$TOKENS_DIR/hf_token"
ANT_FILE="$TOKENS_DIR/anthropic_key"
GH_FILE="$TOKENS_DIR/gh_token"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

prompt_or_load() {
    # $1 friendly name; $2 dest path; $3 env var name; $4 candidate file (or "")
    local name="$1" dest="$2" envvar="$3" candidate="$4"
    local val=""
    if [ -n "${!envvar:-}" ]; then
        val="${!envvar}"
        echo "[$name] using value from \$$envvar"
    elif [ -f "$dest" ] && [ -s "$dest" ]; then
        val="$(cat "$dest")"
        echo "[$name] using existing token at $dest"
    elif [ -n "$candidate" ] && [ -f "$candidate" ] && [ -s "$candidate" ]; then
        echo -n "[$name] found candidate at $candidate. Use it? [Y/n]: "
        read -r ans
        if [[ "$ans" != "n" && "$ans" != "N" ]]; then
            val="$(cat "$candidate" | head -1)"
        fi
    fi
    if [ -z "$val" ]; then
        echo -n "[$name] paste token (input hidden): "
        read -rs val
        echo
    fi
    if [ -z "$val" ]; then
        echo "[$name] empty; skipping"
        return 1
    fi
    printf '%s' "$val" > "$dest"
    chmod 600 "$dest"
    return 0
}

# ─── HuggingFace ──────────────────────────────────────────────────
echo
echo "=== HuggingFace ==="
HF_CANDIDATE="$HOME/.cache/huggingface/token"
if prompt_or_load "HuggingFace" "$HF_FILE" "HF_TOKEN" "$HF_CANDIDATE"; then
    HF=$(cat "$HF_FILE")
    if curl -fsSL -H "Authorization: Bearer $HF" \
        https://huggingface.co/api/whoami-v2 > /tmp/_hf_check.json 2>/dev/null; then
        name=$(python3 -c '
import json,sys
d=json.load(sys.stdin)
print(d.get("name", d.get("type","?")))
' < /tmp/_hf_check.json 2>/dev/null || echo "?")
        echo "  ✓ HF API: $name"
    else
        echo "  ✗ HF API check failed (token bad?)"
    fi
fi

# ─── Anthropic ────────────────────────────────────────────────────
echo
echo "=== Anthropic ==="
# Look for a wasteland-style .env_autointerp file with the key.
ANT_CANDIDATE=""
for c in "$HOME/.env_autointerp" "$REPO_ROOT/.env_autointerp"; do
    if [ -f "$c" ] && grep -q ANTHROPIC "$c" 2>/dev/null; then
        ANT_CANDIDATE="$c"
        break
    fi
done

if [ -n "$ANT_CANDIDATE" ]; then
    # Extract just the key value
    EXTRACTED=$(grep -E 'ANTHROPIC[^=]*=' "$ANT_CANDIDATE" | head -1 | sed -E 's/^[^=]*=//' | tr -d '"' | tr -d "'")
    if [ -n "$EXTRACTED" ]; then
        TMP_ANT=$(mktemp)
        printf '%s' "$EXTRACTED" > "$TMP_ANT"
        chmod 600 "$TMP_ANT"
        ANT_CANDIDATE="$TMP_ANT"
    fi
fi

if prompt_or_load "Anthropic" "$ANT_FILE" "ANTHROPIC_API_KEY" "$ANT_CANDIDATE"; then
    ANT=$(cat "$ANT_FILE")
    if curl -fsSL -X POST https://api.anthropic.com/v1/messages \
        -H "x-api-key: $ANT" \
        -H "anthropic-version: 2023-06-01" \
        -H "Content-Type: application/json" \
        -d '{"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
        > /tmp/_ant_check.json 2>/dev/null; then
        if grep -q '"type":"message"' /tmp/_ant_check.json; then
            echo "  ✓ Anthropic API: ok (Haiku 4.5)"
        else
            echo "  ✗ Anthropic API non-message response"
            head -c 200 /tmp/_ant_check.json
        fi
    fi
fi

# ─── GitHub ───────────────────────────────────────────────────────
echo
echo "=== GitHub ==="
GH_CANDIDATE=""
if command -v gh >/dev/null 2>&1; then
    if gh auth token >/dev/null 2>&1; then
        GH_CANDIDATE=$(mktemp)
        gh auth token > "$GH_CANDIDATE" 2>/dev/null
        chmod 600 "$GH_CANDIDATE"
    fi
fi
if [ -z "$GH_CANDIDATE" ] && [ -z "${GH_TOKEN:-}" ] && [ ! -f "$GH_FILE" ]; then
    echo "  (gh CLI not authenticated. If you only push via SSH locally,"
    echo "   you can skip this — but RunPod pods will still need a PAT."
    echo "   Press Enter to skip, or paste a PAT.)"
fi
if prompt_or_load "GitHub" "$GH_FILE" "GH_TOKEN" "$GH_CANDIDATE"; then
    GH=$(cat "$GH_FILE")
    if curl -fsSL -H "Authorization: token $GH" \
        https://api.github.com/user > /tmp/_gh_check.json 2>/dev/null; then
        login=$(python3 -c '
import json,sys
print(json.load(sys.stdin).get("login","?"))
' < /tmp/_gh_check.json 2>/dev/null || echo "?")
        echo "  ✓ GitHub API: $login"
    else
        echo "  ✗ GitHub API check failed"
    fi
fi

echo
echo "=== Done ==="
echo "Tokens stored at $TOKENS_DIR/"
ls -la "$TOKENS_DIR" 2>/dev/null | tail -n +2

echo
echo "Verify resolution:"
echo "  python -c 'from temp_bench.utils import token_status; import json; print(json.dumps(token_status(), indent=2))'"
