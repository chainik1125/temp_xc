#!/bin/bash
# trillium_debug_anthropic.sh — Diagnose why the Anthropic API returns 401.
#
# Runs three checks, reading the key directly from ~/.txc_secrets.env:
#   1. Byte-level look at the key — length, hex of first/last bytes, any
#      hidden whitespace or CR characters
#   2. Raw curl to api.anthropic.com (no Python SDK in the loop)
#   3. Same curl with the official anthropic-version header
#
# Run on the LOGIN NODE:
#   bash $SCRATCH/temp_xc/scripts/trillium_debug_anthropic.sh

set -uo pipefail

SECRETS="$HOME/.txc_secrets.env"

if [ ! -f "$SECRETS" ]; then
    echo "FAIL: $SECRETS does not exist."
    exit 1
fi

# ─── 1. byte-level view ────────────────────────────────────────────────────
echo "=== 1. byte-level view of the key ==="

# Grab the raw line, strip the "export ANTHROPIC_API_KEY=" prefix and outer quotes
RAW=$(grep -E '^export ANTHROPIC_API_KEY=' "$SECRETS" || true)
if [ -z "$RAW" ]; then
    echo "FAIL: no 'export ANTHROPIC_API_KEY=' line in $SECRETS"
    exit 2
fi
echo "  raw line in file:"
echo "    $RAW" | cat -A     # cat -A shows \t as ^I, \r as ^M, trailing $

# Extract the value (strips "export ANTHROPIC_API_KEY=" and surrounding quotes)
KEY=$(sed -n 's/^export ANTHROPIC_API_KEY=["'\'']\{0,1\}\([^"'\'']*\)["'\'']\{0,1\}.*$/\1/p' "$SECRETS")
LEN=${#KEY}
echo "  parsed length: $LEN"

if [ "$LEN" -eq 0 ]; then
    echo "FAIL: parsed key is empty. Check quoting in $SECRETS."
    exit 3
fi

echo "  first 12 chars: ${KEY:0:12}"
echo "  last  4 chars:  ${KEY: -4}"
echo "  hexdump of first 4 and last 4 bytes:"
printf '%s' "${KEY:0:4}" | od -c | head -1
printf '%s' "${KEY: -4}" | od -c | head -1

# Detect hidden characters
if [[ "$KEY" == *$'\r'* ]]; then
    echo "  WARN: key contains a carriage return (\\r) — likely from Windows-style line endings"
fi
if [[ "$KEY" == *' '* ]]; then
    echo "  WARN: key contains a literal space"
fi
if [[ "$KEY" == *$'\t'* ]]; then
    echo "  WARN: key contains a tab"
fi
if [[ "$KEY" == *'"'* ]] || [[ "$KEY" == *\'* ]]; then
    echo "  WARN: key contains a quote — probably a copy-paste artifact"
fi

# Also compare against what's actually in the shell env after sourcing
source "$SECRETS"
if [ "${ANTHROPIC_API_KEY:-}" != "$KEY" ]; then
    echo "  WARN: sourced ANTHROPIC_API_KEY differs from file-parsed value"
    echo "    file-parsed len: ${#KEY}"
    echo "    sourced   len:   ${#ANTHROPIC_API_KEY}"
fi

# ─── 2. raw curl without SDK ───────────────────────────────────────────────
echo ""
echo "=== 2. raw curl to api.anthropic.com/v1/messages ==="
echo "  (bypasses the python SDK entirely)"

HTTP_CODE=$(curl -s -o /tmp/txc_anth.json -w '%{http_code}' \
    https://api.anthropic.com/v1/messages \
    --header "x-api-key: $KEY" \
    --header "anthropic-version: 2023-06-01" \
    --header "content-type: application/json" \
    --data '{
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Say ok."}]
    }')

echo "  HTTP status: $HTTP_CODE"
echo "  body:"
cat /tmp/txc_anth.json
echo ""

if [ "$HTTP_CODE" = "200" ]; then
    echo ""
    echo "=== curl succeeded — the key works at the wire level ==="
    echo "If the python SDK still 401s, it's an env-loading issue, not the key."
    exit 0
fi

echo ""
echo "=== curl also returned $HTTP_CODE ==="
echo "This confirms the 401 is a real auth failure at Anthropic's end, not"
echo "a Python/env problem. Most likely causes:"
echo "  - key was created in a workspace with no billing attached"
echo "  - key belongs to a different workspace than the one you think"
echo "  - key was regenerated after you copied it"
echo "  - copy-paste dropped/added a character (see warnings above)"
echo ""
echo "Go to https://console.anthropic.com/settings/keys and:"
echo "  1. Confirm the key's 'Last used' timestamp updated just now"
echo "     (if yes, the key IS being received — the issue is workspace billing)"
echo "  2. Confirm the active workspace dropdown is SPAR"
echo "  3. If needed: revoke, create a fresh key under SPAR, paste into"
echo "     ~/.txc_secrets.env carefully (no quotes, no trailing whitespace)"
rm -f /tmp/txc_anth.json
exit 4
