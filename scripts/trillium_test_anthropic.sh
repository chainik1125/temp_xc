#!/bin/bash
# trillium_test_anthropic.sh — Smoke-test the Anthropic API key on Trillium.
#
# MUST be run on the LOGIN NODE (trig-login01 / trig-login02 — the machine
# you land on when you SSH in). Compute nodes have no outbound internet and
# every call to api.anthropic.com will fail with "Network is unreachable".
#
# What it checks:
#   1. You're actually on a login node (hostname prefix check)
#   2. ANTHROPIC_API_KEY is set and non-empty
#   3. A real Claude Haiku request round-trips successfully
#   4. Reports token counts so you know billing is wired to the right org
#
# Usage:
#   ssh aniketrd@trillium-gpu.scinet.utoronto.ca
#   bash $SCRATCH/temp_xc/scripts/trillium_test_anthropic.sh
#
# Exit codes:
#   0 = all good, key works
#   1 = not on login node
#   2 = ANTHROPIC_API_KEY not set
#   3 = API call failed (401, network, etc.)

set -euo pipefail

echo "=== Anthropic API smoke test ==="

# ─── 1. hostname check ─────────────────────────────────────────────────────
HOST=$(hostname)
if [[ "$HOST" != trig-login* ]]; then
    echo ""
    echo "FAIL: you are on '$HOST', which looks like a compute node."
    echo "      Compute nodes have no outbound internet. Exit the"
    echo "      allocation (Ctrl-D from srun --pty or scancel the sbatch)"
    echo "      and rerun this script from the login node."
    exit 1
fi
echo "  host: $HOST  (login node, has internet)"

# ─── 2. activate env + source secrets ──────────────────────────────────────
source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo ""
    echo "FAIL: ANTHROPIC_API_KEY is not set in the environment."
    echo "      Edit ~/.txc_secrets.env, paste your key, save, and rerun."
    echo "      Make sure the line reads:"
    echo "        export ANTHROPIC_API_KEY=\"sk-ant-api03-...\""
    exit 2
fi

MASK="${ANTHROPIC_API_KEY:0:12}...${ANTHROPIC_API_KEY: -4}"
echo "  key:  $MASK  (len=${#ANTHROPIC_API_KEY})"

# ─── 3. live round-trip ────────────────────────────────────────────────────
echo ""
echo "=== calling claude-haiku-4-5 ==="

python - <<'PY' || { echo ""; echo "FAIL: API call errored — see traceback above."; exit 3; }
import os, sys
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=50,
    messages=[
        {"role": "user",
         "content": "Respond with the single word 'ok' and nothing else."}
    ],
)

text = resp.content[0].text.strip()
print(f"  response: {text!r}")
print(f"  model:    {resp.model}")
print(f"  tokens:   input={resp.usage.input_tokens} "
      f"output={resp.usage.output_tokens}")
print(f"  stop:     {resp.stop_reason}")

# Sanity: the assistant should actually say "ok"
if "ok" not in text.lower():
    print(f"  WARN: response did not contain 'ok', got: {text!r}")
    sys.exit(4)
PY

echo ""
echo "=== all checks passed ==="
echo ""
echo "Your key is live and billed to whichever Anthropic workspace you"
echo "created it under. Double-check it's the SPAR workspace in the"
echo "console (https://console.anthropic.com/settings/workspaces) before"
echo "running anything expensive."
