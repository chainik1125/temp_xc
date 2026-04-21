#!/bin/bash
# setup_env.sh — write a .env template with quoted empty values so you can
# paste your keys between the "" on each line.
#
# Usage:
#   bash scripts/setup_env.sh              # write .env (refuses if exists)
#   bash scripts/setup_env.sh --force      # overwrite existing .env
#
# After running: open .env in an editor and fill in the keys, then:
#   set -a; source .env; set +a

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

if [[ -f .env && "$FORCE" != "1" ]]; then
    echo "[error] .env already exists. Pass --force to overwrite." >&2
    exit 1
fi

cat > .env <<'EOF'
# ── Required for the hybrid Gap Recovery run (Phase 2 + 3) ──

# HuggingFace token — needed to download meta-llama/Llama-3.1-8B (gated)
# and deepseek-ai/DeepSeek-R1-Distill-Llama-8B. Get one at
# https://huggingface.co/settings/tokens and accept the Llama-3.1 license.
HF_TOKEN=""

# ── Required only for the taxonomy side-channel (Phase 1 / MODE=full) ──

# Anthropic Claude API — Haiku 4.5 judges the three taxonomy metrics.
# Not needed for MODE=hybrid.
ANTHROPIC_API_KEY=""

# OpenAI API — used only by the GPT-4o drift bridge at smoke time.
# Not needed for MODE=hybrid.
OPENAI_API_KEY=""

# OpenRouter API — REQUIRED for MODE=hybrid. Venhoff's hybrid_token.py
# routes its answer-grading judge (openai/gpt-5.2) through OpenRouter
# via chat_limiter. We use the same judge for apples-to-apples
# comparability with their 3.5% baseline. Get a key at
# https://openrouter.ai/keys — free tier + credit-based thereafter.
OPENROUTER_API_KEY=""

# ── Optional ──

# Weights & Biases — training telemetry. Leave empty to disable W&B.
WANDB_API_KEY=""

# ── Fixed (don't edit unless you know why) ──

# Redirect HF cache to the persistent volume so model weights survive
# pod restarts and don't fill the root disk.
HF_HOME="/workspace/hf_cache"

# Enable hf_transfer for faster Llama-8B downloads (~3× speedup).
HF_HUB_ENABLE_HF_TRANSFER="1"

# Unbuffer python stdout so tail -f shows log lines live.
PYTHONUNBUFFERED="1"
EOF

chmod 600 .env

echo "[done] wrote .env template at $(pwd)/.env"
echo "[info] next steps:"
echo "  1. Edit .env and paste your keys between the quotes."
echo "  2. Load into shell:    set -a; source .env; set +a"
echo "  3. Verify HF login:    huggingface-cli login --token \"\$HF_TOKEN\""
