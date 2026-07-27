#!/usr/bin/env bash
# Fetch shared HF artifacts (activation caches / checkpoints) for the
# temp_bench real-task experiments. No fetch script existed before the
# 2026-07-27 stacked-SAE sprint; upload conventions are in
# checkpoints/README.md and docs/huggingface-artifacts.md (aniket-phase7-y).
#
# Usage:
#   scripts/fetch_hf_artifacts.sh probing            # gemma-IT L13 act cache + probe cache anchor
#   scripts/fetch_hf_artifacts.sh rlhf               # phase-7 BASE stream for actmix_rlhf
#   scripts/fetch_hf_artifacts.sh ckpt <train_key>   # one checkpoint dir from temp-bench-models
#
# Requires: `huggingface_hub[cli]` in the venv, HF_TOKEN exported (public
# repos work without it, but rate limits are kinder with a token).

set -euo pipefail

DATA_REPO="han1823123123/temp-bench-data"
MODELS_REPO="han1823123123/temp-bench-models"
RLHF_STREAM_REPO="han1823123123/txcdr-base-data"
MIRROR_ROOT="${TEMP_BENCH_HF_MIRROR:-/workspace/caches/hf_mirror}"

HF_CLI=(python -m huggingface_hub.commands.huggingface_cli)
command -v huggingface-cli >/dev/null 2>&1 && HF_CLI=(huggingface-cli)

case "${1:?mode: probing | rlhf | ckpt <train_key>}" in
  probing)
    # v1 act-cache anchor consumed by experiments/probing/actmix/prep_cache.py
    "${HF_CLI[@]}" download "$DATA_REPO" --repo-type dataset \
      --include "act_cache/e4916bcae1881963/*" "probe_cache/gemma_2_2b_it_l13_fineweb_24k128/*" \
      --local-dir "$MIRROR_ROOT/temp-bench-data"
    echo "[fetch] now run: python experiments/probing/actmix/prep_cache.py --mirror $MIRROR_ROOT/temp-bench-data"
    ;;
  rlhf)
    "${HF_CLI[@]}" download "$RLHF_STREAM_REPO" --repo-type dataset \
      --include "activation_cache/resid_L12.npy" \
      --local-dir "/workspace/caches/rlhf/txcdr-base-data"
    echo "[fetch] now run: python experiments/explorations/actmix_rlhf/convert_train_cache.py"
    ;;
  ckpt)
    TRAIN_KEY="${2:?train_key required}"
    "${HF_CLI[@]}" download "$MODELS_REPO" \
      --include "$TRAIN_KEY/*" \
      --local-dir "checkpoints_hf_mirror"
    mkdir -p "checkpoints/$TRAIN_KEY"
    cp -v "checkpoints_hf_mirror/$TRAIN_KEY/"* "checkpoints/$TRAIN_KEY/"
    ;;
  *)
    echo "unknown mode: $1" >&2; exit 1
    ;;
esac
