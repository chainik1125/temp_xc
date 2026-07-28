#!/bin/bash
# mac-d — post-bootstrap chain for the RLHF pf grid pods.
#
#   bash pod_cache_and_receipt.sh
#
# Self-driving so a pod needs one launch, not a babysitter:
#   1. wait for BOOTSTRAP-DONE (bootstrap may still be running)
#   2. fetch the 14.16 GB activation cache from HF (the long pole)
#   3. install it into the keyed layout via convert_train_cache.py
#   4. run the resident-buffer equivalence receipt ON CUDA
#
# Writes markers the mac polls for: CACHE-OK, RECEIPT-PASS/RECEIPT-FAIL,
# CHAIN-DONE. Any failure writes CHAIN-FAIL and stops — an idle pod that
# says why beats a pod that silently does nothing.
set -uo pipefail
LOG=/workspace/chain.log
exec >>"$LOG" 2>&1
echo "=== chain start $(date -u +%FT%TZ) ==="

fail() { echo "CHAIN-FAIL: $*"; exit 1; }

# 1) wait for bootstrap
for i in $(seq 1 120); do
    grep -q BOOTSTRAP-DONE /workspace/bootstrap.log 2>/dev/null && break
    [ "$i" = 120 ] && fail "bootstrap did not finish in 60 min"
    sleep 30
done
echo "bootstrap ok"

cd /workspace/temp_xc || fail "no repo"
export HF_TOKEN="$(cat /workspace/.tokens/hf_token 2>/dev/null)"
SRC=/workspace/caches/rlhf/txcdr-base-data

# 2) fetch cache (idempotent; hf transfer is resumable)
if [ ! -f "$SRC/activation_cache/resid_L12.npy" ]; then
    echo "--- fetching 14.16 GB cache ---"
    # Python API, not the CLI: `huggingface-cli` is deprecated in
    # huggingface_hub 1.13 (renamed `hf`) and its flags moved, which is
    # what failed the first launch. snapshot_download is stable and
    # resumable.
    .venv/bin/python - <<'PY' || fail "hf download"
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="han1823123123/txcdr-base-data",
    repo_type="dataset",
    local_dir="/workspace/caches/rlhf/txcdr-base-data",
    allow_patterns=["activation_cache/*"],
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=8,
)
print("downloaded ->", p)
PY
fi
[ -f "$SRC/activation_cache/resid_L12.npy" ] || fail "cache file absent after download"
echo "CACHE-OK $(du -sh "$SRC" | cut -f1)"

# 3) keyed install (hardlink; near-free)
ACTMIX_RLHF_CACHE_SRC="$SRC/activation_cache" \
    .venv/bin/python -m experiments.explorations.actmix_rlhf.convert_train_cache \
    || fail "convert_train_cache"
echo "INSTALL-OK"

# 4) equivalence receipt on CUDA — the claim is device-independent by
#    construction but has only been shown on MPS.
if .venv/bin/python scripts/verify_resident_buffer.py \
        gemma_2_2b_base_l12_phase7 10 1024 2>&1 | tee /workspace/receipt.log \
        | grep -q "VERDICT: PASS"; then
    echo "RECEIPT-PASS"
else
    echo "RECEIPT-FAIL"; fail "resident buffer not bitwise identical on CUDA"
fi

echo "CHAIN-DONE $(date -u +%FT%TZ)"
