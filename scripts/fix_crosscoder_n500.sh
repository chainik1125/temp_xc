#!/bin/bash
# Fix crosscoder N=500 with active-feature-gated span-weighted picker.
#   1. Re-scan crosscoder with --top-features 3000 to get mass for all
#      ~2970 active features (L25) / ~2435 active (L13).
#   2. Pick top-500 by (1 - decoder_conc) * mass (mass from the rescan).
#   3. Targeted scan_specific to pull top-10 exemplars for those 500.
#   4. Haiku label top-100.

set -euo pipefail
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/hf_cache
PYTHON="${PYTHON:-/workspace/temp_xc/.venv/bin/python}"
SCAN_DIR="results/nlp_sweep/gemma/scans"
CKPT_DIR="results/nlp_sweep/gemma/ckpts"

echo "==== FIX CROSSCODER N=500 — $(date) ===="

# 1. Rescan crosscoder, saving top-3000 features (= all active)
for LAYER in resid_L25 resid_L13; do
    echo ""
    echo "==== crosscoder scan top-3000 @ $LAYER ===="
    $PYTHON -u -m temporal_crosscoders.NLP.scan_features \
        --arches crosscoder \
        --layer-key "$LAYER" --k 50 --T 5 \
        --sample-chains 1000 \
        --top-features 3000 \
        --top-k 10 \
        --ckpt-dir "$CKPT_DIR" \
        --out-dir /tmp/cx_scan
done

# 2. Re-pick with active-scan flag
for LAYER in resid_L25 resid_L13; do
    CKPT="$CKPT_DIR/crosscoder__gemma-2-2b-it__fineweb__${LAYER}__k50__seed42.pt"
    ACTIVE="/tmp/cx_scan/scan__crosscoder__${LAYER}__k50.json"
    echo ""
    echo "==== crosscoder span picker (active-gated) @ $LAYER ===="
    $PYTHON -u -m temporal_crosscoders.NLP.crosscoder_span_picker \
        --ckpt "$CKPT" --layer-key "$LAYER" --top-n 500 \
        --active-scan "$ACTIVE" \
        --out "$SCAN_DIR/crosscoder_decoder_span_top500__${LAYER}__k50.json"
done

# 3. scan_specific for the new 500 features
for LAYER in resid_L25 resid_L13; do
    FEATS=$(
        $PYTHON -c "
import json
d = json.load(open('$SCAN_DIR/crosscoder_decoder_span_top500__${LAYER}__k50.json'))
feats = [e['feat_idx'] for e in d['top_span_weighted']]
print(' '.join(str(f) for f in feats))
"
    )
    echo ""
    echo "==== crosscoder scan_specific @ $LAYER ===="
    $PYTHON -u -m temporal_crosscoders.NLP.scan_specific_features \
        --arch crosscoder --layer-key "$LAYER" --k 50 --T 5 \
        --feats $FEATS \
        --out "$SCAN_DIR/span_weighted_scan500__crosscoder__${LAYER}__k50.json"
done

# 4. Haiku label top-100
echo ""
echo "==== LABELING TOP-100 CROSSCODER ===="
export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)
for LAYER in resid_L25 resid_L13; do
    # overwrite old labels (which had only 25/12 active features)
    rm -f "$SCAN_DIR/span_weighted_labels500__crosscoder__${LAYER}__k50.json"
    $PYTHON -u -m temporal_crosscoders.NLP.explain_features \
        --scan "$SCAN_DIR/span_weighted_scan500__crosscoder__${LAYER}__k50.json" \
        --out  "$SCAN_DIR/span_weighted_labels500__crosscoder__${LAYER}__k50.json" \
        --top-features 100 --concurrency 2
done

echo ""
echo "==== DONE — $(date) ===="
