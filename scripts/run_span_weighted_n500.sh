#!/bin/bash
# N=500 span-weighted analysis. Picks span-weighted top-500 per arch
# per layer, runs targeted scan for exemplars, labels top-100 with
# Haiku.
#
# For crosscoder, uses decoder-based concentration (the architectural
# fit) since its code has no T dim.

set -euo pipefail
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/hf_cache
PYTHON="${PYTHON:-/workspace/temp_xc/.venv/bin/python}"
SCAN_DIR="results/nlp_sweep/gemma/scans"
CKPT_DIR="results/nlp_sweep/gemma/ckpts"

echo "==== SPAN-WEIGHTED N=500 — $(date) ===="

# 1. TFA + Stacked picker (activation-based span, top-500)
for LAYER in resid_L25 resid_L13; do
    $PYTHON -u -m temporal_crosscoders.NLP.span_weighted_picker \
        --layer-key "$LAYER" --top-n 500 \
        --out "$SCAN_DIR/span_weighted_top500__${LAYER}__k50.json"
done

# 2. Crosscoder picker (decoder-based span, top-500)
for LAYER in resid_L25 resid_L13; do
    CKPT="$CKPT_DIR/crosscoder__gemma-2-2b-it__fineweb__${LAYER}__k50__seed42.pt"
    $PYTHON -u -m temporal_crosscoders.NLP.crosscoder_span_picker \
        --ckpt "$CKPT" --layer-key "$LAYER" --top-n 500 \
        --out "$SCAN_DIR/crosscoder_decoder_span_top500__${LAYER}__k50.json"
done

# 3. Scan top-500 feats for each (arch, layer)
for LAYER in resid_L25 resid_L13; do
    for ARCH in tfa_pos_pred tfa_pos stacked_sae; do
        FEATS=$(
            $PYTHON -c "
import json
d = json.load(open('$SCAN_DIR/span_weighted_top500__${LAYER}__k50.json'))
feats = [e['feat_idx'] for e in d['per_arch']['$ARCH']['top_span_weighted']]
print(' '.join(str(f) for f in feats))
"
        )
        echo ""
        echo "==== $ARCH @ $LAYER — scanning span-weighted top-500 ===="
        $PYTHON -u -m temporal_crosscoders.NLP.scan_specific_features \
            --arch "$ARCH" --layer-key "$LAYER" --k 50 --T 5 \
            --feats $FEATS \
            --out "$SCAN_DIR/span_weighted_scan500__${ARCH}__${LAYER}__k50.json"
    done

    # Crosscoder with decoder-based top-500
    FEATS=$(
        $PYTHON -c "
import json
d = json.load(open('$SCAN_DIR/crosscoder_decoder_span_top500__${LAYER}__k50.json'))
feats = [e['feat_idx'] for e in d['top_span_weighted']]
print(' '.join(str(f) for f in feats))
"
    )
    echo ""
    echo "==== crosscoder @ $LAYER — scanning decoder-span top-500 ===="
    $PYTHON -u -m temporal_crosscoders.NLP.scan_specific_features \
        --arch crosscoder --layer-key "$LAYER" --k 50 --T 5 \
        --feats $FEATS \
        --out "$SCAN_DIR/span_weighted_scan500__crosscoder__${LAYER}__k50.json"
done

# 4. Haiku label top-100 per arch-layer
echo ""
echo "==== LABELING TOP-100 — $(date) ===="
export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)
for LAYER in resid_L25 resid_L13; do
    for ARCH in stacked_sae crosscoder tfa_pos tfa_pos_pred; do
        $PYTHON -u -m temporal_crosscoders.NLP.explain_features \
            --scan "$SCAN_DIR/span_weighted_scan500__${ARCH}__${LAYER}__k50.json" \
            --out  "$SCAN_DIR/span_weighted_labels500__${ARCH}__${LAYER}__k50.json" \
            --top-features 100 --concurrency 2
    done
done

echo ""
echo "==== DONE — $(date) ===="
