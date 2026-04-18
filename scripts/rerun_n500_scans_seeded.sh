#!/bin/bash
# Re-run all N=500 scan_specific invocations with seeded RNG so the
# scans are deterministic / reproducible.

set -euo pipefail
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/hf_cache
PYTHON="${PYTHON:-/workspace/temp_xc/.venv/bin/python}"
SCAN_DIR="results/nlp_sweep/gemma/scans"

echo "==== RERUN N=500 scans (seeded) — $(date) ===="

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
        echo "==== $ARCH @ $LAYER — seeded scan ===="
        $PYTHON -u -m temporal_crosscoders.NLP.scan_specific_features \
            --arch "$ARCH" --layer-key "$LAYER" --k 50 --T 5 \
            --feats $FEATS \
            --out "$SCAN_DIR/span_weighted_scan500__${ARCH}__${LAYER}__k50.json"
    done
    # crosscoder
    FEATS=$(
        $PYTHON -c "
import json
d = json.load(open('$SCAN_DIR/crosscoder_decoder_span_top500__${LAYER}__k50.json'))
feats = [e['feat_idx'] for e in d['top_span_weighted']]
print(' '.join(str(f) for f in feats))
"
    )
    echo ""
    echo "==== crosscoder @ $LAYER — seeded scan ===="
    $PYTHON -u -m temporal_crosscoders.NLP.scan_specific_features \
        --arch crosscoder --layer-key "$LAYER" --k 50 --T 5 \
        --feats $FEATS \
        --out "$SCAN_DIR/span_weighted_scan500__crosscoder__${LAYER}__k50.json"
done

# Re-label with the new scan content
echo ""
echo "==== RELABEL — $(date) ===="
export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)
for LAYER in resid_L25 resid_L13; do
    for ARCH in stacked_sae crosscoder tfa_pos tfa_pos_pred; do
        # force re-label by clearing old file
        rm -f "$SCAN_DIR/span_weighted_labels500__${ARCH}__${LAYER}__k50.json"
        $PYTHON -u -m temporal_crosscoders.NLP.explain_features \
            --scan "$SCAN_DIR/span_weighted_scan500__${ARCH}__${LAYER}__k50.json" \
            --out  "$SCAN_DIR/span_weighted_labels500__${ARCH}__${LAYER}__k50.json" \
            --top-features 100 --concurrency 2
    done
done

echo ""
echo "==== DONE — $(date) ===="
