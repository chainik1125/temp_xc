#!/bin/bash
# Span-weighted feature pipeline for L25 and L13:
#   1. span_weighted_picker: rank by (1 - conc) * mass, top-25 per arch
#   2. scan_specific_features: pull top-10 exemplars for each selected feat
#   3. explain_features: Haiku label them
#
# Re-uses precomputed span_all__*.json from all_features_span.py.

set -euo pipefail

export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

PYTHON="${PYTHON:-/workspace/temp_xc/.venv/bin/python}"
SCAN_DIR="results/nlp_sweep/gemma/scans"

echo "============================================================"
echo "  SPAN-WEIGHTED PIPELINE — $(date)"
echo "============================================================"

for LAYER in resid_L25 resid_L13; do
    for ARCH in tfa_pos_pred tfa_pos crosscoder stacked_sae; do
        # Pull feats needing scan for this arch-layer
        FEATS=$(
            $PYTHON -c "
import json
d = json.load(open('$SCAN_DIR/span_weighted_top__${LAYER}__k50.json'))
feats = [e['feat_idx'] for e in d['per_arch']['$ARCH']['top_span_weighted']]
print(' '.join(str(f) for f in feats))
"
        )
        echo ""
        echo "==== $ARCH @ $LAYER — scanning span-weighted top-25 feats ===="
        echo "    feats: $FEATS"
        $PYTHON -u -m temporal_crosscoders.NLP.scan_specific_features \
            --arch "$ARCH" \
            --layer-key "$LAYER" \
            --k 50 --T 5 \
            --feats $FEATS \
            --out "$SCAN_DIR/span_weighted_scan__${ARCH}__${LAYER}__k50.json"
    done
done

echo ""
echo "==== LABELING WITH HAIKU — $(date) ===="
export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)
for LAYER in resid_L25 resid_L13; do
    for ARCH in stacked_sae crosscoder tfa_pos tfa_pos_pred; do
        $PYTHON -u -m temporal_crosscoders.NLP.explain_features \
            --scan "$SCAN_DIR/span_weighted_scan__${ARCH}__${LAYER}__k50.json" \
            --out  "$SCAN_DIR/span_weighted_labels__${ARCH}__${LAYER}__k50.json" \
            --top-features 25 --concurrency 2
    done
done

echo ""
echo "============================================================"
echo "  SPAN-WEIGHTED PIPELINE COMPLETE — $(date)"
echo "============================================================"
