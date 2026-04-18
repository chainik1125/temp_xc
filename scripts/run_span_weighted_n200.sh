#!/bin/bash
# Extended span-weighted pipeline at N=200 (vs N=25 prior). Provides
# statistical power. Skip Haiku labels on all 200 — only label a random
# sample of 50 per arch-layer for the label-collapse metric.

set -euo pipefail

export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

PYTHON="${PYTHON:-/workspace/temp_xc/.venv/bin/python}"
SCAN_DIR="results/nlp_sweep/gemma/scans"

echo "==== SPAN-WEIGHTED N=200 PIPELINE — $(date) ===="

# 1. Scan top-200 feats for each (arch, layer). Skip crosscoder — its
# span_all is buggy (feat_acts has no T dim).
for LAYER in resid_L25 resid_L13; do
    for ARCH in tfa_pos_pred tfa_pos stacked_sae; do
        FEATS=$(
            $PYTHON -c "
import json
d = json.load(open('$SCAN_DIR/span_weighted_top200__${LAYER}__k50.json'))
feats = [e['feat_idx'] for e in d['per_arch']['$ARCH']['top_span_weighted']]
print(' '.join(str(f) for f in feats))
"
        )
        echo ""
        echo "==== $ARCH @ $LAYER — scanning span-weighted top-200 ===="
        $PYTHON -u -m temporal_crosscoders.NLP.scan_specific_features \
            --arch "$ARCH" --layer-key "$LAYER" --k 50 --T 5 \
            --feats $FEATS \
            --out "$SCAN_DIR/span_weighted_scan200__${ARCH}__${LAYER}__k50.json"
    done
done

# 2. Haiku-label the top-50 (not all 200) of each — label cost ~1 min/arch
echo ""
echo "==== LABELING TOP-50 OF TOP-200 ===="
export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)
for LAYER in resid_L25 resid_L13; do
    for ARCH in stacked_sae tfa_pos tfa_pos_pred; do
        $PYTHON -u -m temporal_crosscoders.NLP.explain_features \
            --scan "$SCAN_DIR/span_weighted_scan200__${ARCH}__${LAYER}__k50.json" \
            --out  "$SCAN_DIR/span_weighted_labels200__${ARCH}__${LAYER}__k50.json" \
            --top-features 50 --concurrency 2
    done
done

echo ""
echo "==== DONE — $(date) ===="
