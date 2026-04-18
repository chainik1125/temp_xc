#!/bin/bash
# Rerun N=200 span-weighted scans seeded so the top-10 exemplars are
# reproducible. Uses scan_specific_features with the np.random.seed(42)
# / torch.manual_seed(42) patch just before finder.run().

set -euo pipefail
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/hf_cache
PYTHON="${PYTHON:-/workspace/temp_xc/.venv/bin/python}"
SCAN_DIR="results/nlp_sweep/gemma/scans"
CKPT_DIR="results/nlp_sweep/gemma/ckpts"

echo "==== N=200 SEEDED RERUN — $(date) ===="

# Ensure top-200 picker is present for each layer
for LAYER in resid_L25 resid_L13; do
    if [ ! -s "$SCAN_DIR/span_weighted_top200__${LAYER}__k50.json" ]; then
        $PYTHON -u -m temporal_crosscoders.NLP.span_weighted_picker \
            --layer-key "$LAYER" --top-n 200 \
            --out "$SCAN_DIR/span_weighted_top200__${LAYER}__k50.json"
    fi
done

# Scan specific for tfa/stacked
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
        echo "==== $ARCH @ $LAYER — seeded scan (N=200) ===="
        $PYTHON -u -m temporal_crosscoders.NLP.scan_specific_features \
            --arch "$ARCH" --layer-key "$LAYER" --k 50 --T 5 \
            --feats $FEATS \
            --out "$SCAN_DIR/span_weighted_scan200__${ARCH}__${LAYER}__k50.json"
    done
done

# Crosscoder: decoder-based picker, active-gated from /tmp/cx_scan
for LAYER in resid_L25 resid_L13; do
    ACTIVE="/tmp/cx_scan/scan__crosscoder__${LAYER}__k50.json"
    CKPT="$CKPT_DIR/crosscoder__gemma-2-2b-it__fineweb__${LAYER}__k50__seed42.pt"
    if [ ! -s "$ACTIVE" ]; then
        echo "WARNING: no active scan at $ACTIVE"; continue
    fi
    $PYTHON -u -m temporal_crosscoders.NLP.crosscoder_span_picker \
        --ckpt "$CKPT" --layer-key "$LAYER" --top-n 200 \
        --active-scan "$ACTIVE" \
        --out "$SCAN_DIR/crosscoder_decoder_span_top200__${LAYER}__k50.json"

    FEATS=$(
        $PYTHON -c "
import json
d = json.load(open('$SCAN_DIR/crosscoder_decoder_span_top200__${LAYER}__k50.json'))
feats = [e['feat_idx'] for e in d['top_span_weighted']]
print(' '.join(str(f) for f in feats))
"
    )
    echo ""
    echo "==== crosscoder @ $LAYER — seeded scan (N=200) ===="
    $PYTHON -u -m temporal_crosscoders.NLP.scan_specific_features \
        --arch crosscoder --layer-key "$LAYER" --k 50 --T 5 \
        --feats $FEATS \
        --out "$SCAN_DIR/span_weighted_scan200__crosscoder__${LAYER}__k50.json"
done

# Relabel top-50 per arch
echo ""
echo "==== LABELING TOP-50 — $(date) ===="
export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)
for LAYER in resid_L25 resid_L13; do
    for ARCH in stacked_sae crosscoder tfa_pos tfa_pos_pred; do
        rm -f "$SCAN_DIR/span_weighted_labels200__${ARCH}__${LAYER}__k50.json"
        $PYTHON -u -m temporal_crosscoders.NLP.explain_features \
            --scan "$SCAN_DIR/span_weighted_scan200__${ARCH}__${LAYER}__k50.json" \
            --out  "$SCAN_DIR/span_weighted_labels200__${ARCH}__${LAYER}__k50.json" \
            --top-features 50 --concurrency 2
    done
done

echo ""
echo "==== DONE — $(date) ===="
