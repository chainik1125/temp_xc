#!/bin/bash
# runpod_saebench_cache.sh — Cache Gemma-2-2B FineWeb activations at the
# 5-layer window {L10..L14} for training all SAEBench-experiment
# architectures. Single Gemma forward; all 5 hooks captured per pass.
#
# Output: data/cached_activations/gemma-2-2b/fineweb/resid_L{10..14}.npy
#   Size: ~6 GB per layer × 5 = ~30 GB total.
#   SAE + TempXC train on resid_L12.npy (one layer).
#   MLC trains on all 5 stacked via data_format=multi_layer_activations.
#
# Resumable: cache_activations.py skips per-layer files that already exist.
#
# Usage:
#   bash scripts/runpod_saebench_cache.sh           # default 6000 seqs × 128 tok
#   NUM_SEQS=12000 SEQ_LEN=256 bash scripts/runpod_saebench_cache.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

NUM_SEQS="${NUM_SEQS:-6000}"
SEQ_LEN="${SEQ_LEN:-128}"

echo "=== saebench multi-layer cache ==="
echo "  model:     gemma-2-2b"
echo "  dataset:   fineweb"
echo "  layers:    10, 11, 12, 13, 14"
echo "  n_seqs:    $NUM_SEQS"
echo "  seq_len:   $SEQ_LEN"
echo ""

python -m temporal_crosscoders.NLP.cache_activations \
    --model gemma-2-2b \
    --dataset fineweb \
    --mode forward \
    --num-sequences "$NUM_SEQS" \
    --seq-length "$SEQ_LEN" \
    --layer_indices 10 11 12 13 14 \
    --components resid

echo ""
echo "=== cache done ==="
ls -lh data/cached_activations/gemma-2-2b/fineweb/resid_L*.npy
