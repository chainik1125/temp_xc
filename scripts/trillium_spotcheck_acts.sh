#!/bin/bash
# trillium_spotcheck_acts.sh — Quick sanity on cached activation tensors.
# No GPU needed — runs on login node. Prints shapes, nonzero fraction,
# mean trace length, and the layer_specs sidecar.
#
#   bash scripts/trillium_spotcheck_acts.sh                                          # defaults
#   bash scripts/trillium_spotcheck_acts.sh deepseek-r1-distill-llama-8b gsm8k resid_L12
set -euo pipefail

MODEL="${1:-deepseek-r1-distill-llama-8b}"
DATASET="${2:-gsm8k}"
LAYER="${3:-resid_L12}"

cd "$SCRATCH/temp_xc"
source scripts/trillium_activate.sh

DIR="data/cached_activations/$MODEL/$DATASET"
if [ ! -d "$DIR" ]; then
    echo "No cache dir: $DIR"
    echo "Run trillium_cache_reasoning.sh first."
    exit 1
fi

python - <<PY
import json, os, numpy as np
d = "$DIR"
layer = "$LAYER"
acts_path = os.path.join(d, f"{layer}.npy")
print(f"=== {d} ===")
print("files:", sorted(os.listdir(d)))
acts = np.load(acts_path, mmap_mode="r")
print(f"{layer}.npy shape: {acts.shape}  dtype={acts.dtype}")
print(f"  nonzero fraction: {(acts != 0).mean():.3f}")
print(f"  mean abs        : {np.abs(acts[:32]).mean():.3f}")

lens_p = os.path.join(d, "trace_lengths.npy")
if os.path.exists(lens_p):
    lens = np.load(lens_p)
    print(f"trace_lengths   : mean={lens.mean():.1f}  min={lens.min()}  max={lens.max()}")

meta_p = os.path.join(d, "layer_specs.json")
if os.path.exists(meta_p):
    with open(meta_p) as f:
        print("meta:", json.dumps(json.load(f), indent=2))
PY
