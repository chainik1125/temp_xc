#!/bin/bash
# prefetch_text_dataset.sh — Pull a slice of a streamed HuggingFace text
# dataset into a local JSONL file so compute nodes (no network) can read it.
#
# Runs on a machine WITH internet access. On Trillium that means the login
# node, NOT inside an sbatch/srun allocation.
#
# Usage:
#   bash scripts/prefetch_text_dataset.sh fineweb 24000
#   bash scripts/prefetch_text_dataset.sh coding  10000
#   bash scripts/prefetch_text_dataset.sh fineweb 24000 "HuggingFaceFW/fineweb" "sample-10BT" "train"
#
# Output: data/prefetched/<dataset>_<N>.jsonl  (one JSON object per line,
# each with a "text" field). cache_activations.py checks for this file
# automatically.

set -euo pipefail

DATASET="${1:-fineweb}"
N="${2:-24000}"
HF_PATH="${3:-}"
SUBSET="${4:-}"
SPLIT="${5:-}"

# Defaults per known dataset
if [ -z "$HF_PATH" ]; then
    case "$DATASET" in
        fineweb)
            HF_PATH="HuggingFaceFW/fineweb"
            SUBSET="sample-10BT"
            SPLIT="train"
            ;;
        coding)
            HF_PATH="codeparrot/codeparrot-clean"
            SUBSET=""
            SPLIT="train"
            ;;
        *)
            echo "Unknown dataset '$DATASET'. Pass hf_path/subset/split explicitly."
            exit 1
            ;;
    esac
fi

# On Trillium this refuses to run inside an allocation (no internet)
HOST=$(hostname)
if [[ "$HOST" == trig0* ]]; then
    echo "FAIL: on '$HOST', a compute node with no outbound internet."
    echo "      Run this from the login node (trig-login01)."
    exit 1
fi

cd "$SCRATCH/temp_xc" 2>/dev/null || cd "$(dirname "$0")/.."
source "$SCRATCH/temp_xc/scripts/trillium_activate.sh" 2>/dev/null || true

REPO="$(pwd)"
OUT_DIR="$REPO/data/prefetched"
OUT="$OUT_DIR/${DATASET}_${N}.jsonl"
mkdir -p "$OUT_DIR"

if [ -f "$OUT" ]; then
    EXISTING=$(wc -l < "$OUT")
    if [ "$EXISTING" -ge "$N" ]; then
        echo ">> $OUT already has $EXISTING lines, skipping."
        exit 0
    fi
    echo ">> $OUT only has $EXISTING / $N lines, re-fetching."
fi

echo ">> streaming $HF_PATH ${SUBSET:+($SUBSET)} / $SPLIT  target=$N samples"
echo ">> output:   $OUT"

python - <<PY
import json
from datasets import load_dataset
from tqdm.auto import tqdm

hf_path = "$HF_PATH"
subset  = "$SUBSET" or None
split   = "$SPLIT"
n       = $N
out_path = "$OUT"

kwargs = dict(split=split, streaming=True)
if subset:
    ds = load_dataset(hf_path, subset, **kwargs)
else:
    ds = load_dataset(hf_path, **kwargs)

kept = 0
with open(out_path, "w") as fout:
    for sample in tqdm(ds, total=n, desc="prefetch"):
        txt = None
        for col in ("text", "content", "question", "problem"):
            if col in sample and sample[col]:
                txt = str(sample[col])
                break
        if not txt or len(txt) < 100:
            continue
        fout.write(json.dumps({"text": txt}) + "\n")
        kept += 1
        if kept >= n:
            break

print(f"  wrote {kept} samples to {out_path}")
PY

ls -lh "$OUT"
