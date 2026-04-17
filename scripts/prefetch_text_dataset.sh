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
        stack-python)
            # Python source code with inline content. bigcode/the-stack-v2
            # is unusable here because it only ships blob_ids (content
            # lives on Software Heritage S3). Use starcoderdata's Python
            # subset (gated BigCode license — accept at
            # https://huggingface.co/datasets/bigcode/starcoderdata) or
            # fall back to codeparrot/codeparrot-clean (open access).
            HF_PATH="bigcode/starcoderdata"
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

# Compute Canada's pyarrow has a known interpreter-teardown race that fires
# AFTER all work completes and the file is fully written. It surfaces as:
#     Fatal Python error: PyGILState_Release
# Harmless — just Python's cleanup racing pyarrow's cleanup. We run the
# Python block without `set -e` and judge success by whether the output
# file ended up with the expected number of lines, ignoring the exit code.
set +e
python - <<PY
import json
from datasets import load_dataset
from tqdm.auto import tqdm

hf_path  = "$HF_PATH"
subset   = "$SUBSET" or None
split    = "$SPLIT"
n        = $N
out_path = "$OUT"
dataset  = "$DATASET"

kwargs = dict(split=split, streaming=True)
if subset:
    ds = load_dataset(hf_path, subset, **kwargs)
else:
    ds = load_dataset(hf_path, **kwargs)

# stack-python: language filter + line-count filter (100-2000 lines)
if dataset == "stack-python":
    ds = ds.filter(lambda s: s.get("language") == "Python")

def accept(txt: str) -> bool:
    if not txt or len(txt) < 100:
        return False
    if dataset == "stack-python":
        n_lines = txt.count("\n") + 1
        if n_lines < 100 or n_lines > 2000:
            return False
    return True

kept = 0
with open(out_path, "w") as fout:
    for sample in tqdm(ds, total=n, desc="prefetch"):
        txt = None
        for col in ("text", "content", "question", "problem"):
            if col in sample and sample[col]:
                txt = str(sample[col])
                break
        if not accept(txt):
            continue
        fout.write(json.dumps({"text": txt}) + "\n")
        kept += 1
        if kept >= n:
            break

print(f"  wrote {kept} samples to {out_path}", flush=True)
PY
set -e

# Judge success by file state, not python exit code.
if [ ! -f "$OUT" ]; then
    echo "FAIL: $OUT was not created."
    exit 1
fi
LINES=$(wc -l < "$OUT")
echo ">> wrote $LINES lines to $OUT"
if [ "$LINES" -lt "$N" ]; then
    echo "FAIL: expected $N lines but got $LINES. Re-run the prefetch."
    exit 1
fi
ls -lh "$OUT"
