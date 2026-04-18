#!/bin/bash
# =================================================================
# 16-Hour NLP Sweep: TFA-pos vs TXCDR vs Stacked SAE
# on Gemma-2-2B-IT and DeepSeek-R1-Distill-Llama-8B
#
# Prerequisites:
#   - HF_TOKEN set (Gemma is gated)
#   - uv venv at /workspace/temp_xc/.venv (see RUNPOD_INSTRUCTIONS.md)
#   - ~120 GB free disk
#   - RTX 5090 (32 GB VRAM)
#
# Usage:
#   TQDM_DISABLE=1 bash scripts/run_nlp_sweep_16h.sh 2>&1 | tee logs/nlp_sweep_16h.log
# =================================================================

set -euo pipefail

export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

PYTHON="${PYTHON:-/workspace/temp_xc/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: python not found at $PYTHON" >&2
    echo "  Run the setup steps in RUNPOD_INSTRUCTIONS.md to create the uv venv." >&2
    exit 1
fi
RESULTS_DIR="results/nlp_sweep"
STEPS=10000

mkdir -p "$RESULTS_DIR" logs

echo "============================================================"
echo "  NLP SWEEP — $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "  Disk free: $(df -h /workspace/temp_xc | tail -1 | awk '{print $4}')"
echo "============================================================"

# ── PHASE 1: Cache Gemma-2-2B-IT activations ────────────────────
echo ""
echo "==== PHASE 1: Cache Gemma-2-2B-IT (24K x 128 tok, layers 13+25) ===="
echo "  Started: $(date)"

$PYTHON -u -m src.data.nlp.cache_activations \
    --model gemma-2-2b-it \
    --dataset fineweb \
    --mode forward \
    --num-sequences 24000 \
    --seq-length 128 \
    --batch-size 64 \
    --layer_indices 13 25 \
    --components resid

echo "  Finished: $(date)"

# ── PHASE 2: Gemma sweeps (24 runs) ─────────────────────────────
echo ""
echo "==== PHASE 2: Gemma training sweeps ===="

for LAYER in resid_L25 resid_L13; do
    for SHUF_FLAG in "" "--shuffle-within-sequence"; do
        MODE="unshuffled"
        if [ -n "$SHUF_FLAG" ]; then MODE="shuffled"; fi

        echo ""
        echo "---- Gemma | ${LAYER} | ${MODE} | $(date) ----"

        $PYTHON -u -m src.pipeline.sweep \
            --dataset-type cached_activations \
            --model-name gemma-2-2b-it \
            --cached-dataset fineweb \
            --cached-layer-key "$LAYER" \
            --models tfa_pos stacked_sae crosscoder \
            --k 50 100 \
            --T 5 \
            --steps $STEPS \
            --expansion-factor 8 \
            --tfa-bottleneck-factor 8 \
            --tfa-batch-size 16 \
            --results-dir "${RESULTS_DIR}/gemma" \
            $SHUF_FLAG
    done
done

echo ""
echo "  Gemma sweeps complete: $(date)"

# ── PHASE 3: Cache DeepSeek-R1-Distill-Llama-8B activations ─────
echo ""
echo "==== PHASE 3: Cache DeepSeek-R1-8B (12K x 128 tok, layers 12+24) ===="
echo "  Started: $(date)"

$PYTHON -u -m src.data.nlp.cache_activations \
    --model deepseek-r1-distill-llama-8b \
    --dataset fineweb \
    --mode forward \
    --num-sequences 12000 \
    --seq-length 128 \
    --batch-size 32 \
    --layer_indices 12 24 \
    --components resid

echo "  Finished: $(date)"

# ── PHASE 4: DeepSeek sweeps (12 runs) ──────────────────────────
echo ""
echo "==== PHASE 4: DeepSeek training sweeps ===="

for SHUF_FLAG in "" "--shuffle-within-sequence"; do
    MODE="unshuffled"
    if [ -n "$SHUF_FLAG" ]; then MODE="shuffled"; fi

    echo ""
    echo "---- DeepSeek | resid_L12 | ${MODE} | $(date) ----"

    $PYTHON -u -m src.pipeline.sweep \
        --dataset-type cached_activations \
        --model-name deepseek-r1-distill-llama-8b \
        --cached-dataset fineweb \
        --cached-layer-key resid_L12 \
        --models tfa_pos stacked_sae crosscoder \
        --k 50 100 \
        --T 5 \
        --steps $STEPS \
        --expansion-factor 4 \
        --tfa-bottleneck-factor 8 \
        --tfa-batch-size 8 \
        --results-dir "${RESULTS_DIR}/deepseek" \
        $SHUF_FLAG
done

echo ""
echo "  DeepSeek sweeps complete: $(date)"

# ── PHASE 5: Aggregate results ──────────────────────────────────
echo ""
echo "==== PHASE 5: Aggregate ===="

$PYTHON -u -c "
import json, glob, os
results = []
for path in sorted(glob.glob('${RESULTS_DIR}/**/results_*.json', recursive=True)):
    with open(path) as f:
        results.extend(json.load(f))
out = '${RESULTS_DIR}/all_results.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Aggregated {len(results)} runs -> {out}')

# Quick summary table
print()
print(f'{\"Model\":30s} {\"Layer\":12s} {\"Shuf\":>5s} {\"k\":>4s} {\"NMSE\":>8s} {\"FVU\":>8s} {\"L0\":>6s}')
print('-' * 80)
for r in sorted(results, key=lambda x: (x.get('subject_model',''), x.get('layer_key',''), x.get('shuffled',False), x.get('k',0), x.get('arch',''))):
    fvu = r.get('nmse', 0)  # NMSE is FVU for centered data
    print(f'{r.get(\"arch\",\"?\"):30s} {r.get(\"layer_key\",\"?\"):12s} {str(r.get(\"shuffled\",\"\")):>5s} {r.get(\"k\",\"?\"):>4} {r.get(\"nmse\",0):>8.4f} {fvu:>8.4f} {r.get(\"l0\",0):>6.1f}')
"

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results: ${RESULTS_DIR}/"
echo "============================================================"
