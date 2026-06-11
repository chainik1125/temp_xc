#!/bin/bash
cd /workspace/c7
export HF_HOME=/workspace/hf
ROOT=data/llama_3_1_8b/resid/L10
# wait for cache
until [ -f "$ROOT/activations.fp16.npy" ] && python - <<'PY'
import json, sys
import os
p = "data/llama_3_1_8b/resid/L10/progress.json"
m = "data/llama_3_1_8b/resid/L10/meta.json"
sys.exit(0 if os.path.exists(m) else 1)
PY
do sleep 20; done
echo "=== cache ready $(date -u)"
COMMON="--hook resid --layer 10 --d-sae 32768 --k-pos 20 --steps 10000 --batch 1024 --lr 3e-4"
python -m experiments.phase7_unification.case_studies.backtracking.train_llama_txc --arch topk_sae $COMMON 2>&1 | tail -5
echo "=== topk_sae done $(date -u)"
python -m experiments.phase7_unification.case_studies.backtracking.train_llama_txc --arch txc_bare --T 5 $COMMON 2>&1 | tail -5
echo "=== txc_bare done $(date -u)"
python c7_spectral_arm.py --T 5 --d-sae 32768 --k-pos 20 --steps 10000 --batch 1024 --lr 3e-4 2>&1 | tail -5
echo "=== spectral done $(date -u)"
python c7_detect.py 2>&1 | tail -20
echo "=== ALL DONE $(date -u)"
