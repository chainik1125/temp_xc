#!/bin/bash
# trillium_sanity_pipeline.sh — End-to-end dry run of the bench data pipeline
# on cached activations: build_pipeline(), sample a flat / seq / window batch,
# print shapes. Exercises the full config → data.py → model_registry chain
# without training anything. Needs a GPU because we push tensors to cuda.
#
#   bash scripts/trillium_sanity_pipeline.sh                                       # defaults
#   bash scripts/trillium_sanity_pipeline.sh gemma-2-2b fineweb resid_L13
set -euo pipefail

MODEL="${1:-deepseek-r1-distill-llama-8b}"
DATASET="${2:-gsm8k}"
LAYER="${3:-resid_L12}"

cd "$SCRATCH/temp_xc"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    exec srun --account=rrg-aspuru --nodes=1 --gpus-per-node=1 --cpus-per-task=4 \
         --time=00:30:00 --job-name=txc-sanity --pty bash -lc "
bash scripts/trillium_sanity_pipeline.sh $MODEL $DATASET $LAYER
"
fi

source scripts/trillium_activate.sh

python - <<PY
import torch
from src.bench.config import DataConfig
from src.bench.data import build_pipeline

cfg = DataConfig(
    dataset_type="cached_activations",
    model_name="$MODEL",
    cached_dataset="$DATASET",
    cached_layer_key="$LAYER",
    eval_n_seq=64,
)
p = build_pipeline(cfg, torch.device("cuda"), window_sizes=[2, 5])
print(f"eval_hidden : {tuple(p.eval_hidden.shape)}")
print(f"flat (256)  : {tuple(p.gen_flat(256).shape)}")
print(f"seq  (32)   : {tuple(p.gen_seq(32).shape)}")
for T, g in p.gen_windows.items():
    print(f"win  T={T}   : {tuple(g(128).shape)}")
print("OK")
PY
