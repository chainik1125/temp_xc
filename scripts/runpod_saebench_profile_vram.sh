#!/bin/bash
# runpod_saebench_profile_vram.sh — profile peak VRAM for TempXC at a
# given T on the current pod's GPU. Used by the T-sweep orchestrator to
# find the max feasible T before launching the full sweep.
#
# Trains for only 10 steps, reports peak VRAM in GB (machine-parseable),
# then exits cleanly. Cheap — ~30 seconds per T value.
#
# Usage:
#   bash scripts/runpod_saebench_profile_vram.sh --t 20
#   bash scripts/runpod_saebench_profile_vram.sh --t 40
#
# Output format (last line only): `VRAM_PEAK_GB=<float>`
# Non-zero exit on OOM.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

T=5
PROTOCOL="A"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --t)        T="$2";        shift 2;;
        --protocol) PROTOCOL="$2"; shift 2;;
        *) echo "unknown flag: $1"; exit 2;;
    esac
done

echo "=== profile VRAM: TempXC T=$T, protocol=$PROTOCOL ==="

python - <<PY
import torch
from src.bench.architectures.crosscoder import CrosscoderSpec
from src.bench.saebench.configs import D_MODEL, D_SAE
from src.bench.saebench.matching_protocols import protocol_k

T = $T
protocol = "$PROTOCOL"
k = protocol_k("tempxc", protocol, t=T)

device = torch.device("cuda:0")
torch.cuda.reset_peak_memory_stats(device)

spec = CrosscoderSpec(T=T)
model = spec.create(d_in=D_MODEL, d_sae=D_SAE, k=k, device=device)

# Run 10 training-like steps on random data to populate activations,
# optimizer state, grad buffers
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
batch = 32
for step in range(10):
    x = torch.randn(batch, T, D_MODEL, device=device)
    loss, _, _ = model(x)
    opt.zero_grad()
    loss.backward()
    opt.step()

peak_bytes = torch.cuda.max_memory_allocated(device)
peak_gb = peak_bytes / (1024 ** 3)
total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

print(f"  T={T}  k={k}  batch={batch}")
print(f"  peak allocated : {peak_gb:.2f} GB")
print(f"  total device   : {total_gb:.2f} GB")
print(f"  headroom       : {total_gb - peak_gb:.2f} GB")
# Machine-parseable last line
print(f"VRAM_PEAK_GB={peak_gb:.3f}")
PY
