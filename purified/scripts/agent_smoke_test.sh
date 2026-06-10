#!/usr/bin/env bash
# agent_smoke_test.sh — verify that an agent's environment is healthy.
#
# Run from inside `purified/` after `uv sync`. Verifies:
#   1. .venv exists and torch/cuda are visible
#   2. temp_bench (core.config/cache/runner/schemas) imports cleanly
#   3. configs/archs.yaml + configs/data.yaml registries load
#   4. cache-key + schema tests pass (pytest)
#   5. registry self-check + token resolution are reported
#
# Run on every agent session start. CI doesn't exist on `arxiv`; this
# script is the cheap proxy.

set -eu

# This script always operates inside purified/ even if invoked from
# elsewhere (e.g. `bash purified/scripts/agent_smoke_test.sh` from repo
# root). But the AGENT's shell should already be in purified/ so that
# every other command they run uses purified/.venv directly.
cd "$(dirname "$0")/.."   # purified/

if [ "$(basename "$PWD")" != "purified" ]; then
    echo "✗ smoke test must run from purified/ (cwd: $PWD)" >&2
    exit 1
fi

echo "[smoke] purified root: $(pwd)"

# 1. venv exists
if [ ! -d ".venv" ]; then
    echo "✗ no .venv — run: uv sync"
    exit 1
fi

# 2. minimum imports
TQDM_DISABLE=1 .venv/bin/python -c "
import torch
print(f'✓ torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')

import temp_bench
print(f'✓ temp_bench {temp_bench.__version__}')

from temp_bench.core import config, cache, runner, schemas
print(f'✓ framework modules: core.config, core.cache, core.runner, core.schemas')

archs = config.list_archs()
print(f'✓ archs ({len(archs)}): {archs}')

dss = config.list_datasources()
print(f'✓ datasources ({len(dss)}): {dss}')
" || { echo "✗ import smoke failed"; exit 1; }

# 3. tests
echo
echo "[smoke] running pytest…"
TQDM_DISABLE=1 .venv/bin/python -m pytest tests/ -q 2>&1 || {
    echo "✗ tests failed"
    exit 1
}

# 4. registry self-check + token resolution (the RunPod-relevant bits)
echo
echo "[smoke] registry + token check…"
TQDM_DISABLE=1 .venv/bin/python -c "
from temp_bench.core.config import list_archs, list_datasources, list_experiments
print(f'✓ registries: {len(list_archs())} archs, {len(list_datasources())} datasources, {len(list_experiments())} experiments')

# GPU lock state (if any held)
try:
    from temp_bench.utils.gpu_locks import gpu_lock_status
    status = gpu_lock_status()
    held = {i: v for i, v in (status or {}).items() if v}
    if held:
        print('  GPU lock state:')
        for idx, info in sorted(held.items()):
            print(f'   GPU {idx}: {info.get(\"agent\")} (PID {info.get(\"pid\")})')
    else:
        print('  GPU locks: none held')
except Exception as e:
    print(f'  GPU locks: (unavailable: {e})')

# Token resolution — where each key resolves from (env / .tokens / missing)
from temp_bench.utils.tokens import token_status, tokens_dir
ts = token_status()
print(f'  Token store: {tokens_dir()}')
for kind in ('hf', 'anthropic', 'gh'):
    src = ts[kind]['resolved_from']
    print(f'   {kind:9s} ← {src or \"(missing — bootstrap_local.sh / bootstrap_runpod.sh)\"}')
"

echo
echo "✓ smoke test passed"
