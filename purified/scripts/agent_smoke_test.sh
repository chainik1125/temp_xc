#!/usr/bin/env bash
# agent_smoke_test.sh — verify that an agent's environment is healthy.
#
# Run from inside `purified/` after `uv sync`. Verifies:
#   1. .venv exists and torch/cuda are visible
#   2. temp_bench imports cleanly
#   3. configs/locked_archs.yaml + configs/datasources.yaml validate
#   4. cache-key + schema tests pass
#   5. runner.preflight() reports no warnings (or warns are listed)
#
# Run on every agent session start. CI doesn't exist on `final`; this
# script is the cheap proxy.

set -eu

cd "$(dirname "$0")/.."   # always run from purified/

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

from temp_bench import config, cache, runner, schemas
print(f'✓ framework modules: config, cache, runner, schemas')

archs = config.list_archs()
print(f'✓ locked archs ({len(archs)}): {archs}')

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

# 4. preflight — GPU pinning + arch class imports + datasources
echo
echo "[smoke] runner preflight…"
TQDM_DISABLE=1 .venv/bin/python -c "
from temp_bench.runner import preflight
warns = preflight()
critical = [w for w in warns if w.startswith('CRITICAL')]
gaps = [w for w in warns if not w.startswith('CRITICAL')]
if critical:
    print('✗ CRITICAL preflight failures:')
    for w in critical:
        print(f'   - {w}')
    raise SystemExit(1)
if not warns:
    print('✓ preflight clean')
else:
    print(f'⚠  preflight reports {len(gaps)} expected gaps (architectures not yet implemented):')
    for w in gaps:
        print(f'   - {w}')
"

echo
echo "✓ smoke test passed"
