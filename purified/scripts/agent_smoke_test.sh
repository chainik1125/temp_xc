#!/usr/bin/env bash
# agent_smoke_test.sh — verify that an agent's environment is healthy.
#
# Run from inside `purified/` after `uv sync`.

set -eu

cd "$(dirname "$0")/.."   # always run from purified/

echo "[smoke] purified root: $(pwd)"

# venv exists
if [ ! -d ".venv" ]; then
    echo "✗ no .venv — run: uv sync"
    exit 1
fi

# minimum imports
TQDM_DISABLE=1 .venv/bin/python -c "
import torch
print(f'✓ torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')

import temp_bench
print(f'✓ temp_bench {temp_bench.__version__}')

from temp_bench import architectures, case_studies
print(f'✓ archs registry: {architectures.names()}')
print(f'✓ case studies registry: {case_studies.names()}')

from temp_bench.utils import make_run_id
rid = make_run_id('c0', 'smoke_test', seed=0)
print(f'✓ run_id allocator: {rid}')

from temp_bench.eval import CaseStudy
print(f'✓ CaseStudy ABC importable')
"

echo "✓ smoke test passed"
