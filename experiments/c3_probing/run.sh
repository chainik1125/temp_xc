#!/usr/bin/env bash
# C3 — sparse-probing component runner wrapper.
#
# Usage:
#   ./experiments/c3_probing/run.sh                     # full sweep
#   ./experiments/c3_probing/run.sh --smoke             # 1-cell smoke
#   ./experiments/c3_probing/run.sh --archs tsae_paper  # subset
#
# Pre-conditions:
#   - cwd is 
#   - `bash scripts/smoke_test.sh` passes
#   - probe cache built at results/probe_cache/<datasource>/
#     (run `.venv/bin/python -c "from temp_bench.data.nlp import \
#       build_probe_cache; build_probe_cache('gemma_2_2b_it_l13_fineweb_24k128')"`
#     once if missing)
set -euo pipefail
cd "$(dirname "$0")/../.."  # 

export TQDM_DISABLE=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

exec .venv/bin/python -u -m experiments.c3_probing.run "$@"
