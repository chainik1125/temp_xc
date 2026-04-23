#!/bin/bash
# Prefetch both 8B models into HF_HOME so the first scrappy cycle
# doesn't pay a ~10 min model-shard download.
set -eu
cd /workspace/spar-temporal-crosscoders
exec .venv/bin/python scripts/prefetch_scrappy_models.py
