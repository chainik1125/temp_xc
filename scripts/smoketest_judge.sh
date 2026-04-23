#!/bin/bash
# Confirm the dated Anthropic Haiku judge id is live before burning
# autoresearch cycles. Single-line bash wrapper around the python impl.
set -eu
cd /workspace/spar-temporal-crosscoders
exec .venv/bin/python scripts/smoketest_judge.py
