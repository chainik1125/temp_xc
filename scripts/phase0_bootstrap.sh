#!/bin/bash
# Phase 0 bootstrap for the scrappy autoresearch loop — runs ONCE per
# pod, then every cycle symlinks the cached traces + activations in.
#
# Generates:
#   - traces.json  (thinking-model reasoning traces on the scrappy slice)
#   - activations_path1.pkl + activations_path3.pkl (per-sentence-mean
#     and T-window activations at the configured steering_layer)
#   - sentence/mean sidecar files
#
# Output goes under:
#   experiments/venhoff_scrappy/results/phase0/<identity_slug>/
# where <identity_slug> = `<thinking-short>_<dataset>-<split>_n<N>_L<layer>_seed<seed>`
# — same convention as ArtifactPaths.
#
# Reads the scrappy defaults from experiments/venhoff_scrappy/config.yaml.
# n_traces=200 is set there; generating 200 MATH500 traces on a single
# H100 takes ~15–25 min. Subsequent cycles cost zero Phase 0 time.
#
# Usage: bash scripts/phase0_bootstrap.sh

set -euo pipefail
cd /workspace/spar-temporal-crosscoders

exec .venv/bin/python -m experiments.venhoff_scrappy.phase0_bootstrap "$@"
