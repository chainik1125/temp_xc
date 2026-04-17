#!/bin/bash
# fetch_runpod_fmap.sh — Pull the 4 labeled feature-map PNGs + HTMLs
# from a RunPod pod (where runpod_fmap_5k_labeled.sh just ran) down to
# the laptop's local reports/ dir.
#
# Usage (from the laptop, inside the repo root):
#   bash scripts/fetch_runpod_fmap.sh 'root@<pod-ip> -p <port>'
#
# Example:
#   bash scripts/fetch_runpod_fmap.sh 'root@213.173.102.123 -p 12345'
#
# The ssh target goes inside the single quotes exactly as RunPod gave it
# to you (the "Connect" button in their UI shows the command). The -p
# flag is required for non-standard ports.
#
# Only pulls PNGs + HTMLs (~30 MB total). If you want the autointerp
# JSONs or checkpoints on your laptop, use scripts/fetch_trillium_results.sh
# against Trillium instead — that's still the canonical source.

set -euo pipefail

SSH_TARGET="${1:-}"
if [ -z "$SSH_TARGET" ]; then
    echo "Usage: bash scripts/fetch_runpod_fmap.sh 'root@<pod-ip> -p <port>'"
    echo ""
    echo "Get the SSH target from the RunPod UI's 'Connect' dialog."
    exit 2
fi

cd "$(git rev-parse --show-toplevel)"

# Split ssh target into host and port for rsync -e
# "root@1.2.3.4 -p 12345" → SSH_HOST="root@1.2.3.4", SSH_PORT="12345"
SSH_HOST="$(echo "$SSH_TARGET" | awk '{print $1}')"
SSH_PORT="$(echo "$SSH_TARGET" | grep -oE '\-p [0-9]+' | awk '{print $2}')"
if [ -z "$SSH_PORT" ]; then
    SSH_PORT=22
fi

echo "=== fetching labeled feature maps from $SSH_HOST:$SSH_PORT ==="
echo ""

REMOTE_BASE="/workspace/temp_xc/reports"

for subdir in step1-gemma-replication step2-deepseek-reasoning; do
    echo ">> reports/${subdir}/"
    mkdir -p "reports/${subdir}"
    rsync -avzP \
        -e "ssh -p $SSH_PORT" \
        --include='*/' \
        --include='feature_map_*.png' \
        --include='feature_map_*.html' \
        --include='feature_map_*_clusters.json' \
        --exclude='*' \
        "${SSH_HOST}:${REMOTE_BASE}/${subdir}/" \
        "reports/${subdir}/"
done

echo ""
echo "=== done ==="
echo ""
echo "Files pulled:"
ls -lh reports/step1-gemma-replication/feature_map_*.{png,html} \
       reports/step2-deepseek-reasoning/feature_map_*.{png,html} 2>/dev/null \
    | awk '{print "  " $5 "  " $9}'
echo ""
echo "Open the interactive HTMLs to see gemma-labeled clusters:"
echo "  open reports/step1-gemma-replication/feature_map_step1-unshuffled.html"
echo "  open reports/step2-deepseek-reasoning/feature_map_step2-unshuffled.html"
echo ""
echo "Safe to destroy the RunPod pod now."
