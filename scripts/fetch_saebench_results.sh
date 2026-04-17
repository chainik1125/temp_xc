#!/bin/bash
# fetch_saebench_results.sh — Pull the SAEBench sparse-probing
# experiment outputs from the RunPod pod down to the laptop's local
# repo, organized to match the on-pod layout.
#
# What gets pulled:
#   results/saebench/results/*.jsonl     — aggregate per-arch JSONLs
#   results/saebench/results/saebench_json/  — raw per-run SAEBench JSON
#   results/saebench/logs/               — training logs (keep for FLOPs)
#   results/saebench/preflight/          — preflight validation output
#
# What gets SKIPPED:
#   results/saebench/ckpts/*.pt          — 10 × ~1.5 GB checkpoints;
#                                          stays on the pod volume.
#                                          Re-pull individually if needed.
#   data/cached_activations/             — ~30 GB multi-layer cache; not
#                                          useful locally.
#
# Usage (from laptop, in repo root):
#   bash scripts/fetch_saebench_results.sh
#
# Pod SSH target is baked in from the user's Connect dialog; override
# with POD_SSH env var if you rotate pods:
#   POD_SSH='c2o4it0x73x88e-64412168@ssh.runpod.io' \
#     bash scripts/fetch_saebench_results.sh

set -euo pipefail

POD_SSH="${POD_SSH:-c2o4it0x73x88e-64412168@ssh.runpod.io}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_BASE="${REMOTE_BASE:-/workspace/temp_xc}"

cd "$(git rev-parse --show-toplevel)"
mkdir -p results/saebench/results results/saebench/logs results/saebench/preflight

echo "=== fetching SAEBench results from RunPod ==="
echo "  pod:     $POD_SSH"
echo "  remote:  $REMOTE_BASE/results/saebench/"
echo "  local:   $(pwd)/results/saebench/"
echo ""

# All transfers use the same ssh options (identity file, pod SSH target)
SSH_OPTS="-e 'ssh -i $SSH_KEY -o StrictHostKeyChecking=no'"

# Aggregate JSONLs + raw SAEBench JSONs — the primary deliverable
echo ">> results/saebench/results/ (JSONLs + raw SAEBench output)"
rsync -avzP \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    "${POD_SSH}:${REMOTE_BASE}/results/saebench/results/" \
    results/saebench/results/

# Training logs — keep for FLOPs extraction, post-hoc analysis
echo ""
echo ">> results/saebench/logs/ (training logs)"
rsync -avzP \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    "${POD_SSH}:${REMOTE_BASE}/results/saebench/logs/" \
    results/saebench/logs/

# Preflight output — sanity-check receipt
echo ""
echo ">> results/saebench/preflight/"
rsync -avzP \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    "${POD_SSH}:${REMOTE_BASE}/results/saebench/preflight/" \
    results/saebench/preflight/ 2>/dev/null || echo "  (no preflight dir on pod — skipping)"

echo ""
echo "=== done ==="
echo ""
echo "Summary of what landed:"
jsonl_count=$(find results/saebench/results -name '*.jsonl' 2>/dev/null | wc -l | tr -d ' ')
jsonl_lines=$(cat results/saebench/results/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
saebench_json_count=$(find results/saebench/results/saebench_json -name '*_eval_results.json' 2>/dev/null | wc -l | tr -d ' ')
log_count=$(find results/saebench/logs -name '*.log' 2>/dev/null | wc -l | tr -d ' ')
echo "  ${jsonl_count} JSONL files  (${jsonl_lines} probing records total)"
echo "  ${saebench_json_count} raw SAEBench eval JSONs"
echo "  ${log_count} training logs"
echo ""
echo "Next: re-run summary locally (or look at the pod's orchestrator tail)"
echo "  python -c \"from src.bench.saebench.probing_runner import *; ...\""
echo "or inspect raw JSONLs:"
echo "  cat results/saebench/results/*.jsonl | jq ."
