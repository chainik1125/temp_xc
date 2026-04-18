#!/bin/bash
# fetch_saebench_b64.sh — Pull SAEBench results from the pod via a
# base64-piped ssh stream. Robust to .bashrc banner pollution and
# missing SFTP subsystem on RunPod's SSH server, which break rsync
# and modern scp respectively. Designed for small payloads (JSONLs +
# logs are <1 MB total); not suitable for large checkpoints.
#
# Usage (from laptop, in repo root):
#   bash scripts/fetch_saebench_b64.sh
#
# Override the pod target with POD_SSH:
#   POD_SSH='abc123-deadbeef@ssh.runpod.io' bash scripts/fetch_saebench_b64.sh

set -euo pipefail

POD_SSH="${POD_SSH:-fwmg8l9jbsco6n-644121b2@ssh.runpod.io}"
REMOTE_BASE="${REMOTE_BASE:-/workspace/temp_xc}"
TARBALL_REMOTE="/workspace/saebench_pull.tar.gz"
TARBALL_LOCAL="$(mktemp -t saebench_pull).tar.gz"

cd "$(git rev-parse --show-toplevel)"

echo "=== fetching SAEBench results via base64 stream ==="
echo "  pod:       $POD_SSH"
echo "  remote:    $REMOTE_BASE/results/saebench/"
echo "  tarball:   $TARBALL_REMOTE"
echo ""

echo ">> [1/3] tar results+logs on pod into $TARBALL_REMOTE"
# RunPod's SSH gateway requires PTY allocation (-tt). PTY pollutes
# stdout with carriage returns and banner text; we don't care here
# because we only check the exit code.
ssh -tt -q "$POD_SSH" "cd $REMOTE_BASE && tar cz results/saebench/results results/saebench/logs > $TARBALL_REMOTE && ls -lh $TARBALL_REMOTE && echo TAR_OK" \
    | tee /tmp/saebench_tar.log
if ! grep -q TAR_OK /tmp/saebench_tar.log; then
    echo "FAIL: tar step did not report TAR_OK — see /tmp/saebench_tar.log"
    exit 1
fi

echo ""
echo ">> [2/3] stream tarball back as base64 (banner-tolerant)"
# -tt allocates PTY (RunPod requires); banner + CR pollution is
# filtered by `tr -cd 'A-Za-z0-9+/='` which only keeps base64 chars.
ssh -tt -q "$POD_SSH" "base64 -w0 $TARBALL_REMOTE" 2>/dev/null \
    | tr -cd 'A-Za-z0-9+/=' \
    | base64 -D > "$TARBALL_LOCAL"

actual_size=$(wc -c < "$TARBALL_LOCAL" | tr -d ' ')
echo "   wrote $actual_size bytes to $TARBALL_LOCAL"

echo ""
echo ">> [3/3] verify + extract"
file "$TARBALL_LOCAL"
tar tzf "$TARBALL_LOCAL" | head -10
echo "   ..."

tar xzf "$TARBALL_LOCAL"
rm -f "$TARBALL_LOCAL"

echo ""
echo "=== done ==="
jsonl_count=$(find results/saebench/results -name '*.jsonl' 2>/dev/null | wc -l | tr -d ' ')
jsonl_lines=$(cat results/saebench/results/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
log_count=$(find results/saebench/logs -name '*.log' 2>/dev/null | wc -l | tr -d ' ')
echo "  $jsonl_count JSONL files ($jsonl_lines records total)"
echo "  $log_count training logs"
echo ""
echo "Next: run summary locally, e.g."
echo "  python -c \"import json,glob,statistics; from collections import defaultdict; \\"
echo "    rs=[json.loads(l) for p in glob.glob('results/saebench/results/*.jsonl') \\"
echo "        for l in open(p) if l.strip()]; \\"
echo "    print(f'{len(rs)} records'); \\"
echo "    by=defaultdict(list); \\"
echo "    [by[(r['architecture'],r['t'],r['matching_protocol'],r['aggregation'])].append(r['accuracy']) \\"
echo "     for r in rs if r['k']==5]; \\"
echo "    [print(f'{k} -> mean={statistics.mean(v):.4f} n={len(v)}') for k,v in sorted(by.items())]\""
