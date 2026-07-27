#!/usr/bin/env bash
# mac-d repatriation — pull result rows/manifests off MY pod and merge locally.
# Containers never push (mac-d STATUS rule): rows travel by scp, merge happens
# here with dup-key checks, and the push to origin is done from the mac.
#
#   repatriate.sh <ssh-host> <ssh-port> [remote-repo=/workspace/temp_xc]
#
# Dry-runs the merges first, then applies. Stage copies are kept for audit.
set -euo pipefail
HOST="${1:?ssh host}"; PORT="${2:?ssh port}"; RREPO="${3:-/workspace/temp_xc}"
cd "$(git rev-parse --show-toplevel)"

STAGE="${TMPDIR:-/tmp}/mac-d-repatriate-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$STAGE"
echo "staging to $STAGE"

scp -P "$PORT" -o StrictHostKeyChecking=accept-new \
  "root@$HOST:$RREPO/results/leaderboard.jsonl" "$STAGE/leaderboard.pod.jsonl"
scp -P "$PORT" "root@$HOST:$RREPO/checkpoints/manifest.jsonl" \
  "$STAGE/manifest.pod.jsonl" 2>/dev/null || echo "(no checkpoint manifest on pod)"

/usr/bin/python3 agents/mac-d/merge_rows.py \
  --incoming "$STAGE/leaderboard.pod.jsonl" --target results/leaderboard.jsonl --key eval_key
/usr/bin/python3 agents/mac-d/merge_rows.py \
  --incoming "$STAGE/leaderboard.pod.jsonl" --target results/leaderboard.jsonl --key eval_key --apply

if [ -s "$STAGE/manifest.pod.jsonl" ]; then
  /usr/bin/python3 agents/mac-d/merge_rows.py \
    --incoming "$STAGE/manifest.pod.jsonl" --target checkpoints/manifest.jsonl --key train_key
  /usr/bin/python3 agents/mac-d/merge_rows.py \
    --incoming "$STAGE/manifest.pod.jsonl" --target checkpoints/manifest.jsonl --key train_key --apply
fi

echo "merged. Review 'git diff --stat', then pull-rebase and push from the mac."
echo "stage kept at $STAGE"
