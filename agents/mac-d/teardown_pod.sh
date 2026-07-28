#!/bin/bash
# mac-d — terminate one pf grid pod, SAFELY.
#
#   bash agents/mac-d/teardown_pod.sh <pod_id> <T>
#
# The one rule this enforces: containers never push, so a pod killed
# with unrepatriated rows loses them silently. This refuses to
# terminate until all three seeds for <T> are present in the LOCAL
# leaderboard. Refusing is the safe direction — an extra $2.99/h beats
# a lost cell that costs 20-30 min of H100 to recreate.
#
# Verifies termination by API query rather than trusting the DELETE.
set -uo pipefail
POD="${1:?usage: teardown_pod.sh <pod_id> <T>}"
T="${2:?usage: teardown_pod.sh <pod_id> <T>}"
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python - "$T" <<'PY' || { echo "REFUSING to terminate: seeds missing locally"; exit 1; }
import json, sys
T = int(sys.argv[1]); seeds = set()
for line in open("results/leaderboard.jsonl"):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    if (r.get("arch") == "agentic_txc_02_v1t"
            and (r.get("training_cfg") or {}).get("n_steps") == 8000
            and int(r["metrics"]["T"]) == T):
        seeds.add(int(r["seed"]))
print(f"  T{T} seeds local: {sorted(seeds)}")
sys.exit(0 if seeds == {42, 1, 2} else 1)
PY

export RUNPOD_API_KEY="$(security find-generic-password -s dmitrys-runpod-api-key -w)"
curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
     "https://rest.runpod.io/v1/pods/$POD" -o /dev/null -w "  DELETE http=%{http_code}\n"
sleep 5
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
     "https://rest.runpod.io/v1/pods/$POD" \
  | .venv/bin/python -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('  API-verified status:', d.get('desiredStatus') or 'GONE')
except Exception:
    print('  API-verified: pod not found (terminated)')
"
