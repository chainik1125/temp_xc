#!/bin/bash
# mac-d — terminate one grid pod, SAFELY.
#
#   bash agents/mac-d/teardown_pod.sh <pod_id> <arm> <T> [T ...]
#     arm = pf   -> agentic_txc_02_v1t        @ n_steps 8000
#     arm = btk  -> txc_batchtopk_post_btkonly @ n_steps 25000
#
# The one rule this enforces: containers never push, so a pod killed
# with unrepatriated rows loses them silently. This refuses to
# terminate until all three seeds of EVERY listed T are present in the
# LOCAL leaderboard, for the ARM that pod actually ran.
#
# ⚑ v1 of this script hardcoded the pf arm. On the last pod — whose
# final work is btk gap cells — it would have reported "seeds local ✓,
# ALLOW TERMINATION" off the pf rows while three btk cells (~3 GPU-h)
# sat unrepatriated on the pod. That is the day's recurring failure
# (a guard reporting success while doing nothing), committed inside the
# guard written to prevent exactly this loss. Hence: arm is REQUIRED,
# never defaulted — a wrong-arm check is worse than no check, because
# it manufactures confidence.
set -uo pipefail
POD="${1:?usage: teardown_pod.sh <pod_id> <arm:pf|btk> <T> [T ...]}"
ARM="${2:?usage: teardown_pod.sh <pod_id> <arm:pf|btk> <T> [T ...]}"
shift 2
[ $# -ge 1 ] || { echo "usage: teardown_pod.sh <pod_id> <arm> <T> [T ...]"; exit 2; }
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python - "$ARM" "$@" <<'PY' || { echo "REFUSING to terminate"; exit 1; }
import json, sys
ARMS = {
    "pf":  ("agentic_txc_02_v1t", 8000),
    "btk": ("txc_batchtopk_post_btkonly", 25000),
}
arm = sys.argv[1]
if arm not in ARMS:
    print(f"  unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    sys.exit(1)
arch, nsteps = ARMS[arm]
want = {int(t) for t in sys.argv[2:]}
have = {t: set() for t in want}
for line in open("results/leaderboard.jsonl"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("arch") != arch:
        continue
    if (r.get("training_cfg") or {}).get("n_steps") != nsteps:
        continue
    # untrained twins / smoke rows carry no metrics.T — skip rather
    # than KeyError. A crash exits non-zero so it would "refuse", but
    # for the wrong reason and without showing the real coverage.
    tval = (r.get("metrics") or {}).get("T")
    if tval is None:
        continue
    T = int(tval)
    if T in want:
        have[T].add(int(r["seed"]))
ok = True
for T in sorted(want):
    complete = have[T] == {42, 1, 2}
    ok &= complete
    print(f"  {arm} T{T} seeds local: {sorted(have[T])} {'OK' if complete else '<-- INCOMPLETE'}")
sys.exit(0 if ok else 1)
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
