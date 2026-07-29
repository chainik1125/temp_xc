"""Full RunPod inventory — ALL running pods, never truncated.

Exists because at 01:47 on 2026-07-29 I piped the pod list through `tail -8`,
lost a row off the top, and reported a pod as GONE that was still running.
The reported unattributed burn was $9.41/h when it was $12.40/h, and I built
a narrative ("the set churned again at a near-constant total") that was exactly
backwards: the total was constant BECAUSE the set was constant — my view of the
set was truncated.

Never `tail` a list whose length you do not control. Print the count, print
every row, and let the caller scroll.

    export RUNPOD_API_KEY="$(security find-generic-password -s dmitrys-runpod-api-key -w)"
    curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
      -d '{"query":"query { myself { pods { id name desiredStatus costPerHr } } }"}' \
      https://api.runpod.io/graphql | .venv/bin/python scripts/pod_inventory.py

Reads stdin so the key is never an argument (process listings leak argv).
Only RUNNING pods are summed — EXITED records carry a costPerHr and summing
them once produced "128 pods, $109.03/h".
"""
import sys, json
p = (json.load(sys.stdin).get("data") or {}).get("myself", {}).get("pods") or []
r = [x for x in p if x.get("desiredStatus") == "RUNNING"]
print(f"RUNNING pods: {len(r)}  (ALL shown, no truncation)")
tot = ours = 0.0
for x in sorted(r, key=lambda z: z.get("name") or ""):
    c = float(x.get("costPerHr") or 0)
    tot += c
    mine = (x.get("name") or "").startswith("mac-")
    if mine:
        ours += c
    tag = "OURS" if mine else "unattributed"
    print(f"   {x.get('name'):36s} ${c:5.2f}/h  {tag}")
print(f"\ntotal ${tot:.2f}/h   ours ${ours:.2f}/h   unattributed ${tot-ours:.2f}/h")
