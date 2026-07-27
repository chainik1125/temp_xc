#!/usr/bin/env bash
# mac-d pod lifecycle — RunPod REST API under Dmitry's key (on loan).
#
# Governance (briefings/actmix-shared.md, BINDING): key env-injected from the
# macOS keychain inside this script only — never echoed, never written to a
# file, never passed as an argument. Only pods named mac-d-* are ever mutated;
# every write is verified by a follow-up API query. Terminate > stop.
#
#   podctl.sh create [name] [--dry-run]   # 2xH100 SXM secure, torch-2.8.0 image
#   podctl.sh mine                        # list mac-d-* pods only
#   podctl.sh status <podId>
#   podctl.sh ssh <podId>                 # print ssh coordinates (port 22/tcp mapping)
#   podctl.sh terminate <podId>           # refuses non-mac-d pods; verifies after
set -euo pipefail

API=https://rest.runpod.io/v1
DEFAULT_NAME="mac-d-huntretrain-0727"

_key() { security find-generic-password -s dmitrys-runpod-api-key -w; }

_req() { # method path [body]
  local m="$1" p="$2" body="${3:-}"
  if [ -n "$body" ]; then
    curl -sS --max-time 90 -X "$m" "$API$p" \
      -H "Authorization: Bearer $(_key)" -H "Content-Type: application/json" \
      -d "$body"
  else
    curl -sS --max-time 90 -X "$m" "$API$p" -H "Authorization: Bearer $(_key)"
  fi
}

_body() { # create-body JSON on stdout; pubkey embedded via python for safe quoting
  local name="$1"
  PODNAME="$name" /usr/bin/python3 - <<'PY'
import json, os, pathlib
pub = pathlib.Path.home().joinpath(".ssh/id_ed25519.pub").read_text().strip()
print(json.dumps({
    "name": os.environ["PODNAME"],
    "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "cloudType": "SECURE",
    "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
    "gpuCount": 2,
    "containerDiskInGb": 30,
    "volumeInGb": 300,
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp", "8888/http"],
    "env": {"PUBLIC_KEY": pub},
}))
PY
}

_show() { # compact pod line(s) from a JSON blob on stdin (list or single object)
  /usr/bin/python3 -c '
import json, sys
d = json.load(sys.stdin)
for p in (d if isinstance(d, list) else [d]):
    if not isinstance(p, dict): continue
    print(p.get("id"), p.get("name"), p.get("desiredStatus"),
          str(p.get("costPerHr")) + "/h", "ip=" + str(p.get("publicIp")),
          "ports=" + json.dumps(p.get("portMappings") or p.get("ports")))'
}

cmd="${1:?usage: podctl.sh create|mine|status|ssh|terminate ...}"; shift || true
case "$cmd" in
  create)
    name="$DEFAULT_NAME"; dry=0
    for arg in "$@"; do
      case "$arg" in --dry-run) dry=1;; *) name="$arg";; esac
    done
    case "$name" in mac-d-*) ;; *) echo "REFUSE: pod name must start mac-d-"; exit 1;; esac
    body="$(_body "$name")"
    if [ "$dry" = 1 ]; then
      echo "$body" | /usr/bin/python3 -m json.tool; exit 0
    fi
    echo "creating $name (2xH100 SXM secure, ~\$5.98/h) ..."
    resp="$(_req POST /pods "$body")"
    echo "$resp" | _show || { echo "create response:"; echo "$resp"; exit 1; }
    id="$(echo "$resp" | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')"
    [ -n "$id" ] || { echo "NO POD ID — inspect response above"; exit 1; }
    echo "pod id: $id — polling until RUNNING (ledger line is due NOW per governance)"
    for _ in $(seq 1 60); do
      sleep 10
      s="$(_req GET "/pods/$id")"
      st="$(echo "$s" | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin).get("desiredStatus","?"))' 2>/dev/null || echo '?')"
      echo "  status: $st"
      [ "$st" = "RUNNING" ] && { echo "$s" | _show; exit 0; }
    done
    echo "TIMEOUT waiting for RUNNING — check 'podctl.sh status $id'"; exit 1
    ;;
  mine)
    _req GET /pods | /usr/bin/python3 -c '
import json, sys
pods = [p for p in json.load(sys.stdin) if str(p.get("name","")).startswith("mac-d-")]
print(f"mac-d pods: {len(pods)}")
for p in pods:
    print(" ", p.get("id"), p.get("name"), p.get("desiredStatus"), str(p.get("costPerHr"))+"/h")'
    ;;
  status)
    id="${1:?podId}"; _req GET "/pods/$id" | _show
    ;;
  ssh)
    id="${1:?podId}"
    _req GET "/pods/$id" | /usr/bin/python3 -c '
import json, sys
p = json.load(sys.stdin)
ip = p.get("publicIp"); pm = p.get("portMappings") or {}
port = pm.get("22") if isinstance(pm, dict) else None
print(f"ssh root@{ip} -p {port}  # direct tcp" if ip and port else
      f"no direct mapping yet: publicIp={ip} portMappings={pm}")'
    ;;
  terminate)
    id="${1:?podId}"
    name="$(_req GET "/pods/$id" | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin).get("name",""))')"
    case "$name" in
      mac-d-*) ;;
      *) echo "REFUSE: pod $id name='$name' is not mine (mac-d-*) — governance rule 3"; exit 1;;
    esac
    echo "terminating $id ($name) ..."
    _req DELETE "/pods/$id" || true
    sleep 5
    v="$(_req GET "/pods/$id" 2>/dev/null || true)"
    echo "$v" | grep -q '"desiredStatus"' && echo "$v" | _show || echo "verified: pod $id gone (API returns no pod)"
    ;;
  *) echo "unknown cmd $cmd"; exit 1;;
esac
