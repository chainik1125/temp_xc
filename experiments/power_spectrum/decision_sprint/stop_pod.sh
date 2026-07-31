#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNPOD_API_KEY:-}" || -z "${RUNPOD_POD_ID:-}" ]]; then
  echo "RUNPOD_API_KEY/RUNPOD_POD_ID absent; refusing silent shutdown success"
  exit 2
fi

curl -fsS -X POST \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop"
