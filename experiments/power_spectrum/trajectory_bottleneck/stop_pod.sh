#!/usr/bin/env bash
set -euo pipefail

# Installed only on the RunPod volume. The API token is transferred separately
# with mode 0600 and is never committed or printed.

RUN_ROOT="${TRAJECTORY_RUN_ROOT:-/workspace/trajectory_bottleneck_c7}"
TOKEN_FILE="${RUNPOD_TOKEN_FILE:-${RUN_ROOT}/.runpod_api_key}"
POD_ID_FILE="${RUNPOD_ID_FILE:-${RUN_ROOT}/runpod_pod_id}"

if [[ ! -s "${TOKEN_FILE}" || ! -s "${POD_ID_FILE}" ]]; then
  echo "RunPod stop credential or pod id is missing." >&2
  exit 1
fi

pod_id="$(<"${POD_ID_FILE}")"
api_key="$(<"${TOKEN_FILE}")"
curl --fail --silent --show-error --request POST \
  --url "https://rest.runpod.io/v1/pods/${pod_id}/stop" \
  --header "Authorization: Bearer ${api_key}"
unset api_key
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_ROOT}/stop_requested_utc"
