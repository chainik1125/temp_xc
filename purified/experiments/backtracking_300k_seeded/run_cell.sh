#!/usr/bin/env bash
set -euo pipefail

log_file="${1:?missing log file}"
shift
exit_file="${log_file}.exit"

set +e
"$@" 2>&1 | tee "$log_file"
status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$status" > "$exit_file"
exit "$status"
