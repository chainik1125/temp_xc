#!/bin/bash
# fetch_mess3_ablation.sh — Pull the Mess3 Mat-TopK-SAE ablation results
# from the RunPod pod down to the laptop's local repo.
#
# Usage (from the laptop, inside the repo root):
#   bash scripts/fetch_mess3_ablation.sh
#
# Or override the ssh target:
#   SSH_TARGET='0p5f3ic7qs7dtv-64411fec@ssh.runpod.io' bash scripts/fetch_mess3_ablation.sh
#
# Strategy: build the tarball on the remote first (in /tmp), then fetch
# it as a single base64-wrapped stream over ssh. base64 makes the bytes
# printable so we can strip leading shell pollution with sed.
# Saves the raw stream to /tmp locally so we can debug if anything fails.

set -euo pipefail

SSH_TARGET="${SSH_TARGET:-0p5f3ic7qs7dtv-64411fec@ssh.runpod.io}"

cd "$(git rev-parse --show-toplevel)"

REMOTE_BASE="/workspace/spar-temporal-crosscoders"
LOCAL_EXP="experiments/mess3_mat_ablation"
LOCAL_DOCS="docs/aniket/experiments/mess3_mat_ablation/plots"

mkdir -p "${LOCAL_EXP}/results" "${LOCAL_EXP}/plots" "${LOCAL_DOCS}"

REMOTE_TGZ="/tmp/mess3_ablation_$(date +%s).tgz"
LOCAL_RAW="/tmp/mess3_ablation_raw.b64"
LOCAL_TGZ="/tmp/mess3_ablation.tgz"

echo "=== fetching mess3 ablation from ${SSH_TARGET} ==="
echo ""

# RunPod's ssh proxy requires PTY (-T fails with "doesn't support PTY"),
# so we force TTY with -tt. PTY mode adds \r to every line — fine for
# base64 (tr -d '\r' strips them later) but lethal to raw binary streams,
# which is why we wrap in base64 in the first place.
SSH="ssh -tt -o LogLevel=ERROR"

echo ">> step 1: build remote tarball at ${REMOTE_TGZ}"
$SSH "${SSH_TARGET}" "
cd ${REMOTE_BASE} && \
tar czf ${REMOTE_TGZ} \
    ${LOCAL_EXP}/results \
    ${LOCAL_EXP}/plots \
    ${LOCAL_DOCS} && \
ls -la ${REMOTE_TGZ}
"

echo ""
echo ">> step 2: stream tarball as base64 → ${LOCAL_RAW}"
$SSH "${SSH_TARGET}" "
echo ===B64START===
base64 < ${REMOTE_TGZ}
echo ===B64END===
" > "${LOCAL_RAW}"

echo "raw stream bytes: $(wc -c < "${LOCAL_RAW}")"
echo "first 3 lines of raw stream:"
sed -n '1,3p' "${LOCAL_RAW}"
echo "..."
echo "last 3 lines:"
tail -3 "${LOCAL_RAW}"

echo ""
echo ">> step 3: strip CRs + sentinels and decode → ${LOCAL_TGZ}"
# tr -d '\r' first because PTY adds \r\n; then sentinel-bracket the b64 body
tr -d '\r' < "${LOCAL_RAW}" \
    | sed -n '/^===B64START===$/,/^===B64END===$/p' \
    | sed '1d;$d' \
    | base64 -d > "${LOCAL_TGZ}"

echo "decoded tarball bytes: $(wc -c < "${LOCAL_TGZ}")"

echo ""
echo ">> step 4: extract"
tar xzvf "${LOCAL_TGZ}" -C .

echo ""
echo ">> step 5: cleanup remote tarball"
$SSH "${SSH_TARGET}" "rm -f ${REMOTE_TGZ}" || true

echo ""
echo "=== done ==="
echo ""
echo "Plots pulled:"
ls -lh "${LOCAL_EXP}/plots/"*.pdf 2>/dev/null | awk '{print "  " $5 "  " $9}'
echo ""
echo "Open the gap-recovery figure:"
echo "  open ${LOCAL_EXP}/plots/fig1_gap_recovery_2x2.pdf"
