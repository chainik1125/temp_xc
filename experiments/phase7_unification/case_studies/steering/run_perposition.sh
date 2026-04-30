#!/usr/bin/env bash
# Run per-position write-back (Q2.C) intervene + grade for an already-trained arch.
# Assumes the arch has already gone through select_features + diagnose_z_magnitudes
# (so feature_selection.json + z_orig_magnitudes.json exist).
#
# Usage (from repo root):
#   ./experiments/phase7_unification/case_studies/steering/run_perposition.sh ARCH_ID [SEED]
#
# Output:
#   results/case_studies/steering_paper_window_perposition/<ARCH_ID>/{generations,grades}.jsonl

set -euo pipefail
export TQDM_DISABLE=1
export HF_TOKEN="$(cat /workspace/.tokens/hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export ANTHROPIC_API_KEY="$(cat /workspace/.tokens/anthropic_key)"

ARCH=${1:?"usage: $0 ARCH_ID [SEED]"}
SEED=${2:-42}
ROOT=/workspace/temp_xc
RESULTS=${ROOT}/experiments/phase7_unification/results/case_studies
ZMAG=${RESULTS}/diagnostics_kpos20/z_orig_magnitudes.json
GEN_FILE=${RESULTS}/steering_paper_window_perposition/${ARCH}/generations.jsonl
GRADE_FILE=${RESULTS}/steering_paper_window_perposition/${ARCH}/grades.jsonl

cd "${ROOT}"

if [[ ! -f "${ZMAG}" ]]; then
    echo "ERROR: missing z_orig_magnitudes.json at ${ZMAG}"
    echo "Run diagnose_z_magnitudes for ${ARCH} first."
    exit 1
fi

# Verify the arch's z entry is in the file
if ! .venv/bin/python -c "import json; d=json.load(open('${ZMAG}')); assert '${ARCH}' in d, 'arch missing from z_mag file'" 2>/dev/null; then
    echo "ERROR: '${ARCH}' not in ${ZMAG}. Re-run diagnose_z_magnitudes:"
    echo "  .venv/bin/python -m experiments.phase7_unification.case_studies.steering.diagnose_z_magnitudes --archs ${ARCH} --out-dir ${RESULTS}/diagnostics_kpos20 --seed ${SEED}"
    echo "Then merge with the existing file (other agent's entries)."
    exit 2
fi

step() { echo ""; echo "════════════════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════════════════"; }

step "[1/2] per-position intervene for ${ARCH}"
if [[ -f "${GEN_FILE}" ]] && [[ $(wc -l < "${GEN_FILE}") -ge 210 ]]; then
    echo "  generations exist (${GEN_FILE}, $(wc -l < "${GEN_FILE}") rows), skip"
else
    .venv/bin/python -m experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_window_perposition \
        --archs "${ARCH}" \
        --normalised \
        --z-mag "${ZMAG}" \
        --seed "${SEED}"
fi

step "[2/2] grade per-position for ${ARCH} (--n-workers 1; shared rate limit with Y)"
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.grade_with_sonnet \
    --archs "${ARCH}" \
    --subdir steering_paper_window_perposition \
    --n-workers 1

echo ""
echo "Per-position pipeline complete for ${ARCH}."
echo "  generations: ${GEN_FILE}"
echo "  grades:      ${GRADE_FILE}"
