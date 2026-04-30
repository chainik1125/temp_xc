#!/usr/bin/env bash
# Chained post-training pipeline for a Hail Mary kpos20 cell.
#
# Run from repo root (/workspace/temp_xc):
#   ./experiments/phase7_unification/case_studies/steering/run_kpos20_pipeline.sh <arch_id> [seed=42]
#
# Subdirs depend on seed:
#   seed=42:  steering/, steering_paper_normalised/, steering_paper_window_perposition/
#   seed!=42: steering_seed{N}/, steering_paper_normalised_seed{N}/, steering_paper_window_perposition_seed{N}/
#
# Stages (skip any whose output exists):
#   1. select_features      → results/case_studies/<sel_subdir>/<arch>/feature_selection.json
#   2. diagnose_z_magnitudes → results/case_studies/diagnostics_kpos20/z_orig_magnitudes.json
#   3. intervene_paper_clamp_normalised → results/case_studies/<gen_subdir>/<arch>/generations.jsonl
#   4. grade_with_sonnet (--n-workers 1, shared with W) → ...../grades.jsonl

set -euo pipefail
export TQDM_DISABLE=1

ARCH=${1:?"usage: $0 <arch_id> [seed=42]"}
SEED=${2:-42}

# Subdir naming convention (matches intervene_paper_clamp_normalised.py + select_features.py)
if [[ "${SEED}" == "42" ]]; then
    SEL_SUBDIR="steering"
    GEN_SUBDIR="steering_paper_normalised"
else
    SEL_SUBDIR="steering_seed${SEED}"
    GEN_SUBDIR="steering_paper_normalised_seed${SEED}"
fi

ROOT=/workspace/temp_xc
RESULTS=${ROOT}/experiments/phase7_unification/results/case_studies
SEL_FILE=${RESULTS}/${SEL_SUBDIR}/${ARCH}/feature_selection.json
DIAG_DIR=${RESULTS}/diagnostics_kpos20
GEN_FILE=${RESULTS}/${GEN_SUBDIR}/${ARCH}/generations.jsonl
GRADE_FILE=${RESULTS}/${GEN_SUBDIR}/${ARCH}/grades.jsonl

cd "${ROOT}"

step() { echo ""; echo "════════════════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════════════════"; }

step "[1/4] select_features for ${ARCH} (seed=${SEED}; sel_subdir=${SEL_SUBDIR})"
if [[ -f "${SEL_FILE}" ]]; then
    echo "  exists: ${SEL_FILE}, skip"
else
    .venv/bin/python -m experiments.phase7_unification.case_studies.steering.select_features \
        --archs "${ARCH}" --seed "${SEED}"
fi

step "[2/4] diagnose_z_magnitudes for ${ARCH} (seed=${SEED})"
mkdir -p "${DIAG_DIR}"
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.diagnose_z_magnitudes \
    --archs "${ARCH}" --out-dir "${DIAG_DIR}" --seed "${SEED}"

step "[3/4] intervene_paper_clamp_normalised for ${ARCH} (seed=${SEED}; gen_subdir=${GEN_SUBDIR})"
if [[ -f "${GEN_FILE}" ]] && [[ $(wc -l < "${GEN_FILE}") -ge 210 ]]; then
    echo "  generations exist (${GEN_FILE}, $(wc -l < "${GEN_FILE}") rows), skip"
else
    .venv/bin/python -m experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_normalised \
        --archs "${ARCH}" --z-mag "${DIAG_DIR}/z_orig_magnitudes.json" --seed "${SEED}"
fi

step "[4/4] grade_with_sonnet for ${ARCH} (--n-workers 1; subdir=${GEN_SUBDIR})"
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.grade_with_sonnet \
    --archs "${ARCH}" --subdir "${GEN_SUBDIR}" --n-workers 1

echo ""
echo "Pipeline complete. Outputs:"
echo "  feature_selection: ${SEL_FILE}"
echo "  z_magnitudes:      ${DIAG_DIR}/z_orig_magnitudes.json"
echo "  generations:       ${GEN_FILE}"
echo "  grades:            ${GRADE_FILE}"
