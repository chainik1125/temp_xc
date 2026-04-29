#!/usr/bin/env bash
# Run a single W Phase 1 cell end-to-end: train → select → diagnose → intervene → grade.
#
# Usage (from repo root /workspace/temp_xc):
#   ./experiments/phase7_unification/case_studies/steering/run_w_phase1_cell.sh CELL [WARM_START]
#
# CELL ∈ {C, D, E, F, D_warmstart, C_warmstart, F_warmstart}
#   C = T=3, k_pos=20, TXCBareAntidead, random-init (arch_id txc_bare_antidead_t3_kpos20)
#   D_warmstart = T=5, k_pos=20, TXCBareAntidead, warm-start (arch_id txc_bare_antidead_t5_kpos20_warmstart)
#   E = T=5, k_pos=20, MatryoshkaTXCDRContrastiveMultiscale (arch_id agentic_txc_02_kpos20)
#   F = T=10, k_pos=20, TXCBareAntidead, random-init (arch_id txc_bare_antidead_t10_kpos20)
#
# Cell D itself (random-init T=5) is OWNED BY AGENT Y per coordination
# (commit fd117ca9). This script will refuse to run cell D random-init
# unless the [meeting cell] commit appears in git log.

set -euo pipefail
export TQDM_DISABLE=1
export HF_TOKEN="$(cat /workspace/.tokens/hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export ANTHROPIC_API_KEY="$(cat /workspace/.tokens/anthropic_key)"

CELL=${1:?"usage: $0 CELL  (C, D, E, F, D_warmstart, C_warmstart, F_warmstart)"}
SEED=${2:-42}
ROOT=/workspace/temp_xc

cd "${ROOT}"

# Cell → trainer + arch_id mapping
case "${CELL}" in
    C)            ARCH_ID="txc_bare_antidead_t3_kpos20";              TRAINER="train_kpos20_txc.py";   T_FLAG="--T 3";  WARM='--warm-start ""' ;;
    F)            ARCH_ID="txc_bare_antidead_t10_kpos20";             TRAINER="train_kpos20_txc.py";   T_FLAG="--T 10"; WARM='--warm-start ""' ;;
    E)            ARCH_ID="agentic_txc_02_kpos20";                    TRAINER="train_kpos20_matry.py"; T_FLAG="--T 5";  WARM="" ;;
    D_warmstart)  ARCH_ID="txc_bare_antidead_t5_kpos20_warmstart";    TRAINER="train_kpos20_txc.py";   T_FLAG="--T 5";  WARM='--warm-start auto' ;;
    C_warmstart)  ARCH_ID="txc_bare_antidead_t3_kpos20_warmstart";    TRAINER="train_kpos20_txc.py";   T_FLAG="--T 3";  WARM='--warm-start auto' ;;
    F_warmstart)  ARCH_ID="txc_bare_antidead_t10_kpos20_warmstart";   TRAINER="train_kpos20_txc.py";   T_FLAG="--T 10"; WARM='--warm-start auto' ;;
    D)
        echo "Cell D random-init is owned by Agent Y (coordination commit fd117ca9)."
        echo "Aborting. To override (after Y abandons): use 'D_warmstart' for the warm-start variant."
        exit 1
        ;;
    *)
        echo "unknown cell: ${CELL}"
        exit 2
        ;;
esac

step() { echo ""; echo "════════════════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════════════════"; }

step "[CELL=${CELL}] arch_id=${ARCH_ID} trainer=${TRAINER} ${T_FLAG} ${WARM} seed=${SEED}"

# 1. Train (idempotent — trainer skips if ckpt exists)
step "[1/5] train ${ARCH_ID}"
eval ".venv/bin/python -m experiments.phase7_unification.case_studies.${TRAINER%.py} ${T_FLAG} --k-pos 20 --seed ${SEED} ${WARM}"

# 2-5. Run Y's pipeline.sh for the case-study post-training stages
step "[2-5/5] run_kpos20_pipeline.sh ${ARCH_ID} ${SEED}"
./experiments/phase7_unification/case_studies/steering/run_kpos20_pipeline.sh "${ARCH_ID}" "${SEED}"

step "Cell ${CELL} (${ARCH_ID}) DONE"
