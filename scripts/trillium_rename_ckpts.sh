#!/bin/bash
# trillium_rename_ckpts.sh — Rename legacy sweep checkpoints (display-name
# format) to the registry-key format that finalize scripts expect.
#
# The initial run_cached_sweep saved checkpoints using entry.name (e.g.
# "TopKSAE__...", "Stacked T=5__...", "TXCDR T=5__...") rather than the
# registry key. This one-shot renames those to "topk_sae__...",
# "stacked_sae__..." and "crosscoder__..." respectively. Idempotent: files
# already in the new format are skipped.
#
# Run on the LOGIN NODE (or anywhere — just touches filenames):
#   bash scripts/trillium_rename_ckpts.sh

set -euo pipefail

cd "$SCRATCH/temp_xc"

ROOTS=(
    results/nlp/step1-unshuffled/ckpts
    results/nlp/step1-shuffled/ckpts
    results/nlp/step2-unshuffled/ckpts
    results/nlp/step2-shuffled/ckpts
)

total_renamed=0

for ROOT in "${ROOTS[@]}"; do
    [ -d "$ROOT" ] || continue
    echo ">> $ROOT"
    for OLD in "$ROOT"/*.pt; do
        [ -f "$OLD" ] || continue
        BASE=$(basename "$OLD")

        case "$BASE" in
            "TopKSAE__"*)     NEW="topk_sae__${BASE#TopKSAE__}" ;;
            "Stacked T="*)    NEW="stacked_sae__${BASE#Stacked T=*__}" ;;
            "TXCDR T="*)      NEW="crosscoder__${BASE#TXCDR T=*__}" ;;
            "TFA__"*)         NEW="tfa__${BASE#TFA__}" ;;
            "TFA-pos__"*)     NEW="tfa_pos__${BASE#TFA-pos__}" ;;
            *)                NEW="" ;;  # already normalized or unknown
        esac

        if [ -z "$NEW" ] || [ "$NEW" = "$BASE" ]; then
            echo "   keep:    $BASE"
            continue
        fi
        mv -v "$OLD" "$ROOT/$NEW"
        total_renamed=$((total_renamed + 1))
    done
done

echo ""
echo "=== renamed $total_renamed checkpoint(s) ==="
