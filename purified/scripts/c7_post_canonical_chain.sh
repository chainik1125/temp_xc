#!/usr/bin/env bash
# One-shot post-canonical chain for txc_pro|1024 (the only outstanding
# cell after the 2026-05-06 pod-crash recovery): runs optimal-mag and
# extended-mags evals, then re-renders the paper bundle, regenerates
# tex snippets, runs the analyzer, and refreshes the synthetic bundle.
#
# Pinned to GPU 0 (the only GPU on this pod).
#
# Usage:
#   set -a; source /workspace/aniket/temp_xc/.env; set +a
#   cd /workspace/aniket/temp_xc-final/purified
#   bash scripts/c7_post_canonical_chain.sh

set -eu
cd /workspace/aniket/temp_xc-final/purified

GPU=0
ARCH=txc_pro
BS=1024
SEED=42
N_STEPS=300000

PAPER_PURIFIED=/workspace/aniket/temp_xc_paper/purified
COMPONENTS_DIR="$PAPER_PURIFIED/docs/components"
FIGS_DIR="$PAPER_PURIFIED/docs/aniket/figs"
ASSETS_DIR="$COMPONENTS_DIR/c7_paper_assets"
ANALYZE_OUT="$COMPONENTS_DIR/c7_optimal_analysis.md"

log() { printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*"; }

mkdir -p logs "$FIGS_DIR" "$ASSETS_DIR"

log "[1/6] eval_optimal_mag arch=$ARCH bs=$BS"
CUDA_VISIBLE_DEVICES=$GPU \
AGENT_NAME=agent_back_300k \
TEMP_BENCH_POD_MODE=persistent \
TQDM_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m experiments.c7_backtracking.eval_optimal_mag \
        --arch "$ARCH" --bs "$BS" \
        --n-steps "$N_STEPS" --seed "$SEED" \
        > "logs/c7_optimal_${ARCH}_bs${BS}_resume.log" 2>&1
echo "$ARCH|$BS" >> logs/c7_optimal_seen.txt
log "[1/6] optimal-mag done"

log "[2/6] eval_extended_mags arch=$ARCH bs=$BS"
CUDA_VISIBLE_DEVICES=$GPU \
AGENT_NAME=agent_back_300k \
TEMP_BENCH_POD_MODE=persistent \
TQDM_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m experiments.c7_backtracking.eval_extended_mags \
        --arch "$ARCH" --bs "$BS" \
        --magnitudes -32 -24 0 24 32 \
        > "logs/c7_extended_mags_${ARCH}_bs${BS}_resume.log" 2>&1
echo "$ARCH|$BS" >> logs/c7_extended_mags_seen.txt
log "[2/6] extended-mags done"

log "[3/6] c7_paper_renderer → $COMPONENTS_DIR"
.venv/bin/python -m scripts.c7_paper_renderer --output-dir "$COMPONENTS_DIR"

log "[4/6] sync PNGs → $FIGS_DIR with c7_ prefix"
for f in "$ASSETS_DIR"/*.png; do
    [ -f "$f" ] || continue
    cp -f "$f" "$FIGS_DIR/c7_$(basename "$f")"
done

log "[5/6] c7_tex_snippets → $FIGS_DIR + analyze_optimal → $ANALYZE_OUT (+ tex tables)"
.venv/bin/python -m scripts.c7_tex_snippets --output-dir "$FIGS_DIR"
.venv/bin/python -m experiments.c7_backtracking.analyze_optimal \
    --output "$ANALYZE_OUT" \
    --tex-output-dir "$FIGS_DIR"

log "[6/6] synthetic_paper_renderer → $COMPONENTS_DIR"
.venv/bin/python -m scripts.synthetic_paper_renderer --output-dir "$COMPONENTS_DIR" || true

log "all post-canonical steps done"
