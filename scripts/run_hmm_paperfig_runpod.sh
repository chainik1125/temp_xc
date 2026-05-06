#!/usr/bin/env bash
# Paper figure: HMM denoising bench, 4 archs, 3 seeds, full T x k grid.
#
# Fills two gaps in the midterm bench:
#   1. Adds regular_sae_kT (framing-B per-token-budget baseline).
#   2. Multi-seed (3 seeds) so fig9 can show error bars.
#
# Usage on runpod:
#     cd /temp_xc && git pull
#     tmux new -s paperfig
#     bash scripts/run_hmm_paperfig_runpod.sh 2>&1 | tee paperfig.log
#     # Ctrl-b d to detach. When done, $REPO_ROOT/PAPERFIG_DONE exists.
#
# Output: results/hmm_paperfig/sweep_results.json. Atomic incremental
# save means a crash mid-sweep keeps everything completed up to that point;
# however, the sweep does NOT auto-skip already-done cells on restart, so
# re-running overwrites previously completed cells.

set -uo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

OUT_DIR="results/hmm_paperfig"
SENTINEL="$REPO_ROOT/PAPERFIG_DONE"

echo "[paperfig] starting at $(date)"
echo "[paperfig] repo: $REPO_ROOT"
echo "[paperfig] output: $OUT_DIR"

# Sweep: 4 models x 8 T x 3 k x 3 seeds = ~288 cells (a few txcdr cells
# at large k*T get skipped). Roughly 4x the midterm bench runtime.
uv run python scripts/run_hmm_denoising_sweep.py \
    --models regular_sae regular_sae_kT stacked_sae txcdr \
    --n-seeds 3 \
    --output-dir "$OUT_DIR" \
    --steps 65000 \
    --batch-size 64 \
    --seed 42

EXIT=$?
echo "[paperfig] sweep exited with code $EXIT at $(date)"

if [ $EXIT -eq 0 ]; then
    touch "$SENTINEL"
    echo "[paperfig] DONE — sentinel: $SENTINEL"
    echo "[paperfig] results: $OUT_DIR/sweep_results.json"
    echo "[paperfig] next: render the figure locally with"
    echo "    uv run python scripts/plot_fig9_denoising_vs_T.py \\"
    echo "        --input $OUT_DIR/sweep_results.json \\"
    echo "        --output-dir docs/bill/results/hmm_paperfig"
else
    echo "[paperfig] FAILED — partial results in $OUT_DIR/sweep_results.json"
    echo "[paperfig] re-run this script to resume from the last completed cell"
    exit $EXIT
fi
