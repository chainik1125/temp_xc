#!/usr/bin/env bash
# Detach-and-walk-away launcher: runs paperfig (part 2) and then
# coupled-complexity (part 3) sequentially. Smart-resumes when a
# paperfig run is already in progress: waits for its sentinel
# rather than re-launching.
#
# Usage on runpod:
#     cd /temp_xc && git pull
#     tmux new -s allfigs
#     bash scripts/run_all_paperfigs_runpod.sh 2>&1 | tee allfigs.log
#     # Ctrl-b d to detach. When totally done: $REPO_ROOT/ALL_PAPERFIGS_DONE.

set -uo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

PAPERFIG_DONE="$REPO_ROOT/PAPERFIG_DONE"
COMPLEXITY_DONE="$REPO_ROOT/COMPLEXITY_DONE"
ALL_DONE="$REPO_ROOT/ALL_PAPERFIGS_DONE"

rm -f "$ALL_DONE"

echo "[allfigs] starting at $(date)"

is_paperfig_running() {
    pgrep -f "run_hmm_denoising_sweep.py" >/dev/null 2>&1
}
is_complexity_running() {
    pgrep -f "run_coupled_complexity_sweep.py" >/dev/null 2>&1
}

# ------------------------------------------------------------- step 1: paperfig
if [ -f "$PAPERFIG_DONE" ]; then
    echo "[allfigs] paperfig already done (sentinel: $PAPERFIG_DONE) — skipping."
elif is_paperfig_running; then
    echo "[allfigs] paperfig already running — waiting for sentinel..."
    while ! [ -f "$PAPERFIG_DONE" ]; do
        if ! is_paperfig_running; then
            echo "[allfigs] paperfig processes died without producing sentinel; aborting."
            exit 1
        fi
        sleep 60
    done
    echo "[allfigs] paperfig sentinel detected at $(date)."
else
    echo "[allfigs] launching paperfig..."
    bash scripts/run_hmm_paperfig_runpod.sh 2>&1 | tee paperfig.log
    if ! [ -f "$PAPERFIG_DONE" ]; then
        echo "[allfigs] paperfig finished without sentinel — partial results in"
        echo "          results/hmm_paperfig/. Continuing to complexity anyway."
    fi
fi

# --------------------------------------------------------- step 2: complexity
if [ -f "$COMPLEXITY_DONE" ]; then
    echo "[allfigs] complexity already done (sentinel: $COMPLEXITY_DONE) — skipping."
elif is_complexity_running; then
    echo "[allfigs] complexity already running — waiting for sentinel..."
    while ! [ -f "$COMPLEXITY_DONE" ]; do
        if ! is_complexity_running; then
            echo "[allfigs] complexity processes died without producing sentinel; aborting."
            exit 1
        fi
        sleep 60
    done
else
    echo "[allfigs] launching complexity sweep..."
    bash scripts/run_coupled_complexity_runpod.sh 2>&1 | tee complexity.log
fi

# ----------------------------------------------------------------- finalize
touch "$ALL_DONE"
echo "[allfigs] ALL DONE at $(date) — sentinel: $ALL_DONE"
echo "[allfigs] render figures locally with:"
echo "    uv run python scripts/plot_fig9_denoising_vs_T.py \\"
echo "        --input results/hmm_paperfig/sweep_results.json \\"
echo "        --output-dir docs/bill/results/hmm_paperfig"
echo "    uv run python scripts/plot_fig_complexity.py \\"
echo "        --input results/coupled_complexity/sweep_results.json \\"
echo "        --output-dir docs/bill/results/coupled_complexity"
