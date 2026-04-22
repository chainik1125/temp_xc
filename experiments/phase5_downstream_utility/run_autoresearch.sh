#!/usr/bin/env bash
# Phase 5.7 autoresearch orchestrator.
#
# For each candidate arch passed in $@:
#   1. Train it (seed 42, 25k max steps, plateau-stop) — skipped if ckpt
#      already exists.
#   2. Probe it at last_position_val (val split of train; test untouched).
#   3. Probe its baseline at last_position_val if missing — so we have
#      a Δ_val number on the same val fold.
#   4. Run autoresearch_summarise.py to compute Δ_val + verdict + write
#      a row to autoresearch_index.jsonl.
#   5. Commit + push (per-candidate milestone, so partial progress is
#      safe against pod reboot).
#
# Serializes all GPU work — call sequentially. Waits for any in-flight
# train_primary_archs / run_probing.py before starting (defensive).

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/autoresearch_orchestrator.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

GIT_USER_NAME="Han"
GIT_USER_EMAIL="hxuany0@gmail.com"

CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts

declare -A BASE_OF
BASE_OF[txcdr_contrastive_t5]=txcdr_t5
BASE_OF[txcdr_rotational_t5]=txcdr_t5
BASE_OF[txcdr_basis_expansion_t5]=txcdr_t5
BASE_OF[txcdr_film_t5]=txcdr_t5
BASE_OF[txcdr_smoothness_t5]=txcdr_t5
BASE_OF[matryoshka_txcdr_contrastive_t5]=matryoshka_t5
BASE_OF[matryoshka_feature_idx_t5]=matryoshka_t5
BASE_OF[mlc_temporal_t3]=mlc
BASE_OF[time_layer_contrastive_t5]=time_layer_crosscoder_t5
# Part B α / k sweep variants — compared vs the same vanilla base as the
# α=0.1 reference (A2 -> txcdr_t5, A3 -> matryoshka_t5) to give an
# apples-to-apples absolute Δ_val. Post-hoc summariser also runs a
# second comparison against the α=0.1 reference variant to show
# sensitivity within the contrastive family.
BASE_OF[txcdr_contrastive_t5_alpha003]=txcdr_t5
BASE_OF[txcdr_contrastive_t5_alpha100]=txcdr_t5
BASE_OF[txcdr_contrastive_t5_k2x]=txcdr_t5
BASE_OF[matryoshka_txcdr_contrastive_t5_alpha003]=matryoshka_t5
BASE_OF[matryoshka_txcdr_contrastive_t5_alpha100]=matryoshka_t5
BASE_OF[matryoshka_txcdr_contrastive_t5_alpha300]=matryoshka_t5
BASE_OF[matryoshka_txcdr_contrastive_t5_alpha1000]=matryoshka_t5
BASE_OF[matryoshka_txcdr_contrastive_t5_k2x]=matryoshka_t5
# MLC-family α sweep — tune the strongest non-TXCDR baseline so the
# bench comparison is apples-to-apples on α too.
BASE_OF[mlc_contrastive_alpha003]=mlc
BASE_OF[mlc_contrastive_alpha100]=mlc
# Agentic autoresearch cycles (Karpathy-style hypothesis-driven loop).
# Each new cycle adds its own entry here. See
# docs/han/research_logs/phase5_downstream_utility/2026-04-21-agentic-log.md
BASE_OF[agentic_txc_01]=matryoshka_t5
BASE_OF[agentic_txc_02]=matryoshka_t5
BASE_OF[agentic_txc_03]=matryoshka_t5
BASE_OF[agentic_txc_04]=matryoshka_t5
BASE_OF[agentic_txc_05]=matryoshka_t5
BASE_OF[agentic_txc_06]=matryoshka_t5
BASE_OF[agentic_txc_07]=matryoshka_t5
# MLC agentic cycles (compared against vanilla mlc).
BASE_OF[agentic_mlc_08]=mlc

commit_and_push() {
    local msg="$1"; shift
    local files=("$@")
    git -c user.name="$GIT_USER_NAME" -c user.email="$GIT_USER_EMAIL" \
        add -- "${files[@]}" 2>&1 | tee -a "$MAIN" || true
    if git diff --cached --quiet; then
        say "  (no changes to commit for: $msg)"
        return 0
    fi
    git -c user.name="$GIT_USER_NAME" -c user.email="$GIT_USER_EMAIL" \
        commit -m "$msg

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" 2>&1 | tee -a "$MAIN"
    git push origin han 2>&1 | tee -a "$MAIN" || say "  push FAILED; continuing"
}

MIN_VAL_TASKS=30   # probe baseline again if <30/36 tasks have val rows at k=5

probe_val_if_missing() {
    local rid="$1"
    # Returns 0 (skip) only if we have val rows for at least MIN_VAL_TASKS
    # distinct tasks at k=5. Prior bug: any single matching row caused a
    # skip, so the smoke-test row for ag_news_business made the whole
    # baseline probe get skipped.
    local n
    n=$(grep "\"run_id\": \"$rid\".*\"aggregation\": \"last_position_val\".*\"k_feat\": 5" \
        /workspace/temp_xc/experiments/phase5_downstream_utility/results/probing_results.jsonl 2>/dev/null \
        | wc -l)
    if [ "${n:-0}" -ge "$MIN_VAL_TASKS" ]; then
        return 0
    fi
    return 1
}

# ── Wait for any in-flight train/probe job ──────────────────────────────
# Check for running Python training/probing processes.
# IMPORTANT: use a narrow pattern that matches only actual Python
# invocations, not any shell whose command line happens to contain
# the substring (e.g. bash -c commit messages mentioning these files).
# pgrep -f basic regex doesn't accept `\|` as alternation. Use two checks
# chained with || so we wait on EITHER a probing OR a training process.
while pgrep -f "python.* run_probing\.py" >/dev/null \
   || pgrep -f "train_primary_archs.*run_all" >/dev/null; do
    say "waiting for in-flight train/probe job to exit..."
    sleep 30
done
say "GPU idle — proceeding with $#  candidates: $*"

for CAND in "$@"; do
    BASE="${BASE_OF[$CAND]:-txcdr_t5}"
    say "=== candidate=$CAND  base=$BASE ==="

    # 1. Train candidate if no ckpt
    CKPT="$CKPT_DIR/${CAND}__seed42.pt"
    if [ -f "$CKPT" ]; then
        say "  ckpt exists -> skip training: $CKPT"
    else
        say "  training $CAND (seed=42, 25k max steps)"
        t0=$(date +%s)
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[42], max_steps=25000, archs=['$CAND'])
" > "$LOG_DIR/autoresearch_train_${CAND}.log" 2>&1
        ec=$?
        say "  train exit=$ec in $(( $(date +%s) - t0 ))s"
        if [ $ec -ne 0 ]; then
            say "  TRAIN FAILED for $CAND — skipping probe"
            continue
        fi
    fi

    # 2. Probe baseline at last_position_val if missing
    BASE_RID="${BASE}__seed42"
    if probe_val_if_missing "$BASE_RID"; then
        say "  baseline ($BASE_RID) val rows already present"
    else
        say "  probing baseline $BASE_RID @ last_position_val"
        t0=$(date +%s)
        .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
            --aggregation last_position_val --skip-baselines \
            --run-ids "$BASE_RID" \
            > "$LOG_DIR/autoresearch_probe_${BASE_RID}_val.log" 2>&1
        say "  baseline probe exit=$? in $(( $(date +%s) - t0 ))s"
    fi

    # 3. Probe candidate at last_position_val
    CAND_RID="${CAND}__seed42"
    say "  probing candidate $CAND_RID @ last_position_val"
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation last_position_val --skip-baselines \
        --run-ids "$CAND_RID" \
        > "$LOG_DIR/autoresearch_probe_${CAND_RID}_val.log" 2>&1
    say "  cand probe exit=$? in $(( $(date +%s) - t0 ))s"

    # 4. Summarise + write index
    say "  summarising $CAND vs $BASE"
    .venv/bin/python -u -m \
        experiments.phase5_downstream_utility.autoresearch_summarise \
        --candidate "$CAND" --base "$BASE" --seed 42 --k 5 --write-index \
        > "$LOG_DIR/autoresearch_summary_${CAND}.log" 2>&1
    cat "$LOG_DIR/autoresearch_summary_${CAND}.log" | tee -a "$MAIN"

    # 5. Commit + push milestone
    commit_and_push "Phase 5.7 autoresearch: $CAND (val Δ vs $BASE)" \
        experiments/phase5_downstream_utility/results/ckpts/${CAND}__seed42.pt \
        experiments/phase5_downstream_utility/results/training_index.jsonl \
        experiments/phase5_downstream_utility/results/training_logs \
        experiments/phase5_downstream_utility/results/probing_results.jsonl \
        experiments/phase5_downstream_utility/results/autoresearch_index.jsonl \
        logs/overnight/autoresearch_summary_${CAND}.log

    say "=== $CAND DONE ==="
done

say "=== AUTORESEARCH ORCHESTRATOR END ==="
