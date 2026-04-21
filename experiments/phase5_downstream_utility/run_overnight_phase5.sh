#!/usr/bin/env bash
# Phase 5 overnight orchestrator — T15 → T16 → T17 → final plots + commits.
#
# Prerequisites (T13 + T14 code edits) MUST be on disk before this launches:
#   - plots/make_headline_plot.py iterator drops full_window.
#   - probing/run_probing.py supports --save-predictions.
#
# Serializes all GPU work. Waits for the three earlier overnight orchestrators
# (T-sweep, mean_pool, mlc_contrastive) to exit before starting. Commits +
# pushes after every major milestone so an OOM-induced pod reboot does not
# lose progress.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export UV_LINK_MODE=copy
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/overnight_phase5_orchestrator.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

GIT_USER_NAME="Han"
GIT_USER_EMAIL="hxuany0@gmail.com"

QUOTA_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results
free_gb() { df -BG "$1" | awk 'NR==2{gsub(/G/,"",$4); print $4}'; }
mem_gb_used() {
    # cgroup v1 path (this pod); fall back to v2 if v1 missing.
    local path
    if [ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
        path=/sys/fs/cgroup/memory/memory.usage_in_bytes
    elif [ -f /sys/fs/cgroup/memory.current ]; then
        path=/sys/fs/cgroup/memory.current
    else
        echo "0.0"; return
    fi
    awk '{printf "%.1f", $1/1024/1024/1024}' "$path"
}

commit_and_push() {
    local msg="$1"
    shift
    local files=("$@")
    git -c user.name="$GIT_USER_NAME" -c user.email="$GIT_USER_EMAIL" \
        add -- "${files[@]}" 2>&1 | tee -a "$MAIN" || true
    if git diff --cached --quiet; then
        say "  (no changes to commit for milestone: $msg)"
        return 0
    fi
    git -c user.name="$GIT_USER_NAME" -c user.email="$GIT_USER_EMAIL" \
        commit -m "$msg

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" 2>&1 | tee -a "$MAIN"
    git push origin han 2>&1 | tee -a "$MAIN" || say "  push FAILED; continuing"
}

quota_guard() {
    local min_gb=${1:-5}
    local free
    free=$(free_gb "$QUOTA_DIR")
    if [ "${free:-0}" -lt "$min_gb" ]; then
        say "FATAL: pod volume free ${free} GB < ${min_gb} GB threshold — aborting"
        return 1
    fi
    return 0
}

mem_guard() {
    # Abort if current cgroup usage exceeds 40 GB (46 GB hard limit).
    local used
    used=$(mem_gb_used)
    local used_int=${used%.*}
    if [ "${used_int:-0}" -ge 40 ]; then
        say "WARN: cgroup memory ${used} GB near 46 GB cap — pausing 60s"
        sleep 60
    fi
}

say "=== OVERNIGHT PHASE5 ORCHESTRATOR START ==="
say "pod volume free: $(free_gb "$QUOTA_DIR") GB  cgroup memory: $(mem_gb_used) GB"

# ── Wait for the three earlier overnight orchestrators ───────────────────
for upstream in run_fw_tsweep.sh run_mean_pool_probing.sh run_mlc_contrastive.sh; do
    if pgrep -f "$upstream" >/dev/null; then
        say "waiting for $upstream to exit..."
        while pgrep -f "$upstream" >/dev/null; do sleep 30; done
    fi
    say "$upstream done"
done

# ── Also wait for any live run_probing.py or train_primary_archs jobs ─────
while pgrep -f "run_probing.py\|train_primary_archs" >/dev/null; do
    say "waiting for leftover probing/training jobs to exit..."
    sleep 30
done
say "GPU is idle — proceeding"

# ── Commit + push T13 / T14 code changes (already on disk) ────────────────
say "MILESTONE 0: commit T13 + T14 code changes"
commit_and_push "Phase 5: T13 deprecate full_window in plots; T14 --save-predictions; T16/T17 scripts" \
    experiments/phase5_downstream_utility/plots/make_headline_plot.py \
    experiments/phase5_downstream_utility/probing/run_probing.py \
    experiments/phase5_downstream_utility/analyze_error_overlap.py \
    experiments/phase5_downstream_utility/plots/make_seed_variance_plot.py \
    experiments/phase5_downstream_utility/run_overnight_phase5.sh

# ── T15: re-probe 7 top archs at last_position with predictions ──────────
say "MILESTONE T15: re-probe 7 archs with --save-predictions"
ARCHS=(
    topk_sae__seed42
    mlc__seed42
    mlc_contrastive__seed42
    time_layer_crosscoder_t5__seed42
    txcdr_t5__seed42
    txcdr_tied_t5__seed42
    txcdr_rank_k_dec_t5__seed42
)
for rid in "${ARCHS[@]}"; do
    ckpt="/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts/${rid}.pt"
    if [ ! -f "$ckpt" ]; then
        say "  SKIP $rid (ckpt missing: $ckpt)"
        continue
    fi
    mem_guard
    quota_guard 5 || exit 1
    say "  probing $rid @ last_position --save-predictions"
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation last_position --skip-baselines --save-predictions \
        --run-ids "$rid" \
        > "$LOG_DIR/t15_probe_${rid}.log" 2>&1
    ec=$?
    say "  done (${rid}) exit=$ec in $(( $(date +%s) - t0 ))s"
    if [ $ec -ne 0 ]; then
        say "  WARN $rid probing failed — continuing"
    fi
done

say "MILESTONE T15 commit+push"
commit_and_push "Phase 5 T15: per-example predictions for 7 top archs (last_position, k=5/20/etc.)" \
    experiments/phase5_downstream_utility/results/predictions \
    experiments/phase5_downstream_utility/results/probing_results.jsonl

# ── T16: error-overlap analysis ──────────────────────────────────────────
say "MILESTONE T16: error-overlap analysis"
.venv/bin/python -u experiments/phase5_downstream_utility/analyze_error_overlap.py \
    --aggregation last_position --k 5 \
    > "$LOG_DIR/t16_error_overlap.log" 2>&1
say "T16 exit=$?"

say "MILESTONE T16 commit+push"
commit_and_push "Phase 5 T16: error-overlap plots (McNemar + Jaccard + wins/loss) for 7 top archs" \
    experiments/phase5_downstream_utility/results/plots \
    experiments/phase5_downstream_utility/results/error_overlap_summary_last_position_k5.json

# ── T17: 3-seed autoresearch on top-5 archs ──────────────────────────────
say "MILESTONE T17: 3-seed autoresearch"
# Order matters: time_layer + mlc reuse the 18GB ml_buf on GPU; then the
# three anchor-based txcdr archs. run_all frees ml_buf after mlc, so put
# time_layer FIRST to avoid a 15-sec reload.
T17_ARCHS=(time_layer_crosscoder_t5 mlc txcdr_rank_k_dec_t5 txcdr_t5 txcdr_tied_t5)
for seed in 1 2 3; do
    mem_guard
    quota_guard 10 || { say "FATAL: T17 abort seed=$seed (disk)"; exit 1; }
    say "  T17 seed=$seed training ${T17_ARCHS[*]}"
    t0=$(date +%s)
    .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[$seed], max_steps=25000, archs=['time_layer_crosscoder_t5','mlc','txcdr_rank_k_dec_t5','txcdr_t5','txcdr_tied_t5'])
" > "$LOG_DIR/t17_train_seed${seed}.log" 2>&1
    ec=$?
    say "  T17 seed=$seed train exit=$ec in $(( $(date +%s) - t0 ))s"
    if [ $ec -ne 0 ]; then
        say "  WARN seed=$seed training failed — skipping probes for this seed, continuing"
        continue
    fi

    # Probe the 5 new checkpoints at last_position only (no baselines — already done).
    for arch in "${T17_ARCHS[@]}"; do
        rid="${arch}__seed${seed}"
        ckpt="/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts/${rid}.pt"
        if [ ! -f "$ckpt" ]; then
            say "    SKIP $rid (ckpt missing)"
            continue
        fi
        mem_guard
        quota_guard 5 || { say "FATAL: T17 probe abort"; exit 1; }
        say "    probing $rid @ last_position"
        tp=$(date +%s)
        .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
            --aggregation last_position --skip-baselines \
            --run-ids "$rid" \
            > "$LOG_DIR/t17_probe_${rid}.log" 2>&1
        say "    probe done ($rid) in $(( $(date +%s) - tp ))s"
    done

    # Commit after each seed so partial progress is safe against OOM reboot.
    say "  T17 seed=$seed milestone commit+push"
    commit_and_push "Phase 5 T17 seed=${seed}: 5-arch autoresearch (mlc/time_layer/txcdr_rank_k/t5/tied)" \
        experiments/phase5_downstream_utility/results/training_index.jsonl \
        experiments/phase5_downstream_utility/results/probing_results.jsonl \
        experiments/phase5_downstream_utility/results/training_logs
done

# ── Final plots: headline bars (k=5) + seed-variance plot ────────────────
say "MILESTONE: regenerate final plots"
.venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py \
    > "$LOG_DIR/final_headline_plots.log" 2>&1 || say "WARN headline plot failed"
.venv/bin/python experiments/phase5_downstream_utility/plots/make_seed_variance_plot.py \
    --metric auc --k 5 > "$LOG_DIR/final_seed_variance_plot.log" 2>&1 || say "WARN seed variance plot failed"
.venv/bin/python experiments/phase5_downstream_utility/plots/make_seed_variance_plot.py \
    --metric acc --k 5 >> "$LOG_DIR/final_seed_variance_plot.log" 2>&1 || true

say "MILESTONE final commit+push"
commit_and_push "Phase 5 final: headline plots (last_position+mean_pool) + T17 seed-variance plot" \
    experiments/phase5_downstream_utility/results/plots \
    experiments/phase5_downstream_utility/results/headline_summary.json \
    experiments/phase5_downstream_utility/results \
    docs/han/research_logs/phase5_downstream_utility

say "post-run pod volume free: $(free_gb "$QUOTA_DIR") GB  cgroup memory: $(mem_gb_used) GB"
say "=== OVERNIGHT PHASE5 ORCHESTRATOR END ==="
