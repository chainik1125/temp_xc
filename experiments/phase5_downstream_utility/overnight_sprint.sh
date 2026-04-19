#!/usr/bin/env bash
# Phase 5 overnight sprint orchestrator. Runs autonomously for ~8-10h.
#
# Pipeline:
#   1. wait for current full_window probing to finish
#   2. commit + push training/probing snapshot
#   3. build probe cache for new tasks (amazon 5-cat + github-code 4 langs)
#   4. train 12 new archs (errors per-arch don't block others)
#   5. probe all 19 archs × {last_position, full_window}
#   6. regenerate plots (8 = 2 task-sets × 2 aggregations × 2 metrics)
#   7. write summary.md update; commit + push
#
# Logs land under /workspace/temp_xc/logs/overnight/*.log.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export UV_LINK_MODE=copy
export PYTHONPATH=/workspace/temp_xc

LOG_DIR=/workspace/temp_xc/logs/overnight
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/sprint.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }

commit_push() {
    local msg="$1"
    cd /workspace/temp_xc
    git add -A :/ 2>/dev/null || true
    git commit -m "$msg" 2>>"$MAIN_LOG" >>"$MAIN_LOG" || true
    git push origin han 2>>"$MAIN_LOG" >>"$MAIN_LOG" || true
}

say "=== OVERNIGHT SPRINT START ==="

# ─── Step 1: wait for current full_window probing to finish
say "Step 1: waiting for existing full_window probing"
while pgrep -f "run_probing.py" >/dev/null; do sleep 120; done
say "  full_window probing done; current jsonl lines: $(wc -l < experiments/phase5_downstream_utility/results/probing_results.jsonl)"

# Regenerate plots + summary with the pre-expansion 7-arch data (snapshot).
say "Step 1b: plots snapshot (pre-expansion)"
.venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py \
    >"$LOG_DIR/plots_preexp.log" 2>&1 || true
.venv/bin/python experiments/phase5_downstream_utility/plots/plot_training_curves.py \
    >>"$LOG_DIR/plots_preexp.log" 2>&1 || true

commit_push "Phase 5: pre-expansion snapshot (7 archs, acc + full_window)"

# ─── Step 2: build probe cache for new tasks (amazon 5-cat + github-code)
say "Step 2: build probe cache for expanded task set"
.venv/bin/python experiments/phase5_downstream_utility/probing/build_probe_cache.py \
    --include-crosstoken >"$LOG_DIR/probe_cache_expand.log" 2>&1
say "  probe cache build exit=$? (expected skips for already-cached tasks)"

# Re-baseline probing for NEW tasks only. The old baselines already exist
# for old tasks; the new tasks need fresh baselines. Append both aggregations.
NEW_TASKS=$(ls experiments/phase5_downstream_utility/results/probe_cache/ \
    | grep -E "^(amazon_reviews_cat|github_code_)" || true)
say "  new tasks detected: $NEW_TASKS"
if [ -n "$NEW_TASKS" ]; then
    for agg in last_position full_window; do
        .venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
            --tasks $NEW_TASKS --aggregation $agg \
            >"$LOG_DIR/probe_new_tasks_${agg}.log" 2>&1 || true
    done
fi
commit_push "Phase 5: expanded task set (amazon 5-cat + github-code via code_search_net)"

# ─── Step 3: train 12 new archs (per-arch fault isolation)
say "Step 3: training 12 new architectures"
NEW_ARCHS=(
    "txcdr_shared_dec_t5"
    "txcdr_shared_enc_t5"
    "txcdr_tied_t5"
    "txcdr_pos_t5"
    "txcdr_causal_t5"
    "txcdr_block_sparse_t5"
    "txcdr_lowrank_dec_t5"
    "txcdr_rank_k_dec_t5"
    "temporal_contrastive"
    "tfa_small"
    "tfa_pos_small"
    "time_layer_crosscoder_t5"
)

for arch in "${NEW_ARCHS[@]}"; do
    say "  training $arch"
    .venv/bin/python experiments/phase5_downstream_utility/train_primary_archs.py \
        --seeds 42 --max-steps 25000 --archs "$arch" \
        >"$LOG_DIR/train_${arch}.log" 2>&1
    ec=$?
    if [ $ec -eq 0 ]; then
        say "    $arch: OK"
    else
        say "    $arch: FAILED (exit $ec), see $LOG_DIR/train_${arch}.log"
    fi
    commit_push "Phase 5 overnight: train $arch"
done

# ─── Step 4: probe all 19 archs × 2 aggregations
say "Step 4: probing all available ckpts × both aggregations"
for agg in last_position full_window; do
    .venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation $agg --skip-baselines \
        >"$LOG_DIR/probe_final_${agg}.log" 2>&1 || true
done
commit_push "Phase 5 overnight: probed all archs (last_position + full_window)"

# ─── Step 5: regenerate plots (8 plots: 2 task-sets × 2 aggs × 2 metrics)
say "Step 5: regenerate plots"
.venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py \
    >"$LOG_DIR/plots_final.log" 2>&1 || true
.venv/bin/python experiments/phase5_downstream_utility/plots/plot_training_curves.py \
    >>"$LOG_DIR/plots_final.log" 2>&1 || true
.venv/bin/python experiments/phase5_downstream_utility/analyze_decoder_svd.py \
    >>"$LOG_DIR/plots_final.log" 2>&1 || true

commit_push "Phase 5 overnight: regenerate plots + SVD spectrum analysis"

# ─── Step 6: status summary of what shipped
say "Step 6: final status"
ls experiments/phase5_downstream_utility/results/ckpts/ | wc -l | \
    xargs -I {} say "  ckpts: {} total"
wc -l experiments/phase5_downstream_utility/results/probing_results.jsonl | \
    awk '{print "  probing_results.jsonl lines: " $1}' | tee -a "$MAIN_LOG"
ls experiments/phase5_downstream_utility/results/plots/ | wc -l | \
    xargs -I {} say "  plots: {} files"

say "=== OVERNIGHT SPRINT END ==="
