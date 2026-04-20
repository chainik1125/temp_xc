#!/usr/bin/env bash
# OOM-safe full_window probing: one arch per python subprocess.
# Each arch gets a fresh process → task_cache RAM is released between archs.
# Commits+pushes after each arch per the "push every milestone" rule.
set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export UV_LINK_MODE=copy
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
mkdir -p "$LOG_DIR"
MAIN=/workspace/temp_xc/logs/overnight/fw_per_arch.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

ARCHS=(
    "txcdr_shared_dec_t5__seed42"
    "txcdr_shared_enc_t5__seed42"
    "txcdr_tied_t5__seed42"
    "txcdr_pos_t5__seed42"
    "txcdr_causal_t5__seed42"
    "txcdr_block_sparse_t5__seed42"
    "txcdr_lowrank_dec_t5__seed42"
    "txcdr_rank_k_dec_t5__seed42"
    "temporal_contrastive__seed42"
    "tfa_small__seed42"
    "tfa_pos_small__seed42"
    "time_layer_crosscoder_t5__seed42"
)

commit_push() {
    local msg="$1"
    git add experiments/phase5_downstream_utility/results/probing_results.jsonl \
            experiments/phase5_downstream_utility/results/plots/ \
            experiments/phase5_downstream_utility/results/headline_summary*.json 2>/dev/null || true
    git commit -m "$msg" 2>>"$MAIN" >>"$MAIN" || true
    git push origin han 2>>"$MAIN" >>"$MAIN" || true
}

say "=== FW_PER_ARCH START ==="

for arch in "${ARCHS[@]}"; do
    # Skip if already complete (>=144 records in jsonl)
    n=$(.venv/bin/python -c "
import json
c = 0
with open('experiments/phase5_downstream_utility/results/probing_results.jsonl') as f:
    for l in f:
        r = json.loads(l)
        if r.get('run_id')=='$arch' and r.get('aggregation')=='full_window':
            c += 1
print(c)
" 2>/dev/null)
    if [ "${n:-0}" -ge 144 ]; then
        say "  $arch: already complete ($n records), skip"
        continue
    fi
    say "  $arch: probing full_window ($n/144 existing)"
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation full_window --skip-baselines --run-ids "$arch" \
        > "$LOG_DIR/fw_${arch}.log" 2>&1
    ec=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $ec -eq 0 ]; then
        say "    $arch: OK (${dt}s)"
    else
        say "    $arch: FAILED exit=$ec (${dt}s)"
    fi
    commit_push "Phase 5: probe full_window $arch"
done

say "  regenerating plots"
.venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py \
    > "$LOG_DIR/plots_after_fw.log" 2>&1 || true
.venv/bin/python experiments/phase5_downstream_utility/plots/plot_training_curves.py \
    >> "$LOG_DIR/plots_after_fw.log" 2>&1 || true
.venv/bin/python experiments/phase5_downstream_utility/analyze_decoder_svd.py \
    >> "$LOG_DIR/plots_after_fw.log" 2>&1 || true
commit_push "Phase 5: plots after full_window probing"

say "=== FW_PER_ARCH END ==="
