#!/bin/bash
# Tier-1 fast-signal sweep: 4 candidates, train+probe each, ~2-3 hr total.
#
# Order chosen by hypothesis priority (most likely to teach us something):
#   1. D1 strided_track2 T_eff=5 stride=2  — receptive-field test
#   2. C3 token_subseq pos_mode=learned t_sample=5  — full-rank pos bias (NEW)
#   3. C2 token_subseq pos_mode=sinusoidal t_sample=5  — low-rank pos bias
#   4. B2 subseq_track2 T_max=10 t_sample=5 non-contig  — direct T-scale
#
# Outputs:
#   training_index.jsonl  — Phase 5B
#   probing_results.jsonl — Phase 5B
#   /tmp/phase5b_tier1.log — combined stderr/stdout

set -u
cd /home/elysium/temp_xc

LOG=/tmp/phase5b_tier1.log
PYTHON=".venv/bin/python"
echo "=== TIER-1 SWEEP START $(date -Is) ===" > $LOG

train_one() {
    name=$1; shift
    arch_id=$1; shift
    run_id=$1; shift
    echo "" >> $LOG
    echo "===== [$name] TRAIN ($run_id) $(date -Is) =====" >> $LOG
    TQDM_DISABLE=1 timeout 1800 $PYTHON \
        -m experiments.phase5b_t_scaling_explore.train_phase5b \
        --arch $arch_id "$@" >> $LOG 2>&1
    train_status=$?
    echo "===== [$name] TRAIN EXIT=$train_status $(date -Is) =====" >> $LOG
    if [ $train_status -ne 0 ]; then
        echo "  TRAIN FAILED, skipping probe" >> $LOG
        return $train_status
    fi
    echo "===== [$name] PROBE ($run_id) $(date -Is) =====" >> $LOG
    TQDM_DISABLE=1 timeout 1200 $PYTHON \
        -m experiments.phase5b_t_scaling_explore.run_probing_phase5b \
        --run_ids $run_id \
        --aggregations last_position mean_pool 2>&1 \
        | grep -vE "FutureWarning|UserWarning|deprecated|warnings.warn" >> $LOG
    probe_status=$?
    echo "===== [$name] PROBE EXIT=$probe_status $(date -Is) =====" >> $LOG
    return 0
}

# D1: strided_track2 T_eff=5 stride=2
train_one D1 phase5b_strided_track2 phase5b_strided_track2__seed42 \
    --T_eff 5 --stride 2

# C3: token_subseq learned per-position d_sae bias  (NEW)
train_one C3 phase5b_token_subseq phase5b_token_subseq_learned__seed42 \
    --pos_mode learned --t_sample 5

# C2: token_subseq sinusoidal pos emb
train_one C2 phase5b_token_subseq phase5b_token_subseq_sinusoidal__seed42 \
    --pos_mode sinusoidal --t_sample 5

# B2: subseq_track2 T_max=10 t_sample=5 non-contig
train_one B2 phase5b_subseq_track2 phase5b_subseq_track2__seed42 \
    --T_max 10 --t_sample 5

echo "" >> $LOG
echo "=== TIER-1 SWEEP END $(date -Is) ===" >> $LOG
