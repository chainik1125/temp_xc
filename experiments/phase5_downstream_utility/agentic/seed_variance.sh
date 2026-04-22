#!/usr/bin/env bash
# Seed-variance confirmation for the two agentic-loop winners.
# For each (arch, seed) pair: train → probe val → append row → commit.
#
# Designed to run serial on single GPU. ~2.5 hr total.
#
# Usage: bash seed_variance.sh

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG=/workspace/temp_xc/logs/overnight/autoresearch_orchestrator.log
say() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$LOG"; }

# Pairs: arch seeds
ARCH_SEEDS=(
    "agentic_txc_02 1"
    "agentic_txc_02 2"
    "agentic_mlc_08 1"
    "agentic_mlc_08 2"
)

BASE_OF_agentic_txc_02="matryoshka_t5"
BASE_OF_agentic_mlc_08="mlc"

for pair in "${ARCH_SEEDS[@]}"; do
    arch=$(echo "$pair" | awk '{print $1}')
    seed=$(echo "$pair" | awk '{print $2}')
    base_var="BASE_OF_${arch}"
    base="${!base_var}"
    rid="${arch}__seed${seed}"

    say "=== seed-variance ${rid} (base=${base}) ==="

    # Train if ckpt missing
    ckpt="experiments/phase5_downstream_utility/results/ckpts/${rid}.pt"
    if [[ -f "$ckpt" ]]; then
        say "  ckpt exists; skipping train"
    else
        say "  training ${rid}"
        t0=$(date +%s)
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[${seed}], max_steps=25000, archs=['${arch}'])
" > "logs/overnight/seedvar_train_${rid}.log" 2>&1
        rc=$?
        dt=$(( $(date +%s) - t0 ))
        say "  train exit=${rc} in ${dt}s"
        if [[ $rc -ne 0 ]]; then
            say "  TRAIN FAILED; skipping probe/commit for ${rid}"
            continue
        fi
    fi

    # Probe val
    say "  probing ${rid} @ last_position_val"
    t0=$(date +%s)
    .venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation last_position_val --skip-baselines \
        --run-ids "${rid}" > "logs/overnight/seedvar_probe_${rid}.log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    say "  probe exit=${rc} in ${dt}s"
    if [[ $rc -ne 0 ]]; then
        say "  PROBE FAILED; skipping commit for ${rid}"
        continue
    fi

    # Summarise + write index row
    say "  summarising ${rid} vs ${base}"
    .venv/bin/python experiments/phase5_downstream_utility/autoresearch_summarise.py \
        --candidate "${arch}" --base "${base}" --seed "${seed}" --k 5 \
        --write-index --notes "seed-variance confirmation for agentic winner" 2>&1 | tee -a "$LOG"

    # Commit
    git -c user.name="Han" -c user.email="hxuany0@gmail.com" add \
        experiments/phase5_downstream_utility/results/autoresearch_index.jsonl \
        experiments/phase5_downstream_utility/results/probing_results.jsonl \
        "experiments/phase5_downstream_utility/results/training_logs/${rid}.json" 2>&1 | tee -a "$LOG"
    git -c user.name="Han" -c user.email="hxuany0@gmail.com" commit -m "seed-variance: ${rid} (val d vs ${base})

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" 2>&1 | tee -a "$LOG" || true
    git push origin han 2>&1 | tee -a "$LOG" || say "  push FAILED; continuing"
    say "=== ${rid} DONE ==="
done

say "=== seed-variance COMPLETE ==="
