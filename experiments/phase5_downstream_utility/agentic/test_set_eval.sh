#!/usr/bin/env bash
# Test-set eval for the agentic winners (cycle 02, cycle 08) + the
# Part-B family winners (A3 α=1.0, MLC α=1.0) at both `last_position`
# and `mean_pool` aggregations.
#
# FIRST AND ONLY test-set touch for these archs. Gated on user approval.
#
# Runs probes in two waves:
#   Wave 1: seed=42 for all 4 focal arches (2 agentic + 2 Part-B).
#           Matches the bench's single-seed convention, minimal compute.
#   Wave 2: seed=1, 2 for the two agentic winners only (variance data).
#
# ~2 hr total on single GPU.

set -u
cd /workspace/temp_xc
export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG=/workspace/temp_xc/logs/overnight/autoresearch_orchestrator.log
say() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== TEST-SET EVAL: agentic winners + Part-B winners ==="

# Wave 1: seed=42 for all 4 focal arches
WAVE1=(
    agentic_txc_02__seed42
    agentic_mlc_08__seed42
    matryoshka_txcdr_contrastive_t5_alpha100__seed42
    mlc_contrastive_alpha100__seed42
)

# Wave 2: seeds 1, 2 for agentic winners (variance)
WAVE2=(
    agentic_txc_02__seed1
    agentic_txc_02__seed2
    agentic_mlc_08__seed1
    agentic_mlc_08__seed2
)

for AGG in last_position mean_pool; do
    say "--- Wave 1 at ${AGG} (seed=42 bench) ---"
    t0=$(date +%s)
    .venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "${AGG}" --skip-baselines \
        --run-ids "${WAVE1[@]}" \
        > "logs/overnight/testset_wave1_${AGG}.log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    say "    wave1 ${AGG} exit=${rc} in ${dt}s"
done

for AGG in last_position mean_pool; do
    say "--- Wave 2 at ${AGG} (seeds 1,2 variance) ---"
    t0=$(date +%s)
    .venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "${AGG}" --skip-baselines \
        --run-ids "${WAVE2[@]}" \
        > "logs/overnight/testset_wave2_${AGG}.log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    say "    wave2 ${AGG} exit=${rc} in ${dt}s"
done

say "=== TEST-SET EVAL COMPLETE ==="

# Commit the new test-set rows
git -c user.name="Han" -c user.email="hxuany0@gmail.com" add \
    experiments/phase5_downstream_utility/results/probing_results.jsonl \
    2>&1 | tee -a "$LOG"
git -c user.name="Han" -c user.email="hxuany0@gmail.com" commit -m "test-set eval: agentic winners + Part-B winners at last_position + mean_pool

Probes:
- Wave 1 (seed=42 bench entries): agentic_txc_02, agentic_mlc_08,
  matryoshka_txcdr_contrastive_t5_alpha100, mlc_contrastive_alpha100.
- Wave 2 (seeds 1, 2 variance): agentic_txc_02, agentic_mlc_08.

First and only test-set touch for these archs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" 2>&1 | tee -a "$LOG" || say "  (no changes to commit)"
git push origin han 2>&1 | tee -a "$LOG" || say "  push FAILED"
say "=== COMMIT + PUSH DONE ==="
