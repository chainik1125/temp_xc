#!/usr/bin/env bash
# orchestrate_h100_5_more_seeds.sh — SAE+TXC-base × seeds {2,3} pipeline.
#
# Re-uses the canonical c6_em runner (which knows about sae_arditi
# and txc_base). Just runs more seeds. After each seed completes,
# rsyncs results back so partial progress is observable from local.

set -e
. ~/.env-c6
. /root/c6_venv/bin/activate
# Run from purified/ so `python -m experiments.c6_em.run` resolves.
cd /workspace/temp_xc-c6-extend/purified
export PYTHONPATH=/workspace/temp_xc-c6-extend/purified/src:/workspace/temp_xc-c6-extend/purified${PYTHONPATH:+:$PYTHONPATH}

STATE=/workspace/c6_redteam/state.json
SCRIPTDIR=/workspace/c6_redteam

DATASOURCES=(
    qwen_2_5_14b_instruct_finance_l24_resid_post
    qwen_2_5_7b_instruct_medical_l15_resid_post
)
ARCHS=(sae_arditi txc_base)
# Seed=2 first, then seed=3 (so partial results land sooner).
SEEDS=(2 3)

write_state() {
    local phase="$1" cell="$2" progress="$3"
    cat > "$STATE" <<EOF
{
  "pod": "h100_5_em",
  "experiment": "more_seeds_c6",
  "phase": "$phase",
  "current_cell": "$cell",
  "phase_progress": "$progress",
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

write_state "A" "" "preflight"
echo "=== Phase A: preflight $(date -u) ==="
python -c "
import sys
sys.path.insert(0, '/workspace/temp_xc-c6-extend/purified/src')
sys.path.insert(0, '/workspace/temp_xc-c6-extend/purified')
from temp_bench.config import load_arch
print('SAE c6:', load_arch('sae_arditi', component='c6').hparams)
print('TXC-base c6:', load_arch('txc_base', component='c6').hparams)
print('Phase A OK')
"

# ── Phase B+C: train + Wang full per seed (one full pipeline per seed
#    so partial results land progressively) ─────────────────────────
B_TOTAL=$((${#DATASOURCES[@]} * ${#ARCHS[@]} * ${#SEEDS[@]}))
B_DONE=0
for seed in "${SEEDS[@]}"; do
    for arch in "${ARCHS[@]}"; do
        for ds in "${DATASOURCES[@]}"; do
            write_state "BC" "${arch}/${ds}/seed=${seed}" "$B_DONE/$B_TOTAL"
            echo "--- Cell ${arch}/${ds}/seed=${seed} (${B_DONE}/${B_TOTAL}) $(date -u) ---"
            python /workspace/c6_redteam/run_more_seeds.py \
                --archs "$arch" \
                --seed "$seed" \
                --datasource "$ds"
            B_DONE=$((B_DONE + 1))
            # After each cell, sync to local (best-effort, ignore failure).
            rsync -av --include='*/' --include='wang_full.json' \
                --include='judge_outputs.jsonl' --include='stage*.json' \
                --exclude='*' \
                /workspace/temp_xc-c6-extend/purified/results/runs/c6_*/ \
                /workspace/c6_redteam/seed_results/ 2>/dev/null || true
        done
    done
done

# ── Phase D: dense α-sweep across all new cells ──────────────────────
echo "=== Phase D: dense α-sweep $(date -u) ==="
write_state "D" "all" "0/$B_TOTAL"

# Discover new train_keys (filter by arch sae_arditi/txc_base AND seeds 2,3).
NEW_TKS=$(python -c "
import json
keys = []
for line in open('/workspace/temp_xc-c6-extend/purified/checkpoints/manifest.jsonl'):
    d = json.loads(line)
    if (d.get('arch') in ('sae_arditi','txc_base')
        and 'qwen_2_5' in d.get('datasource','')
        and d.get('seed') in (2, 3)):
        keys.append(d['train_key'])
print(' '.join(sorted(set(keys))))
")
echo "discovered new train_keys: $NEW_TKS"

python /workspace/c6_redteam/extend_alpha_sweep.py \
    --cells $NEW_TKS \
    --alphas 10 20 30 40 50 60 70 80 90 \
             -10 -20 -30 -40 -50 -60 -70 -80 -90 \
             110 120 130 140 150 200 \
             -110 -120 -130 -140 -150 -200

# ── Phase E: detection ─────────────────────────────────────────────
echo "=== Phase E: detection $(date -u) ==="
write_state "E" "all" "0/$B_TOTAL"
python /workspace/c6_redteam/detection_eval.py --cells $NEW_TKS

write_state "F" "" "all $B_TOTAL cells through pipeline"
echo "=== more-seeds pipeline complete $(date -u) ==="
