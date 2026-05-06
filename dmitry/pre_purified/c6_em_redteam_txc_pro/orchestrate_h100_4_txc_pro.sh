#!/usr/bin/env bash
# orchestrate_h100_4_txc_pro.sh — TXC-pro × C6 pipeline.
#
# Phases: A (preflight + smoke) → B (train 4 cells) → C (Wang full)
# → D (dense α-sweep) → E (detection) → F (sync results back).
#
# Updates state.json after each phase. Logs to orchestrate.log.

set -e
. ~/.env-c6
. /root/c6_venv/bin/activate
cd /workspace/temp_xc-c6-extend

STATE=/workspace/c6_redteam/state.json
LOG=/workspace/c6_redteam/orchestrate.log
RUNDIR=/workspace/c6_redteam
SCRIPTDIR=/workspace/c6_redteam

DATASOURCES=(
    qwen_2_5_14b_instruct_finance_l24_resid_post
    qwen_2_5_7b_instruct_medical_l15_resid_post
)
SEEDS=(1 42)

write_state() {
    local phase="$1" cell="$2" progress="$3"
    cat > "$STATE" <<EOF
{
  "pod": "h100_4_em",
  "experiment": "txc_pro_c6",
  "phase": "$phase",
  "current_cell": "$cell",
  "phase_progress": "$progress",
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

# ── Phase A: preflight ──────────────────────────────────────────────
write_state "A" "" "preflight"
echo "=== Phase A: preflight $(date -u) ==="

# Verify Python imports + arch resolution.
python -c "
import sys
sys.path.insert(0, '/workspace/temp_xc-c6-extend/purified/src')
sys.path.insert(0, '/workspace/temp_xc-c6-extend/purified')
from temp_bench.config import load_arch, instantiate_arch
spec = load_arch('txc_pro', component='c6')
print(f'TXC-pro c6 spec: {spec.hparams}')
m = instantiate_arch(spec, d_in=5120)
print(f'W_dec.shape = {tuple(m.W_dec.shape)}')
assert m.W_dec.shape == (32768, 10, 5120), f'unexpected shape {m.W_dec.shape}'
print('Phase A smoke OK')
"

# ── Phase B: train ─────────────────────────────────────────────────
echo "=== Phase B: train $(date -u) ==="
B_DONE=0
B_TOTAL=$((${#DATASOURCES[@]} * ${#SEEDS[@]}))
for ds in "${DATASOURCES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        write_state "B" "${ds}/seed=${seed}" "$B_DONE/$B_TOTAL"
        echo "--- Phase B cell ${ds}/seed=${seed} (${B_DONE}/${B_TOTAL}) $(date -u) ---"
        python "$SCRIPTDIR/run_txc_pro.py" \
            --datasource "$ds" --seed "$seed" --skip-eval
        B_DONE=$((B_DONE + 1))
    done
done
echo "=== Phase B done ($B_TOTAL cells trained) ==="

# ── Phase C: Wang full ─────────────────────────────────────────────
echo "=== Phase C: Wang full $(date -u) ==="
C_DONE=0
for ds in "${DATASOURCES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        write_state "C" "${ds}/seed=${seed}" "$C_DONE/$B_TOTAL"
        echo "--- Phase C cell ${ds}/seed=${seed} (${C_DONE}/${B_TOTAL}) $(date -u) ---"
        python "$SCRIPTDIR/run_txc_pro.py" \
            --datasource "$ds" --seed "$seed"
        C_DONE=$((C_DONE + 1))
    done
done
echo "=== Phase C done ==="

# ── Phase D: dense α-sweep ─────────────────────────────────────────
echo "=== Phase D: dense α-sweep $(date -u) ==="
write_state "D" "all" "0/4"

# Discover the new train_keys from manifest.jsonl (added by Phase B/C).
NEW_TKS=$(python -c "
import json
keys = []
for line in open('/workspace/temp_xc-c6-extend/purified/checkpoints/manifest.jsonl'):
    d = json.loads(line)
    if d.get('arch') == 'txc_pro' and 'qwen_2_5' in d.get('datasource',''):
        keys.append(d['train_key'])
print(' '.join(sorted(set(keys))))
")
echo "discovered new TXC-pro train_keys: $NEW_TKS"

# arch_T_for fix needed for txc_pro=10 — patch the driver locally on pod.
python -c "
import re
p = '/workspace/temp_xc-c6-extend/dmitry/pre_purified/c6_em_redteam/detection_eval.py'
import os
if os.path.exists(p):
    src = open(p).read()
    if 'txc_pro' not in src.split('def arch_T_for')[1].split('def ')[0]:
        src = src.replace(
            'def arch_T_for(arch: str) -> int:\n            return 5 if arch == \"txc_base\" else 1',
            'def arch_T_for(arch: str) -> int:\n            return 10 if arch == \"txc_pro\" else (5 if arch == \"txc_base\" else 1)',
        )
        open(p, 'w').write(src)
        print('patched detection_eval.py for txc_pro T=10')
" 2>&1 || true

python /workspace/c6_redteam/extend_alpha_sweep.py \
    --cells $NEW_TKS \
    --alphas 10 20 30 40 50 60 70 80 90 \
             -10 -20 -30 -40 -50 -60 -70 -80 -90 \
             110 120 130 140 150 200 \
             -110 -120 -130 -140 -150 -200

# ── Phase E: detection ─────────────────────────────────────────────
echo "=== Phase E: detection $(date -u) ==="
write_state "E" "all" "0/4"
python /workspace/c6_redteam/detection_eval.py --cells $NEW_TKS

# ── Phase F: done ──────────────────────────────────────────────────
echo "=== Phase F: pipeline complete $(date -u) ==="
write_state "F" "" "all 4 cells through pipeline"
echo "TXC-pro × C6 pipeline complete. Results under /workspace/c6_redteam/."
