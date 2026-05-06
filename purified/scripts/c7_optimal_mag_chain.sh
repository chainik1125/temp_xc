#!/usr/bin/env bash
# Chain optimal-magnitude re-evals on GPU 0 for every cell whose
# canonical leaderboard row has landed. Each cell takes ~10-15 min
# (61 questions × 2 magnitudes = 122 generations + 244 judge calls).
#
# Usage:
#   set -a; source /workspace/aniket/temp_xc/.env; set +a
#   cd /workspace/aniket/temp_xc-final/purified
#   nohup bash scripts/c7_optimal_mag_chain.sh \
#       > logs/c7_optimal_chain.log 2>&1 &

set -u
cd /workspace/aniket/temp_xc-final/purified

GPU=3
SEEN=logs/c7_optimal_seen.txt
touch "$SEEN"

log() { printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*"; }

# Wait for any running eval_optimal_mag (e.g. the smoke-launched one)
# to exit before starting our queue.
while pgrep -af "eval_optimal_mag" | grep -v "$$" >/dev/null 2>&1; do
    sleep 30
done

# (arch, bs, n_steps, seed) cells to process — same order as the
# Python DEFAULT_CELLS in analyze_optimal.py.
CELLS=(
    "txc_base 256  300000 42"
    "txc_base 1024 300000 42"
    "txc_pro  256  300000 42"
    "mlc      1024 300000 42"
    "tsae_paper 1024 300000 42"
    "txc_pro  1024 300000 42"
    "topk_sae 1024 300000 42"
)

while true; do
    progress=0
    for spec in "${CELLS[@]}"; do
        read arch bs ns seed <<< "$spec"
        key="${arch}|${bs}"
        grep -q "^${key}$" "$SEEN" 2>/dev/null && continue

        # Skip if checkpoint not on disk yet (cell still training).
        train_key=$(.venv/bin/python -c "
from temp_bench.config import compute_train_key, compute_act_cache_key, load_arch, load_datasource
from temp_bench.schemas import TrainingConfig
spec = load_arch('${arch}', component='c7')
ds = load_datasource('llama_3_1_8b_base_l10_ward_nousmirror')
print(compute_train_key(arch=spec, seed=${seed},
    training_cfg=TrainingConfig(n_steps=${ns}, batch_size=${bs}),
    act_cache_key=compute_act_cache_key(ds)))
" 2>/dev/null)
        if [ ! -f "checkpoints/${train_key}/model.safetensors" ]; then
            continue   # cell still training; will retry next pass
        fi
        # Skip if no canonical leaderboard row yet (peak_mag unknown).
        has_row=$(.venv/bin/python -c "
import json
with open('results/leaderboard.jsonl') as f:
    for line in f:
        try: r=json.loads(line)
        except: continue
        if (r.get('component')=='c7'
                and r.get('train_key')=='${train_key}'
                and not r.get('eval_cfg', {}).get('_extended_mags')):
            print('yes'); break
" 2>/dev/null)
        if [ "$has_row" != "yes" ]; then
            continue
        fi

        local_log="logs/c7_optimal_${arch}_bs${bs}.log"
        log "launching ${arch} bs=${bs} → ${local_log}"
        CUDA_VISIBLE_DEVICES=$GPU \
        AGENT_NAME=agent_back_300k \
        TEMP_BENCH_POD_MODE=persistent \
        TQDM_DISABLE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            .venv/bin/python -m experiments.c7_backtracking.eval_optimal_mag \
                --arch "$arch" --bs "$bs" \
                --n-steps "$ns" --seed "$seed" \
                > "$local_log" 2>&1
        rc=$?
        log "${arch} bs=${bs} exit=${rc}"
        if [ $rc -eq 0 ]; then
            echo "$key" >> "$SEEN"
            progress=1
        else
            log "FAILED — will retry on next pass"
        fi
    done

    n_done=$(wc -l < "$SEEN")
    if [ "$n_done" -ge "${#CELLS[@]}" ]; then
        log "all ${#CELLS[@]} cells processed — chain complete"
        break
    fi
    if [ $progress -eq 0 ]; then
        # No cell completed this pass; sleep before retrying (waiting
        # for in-flight training to land).
        sleep 300
    fi
done
