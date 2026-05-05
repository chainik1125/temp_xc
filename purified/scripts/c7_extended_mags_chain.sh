#!/usr/bin/env bash
# Chain extended-magnitude C7 evals on GPU 0 for every cell whose
# canonical (±16 grid) eval has already landed in the leaderboard.
# Each cell takes ~30-45 min on H100 (5 mags × 61 questions).
#
# When new TXC / baseline cells finish their canonical eval (their
# training PIDs exit), this script picks them up automatically.
#
# Usage:
#   set -a; source /workspace/aniket/temp_xc/.env; set +a
#   cd /workspace/aniket/temp_xc-final/purified
#   nohup bash scripts/c7_extended_mags_chain.sh \
#       > logs/c7_extended_mags_chain.log 2>&1 &

set -u
cd /workspace/aniket/temp_xc-final/purified

GPU=0
EXT_MAGS="-32 -24 0 24 32"
SEEN_FILE=logs/c7_extended_mags_seen.txt
touch "$SEEN_FILE"

log() { printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*"; }

# Yield (arch, bs) pairs for every c7 cell whose CANONICAL leaderboard
# row exists (n_steps=300000, _extended_mags absent). Deduped by what's
# already been processed in this run.
list_done_canonical_cells() {
    .venv/bin/python -c "
import json, os
seen = set()
if os.path.exists('$SEEN_FILE'):
    seen = {l.strip() for l in open('$SEEN_FILE') if l.strip()}
out = []
with open('results/leaderboard.jsonl') as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        if r.get('component') != 'c7': continue
        if r.get('eval_cfg', {}).get('_extended_mags'): continue  # skip our own runs
        cfg_path = f'checkpoints/{r[\"train_key\"]}/config.json'
        try: c = json.load(open(cfg_path))
        except: continue
        if c['training_cfg'].get('n_steps') != 300_000: continue
        key = f\"{r['arch']}|{c['training_cfg']['batch_size']}\"
        if key in seen: continue
        out.append((key, r['arch'], c['training_cfg']['batch_size']))
for key, arch, bs in out:
    print(f'{key}\t{arch}\t{bs}')
"
}

run_extended() {
    local arch=$1
    local bs=$2
    local key=$3

    local logfile="logs/c7_extended_mags_${arch}_bs${bs}.log"
    log "launching extended-mags arch=$arch bs=$bs → $logfile"

    CUDA_VISIBLE_DEVICES=$GPU \
    AGENT_NAME=agent_back_300k \
    TEMP_BENCH_POD_MODE=persistent \
    TQDM_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/python -m experiments.c7_backtracking.eval_extended_mags \
            --arch "$arch" --bs "$bs" \
            --magnitudes $EXT_MAGS \
            > "$logfile" 2>&1
    local rc=$?
    log "extended-mags arch=$arch bs=$bs exit=$rc"
    if [ $rc -eq 0 ]; then
        echo "$key" >> "$SEEN_FILE"
    fi
    return $rc
}

log "loop started — polling every 60s for new canonical cells"

while true; do
    while IFS=$'\t' read -r key arch bs; do
        [ -z "$key" ] && continue
        run_extended "$arch" "$bs" "$key"
    done < <(list_done_canonical_cells)

    # Poll: 60s. Exit only when ALL 7 expected canonical cells have
    # been processed (4 TXC + 3 baselines).
    n_seen=$(wc -l < "$SEEN_FILE")
    if [ "$n_seen" -ge 7 ]; then
        log "all 7 cells processed — chain complete"
        break
    fi
    sleep 60
done
