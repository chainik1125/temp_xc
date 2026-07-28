#!/bin/bash
# mac-d — self-driving pf grid worker. One pod owns one T, all seeds.
#
#   PF_T=2 nohup bash pod_drive.sh &
#
# Assignment is one T per pod (uniform memory profile per pod) and seeds
# in the order 42,1,2 — so s42 completes across ALL pods first and a
# full-shape 1-seed curve is renderable early (hub's seed-column-first).
#
# Chain: wait for cache -> keyed install -> CUDA equivalence receipt ->
# run cells. Every stage writes a marker; any failure writes DRIVE-FAIL
# and stops, because an idle pod that says why beats one that is
# silently doing nothing.
set -uo pipefail
LOG=/workspace/drive.log
exec >>"$LOG" 2>&1
T="${PF_T:?PF_T required}"
echo "=== drive start T=$T $(date -u +%FT%TZ) ==="
fail() { echo "DRIVE-FAIL: $*"; exit 1; }

cd /workspace/temp_xc || fail "no repo"
export HF_TOKEN="$(cat /workspace/.tokens/hf_token 2>/dev/null)"
export AGENT_NAME=mac-d
export TEMP_BENCH_BUFFER_RESIDENT=1     # the whole point of this venue
export TEMP_BENCH_ALLOW_DIRTY=1
SRC=/workspace/caches/rlhf/txcdr-base-data/activation_cache

# 1) cache (fetch.sh may already be pulling it; just wait for the file)
for i in $(seq 1 180); do
    [ -f "$SRC/resid_L12.npy" ] && break
    [ "$i" = 180 ] && fail "cache never appeared (90 min)"
    sleep 30
done
# settle: size must stop changing
prev=0
for i in $(seq 1 60); do
    cur=$(stat -c %s "$SRC/resid_L12.npy" 2>/dev/null || echo 0)
    [ "$cur" = "$prev" ] && [ "$cur" -gt 14000000000 ] && break
    prev=$cur; sleep 20
done
echo "CACHE-OK $(du -sh "$SRC" | cut -f1)"

# 2) keyed install (hardlink, idempotent)
ACTMIX_RLHF_CACHE_SRC="$SRC" \
  .venv/bin/python -m experiments.explorations.actmix_rlhf.convert_train_cache \
  || fail "convert_train_cache"
echo "INSTALL-OK"

# 3) equivalence receipt ON CUDA — gate before any grid hour
.venv/bin/python scripts/verify_resident_buffer.py \
    gemma_2_2b_base_l12_phase7 10 1024 > /workspace/receipt.log 2>&1
grep -q "VERDICT: PASS" /workspace/receipt.log \
    || { cat /workspace/receipt.log; fail "resident buffer NOT bitwise identical on CUDA"; }
echo "RECEIPT-PASS"
grep -E 'mean (host|resident)' /workspace/receipt.log

# 4) cells: this pod's T across seeds 42,1,2
for S in 42 1 2; do
    echo "--- CELL T=$T seed=$S start $(date -u +%FT%TZ) ---"
    # NB: no /usr/bin/time — absent on this image (rc=127 killed the
    # first launch). The driver below reports its own wall time.
    .venv/bin/python - "$T" "$S" <<'PY'
import sys, time
from experiments.explorations.actmix_rlhf.cells import pf
from temp_bench.core.runner import run_experiment
T, seed = int(sys.argv[1]), int(sys.argv[2])
c = pf(T, seed)
t0 = time.time()
res = run_experiment(
    experiment="rlhf", arch_name=c["arch"], seed=seed,
    datasource_name=c["datasource"], training_cfg=c["training_cfg"],
    eval_cfg=c.get("eval_cfg", {}), agent="mac-d", allow_dirty=True,
)
m = (res or {}).get("metrics", {}) if isinstance(res, dict) else {}
print(f"CELL-OK T={T} s={seed} wall={time.time()-t0:.0f}s "
      f"auc={m.get('preference_auc_k20')}")
PY
    rc=$?
    [ $rc -eq 0 ] || echo "CELL-FAIL T=$T s=$S rc=$rc"
done
echo "DRIVE-DONE T=$T $(date -u +%FT%TZ)"
