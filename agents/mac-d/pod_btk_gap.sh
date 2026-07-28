#!/bin/bash
# mac-d — run the three DEFERRED btk cells on the surviving pf pod.
#
# Hub 68e146e0f: T6/s2, T10/s1, T10/s2 use the SAME
# gemma_2_2b_base_l12_phase7 cache this pod already holds, so running
# them here costs runtime only; terminating first would re-pay a full
# bootstrap (pod spin + repo + venv + the 14 GB fetch that cost ~35 min
# and three distinct failure modes this afternoon).
#
# These three are the entire gap between the btk exhibit and uniform
# 3-seed coverage.
#
# ⚠ btk is NOT pf: it keeps n_steps=25000 and the framework-default
# warmup_steps=1000. `txc()` builds that correctly — do NOT pass any pf
# constant in here, or the new cells get train_keys that do not match
# the 26 btk rows already on the board.
set -uo pipefail
LOG=/workspace/btk.log
exec >>"$LOG" 2>&1
echo "=== btk gap-fill start $(date -u +%FT%TZ) ==="

cd /workspace/temp_xc || { echo "BTK-FAIL norepo"; exit 1; }
export AGENT_NAME=mac-d
export TEMP_BENCH_BUFFER_RESIDENT=1
export TEMP_BENCH_ALLOW_DIRTY=1

# wait for the in-flight pf cell to finish — by ARTIFACT (3 distinct
# T=8 cells in drive.log), not by pgrep. pgrep -f matches probe shells
# and leftover heredocs and deadlocked two watchers earlier today.
for i in $(seq 1 120); do
    n=$(grep -h CELL-OK /workspace/drive.log 2>/dev/null \
        | grep -oE "T=8 s=[0-9]+" | sort -u | wc -l | tr -d ' ')
    [ "${n:-0}" -ge 3 ] && break
    [ "$i" = 120 ] && { echo "BTK-FAIL pf T8 never reached 3 cells"; exit 1; }
    sleep 30
done
echo "pf T8 lane complete; starting btk gap cells"

for spec in "6 2" "10 1" "10 2"; do
    set -- $spec
    T=$1; S=$2
    echo "--- BTK CELL T=$T seed=$S start $(date -u +%FT%TZ) ---"
    .venv/bin/python - "$T" "$S" <<'PY'
import sys, time
from experiments.explorations.actmix_rlhf.cells import txc
from temp_bench.core.runner import run_experiment
T, seed = int(sys.argv[1]), int(sys.argv[2])
c = txc(T)                      # btk recipe: n_steps 25000, warmup 1000
t0 = time.time()
res = run_experiment(
    experiment="rlhf", arch_name=c["arch"], seed=seed,
    datasource_name=c["datasource"], training_cfg=c["training_cfg"],
    eval_cfg=c.get("eval_cfg", {}), agent="mac-d", allow_dirty=True,
)
print(f"BTK-CELL-OK T={T} s={seed} wall={time.time()-t0:.0f}s")
PY
    rc=$?
    [ $rc -eq 0 ] || echo "BTK-CELL-FAIL T=$T s=$S rc=$rc"
done
echo "BTK-GAP-DONE $(date -u +%FT%TZ)"
