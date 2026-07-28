#!/bin/bash
# RLHF paper-faithful GRID lanes (Han-urgent directive 4e04ae0e3 item 2).
# Authored by runpod-a on pod A; DE-HARDCODED by mac-d 13:14 07-28 when
# the hub designated it fleet property (31930ad8c) and the wave-1 map
# moved it to pod B + old-pod under other agents' hands.
#
#   AGENT_NAME=<your-id> [GPU=<n>] [REPO=<path>] \
#       ./run_pf_grid_lanes.sh <PIN> <lane> [lane ...]
#
# Launches one concurrent process per lane on ONE GPU (default 0, set
# GPU=n), each with explicit thread + GPU-memory caps. Threads are
# divided across the lanes from the REAL container quota (cgroup
# cpu.max, NOT nproc — see LOG 12:52/13:00).
#
# Guards the two traps runpod-b found in run_cells.py (LOG 0be31500b):
#   trap 1  AGENT_NAME setdefault "runpod-2"  -> exported explicitly here
#   trap 2  no thread bound, torch autosizes  -> OMP/MKL exported per lane
# Both are set inline; runpod-2's driver is NOT edited.
#
# trap 3 (mac-d): AGENT_NAME, GPU and REPO were hardcoded to runpod-a /
# GPU 0 / runpod-a's checkout. Used verbatim elsewhere that stamps every
# row `runpod-a` and stacks every lane on GPU 0 — so AGENT_NAME is now
# REQUIRED (fail-fast beats silent misattribution) and GPU/REPO are
# overridable. REPO defaults to this pod's checkout for AGENT_NAME.
set -euo pipefail

: "${AGENT_NAME:?refusing to launch: export AGENT_NAME=<your agent id>. It stamps every leaderboard row; a wrong value is silent provenance corruption.}"
GPU="${GPU:-0}"
REPO="${REPO:-/workspace/agents/${AGENT_NAME}/temp_xc}"
PY="$REPO/.venv/bin/python"
LOGDIR=/workspace/logs
RESERVE_CORES=3          # left for the two agent processes + system

[ -x "$PY" ] || { echo "FATAL: no venv python at $PY — set REPO=<your checkout>"; exit 2; }
[ $# -ge 2 ] || { echo "usage: AGENT_NAME=<id> [GPU=n] $0 <PIN> <lane> [lane ...]"; exit 2; }
PIN="$1"; shift
LANES=("$@")
N=${#LANES[@]}

cd "$REPO"

# ── pin discipline ────────────────────────────────────────────────────
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo "FATAL: HEAD != PIN $PIN"; exit 1; }
git merge-base --is-ancestor "$PIN" origin/arxiv || { echo "FATAL: PIN not on origin/arxiv"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "FATAL: tree dirty"; exit 1; }

# ── thread budget from the cgroup quota, not the host ─────────────────
read -r Q P < /sys/fs/cgroup/cpu.max
if [ "$Q" = "max" ]; then
    echo "FATAL: no cpu quota found; refusing to guess a thread budget"; exit 1
fi
QUOTA=$(( Q / P ))
BUDGET=$(( QUOTA - RESERVE_CORES ))
THREADS=$(( BUDGET / N ))
[ "$THREADS" -ge 1 ] || { echo "FATAL: $N lanes do not fit in $BUDGET cores"; exit 1; }

# ── GPU memory: sized PER LANE from runpod-2's measured table ─────────
# (LOG 624528e85: peak GiB by T — 1:3.6  2:5.2  8:27.7  10:39.0  16:>=72)
# An EVEN split is wrong: per-cell peak spans 3.6->39 GiB, so a uniform
# fraction either OOMs the big-T lanes or strands memory on the small.
# Each lane is sized by the LARGEST T it contains, plus headroom.
GPU_TOTAL_GIB=79.19          # usable on this H100 (runpod-2's measured figure)
HEADROOM=1.25                # margin over measured peak

read -r -d '' PYMEM <<'PYEOF' || true
import sys, json
from experiments.explorations.actmix_rlhf.cells import LANES
# measured peak GiB at the probed T; interpolated between for unmeasured T
MEAS = {1: 3.6, 2: 5.2, 8: 27.7, 10: 39.0, 16: 72.0}
def peak(T):
    if T in MEAS: return MEAS[T]
    lo = max([t for t in MEAS if t < T], default=min(MEAS))
    hi = min([t for t in MEAS if t > T], default=max(MEAS))
    if lo == hi: return MEAS[lo]
    f = (T - lo) / (hi - lo)
    return MEAS[lo] + f * (MEAS[hi] - MEAS[lo])
out = {}
for lane in sys.argv[1:]:
    Ts = [c["training_cfg"].arch_hparams_override.get("T", 1) for c in LANES[lane]()]
    out[lane] = {"max_T": max(Ts), "peak_gib": round(max(peak(t) for t in Ts), 1)}
print(json.dumps(out))
PYEOF
MEMJSON=$("$PY" -c "$PYMEM" "${LANES[@]}")
echo "=== pf grid: $N lane(s) on GPU 0 — $(date -u '+%F %T UTC') ==="
echo "    pin      $PIN"
echo "    quota    ${QUOTA} cores (cgroup), reserve ${RESERVE_CORES}"
echo "    threads  OMP=MKL=$THREADS per lane"
echo "    memory   $MEMJSON"

# refuse to launch a packing that cannot fit — OOM mid-grid is worse than not starting
"$PY" - "$MEMJSON" "$GPU_TOTAL_GIB" "$HEADROOM" <<'PYEOF'
import json, sys
mem, total, head = json.loads(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
need = sum(v["peak_gib"] for v in mem.values()) * head
print(f"    packing  {need:.1f} GiB needed (incl {head}x headroom) of {total} GiB")
if need > total:
    print(f"FATAL: co-resident lanes need {need:.1f} GiB > {total} GiB. "
          f"Split across GPUs or drop a lane (T16 needs a single-tenant card).")
    sys.exit(1)
PYEOF

mkdir -p "$LOGDIR"
pids=()
for lane in "${LANES[@]}"; do
    out="$LOGDIR/pf_grid_${lane}.log"
    FRAC=$("$PY" -c "
import json,sys
m=json.loads(sys.argv[1])[sys.argv[2]]
print(f'{min(0.95, m[\"peak_gib\"]*float(sys.argv[3])/float(sys.argv[4])):.3f}')
" "$MEMJSON" "$lane" "$HEADROOM" "$GPU_TOTAL_GIB")
    echo "--- launching lane $lane (GPU_FRACTION=$FRAC) -> $out"
    env CUDA_VISIBLE_DEVICES="$GPU" \
        AGENT_NAME="$AGENT_NAME" \
        HF_HOME=/workspace/hf_cache \
        OMP_NUM_THREADS="$THREADS" \
        MKL_NUM_THREADS="$THREADS" \
        TEMP_BENCH_GPU_FRACTION="$FRAC" \
        TEMP_BENCH_ALLOW_DIRTY=1 \
        "$PY" -m experiments.explorations.actmix_rlhf.run_cells \
        --lane "$lane" --pin "$PIN" > "$out" 2>&1 &
    pids+=($!)
done

echo "=== ${#pids[@]} lane(s) running: ${pids[*]} ==="
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "=== pf grid lanes COMPLETE (rc=$rc) — $(date -u '+%F %T UTC') ==="
exit $rc
