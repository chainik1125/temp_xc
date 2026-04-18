#!/bin/bash
# runpod_saebench_preflight.sh — quick end-to-end validation before the
# full overnight sweep. Trains one SAE at protocol A for 500 steps, runs
# probing on one task (ag_news), verifies the JSONL schema has the
# expected fields. ~30 minutes of compute. Saves a 14-hour run on a
# silent schema mismatch.
#
# Exits non-zero on any step failure so the orchestrator can block the
# full sweep if pre-flight tripped.
#
# Usage:
#   bash scripts/runpod_saebench_preflight.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

PREFLIGHT_DIR="results/saebench/preflight"
mkdir -p "$PREFLIGHT_DIR"

echo "=== saebench pre-flight ==="
echo "  goal: validate train + probing + JSONL schema end-to-end"
echo "  checkpoint: SAE protocol A, 500 training steps"
echo "  probing:    ag_news only, aggregation=last, k=5"
echo ""

# ─── 1. quick SAE training (500 steps, no full sweep) ─────────────────
echo ">> [1/3] quick SAE train (500 steps)"
STEPS=500 bash scripts/runpod_saebench_train.sh --arch sae --protocol A

# ─── 2. probing eval on one task ──────────────────────────────────────
echo ""
echo ">> [2/3] probing eval (ag_news, last aggregation, k=5)"
python - <<PY
from src.bench.saebench.probing_runner import run_probing
from src.bench.saebench.configs import CKPT_DIR, ckpt_name

ckpt = f"{CKPT_DIR}/{ckpt_name('sae', 'A')}"
summary = run_probing(
    arch="sae",
    ckpt_path=ckpt,
    protocol="A",
    t=5,
    aggregation="last",
    output_jsonl="results/saebench/preflight/preflight.jsonl",
    k=0,  # swept internally
    k_values=(5,),  # single k for preflight
    device="cuda:0",
    random_seed=42,
)
print(f"  wrote {summary['n_records_written']} records in {summary['elapsed_sec']:.1f}s")
PY

# ─── 3. verify JSONL schema ───────────────────────────────────────────
echo ""
echo ">> [3/4] verify JSONL schema"
python - <<'PY'
import json
import sys

path = "results/saebench/preflight/preflight.jsonl"
records = [json.loads(line) for line in open(path)]
if not records:
    print("FAIL: empty JSONL — probing produced zero records")
    sys.exit(1)

required_fields = {
    "architecture", "t", "matching_protocol", "aggregation",
    "task", "k", "accuracy", "param_count", "checkpoint_path",
}
r0 = records[0]
missing = required_fields - set(r0.keys())
if missing:
    print(f"FAIL: JSONL record missing required fields: {missing}")
    print(f"  first record: {r0}")
    sys.exit(2)

if not (0.0 <= r0["accuracy"] <= 1.0):
    print(f"FAIL: accuracy out of range: {r0['accuracy']}")
    sys.exit(3)

print(f"  OK — {len(records)} records, schema valid")
print(f"  sample: {r0}")
PY

# ─── 4. Clean up preflight-trained ckpt so the real sweep trains fresh ─
# The preflight SAE shares its ckpt path with the orchestrator's real
# SAE × protA × T=5 cell. Without this cleanup, runpod_saebench_train.sh's
# skip-if-exists guard would silently pick up the 500-step preflight ckpt
# as if it were a full 5000-step run.
echo ""
echo ">> [4/4] cleanup preflight-polluted ckpt (500-step; real sweep needs full steps)"
PREFLIGHT_CKPT=$(python - <<'PY'
from src.bench.saebench.configs import CKPT_DIR, ckpt_name
print(f"{CKPT_DIR}/{ckpt_name('sae', 'A')}")
PY
)
rm -f "$PREFLIGHT_CKPT"
rm -rf results/saebench/sweeps/sae_protA_T5
echo "   removed $PREFLIGHT_CKPT"
echo "   removed results/saebench/sweeps/sae_protA_T5/"

echo ""
echo "=== pre-flight PASSED ==="
echo "  safe to launch: bash scripts/runpod_saebench_launch.sh"
