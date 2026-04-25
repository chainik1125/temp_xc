#!/usr/bin/env bash
# Run one Phase 6.2 autoresearch cycle end-to-end.
#
# Usage:
#   bash experiments/phase6_2_autoresearch/run_phase62_cycle.sh <CANDIDATE_ID>
#
# Example:
#   bash experiments/phase6_2_autoresearch/run_phase62_cycle.sh C5
#
# Pulls the candidate's dispatch + min_steps settings from
# candidates.py, then:
#   1. Trains the candidate (if no matching seed=42 ckpt exists).
#   2. Encodes concat_A/B/random at seed=42.
#   3. Runs upgraded autointerp across A/B/random.
#   4. Appends a results row to results/phase62_results.jsonl.
#
# NOTE: Currently only C5, C6 are implementation-ready (reuse
# existing dispatcher branches + a min_steps override). C1-C4
# require new arch classes; see candidates.py implementation_note.

set -euo pipefail
cd /workspace/temp_xc
source /workspace/temp_xc/.envrc

CAND_ID="${1:?usage: run_phase62_cycle.sh <CANDIDATE_ID>}"
mkdir -p logs experiments/phase6_2_autoresearch/results

# Resolve candidate
python_readback=$(.venv/bin/python - <<EOF
import sys
sys.path.insert(0, 'experiments/phase6_2_autoresearch')
from candidates import by_id
c = by_id("${CAND_ID}")
import shlex
print(f"DISPATCH={shlex.quote(c.dispatch)}")
print(f"NAME={shlex.quote(c.name)}")
print(f"AXIS={shlex.quote(c.axis_tested)}")
print(f"COST_MIN={c.cost_min}")
print(f"IMPLEMENTED={int(c.implemented)}")
EOF
)
eval "$python_readback"

if [ "$IMPLEMENTED" != "1" ]; then
  echo "ERROR: candidate $CAND_ID not yet implemented in dispatcher." >&2
  echo "See experiments/phase6_2_autoresearch/candidates.py for the TODO note." >&2
  exit 2
fi

echo "============================================================"
echo "Phase 6.2 cycle: $CAND_ID  ($NAME)"
echo "dispatch: $DISPATCH  axis: $AXIS"
echo "============================================================"

# Custom min_steps override for C5 and C6
MIN_STEPS=""
case "$CAND_ID" in
  C5|C6) MIN_STEPS="--min-steps 10000" ;;
esac

CKPT="experiments/phase5_downstream_utility/results/ckpts/${NAME}__seed42.pt"

# Step 1: train if no ckpt (we save candidate ckpts under the
# candidate name, not the dispatch name, so multiple variants with
# the same dispatcher+different cfg don't collide).
if [ -f "$CKPT" ]; then
  echo "[skip train] $CKPT already exists"
else
  echo "--- train ---"
  # Override ckpt path via env var. train_primary_archs.py respects
  # CKPT_DIR and uses run_id="${arch}__seed{seed}.pt" — we need to
  # emit a candidate-specific run_id. Accomplished via a post-train
  # rename once the default ckpt lands.
  DEFAULT_CKPT="experiments/phase5_downstream_utility/results/ckpts/${DISPATCH}__seed42.pt"
  [ -f "$DEFAULT_CKPT" ] && mv "$DEFAULT_CKPT" "${DEFAULT_CKPT}.preserved_for_phase62"

  TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
    experiments/phase5_downstream_utility/train_primary_archs.py \
    --archs "$DISPATCH" --seeds 42 --max-steps 25000 $MIN_STEPS \
    2>&1 | tee "logs/phase62_${CAND_ID}_train.log"

  # Rename freshly-trained ckpt to the candidate name (no-op if same path)
  if [ "$DEFAULT_CKPT" != "$CKPT" ]; then
    mv "$DEFAULT_CKPT" "$CKPT"
  fi
  # Restore pre-existing dispatch ckpt if any
  if [ "$DEFAULT_CKPT" != "$CKPT" ] \
     && [ -f "${DEFAULT_CKPT}.preserved_for_phase62" ]; then
    mv "${DEFAULT_CKPT}.preserved_for_phase62" "$DEFAULT_CKPT"
  fi
fi

# Step 2: encode on A/B/random — uses the candidate-named ckpt.
# encode_archs.py reads {arch}__seed42.pt, so we pass --archs
# NAME instead of DISPATCH. Requires NAME to be a known arch in
# encode_archs.load_arch — for C5/C6 reuse we alias via symlink:
# ln -sf agentic_txc_10_bare__seed42.pt phase62_c5_track2_longer__seed42.pt
# but also need load_arch to handle the name. Simpler: encode
# under the dispatch name instead.

# Actually simpler path: encode via a temporary alias ckpt at the
# dispatch name, then move back. (Matches what run_cycle_eval.sh does.)
# Skip the alias swap entirely if DISPATCH == NAME (no rename needed).
ALIAS="experiments/phase5_downstream_utility/results/ckpts/${DISPATCH}__seed42.pt"
ORIG_ALIAS="${ALIAS}.original"
ALIAS_SWAPPED=0
if [ "$DISPATCH" != "$NAME" ]; then
  if [ -f "$ALIAS" ]; then mv "$ALIAS" "$ORIG_ALIAS"; fi
  cp "$CKPT" "$ALIAS"
  ALIAS_SWAPPED=1
fi

echo "--- encode ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/encode_archs.py \
  --archs "$DISPATCH" --sets A B random --seed 42 \
  2>&1 | tee "logs/phase62_${CAND_ID}_encode.log"

echo "--- autointerp ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/run_autointerp.py \
  --archs "$DISPATCH" --seeds 42 --concats A B random \
  2>&1 | tee "logs/phase62_${CAND_ID}_autointerp.log"

# Restore original ckpt alias (skip if we never swapped)
if [ "$ALIAS_SWAPPED" = "1" ]; then
  rm -f "$ALIAS"
  if [ -f "$ORIG_ALIAS" ]; then mv "$ORIG_ALIAS" "$ALIAS"; fi
fi

# Step 3: collect metrics + append to jsonl
.venv/bin/python - <<EOF
import json, pathlib
out = pathlib.Path("experiments/phase6_2_autoresearch/results/phase62_results.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

metrics = {}
for concat in ("A", "B", "random"):
    p = pathlib.Path(
        f"experiments/phase6_qualitative_latents/results/autointerp/"
        f"${DISPATCH}__seed42__concat{concat}__labels.json"
    )
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    metrics[concat] = d["metrics"]

row = {
    "candidate_id": "$CAND_ID",
    "candidate_name": "$NAME",
    "dispatch": "$DISPATCH",
    "seed": 42,
    "metrics": metrics,
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")

# Pretty-print summary
for concat in ("A", "B", "random"):
    m = metrics.get(concat)
    if m:
        print(f"  concat_{concat}: {m['semantic_count']}/{m['N']} sem, "
              f"{m['passage_coverage_count']}/{m['n_passages']} cov, "
              f"{m['judge_disagreement_rate']:.2f} disagree")
print(f"row written to {out}")
EOF

echo "=== Phase 6.2 cycle $CAND_ID DONE: $(date -u) ==="
