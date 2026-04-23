#!/usr/bin/env bash
# Autoresearch orchestrator for venhoff_scrappy.
#
# Pattern adopted from experiments/phase5_downstream_utility/run_autoresearch.sh
# (Han's Phase 5.7 setup). For each candidate passed in $@:
#   1. Run Phase 0→3 pipeline with scrappy budget + candidate overrides
#      (via run_cycle.py).
#   2. Compute Gap Recovery from the hybrid results.
#   3. Compute Δ vs the candidate's declared baseline
#      (via autoresearch_summarise.py).
#   4. Append a row to results/autoresearch_index.jsonl.
#   5. Commit + push (per-candidate milestone).
#
# Serializes GPU work — pass multiple candidates sequentially in $@.
# Baseline (baseline_sae) is auto-run if its Gap Recovery is missing.

set -u

# Resolve repo root relative to this script so it runs from pod or laptop.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO"

LOG_DIR="$SCRIPT_DIR/results/logs"
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/orchestrator_$(date +%Y%m%d_%H%M%S).log"

say() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"
}

GIT_USER_NAME="${GIT_USER_NAME:-aniket}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-aniketdeshh@gmail.com}"
GIT_BRANCH="${GIT_BRANCH:-aniket}"

commit_and_push() {
    local msg="$1"; shift
    local files=("$@")
    git -c user.name="$GIT_USER_NAME" -c user.email="$GIT_USER_EMAIL" \
        add -- "${files[@]}" 2>&1 | tee -a "$MAIN" || true
    if git diff --cached --quiet; then
        say "  (no changes to commit for: $msg)"
        return 0
    fi
    git -c user.name="$GIT_USER_NAME" -c user.email="$GIT_USER_EMAIL" \
        commit -m "$msg

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" 2>&1 | tee -a "$MAIN"
    git push origin "$GIT_BRANCH" 2>&1 | tee -a "$MAIN" || say "  push FAILED; continuing"
}

# Per-cycle timeout (kills stuck vLLM / nnsight processes instead of
# burning the whole overnight budget). Default 30 min; can override
# via env or config.yaml cycle_timeout_s.
CYCLE_TIMEOUT_S="${CYCLE_TIMEOUT_S:-1800}"

# GPU orphan cleanup — invoked on cycle timeout. Patterns learned the
# hard way during the 2026-04-22 paper-budget run: pkill against
# "run_cycle" misses the actual vendor python subprocess; need to
# grep ps for the vendor venv python explicitly.
gpu_orphan_cleanup() {
    local pids
    pids=$(ps -ef | grep 'vendor/thinking-llms-interp/.venv' | grep -v grep | awk '{print $2}')
    if [[ -n "$pids" ]]; then
        say "  [cleanup] killing orphan vendor-venv pids: $pids"
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
    fi
    # Same for our main-venv subprocess (in case run_cycle.py itself hung).
    local main_pids
    main_pids=$(ps -ef | grep 'spar-temporal-crosscoders/.venv/bin/python' | grep -v grep | grep -v "$$" | awk '{print $2}')
    if [[ -n "$main_pids" ]]; then
        say "  [cleanup] killing orphan main-venv pids: $main_pids"
        echo "$main_pids" | xargs -r kill -9 2>/dev/null || true
    fi
    # Wait for driver to reclaim VRAM (async; ~30-60s).
    say "  [cleanup] waiting 60s for CUDA driver to reclaim VRAM"
    sleep 60
}

write_failed_ledger_row() {
    local cand="$1"
    local reason="$2"
    .venv/bin/python - "$cand" "$reason" <<'PY'
import json, sys, time
from pathlib import Path
REPO = Path("/workspace/spar-temporal-crosscoders")
LEDGER = REPO / "experiments/venhoff_scrappy/results/autoresearch_index.jsonl"
LEDGER.parent.mkdir(parents=True, exist_ok=True)
cand, reason = sys.argv[1], sys.argv[2]
row = {
    "schema_version": 1,
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "candidate": cand,
    "verdict": "FAILED",
    "failure_reason": reason,
}
with LEDGER.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"[info] wrote FAILED row for {cand} to {LEDGER}")
PY
}

run_one_candidate() {
    local cand="$1"
    local cfg="$SCRIPT_DIR/candidates/${cand}.yaml"
    if [[ ! -f "$cfg" ]]; then
        say "  [error] candidate config not found: $cfg"
        return 1
    fi

    local result_dir="$SCRIPT_DIR/results/cycles/${cand}"
    mkdir -p "$result_dir"
    local cycle_log="$result_dir/cycle.log"

    say "=== cycle: $cand (timeout=${CYCLE_TIMEOUT_S}s) ==="

    # Step 1-2: run Phase 0→3 pipeline with the candidate's config.
    # Wrapped in `timeout` — on exceed, SIGTERM then SIGKILL after 30 s.
    # run_cycle.py handles all the vendor-pipeline wiring and writes
    # grade_results.json to the result_dir.
    timeout --kill-after=30 "${CYCLE_TIMEOUT_S}" \
        .venv/bin/python "$SCRIPT_DIR/run_cycle.py" \
            --candidate "$cand" \
            --config "$cfg" \
            --result-dir "$result_dir" \
            2>&1 | tee -a "$cycle_log"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        say "  [timeout] $cand exceeded ${CYCLE_TIMEOUT_S}s — cleaning orphans"
        gpu_orphan_cleanup
        write_failed_ledger_row "$cand" "cycle_timeout_${CYCLE_TIMEOUT_S}s"
        commit_and_push "venhoff_scrappy cycle FAILED (timeout): $cand" \
            "$SCRIPT_DIR/results/autoresearch_index.jsonl" \
            "$result_dir"
        return $rc
    fi
    if [[ $rc -ne 0 ]]; then
        say "  [error] run_cycle.py failed for $cand (rc=$rc) — see $cycle_log"
        write_failed_ledger_row "$cand" "run_cycle_rc_${rc}"
        commit_and_push "venhoff_scrappy cycle FAILED (rc=$rc): $cand" \
            "$SCRIPT_DIR/results/autoresearch_index.jsonl" \
            "$result_dir"
        return $rc
    fi

    # Step 3-4: compute Δ vs baseline, append ledger row.
    .venv/bin/python "$SCRIPT_DIR/autoresearch_summarise.py" \
        --candidate "$cand" \
        --write-index \
        2>&1 | tee -a "$cycle_log"
    local rc2=${PIPESTATUS[0]}
    if [[ $rc2 -ne 0 ]]; then
        say "  [error] autoresearch_summarise.py failed for $cand (rc=$rc2)"
        return $rc2
    fi

    # Step 5: commit + push the new result + ledger row.
    commit_and_push \
        "venhoff_scrappy cycle: $cand (Δ vs baseline)" \
        "$SCRIPT_DIR/candidates/${cand}.yaml" \
        "$SCRIPT_DIR/results/autoresearch_index.jsonl" \
        "$result_dir"
}

# Vendor-patch smoke test — per §11.1 of the plan, the vendor tree MUST
# have all 8 transformers-modernisation patches applied before any
# cycle runs. A dry-run here catches upstream drift before we waste
# 10 min on a doomed cycle.
verify_vendor_patches() {
    say "=== verifying vendor patches ==="
    .venv/bin/python - <<'PY' || { echo "[fatal] vendor patches failed verification"; exit 3; }
from pathlib import Path
from src.bench.venhoff.vendor_patches import (
    ensure_hybrid_judge_patched, ensure_steering_patched,
)
root = Path("vendor/thinking-llms-interp")
ensure_hybrid_judge_patched(root)
ensure_steering_patched(root)
print("[ok] all vendor patches applied/confirmed")
PY
}

main() {
    if [[ $# -eq 0 ]]; then
        say "usage: $0 <candidate_name> [<candidate_name> ...]"
        say "  (candidate configs live in candidates/<name>.yaml)"
        exit 1
    fi

    # Pre-flight: confirm vendor patches are in the state we expect.
    # Fails fast if upstream drifted or our patch strings went stale.
    verify_vendor_patches

    # Auto-run baseline if missing (so later candidates have a reference).
    local baseline="baseline_sae"
    local baseline_grade="$SCRIPT_DIR/results/cycles/${baseline}/grade_results.json"
    if [[ ! -f "$baseline_grade" ]]; then
        say "baseline $baseline not cached — running it first"
        run_one_candidate "$baseline" || { say "baseline failed; aborting"; exit 2; }
    fi

    for cand in "$@"; do
        if [[ "$cand" == "$baseline" ]] && [[ -f "$baseline_grade" ]]; then
            say "skipping $cand (already cached)"
            continue
        fi
        run_one_candidate "$cand" || say "  continuing after failure of $cand"
    done

    say "=== autoresearch run complete ==="
    say "ledger: $SCRIPT_DIR/results/autoresearch_index.jsonl"
}

main "$@"
