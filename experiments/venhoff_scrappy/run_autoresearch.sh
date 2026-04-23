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

    say "=== cycle: $cand ==="

    # Step 1-2: run Phase 0→3 pipeline with the candidate's config.
    # run_cycle.py handles all the vendor-pipeline wiring and writes
    # grade_results.json to the result_dir.
    .venv/bin/python "$SCRIPT_DIR/run_cycle.py" \
        --candidate "$cand" \
        --config "$cfg" \
        --result-dir "$result_dir" \
        2>&1 | tee -a "$cycle_log"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        say "  [error] run_cycle.py failed for $cand (rc=$rc) — see $cycle_log"
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

main() {
    if [[ $# -eq 0 ]]; then
        say "usage: $0 <candidate_name> [<candidate_name> ...]"
        say "  (candidate configs live in candidates/<name>.yaml)"
        exit 1
    fi

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
