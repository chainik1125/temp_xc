#!/usr/bin/env bash
# wrap_up_session.sh — pre-pod-stop / end-of-session checklist.
#
# Run this BEFORE you stop or destroy a pod, or before you mark your
# briefing as `status: complete`. It:
#   1. git-adds every results/runs/*/metrics.json,
#      results/runs/*/judge_outputs.jsonl, and
#      results/runs/*/phase1_unsteered.json (cell artifacts that
#      individual workers don't always remember to commit).
#   2. Stages results/leaderboard.jsonl + checkpoints/manifest.jsonl
#      in case append flock-guarded writes left the working tree
#      with un-committed appends.
#   3. Commits with a uniform "wrap-up" message.
#   4. Pulls + pushes to origin/final (rebase-safe — no force).
#   5. For ephemeral pods: reminds the agent to confirm HF-side
#      checkpoint state via `hf api`.
#   6. For persistent pods: prints the manual `hf upload` recipe for
#      every train_key in the local manifest — these pods do NOT
#      auto-push, and a pod stop without manual upload destroys
#      hours of training.
#   7. Prints a verdict summarising what's persisted off-pod and
#      what's still at risk.
#
# Idempotent. Safe to run multiple times.
#
# Usage:
#     bash scripts/wrap_up_session.sh

set -eu
cd "$(dirname "$0")/.."

AGENT="${AGENT_NAME:-unknown}"
MODE="${TEMP_BENCH_POD_MODE:-unknown}"

echo
echo "═══════════════════════════════════════════════════════════════"
echo "  wrap-up for AGENT_NAME=$AGENT  (pod mode: $MODE)"
echo "═══════════════════════════════════════════════════════════════"
echo

# 1. Stage cell artifacts that agents don't always remember to commit.
new_files=0
for pattern in 'metrics.json' 'judge_outputs.jsonl' 'phase1_unsteered.json'; do
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
            git add "$f"
            new_files=$((new_files + 1))
        fi
    done < <(find results/runs/ -name "$pattern" 2>/dev/null || true)
done

# Stage the append-only state files in case flocked appends are uncommitted.
git add results/leaderboard.jsonl checkpoints/manifest.jsonl 2>/dev/null || true

echo "[1/5] Staged $new_files new run-dir artifact(s)"

# 2. Stage the experiments/cN_*/results.json files — these are written
# by report.render() and reflect the latest aggregate per component.
for f in experiments/c*/results.json; do
    [ -f "$f" ] || continue
    git add "$f" 2>/dev/null || true
done

# 3. Commit if anything's staged.
if git diff --cached --quiet; then
    echo "[2/5] No staged changes — nothing to commit"
else
    msg="Agent ${AGENT}: wrap_up_session.sh — persist cell artifacts before session end"
    git commit -m "$msg" >/dev/null
    echo "[2/5] Committed: $(git log -1 --oneline)"
fi

# 4. Pull-rebase + push.
echo "[3/5] Pull-rebase…"
git pull --rebase origin final 2>&1 | tail -3 || {
    echo "  ✗ Pull-rebase failed. Resolve conflicts and re-run." >&2
    exit 1
}

echo "[4/5] Push…"
git push origin final 2>&1 | tail -3

# 5. Final verdict — different recipe per pod mode.
echo
echo "═══════════════════════════════════════════════════════════════"
echo "  Verdict"
echo "═══════════════════════════════════════════════════════════════"

# Count what's locally available
n_runs=$(find results/runs/ -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
n_with_judge=$(find results/runs/ -name 'judge_outputs.jsonl' 2>/dev/null | wc -l)
n_ckpts=$(wc -l < checkpoints/manifest.jsonl 2>/dev/null || echo 0)

echo "  Local run-dirs:                 $n_runs"
echo "  Run-dirs with judge_outputs:    $n_with_judge"
echo "  Checkpoint manifest entries:    $n_ckpts"
echo

case "$MODE" in
    ephemeral)
        echo "  Pod mode: EPHEMERAL"
        echo "  → Checkpoints AUTO-PUSH on save (cache.save_checkpoint)"
        echo "  → Run-dir artifacts: just committed to git this run"
        echo
        echo "  Pre-stop check (recommended):"
        echo "    .venv/bin/python -c \"from huggingface_hub import HfApi; "
        echo "      api = HfApi(); "
        echo "      import json; "
        echo "      seen = {f.path for f in api.list_repo_tree('han1823123123/temp-bench-models')}; "
        echo "      manifest = [json.loads(l) for l in open('checkpoints/manifest.jsonl')]; "
        echo "      missing = [r['train_key'] for r in manifest if r['train_key'] not in seen]; "
        echo "      print('Missing on HF:', missing or 'none')\""
        echo
        echo "  ✓ POD CAN BE STOPPED if (a) the missing-on-HF check above is empty"
        echo "    and (b) git status is clean."
        ;;
    persistent)
        echo "  Pod mode: PERSISTENT"
        echo "  → Checkpoints DO NOT auto-push (only ephemeral pods auto-push)"
        echo "  → Trained .safetensors live ONLY on this pod's /workspace"
        echo "  → A pod stop (or volume detach) WIPES un-pushed checkpoints"
        echo
        echo "  Manual-push recipe for every train_key in your manifest:"
        echo
        echo "    .venv/bin/python <<'PY'"
        echo "    import json"
        echo "    from huggingface_hub import HfApi"
        echo "    from temp_bench.config import checkpoint_dir"
        echo "    api = HfApi()"
        echo "    for line in open('checkpoints/manifest.jsonl'):"
        echo "        r = json.loads(line)"
        echo "        if r.get('hf_url'): continue"
        echo "        d = checkpoint_dir(r['train_key'])"
        echo "        if not d.exists(): continue"
        echo "        api.upload_folder("
        echo "            folder_path=str(d), path_in_repo=r['train_key'],"
        echo "            repo_id='han1823123123/temp-bench-models', repo_type='model',"
        echo "        )"
        echo "        print('pushed', r['train_key'])"
        echo "    PY"
        echo
        echo "  ⚠  POD MUST NOT BE STOPPED until the loop above prints 'pushed'"
        echo "     for every train_key (or hf_url is already set in manifest)."
        ;;
    *)
        echo "  Pod mode: UNKNOWN ($MODE)"
        echo "  → Source scripts/set_agent_env.sh <agent_name> first?"
        ;;
esac

echo
echo "[5/5] Wrap-up complete."
echo
