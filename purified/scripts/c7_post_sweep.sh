#!/usr/bin/env bash
# Post-sweep: render AUTO-RESULTS into docs/components/c7.md and push.
#
# Usage:  bash scripts/c7_post_sweep.sh
#
# Idempotent: re-running re-renders from the current leaderboard +
# judge_outputs.jsonl state. Useful after each cell completes if you
# want progressive updates.

set -eu
cd "$(dirname "$0")/.."

GH_TOKEN_FILE="/workspace/.tokens/gh_token"

echo "[c7-post] rendering c7.md AUTO-RESULTS"
TQDM_DISABLE=1 .venv/bin/python -c "
from temp_bench import report
res = report.render(component='c7')
print('rendered:', len(res.markdown), 'chars; results.json keys:', list(res.results.keys()))
"

echo "[c7-post] git commit + push"
git add docs/components/c7.md \
    experiments/c7_backtracking/plots/ \
    experiments/c7_backtracking/results.json \
    results/leaderboard.jsonl \
    results/runs/ \
    checkpoints/manifest.jsonl \
    checkpoints/*/config.json 2>/dev/null || true

if git diff --cached --quiet; then
    echo "[c7-post] no changes to commit"
    exit 0
fi

git -c user.name=agent_back -c user.email=agent_back@temp-bench.local commit -m \
    "Agent BACK: c7 sweep checkpoint — render AUTO-RESULTS

Auto-rendered docs/components/c7.md from leaderboard.jsonl +
judge_outputs.jsonl via temp_bench.report.render(component='c7').

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

git pull --rebase origin final 2>&1 | tail -2 || true
git push origin final 2>&1 | tail -3
