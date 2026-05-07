#!/usr/bin/env bash
# Post-pipeline render + commit + push helper for C6.
#
# Run after `experiments.c6_em.run` completes both cells:
#
#   bash scripts/c6_render_and_push.sh
#
# Steps:
#   1. report.render(component="c6") populates the AUTO-RESULTS block of
#      docs/components/c6.md from the leaderboard.
#   2. Stage docs/components/c6.md, results/leaderboard.jsonl,
#      checkpoints/manifest.jsonl, experiments/c6_em/results.json,
#      results/runs/c6_*/.
#   3. Commit + push.
#
# Idempotent: re-running just re-renders + commits if anything changed.
set -e
cd "$(git rev-parse --show-toplevel)/purified"

echo "[c6.render] running report.render(c6)..."
TQDM_DISABLE=1 .venv/bin/python -c "
from temp_bench import report
result = report.render(component='c6')
import json
print('peak_align(sae_arditi):', result.results.get('sae_arditi_peak', {}).get('peak_align'))
print('peak_align(txc_base):',   result.results.get('txc_base_peak', {}).get('peak_align'))
print('gap:', result.results.get('gap'))
"

echo
echo "[c6.render] git status:"
git status --short docs/components/c6.md results/leaderboard.jsonl \
                   checkpoints/manifest.jsonl experiments/c6_em/results.json \
                   results/runs/

if git diff --quiet docs/components/c6.md \
                    results/leaderboard.jsonl \
                    checkpoints/manifest.jsonl \
                    experiments/c6_em/results.json; then
  echo "[c6.render] no changes to commit"
  exit 0
fi

git add docs/components/c6.md \
        results/leaderboard.jsonl \
        checkpoints/manifest.jsonl \
        experiments/c6_em/results.json
git commit -m "Agent EM: C6 results — abbreviated Wang gap-close test landed

Auto-rendered docs/components/c6.md AUTO-RESULTS block from
the latest C6 leaderboard rows via experiments.c6_em.analysis.

See briefing for caveats (judge swap, Wang abbreviation, corpus
stand-in, hparam mismatch). Relative gap is the headline; absolute
numbers won't match the prior author's published 95.16/91.25.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

GH_TOKEN=$(cat /workspace/.tokens/gh_token)
git fetch origin
git rebase origin/final
git push "https://${GH_TOKEN}@github.com/anonymous-temp-bench.git" final
echo "[c6.render] pushed."
