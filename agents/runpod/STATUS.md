# Working state — agent `runpod`

**Last rewrite:** 2026-07-23, after completing `briefings/expansion-c7.md`.
**State: C7 DONE — STOPPED FOR REVIEW** (briefing's acceptance gate; the
briefing stays until mac-local review deletes it). No active task.

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am `runpod`
(original box — `/workspace/.agent_id` does NOT exist; do not create it).**
Parallel agents: `runpod-b` (freqbench-fb5), `runpod-c` (conversion-depth,
GPU) — their briefings are NOT mine. Shared-branch rules (agents/README.md):
`git pull --rebase origin arxiv` before EVERY push (commit this STATUS
first); shared files append-only; leaderboard/manifest union-merge; **cite
commits by SUBJECT LINE or re-verify SHAs post-push** (rebase rewrites
SHAs). Tokens in `/workspace/.tokens/` (`gh_token`, `anthropic_key` →
export ANTHROPIC_API_KEY). Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
32 CPU / 128 GB; harness-tracked background Bash; python -u.

## Just completed: C7 (2026-07-23) — do not redo
**THE CLOSE: reasoning int/eq cell is NEGATIVE at this corpus resolution.**
Spend $0.19 (close-branch skeptic only; cumulative $11.01/$25). Full
record: `expansion/results/estimator_battery_c7.md` (+ `.json`); LEDGER C7
cycle log + cell row + narrative; BENCHMARKS § B row + prize paragraph;
research STATUS § 0 bullet. What a reviewer will check:
- Freeze order: card commit "expansion C7: monotone estimator card + r4
  mirror substitution FROZEN (pre-build)" strictly before implementation
  commit "expansion C7: mono estimator + battery + close-skeptic runner
  (pre-run commit)" strictly before any battery execution
  (commit-then-run honored — the C6 review process note).
- The candidate WORKED (pre-check ρ=−1.00 monotone; gate-3 robust-PASS
  16%; weak regime 4% vs raw 38%) and still closed: λ*_real=0.906 > 0.85
  inert threshold; gate 1 robust-FAIL — the family's INERT limit (λ=1,
  raw-tilt jump law with tilt_seg≈1) inserts +0.0181 ACF(4) > tol 0.0140
  on run-permuted real streams; gate 2 conservative-FAIL at R=24
  (0.0367 vs 0.0351, persistent ±2SE zone — the variance-aware rule, not
  a seed flip). Both close-conditions independent.
- Close skeptic: `records/proof-operation-phase-runs-c7-close/`
  (skeptic_raw.txt pre-parse + skeptic.json; 5/5 no kills; cache-guarded
  runner refuses re-rolls). r4 NEVER ran (CFG entry + C7 amendment stand
  as the frozen would-have-been protocol).
- mirrors.py: `seg_hier_categorical_mono` (+`_mono_stages`/`_mono_tables`/
  `_deconvolve_target`); λ=0 == r3 to machine epsilon (verified); r3/C6
  estimators untouched. Suite was 173 passed + 1 skip pre-battery.
- NO C8 estimator proposals; reopening = more/longer traces (data lever).

## Earlier completed (reviewed; do not redo)
- **C6 (2026-07-23): empty passing set** — reviewed & APPROVED round 2;
  variance-aware margins adopted as standing rule.
- **Stage-6 #3b: recipe_instruction_phase_runs POSITIVE** (reviewed).
- **C5: r3 ABORT** (reviewed). Stage-6 § 8 STOP → re-scope (reviewed).

## Repo state
After the C7 close commits: working tree clean, in sync with origin/arxiv
(verify `git status -sb` shows no ahead/behind). If resuming: nothing
mid-flight; wait for review or a new briefing.

## Gotchas (this box)
- Harness blocks `sleep` (use `until …; done` or background tasks);
  background python needs `-u`.
- 5-family models reject `temperature`; calibrations SEQUENTIAL (spend
  meter is per-process).
- render_report churns fig-PDF timestamps — `git checkout -- …/figs/` if
  only binaries moved.
- Skeptic verdicts: persisted raw pre-parse, cached, NEVER re-rolled
  (skeptic_c7_close.py refuses if raw exists).
- Rebase rewrites SHAs — cite subjects or re-verify post-push.
