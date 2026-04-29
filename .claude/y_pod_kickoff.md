# Pod-side Claude Code kickoff — Agent Y (Aniket)

Paste the message below as the first prompt to a fresh Claude Code
session running on the A40 RunPod, after you've followed Han's setup
recipe (tmux, non-root user, CC installed with
`--dangerously-skip-permissions`, repo cloned to
`/workspace/aniket/temp_xc` or similar, API keys exported via
`/workspace/aniket/.env`).

---

You are Agent Y on Han's Phase 7 unification — Aniket's instance.
This Claude Code is running on the A40 RunPod where all actual
experiments execute (the Mac side handles only code edits).

**First read, in order, before doing anything else:**

1. `docs/han/research_logs/phase7_unification/agent_y_brief.md` —
   your full brief from Han. Q1 (WHY: magnitude-scale verification),
   Q2 (HOW: defensible TXC steering protocol), and the 50%-time
   TXC-win hunt are all defined here.
2. `docs/han/research_logs/phase7_unification/brief.md` — paper-wide
   context: 12 leaderboard archs, two subject models, three
   concurrent agents (X, Y, Z) all on `han-phase7-unification`,
   directory-ownership rules.
3. `docs/han/research_logs/phase7_unification/2026-04-28-y-orientation.md`
   — my (Mac-side) absorption + the locked-in Q1.1/1.2/1.3 plan,
   Q2 candidate ranking, 50%-time candidate stack, and three open
   questions that still want Han's input.
4. `docs/han/research_logs/phase7_unification/2026-04-26-agent-c-stage1-synthesis.md`
   and the two `2026-04-26-c{1,2}-*` logs — what Agent C produced
   that Y is extending.
5. `git fetch origin dmitry-rlhf` then read
   `docs/dmitry/case_studies/rlhf/{summary,notes/methodology}.md` on
   that branch — Dmitry's paper-clamp result, the methodology
   contrast, and the magnitude-scale hypothesis.

**Hard rules (from Aniket — non-negotiable):**

- Branch: work only on `aniket-phase7-y`. Never on
  `han-phase7-unification` directly. Never `git merge` anything,
  ever (rebase / cherry-pick if needed).
- Never push to a non-`aniket-*` branch.
- Aniket's directory ownership for Y: writes confined to
  `experiments/phase7_unification/case_studies/`,
  `experiments/phase7_unification/results/case_studies/`, and
  `docs/han/research_logs/phase7_unification/2026-04-2*-y-*.md`.
  Do NOT touch `paper_archs.json`, `canonical_archs.json`,
  `probing_results.jsonl`, `training_*`, `results/plots/`, or
  `hill_climb/` — those belong to X or Z.
- Push to `origin/aniket-phase7-y` after each milestone (Q1.1
  done, Q1.2 done, Q1.3 grades landed, etc.) so the Mac side can
  pull and review.

**Pre-flight assumptions (verify before any science run):**

1. Steering pipeline runs end-to-end. Smoke:
   `python -m experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_window --arch agentic_txc_02 --limit-concepts 1 --force`
   on a single concept. Confirm a non-empty `generations.jsonl`.
2. `tsae_paper.py` faithful port loads cleanly against
   `tsae_paper_k20__seed42.pt` from HF (`han1823123123/txcdr-base`).
   Sanity-check: encode a batch, count active features, expect
   k=20.
3. Anthropic API access (Sonnet 4.6) live with prompt caching and
   `ThreadPoolExecutor(max_workers=8)`. Quick 1-call test before
   firing a 2700-call grader run.

If any of these fails: write
`docs/han/research_logs/phase7_unification/2026-04-28-y-blockers.md`
first and surface to Aniket before pushing on the science.

**First action after reading:** smoke-test pre-flight assumption
#1, then assumption #2, then assumption #3, then push a single
commit ("Y pre-flight: pipeline + ckpt + API checks pass").
After that, start Q1.1 (`<z[j]_orig>` distributions for the six
shortlisted archs over the existing `feature_selection.json` data
that Dmitry already produced under
`results/case_studies/steering_paper/<arch>/`).

**Identity / env hygiene reminder:** you are Aniket's pod-side CC.
If multiple users share this pod, your shell should have only
Aniket's env loaded (sourced from `/workspace/aniket/.env`). If
you see Han's `HF_TOKEN` or `ANTHROPIC_API_KEY` in `env`, stop
and surface — env is contaminated.

**Where to file results:**

- Q1.1 plot + JSON: `results/case_studies/steering_magnitude/q1_1_z_orig_distributions.{png,json}`
- Q1.2 plot + fitted-peak table: `results/case_studies/steering_magnitude/q1_2_strength_curves.{png,json}`
- Q1.3 generations + grades: `results/case_studies/steering_paper_normalised/<arch>/{generations,grades}.jsonl`
- Q1 synthesis log: `docs/han/research_logs/phase7_unification/2026-04-29-y-tx-steering-magnitude.md`
- Q2 candidate evaluations + final recommendation: per the brief.
- 50%-time case-study attempts: one log per candidate at
  `docs/han/research_logs/phase7_unification/2026-04-2*-y-csN-<topic>.md`.

When you push milestones, use commit messages prefixed `Y: ` so
the Mac side can grep them out of `git log`.
