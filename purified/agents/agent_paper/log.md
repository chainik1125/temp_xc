---
author: agent_paper
date: 2026-05-03
status: active
---

## 2026-05-03 — Day 0 (kick-off)

- Read briefing.md, project_brief.md, and headline research logs for C1–C7.
- Surveyed wasteland: src/architectures/ has 50+ TXC variants (Phase 5+7
  hill-climbing residue); experiments/phase{2,3,5,6,7}_*/ each contain a
  subset of working code.
- Identified three contradictions vs the briefing:
  1. C6 EM result is actually a **negative TXC result** per Dmitry's
     2026-05-03 paper section (SAE arditi beats TXC k=100 at every cell).
  2. C5 steering hill-climb winners (Y/W's Galaxy variants) **lose** at
     probing per `2026-05-02-yw-T8-benchmark.md`.
  3. C7 backtracking is more salvageable than the briefing suggested —
     TXC peak Δgc=+1.574 (~3× next-best) is a real result.
- Asked Han three questions; locked answers:
  - TXC-base = `txc_bare_antidead_t5`; TXC-pro = `phase5b_subseq_h8`.
  - C6 reframed as honest negative.
  - `final` branch from `han-phase7-unification`, push to origin.
- Built `purified/` scaffold:
  - `pyproject.toml`, `.python-version`, `.gitignore`
  - `README.md`, `CLAUDE.md`, `PROTOCOL.md`
  - `src/temp_bench/{architectures,data,training,eval,case_studies,plotting,utils}/`
    skeletons with locked architecture registry + CaseStudy ABC
  - `docs/components/c{1..7}.md` writeup skeletons
  - `docs/paper/{outline,architecture}.md` paper drafts
  - `agents/agent_paper/{decisions,log}.md`
- Next: bootstrap script + experiment dir READMEs, then commit + push.

## TODO (next sessions)

- [x] First commit + push of `final` to origin.
- [x] Write `purified/scripts/bootstrap_runpod.sh` (port of Phase 7 script).
- [ ] Implement TXC-base + TXC-pro in `src/temp_bench/architectures/`
      (copy + simplify from wasteland).
- [ ] Implement C1 toy data generator + sweep script (5090-local).
- [ ] First C1 multi-seed run (3 seeds × 12 k values × 4 archs ≈ 6 hr local).
- [ ] Spawn Agent NLP brief for C3+C4 caching (1× H100 RunPod).
- [ ] Spawn Agent EM brief for C6 (1× H100 RunPod).

## 2026-05-03 — Day 0, follow-up turn

Han raised three operational questions:

1. **CLAUDE.md scoping.** Verified via claude-code-guide that subdir
   CLAUDE.md files auto-load on demand and `@<path>` import syntax works.
   Decision: rely on the on-demand load; added a one-line pointer in the
   root `CLAUDE.md` so paper-bound agents notice the quarantine.
   Recommend (not enforce) launching from `purified/` cwd.
2. **Branch merging.** Decided NOT to merge `em-nanda` /
   `aniket-ward-stage-b` into `final` (they're still active). Added
   `purified/scripts/wasteland_refresh.sh` (fast `git fetch --all`) and
   extended `PROTOCOL.md` § 2a with the cross-branch read pattern.
3. **HF repos.** Created two private repos under han1823123123:
   `temp-bench-models` (model) + `temp-bench-data` (dataset). Seeded each
   with a README. Updated `purified/CLAUDE.md` checkpoint section + added
   `purified/checkpoints/README.md` with upload helper recipe. Wasteland
   repos (txcdr-*) untouched.

All four artifacts (root CLAUDE.md edit, wasteland_refresh.sh, PROTOCOL
update, checkpoints/README.md, decisions.md additions) committed in one
turn after this entry.

## 2026-05-03 — Day 0, Bricken correction

Han raised: Dmitry mentioned C6 was "tied at 30k TXC with Bricken
re-init, the steering range was quite good" — does this contradict the
honest negative framing?

Investigated. Two distinct organisms confused in the wasteland:

- **Qwen-2.5-7B-Instruct + medical** (older, ~April 2026): TXC
  brickenauxk 30k @ resid_mid 53.87 ties T-SAE 100k @ resid_post 52.39
  at 3.3× less training. SAE arditi 100k @ resid_post 57.42 still wins
  by 3.5 align. Source: `txc_hookpoint_comparison_finding.md`.
- **Qwen-2.5-14B-Instruct + finance** (May 2026 paper section): plain
  TXC k=100 (no anti-dead, no Bricken) loses to SAE arditi by +3.91
  (R1 30k) to +12.58 (R32 ext-α). Source: `em_nanda_results_paper.md`.

The "honest negative" was based on the 14B finance evidence — but that
TXC variant had neither anti-dead stack nor Bricken resample. The
comparison was unfair to TXC.

**First revision**: I proposed making Bricken resample a "trainer-level
default for both TXCs." Han pushed back with the right concern: untested
interactions with TXC-pro's matryoshka × InfoNCE; toy data has no dead
pressure; the brickenauxk recipe co-tunes six knobs that may not
transfer.

**Second revision (locked)**: Bricken is opt-in per component, NOT
part of the architecture spec.

- C1/C2: off (no dead pressure at d_sae=40).
- C3/C4/C5/C7: A/B at small scale first, adopt iff Δ > σ_seeds.
- C6: on (Dmitry's evidence directly supports it).

C6 status changed from `planning (honest negative)` to
`pending-retest`. Headline framing depends on the re-run outcome:
gap ≤ 3 → tied; 3–9 → mixed (step-efficiency vs absolute); >9 → honest
negative.

Updated:
- `docs/paper/architecture.md` — removed the "Training defaults" section,
  added "Per-experiment training knobs" section with per-component table.
- `docs/components/c6.md` — rewritten with the brickenauxk evidence,
  setup includes opt-in disclosure.
- `src/temp_bench/training/bricken.py` — docstring clarifies opt-in
  status.
- `agents/agent_paper/decisions.md` — added Decision #7.

## 2026-05-03 — Day 0, framework day

Han pushed back hard on modularity: the framework must make adding /
removing architectures and switching layer/model essentially free, or
it will bite us the day before deadline. Implemented the
modularity-first design:

**The contract** (`docs/paper/framework.md`):

- 10 principles; 5 are hard rules in PROTOCOL.md § 11.
- Configs as source of truth (`locked_archs.yaml`, `datasources.yaml`).
- Two-tier deterministic cache: `act_cache_key` ⊃ `train_key` ⊃ `eval_key`.
- Single canonical `runner.run_cell` — only writer to leaderboard.
- Schema-checked rows (Pydantic) — malformed rows aborted at append time.
- Per-component `EVAL_PROTOCOL_VERSION` constant for cheap re-eval.

**Implementation** (~600 LoC, all tested):

- `configs/locked_archs.yaml` — 8 archs registered with version + hparams
- `configs/datasources.yaml` — 7 datasources (5 real-LM + 2 toy)
- `src/temp_bench/schemas.py` — Pydantic `LeaderboardRow`,
  `CheckpointManifest`, `TrainingConfig`, `ArchSpec`, `DataSourceSpec`
- `src/temp_bench/config.py` — yaml loaders + cache-key computation
  (`compute_act_cache_key`, `compute_train_key`, `compute_eval_key`)
- `src/temp_bench/cache.py` — checkpoint save/load, leaderboard append
  (flock-protected, schema-validated)
- `src/temp_bench/runner.py` — `run_cell()`, `preflight()`, helpers
- `experiments/_runner_template.py` — ~80-line copy-paste template
  for new components

## 2026-05-03 — Day 0, agent roster trim

Han: agent_synth and agent_qa don't have unique jobs — drop them.
agent_apple was just a placeholder — also drop.

Honest reckoning: I was filling 4 A40 GPU slots with named agents,
not naming agents based on real work. C1/C2 are PAPER's job; Bricken
A/B and judge κ are per-component (lead agents handle their own).
Dropped all three.

New A40 pod topology: 2 named agents (steer + back) + 2 spare GPU
slots (a40_helper_gpu2, a40_helper_gpu3). Spare slots are reachable
via `set_agent_env.sh` for lead agents launching parallel-seed
processes — no new named agent needed unless concrete work emerges.

Files updated: agents/README.md, scripts/set_agent_env.sh,
docs/paper/hardware.md, deleted agents/agent_apple/.

## 2026-05-03 — Day 0, wasteland delete + docs consolidation

Han: "is it worth deleting the wasteland files in the purified branch
and forcing agents to inspect other branches to see the wasteland?"
Recommended yes (asymmetrically — code yes, docs no). Han confirmed.
Han also asked about consolidating purified's three top-level docs
(README, CLAUDE, PROTOCOL) — agreed to merge README into CLAUDE.

Executed:

1. **Deleted code wasteland** from `final`: `src/`, `experiments/`,
   `references/`, `tests/`, `scripts/`, root `pyproject.toml`, `uv.lock`,
   `Dockerfile`, `torchgpu_packages.txt`, `launch-sandbox.sh`,
   `temporal_crosscoders/`, root `results/`. **Kept docs wasteland**:
   `docs/`, `papers/`, root `CLAUDE.md`, `RUNPOD_INSTRUCTIONS.md`,
   `CONTRIBUTING.md`. 3658 → 319 tracked files (~91% reduction).

2. **Merged purified/README.md into purified/CLAUDE.md.** README
   deleted. CLAUDE.md is now the single agent-facing operating manual
   (~250 lines) and is auto-loaded by Claude Code. PROTOCOL.md stays
   as the detailed reference (~330 lines).

3. **Updated PROTOCOL.md § 2** (wasteland boundary) with the new
   `git show origin/han-phase7-unification:<path>` read pattern + a
   header-comment template for porting.

4. **Cleaned wasteland-rooted reproduction commands** in
   `docs/components/c2.md` (was pointing at `src/v2_temporal_schemeC/...`)
   and `docs/components/c3.md` (was suggesting to "port Phase 5's cache").
   Both now reference the `temp_bench` framework.

5. **Documented in `decisions.md` #8 (delete) and #9 (consolidate).**

Tests verified: all green.

## 2026-05-03 — Day 0, root tooling cleanup

Han: "I still see a bunch of skeleton files locally in `src` and stuff
like .markdownlint.jsonc; also wouldn't it be appropriate to completely
replace the root level CLAUDE.md with purified/CLAUDE.md?"

Two issues:

(a) The previous commit message claimed I'd consolidated purified
docs (README → CLAUDE merge) but the actual purified edits didn't make
it into the commit — only the wasteland deletion did. Bundled in this
turn.

(b) Root-level wasteland tooling was still present: `.markdownlint.jsonc`,
`run-checks.sh`, `check-tags.sh`, `get-tags.sh`, `CONTRIBUTING.md`,
`.lycheeignore`, `.dockerignore`, `.vscode/extensions.json`,
`.github/workflows/{agent-review,ci,link-check}.yml`. None of these
operate on `purified/`; they all assumed wasteland-style `docs/`
validation.

Resolved:

1. Root `CLAUDE.md` replaced with single-line `@purified/CLAUDE.md`
   import. When an agent launches at the repo root, the import inlines
   purified's operating manual into context — no duplication, single
   source of truth.

2. Root `README.md` replaced with a brief stub: "this is the final
   branch, see purified/CLAUDE.md, wasteland is on origin".

3. Deleted root wasteland tooling: CONTRIBUTING.md, check-tags.sh,
   get-tags.sh, run-checks.sh, .markdownlint.jsonc, .lycheeignore,
   .dockerignore, .vscode/, .github/workflows/.

4. Local untracked litter (~432 GB across src/, experiments/, tests/,
   results/, data/, logs/) is the user's to clean up:

       rm -rf src/ experiments/ tests/ results/ data/ logs/

   Not running this myself — destructive, and `.venv/`, `.env_autointerp`,
   `.claude/`, `.git/` should NOT be touched.

Tests still 29/29.

## 2026-05-03T20:30Z — handover written

`handovers/2026-05-03T20-30Z-framework-complete.md`. First-ever
handover, mostly because PROTOCOL.md § 14 was just authored and I
wanted to demonstrate compliance. Captures: framework state (29/29 →
38/38 after token tests), key locked decisions (#1–9), what to do
next (start C1+C2 architecture porting + experiment runners), pitfalls
to avoid.

**Tests** (23/23 passing):

- `tests/test_cache_keys.py` — determinism, version-bump invalidation,
  Bricken-toggle separates cache, dict-order invariance
- `tests/test_schemas.py` — strict validation, brickenauxk_a8 recipe
  serialises to a different config than the default
- `tests/test_runner_idempotency.py` — call run_cell twice ⇒ second
  call hits cache, no duplicate leaderboard row; force_eval re-runs
  but force_train doesn't help if act_cache_key is the same; different
  seed makes a separate cell

**Walkthrough scenarios** (all in `framework.md`):

- A: miracle TXC found 1 day before deadline → 1 yaml + 1 class, all
  components consume it automatically, only the new arch's cells
  compute (cached for everything else)
- B: bug in TXC-pro encode → fix code + bump arch_version, only
  TXC-pro re-trains
- C: bug in C3 metric → fix in eval/probing.py + bump
  EVAL_PROTOCOL_VERSION, all checkpoints reused
- D: switch C3/C4/C5 IT → BASE → 1-line yaml change per component,
  act-cache rebuilds once

**Doc updates**:

- `docs/paper/framework.md` — new (the principles)
- `purified/CLAUDE.md` — replaced run-id pattern with run_cell example
- `purified/PROTOCOL.md` — added § 11 framework discipline
- `purified/checkpoints/README.md` — updated upload recipe
- `scripts/agent_smoke_test.sh` — runs pytest + preflight on every session

The smoke test now greenlights agents in ~15s. From here on, every
cell that produces a paper number flows through `run_cell`. Nothing
else is allowed to write to `leaderboard.jsonl`.
