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
