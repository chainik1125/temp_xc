---
author: agent_paper
date: 2026-05-03
status: locked
---

## Locked decisions, 2026-05-03

These three decisions were made in conversation with Han at session start.
They are now policy. Re-opening any of them requires a new conversation
with Han.

### 1. Two TXC architectures (locked)

- **TXC-base** = `txc_bare_antidead_t5` — vanilla TopK + tsae_paper anti-dead stack.
- **TXC-pro** = `phase5b_subseq_h8` — subseq encoder + matryoshka H8 + multi-distance contrastive.

**Rationale**: these are the only two TXCs with consistent top-3 finishes
across both Phase 5 and Phase 7 probing leaderboards. Steering hill-climb
winners (Galaxy 8/11/18) were considered and rejected because they lose
0.005–0.020 probing AUC vs canonical (`2026-05-02-yw-T8-benchmark.md`).

**Trade-off accepted**: C5 steering becomes a "matches T-SAE at high coh"
result rather than a "beats T-SAE" result. C3+C4 wins are stronger.

### 2. C6 EM is reframed as honest negative

**Finding**: Dmitry's `em_nanda_results_paper.md` shows SAE arditi beats
TXC k=100 at every (steps × organism × α-regime) cell. Arch gap *widens*
to +12.58 align in R32 ext-α regime.

**New paper framing**: report the SAE win honestly. Salvage the
**bundle-null architecture-generality** result — both arches' k=30
bundles peak at align ≈ 41.3, falling 13–23 below their single-feat
champions. This falsifies the "distributed misalignment" hypothesis in
both dictionaries: an interpretive contribution despite the probing loss.

**Coordinator note**: needs to be co-signed with Dmitry. Agent EM should
ping Dmitry's brief in `origin/em-nanda` before launching anything.

### 3. Branch model

- `final` is created from `han-phase7-unification` HEAD (the user's
  current state, including the briefing + paper md).
- All future work commits to `final`.
- Wasteland is the rest of the branch's tree — read-only context.
- Push `final` to origin so worker agents can clone it.

### 4. Cross-branch reads (em-nanda, aniket-ward-stage-b)

- **Do not merge** sibling branches into `final`. They are still being
  updated by Dmitry and Aniket; merging would freeze stale state and
  create conflict surface on every refresh.
- Read directly from origin: `git show origin/em-nanda:<path>`.
- `purified/scripts/wasteland_refresh.sh` does the `git fetch` so
  `origin/<branch>` always resolves to the latest pushed state.
- If we need a frozen snapshot of code (e.g. Aniket's
  `experiments/ward_backtracking_txc/`), copy it once into
  `purified/src/temp_bench/` with the source commit hash in a header
  comment, and stop tracking origin from then on.

### 5. CLAUDE.md scoping

- Subdirectory CLAUDE.md files **auto-load on demand** when an agent
  reads files under that directory (verified). So an agent launched at
  the repo root sees the wasteland CLAUDE.md initially, and
  `purified/CLAUDE.md` loads automatically the moment it touches a
  paper file.
- Added a one-line pointer in the root `CLAUDE.md` directing paper-bound
  agents at `purified/CLAUDE.md`.
- Recommended (not enforced) launch pattern for paper-only agents:
  `cd purified && claude` — keeps `git add -A` scoped to paper files.

### 6. HuggingFace repos

- New, private, paper-dedicated:
  - **`han1823123123/temp-bench-models`** — all checkpoints (locked
    archs + baselines), keyed by `<run_id>` prefix.
  - **`han1823123123/temp-bench-data`** — activation caches, judge
    transcripts, pre-tokenised tasks, synthetic data.
- Provisioned 2026-05-03 with seed READMEs.
- Wasteland repos (`han1823123123/txcdr-base`, `txcdr-it`, `txcdr`,
  `txcdr-base-data`, `txcdr-data`) are **untouched** — they remain as
  historical record. Paper artifacts never go into them.
- Visibility flips to public when the paper draft stabilises.

### 7. Bricken resample is opt-in per component, NOT a locked architecture default

**Context**: Dmitry's data on Qwen-7B medical (`txc_hookpoint_comparison_finding.md`)
shows TXC brickenauxk 30k @ resid_mid (53.87) ties T-SAE 100k @
resid_post (52.39) — the "tied at 30k" Han mentioned. That recipe
co-tunes **six** knobs (resample_every=500, min_fires=1, n_check=2048,
max_resample_fraction=0.5, EMA-AuxK α=1/8, dead_threshold=128k tokens),
all jointly tuned for that organism.

**Decision** (revised after Han pushed back on the original
"trainer-level default for both TXCs" framing):

- The locked architectures TXC-base and TXC-pro **do not** include
  Bricken resample. They include only what's listed in
  `docs/paper/architecture.md` proper.
- Bricken resample is exposed as an opt-in `BrickenConfig` knob in
  `src/temp_bench/training/bricken.py`. Components turn it on
  themselves and disclose the choice in their writeup.
- **C6 only by default** (Dmitry's evidence directly supports it on
  Qwen-7B medical organism).
- C1/C2 keep it off (no dead-feature pressure at $d_{\text{sae}}=40$).
- **C3/C4/C5/C7 keep it off** (revised 2026-05-03 with Han). The
  earlier policy demanded an A/B test (TXC-base ± Bricken at 5k×1seed)
  for each of these components before adopting. Han: "we're locking in
  txc_base and txc_pro, we'll only try Bricken resample if time
  persists at the end." Saves ~8 H100-hours of validation work for a
  maybe-marginal effect; the cost is leaving on the table any
  Δ AUC > σ_seeds that Bricken would have lifted.

**Rationale**: untested interactions — TXC-pro's matryoshka × InfoNCE
might break under hard resets; toy d_sae=40 has no dead pressure;
Gemma activations may not need the recipe. "Default for both" was a
premature commitment to a recipe that's only validated on one
organism.

### 8. Wasteland code deleted; wasteland docs kept

Han: "is it worth deleting the wasteland files in the purified branch
and forcing agents to inspect other branches to see the wasteland?"

Decision: yes — but asymmetrically. **Code wasteland deleted** from
`final` (`src/`, `experiments/`, `references/`, `tests/`,
`scripts/` at root, root `pyproject.toml`, `uv.lock`, `Dockerfile`,
`launch-sandbox.sh`, `torchgpu_packages.txt`, `temporal_crosscoders/`,
root `results/`); **docs wasteland kept** (`docs/`, `papers/`, root
`CLAUDE.md`, `RUNPOD_INSTRUCTIONS.md`, `CONTRIBUTING.md`).

**Why asymmetric**: docs are read often (passively, for context — every
component writeup cites ~5 wasteland research logs). Code is read once
per port (actively, for transcription, ~10 ports total). Delete what's
read once; keep what's read often.

**Benefit**: an accidental `from src.architectures.tfa import …` now
raises `ModuleNotFoundError` immediately rather than silently picking
up wasteland code. The "no wasteland imports" rule (PROTOCOL.md § 2)
becomes git-level enforcement, not policy.

**Cost**: agents porting code use
`git show origin/han-phase7-unification:src/...` to read. Mild — one
extra command per port, ~10 times across the paper. Worked example +
header-comment template in PROTOCOL.md § 2.

3658 → 319 tracked files (~91% reduction). Reversible via `git revert`
or `git checkout origin/han-phase7-unification -- <path>` if a specific
file turns out to be needed.

### 9. Consolidate purified/ docs from 3 → 2 files

Han: "is it necessary to have THREE different files?" (referring to
`purified/` having `README.md`, `CLAUDE.md`, `PROTOCOL.md`).

Decision: drop `purified/README.md`. Merge its unique content (Quick
start, Layout, Components table, TXC summary) into `purified/CLAUDE.md`,
which is the auto-loaded source of truth for agents. `PROTOCOL.md` stays
as the detailed protocol reference.

Final layout:
- **CLAUDE.md** (~250 lines): operating manual + brief overview.
  Auto-loaded by Claude Code when an agent reads any file under
  `purified/`. Self-contained for session start.
- **PROTOCOL.md** (~330 lines): detailed contract (§ 1-11 incl. GPU
  pinning § 11.0, multi-GPU access § 11.1, framework discipline § 11).
  Read on first session, referenced as needed.

### 10. One agent doc — the briefing — owns identity AND rolling state

**Earlier design (rejected)**: per-agent `briefing.md` (Han owns) +
`handovers/<ts>-<slug>.md` (dated archive of state snapshots) +
`log.md` (running chronological narrative). Three docs per agent.

**Failure mode**: in the wasteland, `briefing.md` files went stale
because nobody updated them; `handover_*.md` files accumulated by the
dozen and nobody knew which was current; `agentic_log.md` files grew
to thousands of lines and were never read post-compact. Three docs
per agent ⇒ none stays fresh.

**Decision** (Han, 2026-05-03):

- **One file per agent: `briefing.md`**, with explicit section
  ownership inside the file:
  - `## Identity + mandate (Han owns — agents do not edit)` — Han's
    prose at the top, immutable.
  - `## Current state` / `## What I just did` / `## Next action` /
    `## Don't repeat` / `## Open questions for Han` —
    agent-owned, **overwritten at every compact**.
- **No separate `handover.md` and no `log.md`.** Git history
  (`git log -p purified/agents/<name>/briefing.md`) is the audit
  trail; `decisions.md` captures locked decisions; the briefing's
  "What I just did" captures the last 5–10 actions for the next-life
  instance.
- **Component-vs-agent doc separation codified** in PROTOCOL.md § 7:
  `docs/components/cN.md` owns the technical setup (hypothesis, results,
  caveats); agent briefings own identity + state. Briefings point at
  component docs, do not duplicate.

**Why this beats both prior options**:
- Single file ⇒ no "which doc is current" confusion.
- Section ownership ⇒ Han's mandate doesn't drift into the agent's
  rolling state.
- Git log ⇒ chronological history without manual log.md upkeep.
- Component docs ⇒ technical setup outlives any individual agent.

**Implementation**: PROTOCOL.md § 14 rewritten from "Handover protocol"
to "Briefing maintenance"; `_briefing_template.md` added;
`_handover_template.md` deleted; `agents/agent_paper/handovers/`
deleted; `agents/agent_paper/log.md` deleted. The historical content
of agent_paper's log.md is summarised in `git log` of the deletion
commit + the now-merged briefing's "What I just did".

### 11. C3 task suite is `SAEBench+CT` (n=38)

Han: "I think we should just do whatever SAEBench did ... we should
definitely fix the github-code discrepancy and do the HF permissions ...
the benefit of using the faithful SAEBench task set is reviewers won't
complain about cherrypicking."

**Decision**: C3 evaluates on **SAEBench+CT**, defined as the canonical
upstream SAEBench sparse-probing suite (Karvonen et al., 36 binary
one-vs-rest tasks across 8 datasets) augmented with two cross-token
coreference probing tasks (WinoGrande, SuperGLUE WSC). **Total: 38
tasks.** Phase 5's 36-task and Phase 7's 16-task PAPER subset are
both retired as headline candidates.

The SAEBench composition is fixed by `chosen_classes_per_dataset` in
upstream `sae_bench/sae_bench_utils/dataset_info.py` (verified
2026-05-03):

```
bias_in_bios_class_set1: ["0","1","2","6","9"]      → 5
bias_in_bios_class_set2: ["11","13","14","18","19"] → 5
bias_in_bios_class_set3: ["20","21","22","25","26"] → 5
amazon_reviews_mcauley_1and5: ["1","2","3","5","6"] → 5
amazon_reviews_mcauley_1and5_sentiment: ["1.0","5.0"] → 2
codeparrot/github-code: ["C","Python","HTML","Java","PHP"] → 5
ag_news: ["0","1","2","3"] → 4
europarl: ["en","fr","de","es","nl"] → 5
                                                       ──
                                                       36
+ winogrande_correct_completion + wsc_coreference    → 38
```

`probe_training.py` iterates per-class with no special handling, so
2-class amazon_sentiment yields 2 binaries (verified upstream).

**Three implementation deltas** vs the wasteland's "FULL-36" loader:

1. **github-code provider switch.** Use SAEBench's `codeparrot/github-code`
   with the 5 SAEBench languages `["C","Python","HTML","Java","PHP"]`,
   not our wasteland's `code_search_net` python/java/javascript/go.
   The dataset uses a Python loading script (HF web viewer is disabled
   for that reason but the dataset itself is publicly readable, NOT
   gated). Loader requires `trust_remote_code=True` — already set via
   `os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")`. Also
   requires `datasets<4` (the `trust_remote_code` mechanism was removed
   in v4); pinned in `purified/pyproject.toml` 2026-05-03.
2. **amazon_sentiment.** Add the 1.0-vs-rest binary (we currently only
   have 5.0-vs-rest as `amazon_reviews_sentiment_5star`).
3. **amazon_categories.** Hardcode the class list to `["1","2","3","5","6"]`
   and use a non-streaming pull large enough to populate all 5; the
   wasteland's streaming-top-5 approach is non-deterministic and
   missed cat6.

**Why SAEBench-faithful + the 2 coref additions**:

- SAEBench is the recognised standard. Saying "we evaluated on
  SAEBench" defends against the "you cherry-picked tasks that favor
  TXC" review on the headline benchmark axis.
- WinoGrande + WSC retained because they are the cleanest single-task
  evidence for TXC's cross-token inductive bias (winogrande T-slope
  +0.0069/T at k=20 — ~100× the next task; from
  `2026-04-29-per-task-tsweep.md`). Reported transparently as a
  "+CT" extension, not folded silently into "SAEBench".
- The 16-task PAPER subset (Phase 7) inherited the same coref-addition
  problem AND added a cluster-balancing decision the paper would have
  to defend separately. Two unforced critique vectors collapsed into
  one ("we extended SAEBench by 2 well-motivated coref tasks").

**Naming convention for the paper**: refer to the suite as
**SAEBench+CT** in tables and figure captions. First mention in prose:
"the standard SAEBench sparse-probing benchmark (Karvonen et al., 36
tasks across 8 datasets) augmented with two cross-token coreference
probing tasks (WinoGrande, SuperGLUE WSC; n=38 binary one-vs-rest
tasks total)."

### 12. TrainingConfig batch_size raised to 2048 (was 256)

agent_nlp caught the undertraining 2026-05-04 (commits 579efb9a +
8904414d + 3558b303): every production cell across C3/C4/C5/C6/C7
shipped at the framework's default `batch_size=256`, while Phase 7's
reference `TrainCfg` used `batch_size=4096`. Net: ~13-40× less
gradient information per cell than the wasteland reference.

**Most-affected archs** are the contrastive ones — T-SAE (temporal
InfoNCE on adjacent tokens) and TXC-pro (multi-distance contrastive
+ matryoshka). Both lean on in-batch negatives. The C3+C4+C5
relative-arch orderings (TopK-SAE > TXC > T-SAE; T-SAE >> TXC) MAY
shift once these archs train at proper batch.

**Decision** (2026-05-04, overseer): bump `TrainingConfig.batch_size`
default to **2048** (token-equivalent budget at n_steps=30K = 61M,
~60% of Phase 7's 100M but feasible in remaining 72-h window).
A40-pod components (C5 agent_steer, C7 agent_back) override to
`batch_size=1024` in their per-component runner because batch=2048 +
larger d_in (Llama-8B) won't fit in 48 GB.

**Cross-agent re-train directive**: every C3/C4/C5/C6/C7 cell needs
NEW train_keys with the new batch. Old cells stay in the leaderboard
for diff comparison; the new runs use bumped train_keys
automatically (batch_size is part of train_key hash).

**Compute cost**: ~8× per cell. Total re-train ~80-150 GPU-hours
across pods. Worker briefings updated with the directive.

### Non-decisions (to revisit later)

- **MLC scope** — competitive with TXC-base at C3 k=5. Include as related
  work / appendix? Decide before the paper goes to draft.
- **A third agent on the A40 pod** — slot is open. Could be a "synthetic
  helper" agent that runs C1/C2 multi-seed at larger scale (50+ features).
  Defer until C3/C4/C7 land.
