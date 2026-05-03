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
- C6 turns it on by default (Dmitry's evidence directly supports it).
- C1/C2 keep it off (no dead-feature pressure at $d_{\text{sae}}=40$).
- C3/C4/C5/C7 must run an A/B test at small scale before adopting:
  TXC-base ± Bricken at 5k steps × 1 seed × small task subset.
  Adopt iff $\Delta \geq \sigma_{\text{seeds}}$. Verdict recorded in
  `docs/components/cN.md`.

**Rationale**: untested interactions — TXC-pro's matryoshka × InfoNCE
might break under hard resets; toy d_sae=40 has no dead pressure;
Gemma activations may not need the recipe. "Default for both" was a
premature commitment to a recipe that's only validated on one
organism.

### Non-decisions (to revisit later)

- **C3 task suite** — Phase 5's 36-task vs Phase 7's 16-task PAPER subset.
  Agent NLP must pre-register a single suite before launch. See
  `docs/components/c3.md`.
- **MLC scope** — competitive with TXC-base at C3 k=5. Include as related
  work / appendix? Decide before the paper goes to draft.
- **A third agent on the A40 pod** — slot is open. Could be a "synthetic
  helper" agent that runs C1/C2 multi-seed at larger scale (50+ features).
  Defer until C3/C4/C7 land.
