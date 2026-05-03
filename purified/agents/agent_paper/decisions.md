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

### Non-decisions (to revisit later)

- **C3 task suite** — Phase 5's 36-task vs Phase 7's 16-task PAPER subset.
  Agent NLP must pre-register a single suite before launch. See
  `docs/components/c3.md`.
- **MLC scope** — competitive with TXC-base at C3 k=5. Include as related
  work / appendix? Decide before the paper goes to draft.
- **A third agent on the A40 pod** — slot is open. Could be a "synthetic
  helper" agent that runs C1/C2 multi-seed at larger scale (50+ features).
  Defer until C3/C4/C7 land.
