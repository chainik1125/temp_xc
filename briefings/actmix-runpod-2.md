---
status: active
created: 2026-07-26 ~20:30 London
for: runpod-2 (H100 pod, Han-provisioned) — ACTMIX P2: EM shuffle + T-sweep
read-first: briefings/actmix-shared.md
---

# ACTMIX P2 — emergent misalignment: shuffle control + T-window sweep, both arms

**Goal:** same exhibit shape as P1 — **TXC | TXC-shuffled | SAE
(| TSAE)** + T-sweep with the T = 1 limit — for the paper's EM
task. Two EM-specific cautions from the program record:
(1) the PAPER's EM numbers came from `dmitry-em-repl`, NOT
`final`'s EM code — wait for mac-c's pin before calling anything
"paper-match"; (2) the paper's EM negative (TXC underperforms
T-SAE on Wang-style steering + sparse-probe PR-AUC) had NO shuffle
ablation in the paper — this run ADDS the missing control, and the
depth story (EM misalignment readout peaks mid-depth, inverted-U,
L13/L15) means LAYER CHOICE must follow the paper's layer, stated
in the card.

**Phase A — `btk-only` arm (runnable immediately):** retrain at
T ∈ {1, 2, 4, 8, 16}: txc btk-only + SAE + TSAE + untrained twins
on the EM substrate per the paper section's data path
(`experiments/em/`); shuffle-control eval as in P1.

**Phase B — `paper-match` arm (BLOCKED on mac-c):** pinned
composition from dmitry-em-repl's result commits; checkpoints from
Han's HF datasets if the inventory finds them (eval-only shuffle),
else retrain at the pinned composition.

**Deliverables:** table (both arms), T-sweep figure per arm with
untrained + shuffle overlays, realized-l0 per cell, honest side-
by-side with the paper's published EM negative (this ablation may
CONFIRM the negative — that is a fine outcome and goes in at full
prominence), LOG verdict PTR, `RUNPOD` ledger lines. Freeze/pin/
pull-rebase discipline; $150/day cap.
