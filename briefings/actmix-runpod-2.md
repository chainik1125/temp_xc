---
status: active
created: 2026-07-26 ~20:30 London
for: runpod-2 (GPU 2 of the shared 3×H100 pod — setup in briefings/actmix-pod-bootstrap.md) — ACTMIX P2: EM shuffle + T-sweep
read-first: briefings/actmix-shared.md, then actmix-pod-bootstrap.md
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

**Phase A — `btk-only` arm:** cache-build FIRST (composition-
independent); btk-only convention comes from mac-a's Stage-1 LOG
note (single-source rule — never fork one locally). **Match the
paper section's setup EXACTLY** (model, layer, dataset — from
`experiments/em/`; the layer note below); ambiguities are FLAGGED
to mac-local/mac-c, not chosen. Retrain at T ∈ {1, 2, 4, 8, 16}:
txc btk-only + SAE + TSAE + untrained twins; shuffle-control eval
as in P1. Convention alignment: read Aniket's
`origin/neurips-aniket` backtracking ablation
(`purified/experiments/backtracking_window_sweep/`,
`purified/src/temp_bench/utils/shuffles.py`) so shuffle semantics
and table format line up across tasks; never modify or depend on
their in-flight branch.

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

---

## ASSIGNMENT UPDATE (Han, ~22:00 London 2026-07-26) — EM → Dmitry; runpod-2 pivots to RLHF after the in-flight grid

**EM is now Dmitry's lane (human).** Consequences, effective on
pull:
1. Your IN-FLIGHT btk-only grid RUNS TO COMPLETION as planned —
   its deliverable is re-labeled **Dmitry-support input** (the
   composition-harmonized shuffle + T table for the medical cell,
   card unchanged, E1–E5/K1–K3 scored as frozen, verdict PTR).
   Do not start anything EM beyond the frozen cells.
2. **Phase B (EM paper-match) is CANCELLED** — the Gen-1/Gen-2
   provenance question and any paper-match runs belong to Dmitry
   (COMPOSITION_AUDIT § 4 is his handover text). Keep the
   organism-forward Phase-B insurance caches on the volume
   (cheap, may serve his redo).
3. **Your next assignment: the RLHF ablation** (shuffle + T-sweep
   — the fourth paper task, currently unowned). Start prep now in
   parallel with the EM grid (CPU/download side), launch after the
   grid frees GPU 2. Everything you need is pinned in
   COMPOSITION_AUDIT § 6:
   - Task: HH-RLHF preference decomposition, gemma-2-2b BASE L12,
     first N=1000 harmless-base pairs, mean-over-response-tokens,
     rank by mean_rejected − mean_chosen.
   - Paper arms (per-task, NOT global): topk_sae k=500/token;
     tsae_paper_k500 + _k20 (ReLU→threshold at eval);
     TXC = `agentic_txc_02` matryoshka-contrastive T=5 k_win=500
     — note the RIGHT-EDGE window attribution convention (audit
     § 6), which paper-match must copy.
   - **paper-match arm = EVAL-ONLY**: the shipped seed-42
     checkpoints are public in `han1823123123/txcdr-base`
     (`<arch_id>__seed42.pt`) — download, shuffle-eval, T-read.
   - **btk-only arm**: retrain the same shapes with `*_btkonly`
     (mac-a's canonical convention; smoke + neg_frac check first —
     the identity heads-up in LOG ~22:20 applies at these widths
     too). T-sweep on the TXC arm T ∈ {1, 2, 5, 8, 16} at
     k_win = 100·T (the paper's per-token parity), T=1 = the
     controlled limit.
   - Shuffle control: within-window input permutation pre-encode,
     seed 42 (protocol semantics as in your EM card § 3).
   - Deliverable: the same Dmitry table + T-sweep figure + LOG
     verdict PTR + RUNPOD ledger lines. Freeze a card first;
     mac-local freeze-reviews in parallel. gemma-2-2b is small —
     est well under the EM grid's cost.
