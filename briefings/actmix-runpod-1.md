---
status: active
created: 2026-07-26 ~20:30 London
for: runpod-1 (H100 pod, Han-provisioned) — ACTMIX P1: SPARSE PROBING shuffle + T-sweep
read-first: briefings/actmix-shared.md
---

# ACTMIX P1 — sparse probing: shuffle control + T-window sweep, both arms

**Goal (rebuttal exhibit, wanted before 9am PT / 17:00 London
2026-07-27):** the table Dmitry specified — **TXC | TXC-shuffled |
SAE (| TSAE)** — plus the T-sweep curve with the **T = 1 controlled
limit** (TXC ≈ SAE at matched params) as the anchor row, for the
paper's sparse-probing task. Han owns the science read; this pod
does the compute.

**Phase A — `btk-only` arm (runnable immediately).** Retrain on
the probing substrate at T ∈ {1, 2, 4, 8, 16} (extend to 32 if
budget/time allow): txc (the paper-shaped variant with relu_mode
none — coordinate with mac-a's Stage-1 implementation, or
implement the same variant locally if mac-a hasn't landed;
IDENTICAL convention, coordinate in briefings/), per-token SAE,
TSAE, + untrained twins. Then the shuffle control: eval each
trained model with window positions randomly permuted (fixed
permutation per row, seeded) — report recovery vs unshuffled.
Expected shape: TXC falls toward the SAE under shuffle; SAE/TSAE
unmoved (their codes are per-token — state this as the control's
own control).

**Phase B — `paper-match` arm (BLOCKED on mac-c).** mac-c's
COMPOSITION_AUDIT pins the composition sparse probing actually
used (likely `han-phase7-unification`-era TopK→ReLU, k = 8·T — but
WAIT for the pin) and whether checkpoints exist in Han's HF
datasets. Checkpoints found ⇒ shuffle is EVAL-ONLY on them
(cheapest, most faithful). No checkpoints ⇒ retrain at the pinned
composition, same grid as Phase A. Same shuffle + sweep.

**Deliverables:** the table (both arms side by side), T-sweep
figure per arm (WRITEUP style, untrained + shuffle overlays),
realized-l0 per cell (the mixing fingerprint — REQUIRED), LOG
verdict PTR, ledger lines under a `RUNPOD` section in
MODAL_SPEND.md. Freeze-commit-run with origin-history pins; the
pod never pushes results without pull-rebase; $150/day cap.
