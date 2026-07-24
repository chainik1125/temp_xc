---
status: active
created: 2026-07-24
for: runpod
venue: runpod (32C CPU)
---

# Hunt support (synthetic) — mechanism receipts for the Stage-2 λ̂ result

**You are `runpod`** (32C, no GPU — everything here is CPU-feasible;
your dissection ran 1,416 cells on this box). Round-1 hunt review is
APPROVED (`task_hunt/LOG.md`, mac-local review entry — read it first;
the two review notes this briefing discharges are the T=16-dip
interpretation and the T-SAE-fairness rejoinder). Prime directive
unchanged; freeze a short card per item
(`experiments/explorations/task_hunt/support_synthetic/CARD.md`,
committed pre-run); results + figs under `support_synthetic/`; verdict
paragraphs appended to the shared `task_hunt/LOG.md`. Canonical runner
for every trained cell; leaderboard hygiene. **Results by Saturday
morning PT.**

## Item 1 — the budget-dilution receipt (explains the T=16 dip)

The real-data Stage-2 result (RECORD § 3b) shows TXC-pre rising
0.13 → 0.19 → 0.21 over T = 2/4/8 then DIPPING at T = 16; the record
interprets the dip as budget dilution (fixed code budget spread over
more positions). That interpretation is currently asserted, not
measured. Test it on the synthetic mirror
(`toy_backtracking_selfexcite`, the λ̂ bench with the provable DPI
floor), which shares the evaluator (`lambda_recovery`, per-tile
leading-edge) with the real panel:

- **Arm A (fixed budget):** TXC-pre × T ∈ {2, 4, 8, 16, 32} at the
  canonical d_sae/k_pos slice, 3 seeds + untrained. Frozen prediction:
  recovery rises to a peak near the kernel support, then DECLINES —
  reproducing the real panel's shape.
- **Arm B (budget-scaled):** same cells with the window budget raised
  so realized atoms-per-window grows ∝ T (state the knob you use;
  match on MEASURED realized l0 as always). Frozen prediction: the
  decline flattens or vanishes.
- Falsifiers, stated in the card: no dip in arm A ⇒ the real dip needs
  a different explanation (a finding — say so); dip persists in arm B
  ⇒ the dilution story is wrong and the RECORD's interpretation clause
  must be retracted (also a finding).

Deliverable: a two-arm figure (mirror recovery vs T, fixed vs scaled
budget) + a LOG paragraph that either backs or retracts the RECORD's
dip sentence. Either outcome is rebuttal-relevant: if backed, the real
figure's dip becomes "predicted by the budget model, reproduced in the
mirror."

## Item 2 — the T-SAE fairness receipt (closes the rejoinder)

Review note 1-adjacent: in the real panel T-SAE occupies a single
config, so "T-SAE stays flat" is partly by construction. An obvious
reviewer rejoinder: "did you sweep T-SAE's own temporal
hyperparameter?" Close it on the mirror: sweep the registered T-SAE's
contrastive-distance/window knob (whatever `tsae.py` exposes; if not
YAML-exposed, add a plugin-arch variant file — hard rule 3, never edit
core) across ≥ 3 settings at the canonical budget, 3 seeds +
untrained. Frozen prediction: λ̂ recovery is FLAT in its own window
knob within seed noise — the per-token decode, not the training
window, is binding. Falsifier: recovery rises with the knob ⇒ the real
panel's T-SAE cell underestimates the baseline ⇒ flag IMMEDIATELY in
the LOG + to runpod-d (its round-2 panel would need a T-SAE re-run at
the best knob before any rebuttal figure ships).

## Acceptance gate — stop for review

Cards frozen pre-run; both verdicts in the LOG with figures; records
under `support_synthetic/`; leaderboard 0 dup keys; STATUS rewritten.
No reviewer/meeting quotes in tracked files. Briefing stays until
mac-local review.
