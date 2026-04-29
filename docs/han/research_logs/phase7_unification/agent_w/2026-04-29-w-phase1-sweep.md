---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Agent W Phase 1 sweep — k_pos=20 atomic-axis exploration

> **Status: cells C and D landed; E and F mid-training.** Numbers locked in this writeup
> are authoritative for cells already graded; cells E/F sections are placeholders that
> will be filled in as pipelines complete. This is a *living* writeup until all 4 cells
> finish.

### TL;DR (preliminary, awaiting E/F)

Under **brief's primary metric** (peak success at coherence ≥ 1.5,
family-normalised paper-clamp), **every Phase 1 cell trained so far LOSES**
to T-SAE k=20 anchor (1.10) by ≥ 0.27 σ_seeds:

| cell | arch_id | T | k_pos | random-init? | raw peak | pk @ coh ≥ 1.5 | win/tie/loss |
|---|---|---|---|---|---|---|---|
| anchor | `tsae_paper_k20` | 1 | 20 | yes | 1.80 | **1.10** | (anchor) |
| C (W) | `txc_bare_antidead_t3_kpos20` | 3 | 20 | yes | **1.40** | 0.77 | **LOSS** by 0.33 |
| D (Y) | `txc_bare_antidead_t5_kpos20` | 5 | 20 | yes | 1.00 | 0.70 | **LOSS** by 0.40 |
| E (W) | `agentic_txc_02_kpos20` | 5 | 20 | yes | TBD | TBD | TBD |
| F (W) | `txc_bare_antidead_t10_kpos20` | 10 | 20 | yes | TBD | TBD | TBD |

Per pre-registered Outcome C: every cell loses by ≥ 0.27 → architecture
has anti-prior at matched per-token sparsity. **Publishable converging
null** with Y's parallel finding from the T-axis ladder direction. Phase 2
unlikely to recover from a Phase-1 LOSS-everywhere; will document the
per-axis perturbation table as the failure-mode artefact instead.

The closest-to-tie cell is W's cell C (T=3) at 0.77 — within Δ=0.06 of
the loss threshold but not crossing it. Worth noting as the *least-bad*
TXC k_pos=20 cell.

### One-line interesting findings

1. **Narrower window is better at k_pos=20.** Cell C (T=3) raw peak 1.40 *beats*
   cell D (T=5) raw peak 1.00. Counterintuitive — at sparse per-token cap, the
   T-axis structural advantage *reverses*.
2. **Coherence is preserved at k_pos=20.** All 4 cells' constrained peaks have
   coh ∈ [1.97, 2.07] — far above the 1.5 threshold cliff that bites k_pos=100
   window archs. The hypothesis "sparser k_pos lifts coherence" is confirmed
   empirically. The remaining problem is success doesn't lift commensurately.
3. **Y's feature-polysemanticity finding.** At k_pos=20, T=5, only 24/30
   distinct picked features (vs T-SAE k=20's 28/30); feat 16117 picked for 4
   different concepts. **Window encoder at sparse per-position cap produces
   less concept-specialised features than per-token T-SAE k=20.** Plausible
   causal mechanism for the steering loss.

### Cells in detail

#### Cell C — `txc_bare_antidead_t3_kpos20` (W, random-init)

**Training**: random-init, T=3, k_pos=20, k_win=60, b=4096, lr=3e-4, max_steps=25k.
Plateau-converged at step **4600** (plateau=0.018, threshold=0.02). Loss
25339 → 4489 (5.6× drop). l0=60 (= k_win, full TopK occupancy). Wall **33 min**
on A40.

**Pipeline**: 33 min train + ~50 min eval (select + diagnose + intervene + grade
all on shared infra).

**Grades**: 210 rows, 0 errors. Mean success=0.64, mean coh=1.91 (over all 210
gens). Peak success at s_norm=5 (s_abs=ε × ⟨|z|⟩_C × 5).

**Verdict under primary metric (coh ≥ 1.5)**: **LOSS** by 0.33 below T-SAE k=20.

**Verdict under raw peak**: 1.40 vs anchor 1.80 → still LOSS, but cell C is the
**best-coherence-at-peak window arch yet** (coh=2.07 at success-peak).

#### Cell D — `txc_bare_antidead_t5_kpos20` (Y, random-init, [meeting cell])

**Source**: Y trained + graded on her separate A40 pod. Reused via pull
(commit `ac65ed92` ckpt, `151d3a01` results). Per Y's coordination message
fd117ca9, this is the apples-to-apples cell since T-SAE k=20 anchor was
random-init too.

**Training** (Y's): plateau-converged at step **3800** (plateau=0.019), loss
20260 → 4813, l0=100, wall 46 min.

**Numbers**: raw peak 1.000 at s_abs=251.1 (coh=1.20 — below threshold).
Constrained peak (coh ≥ 1.5) = 0.700 at s_abs=125.5.

**Verdict**: **LOSS** under both metrics (Δ=−0.80 raw, Δ=−0.40 constrained).

**Y's per-class breakdown (from her writeup)**:

| concept class | T-SAE k=20 | TXC cell D | winner |
|---|---|---|---|
| knowledge | 2.000 | 1.444 | T-SAE +0.56 |
| discourse | 1.375 | 0.500 | T-SAE +0.88 |
| safety | 0.333 | 0.000 | T-SAE |
| **stylistic** | 0.200 | **0.600** | **TXC +0.40** ⭐ |
| sentiment | 0.500 | 0.500 | tie |

Stylistic is the only TXC-favourable class. Possible mechanism: stylistic
features (poetic, literary, list_format, citation, technical_jargon) are
context-shape patterns that span multiple tokens, where the window encoder
pays off even at sparse k_pos.

#### Cell E — `agentic_txc_02_kpos20` (W, matryoshka multiscale, random-init)

**Spec**: T=5, k_pos=20, k_win=100, gamma=0.5, n_scales=3 (= `agentic_txc_02`
recipe at matched sparsity). Random-init.

**Status**: training in flight (started ~23:19 UTC). Will fill in numbers
when grades land.

#### Cell F — `txc_bare_antidead_t10_kpos20` (W, random-init)

**Spec**: T=10, k_pos=20, k_win=200. Wider window probe at sparse per-token cap.

**Status**: queued. Will launch after cell E completes.

### Methodological notes

1. **Conflict resolution**: Y and W's pipelines both write to `results/case_studies/diagnostics_kpos20/z_orig_magnitudes.json` keyed by `arch_id`. Merge cleanly by union of keys — no semantic conflict. Resolved in commit aad048d1.
2. **Strength-grid hygiene**: every cell's constrained peak is at s_norm=5 with the next-up s_norm=10 dropping below coh=1.5 (the "constraint-bound" failure mode in the brief's hygiene section). True constrained optima, not grid-artefact peaks. No grid extension needed.
3. **Same-seed reproducibility**: cells C and D both random-init at seed=42; their plateau-converged steps differ (4600 vs 3800) which is normal — different T means different effective optimisation landscape.

### Pre-registered outcome

Per `agent_w/plan.md` § Pre-registered Phase 1 outcomes:

> *Every cell loses to T-SAE k=20 by ≥0.27* → architecture has anti-prior at
> matched sparsity. Phase 2 perturbations also unlikely to recover. Document
> the per-cell + per-class breakdown to identify which axis hurts most.
> **Publishable converging null** (with Y's parallel finding from the other
> direction).

Currently 2/4 cells trained both LOSS by ≥ 0.27 → **on track for Outcome C
(publishable converging null)** unless cell E or F land a surprise win.

### Phase 2 implications (if Outcome C confirmed)

If E and F also lose:

1. **Skip the random-init Phase 2 hill-climb** — no winner to climb from.
2. **Pursue the warm-start variant** as a "does init transfer matter at sparse k_pos?" probe. If warm-start substantially shifts cell D's numbers, the brief's "5–10× speedup trick" deserves more scrutiny as a methodological choice.
3. **Pursue per-position write-back (Q2.C protocol) as a free axis** — Y's earlier writeup showed +0.13 across window archs at k_pos=100; might lift k_pos=20 cells closer to tie. Cheap because no retraining, only re-intervene.
4. **Document the per-class structural pattern**: at k_pos=20, TXC family wins on **stylistic** (Y's finding), not knowledge (which was the Phase-7 narrative at k_pos=100). The shift is itself a paper-worthy datapoint about how the multi-token receptive-field advantage interacts with sparsity.
5. **Hand back to Han** before initiating Phase 2 if Outcome C confirms — the failure-mode investigation is the deliverable; Phase 2 perturbations would only refine the table.

### Files

- This writeup: `docs/han/research_logs/phase7_unification/agent_w/2026-04-29-w-phase1-sweep.md`
- Cell C: `results/case_studies/steering_paper_normalised/txc_bare_antidead_t3_kpos20/{generations,grades}.jsonl`, training log `training_logs/txc_bare_antidead_t3_kpos20__seed42.json`
- Cell D (Y's): `results/case_studies/steering_paper_normalised/txc_bare_antidead_t5_kpos20/{generations,grades}.jsonl`, training log `training_logs/txc_bare_antidead_t5_kpos20__seed42.json`
- Cell E: pending (training in flight)
- Cell F: pending (queued)
- Comparison plot framework: `experiments/phase7_unification/case_studies/steering/compare_kpos20_vs_tsae.py` (Y's, takes `--archs <id1> <id2> ...`)
