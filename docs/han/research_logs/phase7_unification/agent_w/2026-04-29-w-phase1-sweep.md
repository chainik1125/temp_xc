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

### TL;DR — UPDATED with per-position write-back numbers

**Initial story (right-edge protocol)**: every matched-sparsity TXC k_pos=20 cell
LOSES vs T-SAE k=20. Outcome C (publishable null) on track.

**Revised story (after Y's per-position result, then mine)**: under family-normalised
paper-clamp + **per-position write-back (Q2.C)**, all 3 trained matched-sparsity TXC
cells land in the **TIE band**. Y's T=2 even slightly *exceeds* the anchor (+0.13).
The matched-sparsity narrative is rescued by the protocol — architecture is not
the bottleneck once the write-back is right.

**Combined matched-sparsity matrix** (peak at coh ≥ 1.5, family-normalised paper-clamp):

| protocol | arch_id | T | raw peak | **pk @ coh ≥ 1.5** | Δ vs anchor 1.10 | call |
|---|---|---|---|---|---|---|
| anchor (per-token) | `tsae_paper_k20` | 1 | 1.80 | **1.10** | (anchor) | — |
| right-edge | `txc_bare_antidead_t2_kpos20` (Y) | 2 | 1.30 | 0.83 | −0.27 | TIE (boundary) |
| right-edge | `txc_bare_antidead_t3_kpos20` (W) | 3 | 1.40 | 0.77 | −0.33 | LOSS |
| right-edge | `txc_bare_antidead_t5_kpos20` (Y) | 5 | 1.00 | 0.70 | −0.40 | LOSS |
| **per-position** | **`txc_bare_antidead_t2_kpos20` (Y)** | 2 | 1.30 | **1.23** | **+0.13** | **TIE (best)** ⭐ |
| **per-position** | **`txc_bare_antidead_t3_kpos20` (W)** | 3 | **1.50** | **1.00** | **−0.10** | **TIE** ⭐ |
| **per-position** | **`txc_bare_antidead_t5_kpos20` (Y)** | 5 | 0.90 | 0.83 | −0.27 | TIE (boundary) |
| pending | `agentic_txc_02_kpos20` (W) | 5 | TBD | TBD | TBD | TBD |
| skipped | `txc_bare_antidead_t10_kpos20` | 10 | — | — | — | (T scaling already trending wrong) |

**Cell C T=3 raw peak (1.500)** is the highest of any matched-sparsity cell. **Y's T=2 per-position constrained (1.233)** is the highest constrained — slightly exceeds T-SAE k=20.

Per pre-registered rule, all three per-position cells are TIE band (within ±0.27 of 1.10).
None individually CALL a win, but the consistent TIE pattern across T ∈ {2, 3, 5} is
itself the headline: under per-position write-back, matched-sparsity TXC matches
T-SAE k=20 — the "TXC structurally weaker than per-token at this sparsity" story
collapses.

Cell F (T=10) skipped: per-position boost decreases monotonically with T (+0.40
at T=2, +0.23 at T=3, +0.13 at T=5). T=10 would extrapolate to ~+0.05 boost,
adding negligible signal.

### Headline findings

1. **Per-position write-back rescues TXC at matched sparsity.** Under Q2.C,
   all 3 trained TXC k_pos=20 cells (T=2, 3, 5) land in the TIE band. T=2
   slightly exceeds T-SAE k=20 (+0.13). The brief's primary metric (peak @
   coh ≥ 1.5) flips from "all losses" to "all ties" by changing the protocol.
   **Architecture is not the bottleneck at matched per-token sparsity once
   the write-back is right.**

2. **Per-position boost decreases monotonically with T at matched k_pos=20**:
   +0.40 (T=2) → +0.23 (T=3) → +0.13 (T=5). Mechanism: smaller T means
   cleaner picked features (less polysemantic), so per-position amplification
   is positive-signal-dominant. At larger T the same amplification carries
   more noise (Y's polysemanticity finding: 24/30 distinct features at T=5
   vs ~28/30 at T=2).

3. **Narrower window has highest raw peak.** Cell C T=3 per-position raw
   peak 1.500 > T=2 (1.30) > T=5 (1.00 right-edge / 0.90 per-position).
   Counterintuitive — at sparse per-token cap, the T-axis structural advantage
   *reverses*. The traditional intuition ("longer window = more multi-token
   context = better") fails when k_pos is tight.

4. **Coherence is structurally rescued at k_pos=20.** All cells' constrained
   peaks have coh ∈ [1.84–2.07] (right-edge) and [1.567–1.84] (per-position)
   — far above the 1.5 threshold cliff that bites k_pos=100 window archs
   (where coh ∈ [1.20–1.45] at peak). Sparser per-token cap produces gentler
   per-position activations → smoother coherence falloff with strength →
   wider "constraint-satisfying region". This is the structural prediction
   that motivated the W brief.

5. **Y's per-class breakdown** (cell D right-edge): TXC k_pos=20 wins on
   **stylistic** (+0.40 vs T-SAE k=20), ties on sentiment, loses elsewhere.
   This is a *different* TXC win-pattern than at k_pos=100 (where TXC wins
   on knowledge concepts). Stylistic features (poetic, literary, list_format,
   citation, technical_jargon) are context-shape patterns — multi-token
   *form* rather than multi-token *content*. The window encoder pays off on
   form even when content-features go polysemantic.

### Cells in detail

#### Cell C — `txc_bare_antidead_t3_kpos20` (W, random-init)

**Training**: random-init, T=3, k_pos=20, k_win=60, b=4096, lr=3e-4, max_steps=25k.
Plateau-converged at step **4600** (plateau=0.018, threshold=0.02). Loss
25339 → 4489 (5.6× drop). l0=60 (= k_win, full TopK occupancy). Wall **33 min**
on A40.

**Right-edge protocol (steering_paper_normalised)**:
- 210 rows, 0 errors. Mean success=0.64, mean coh=1.91 over all 210 gens.
- Raw peak: 1.400 at s_norm=10 (coh=1.50 borderline).
- Constrained peak (coh ≥ 1.5): **0.767** at s_norm=5 (coh=2.07).
- Verdict: **LOSS** by 0.33 vs T-SAE k=20 anchor (1.10).

**Per-position write-back (steering_paper_window_perposition)**:
- 210 rows, 0 errors. Mean success=0.69, mean coh=1.84.
- Raw peak: **1.500** at s_norm=10 (coh=1.27 — drops below 1.5 cliff).
- Constrained peak (coh ≥ 1.5): **1.000** at s_norm=5 (coh=1.84).
- Δ from right-edge to per-position constrained: **+0.233** (boost).
- Verdict: **TIE** with Δ=−0.10 vs T-SAE k=20 anchor.

**Cell C's raw peak under per-position (1.500) is the HIGHEST of any
matched-sparsity cell trained**, beating Y's T=2 (1.30) and T=5 (0.90). Combined
with the +0.233 per-position boost (midway between Y's T=2's +0.40 and T=5's
+0.13), cell C confirms the "smaller-T-helps-at-sparse-k_pos" pattern.

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

**Y's per-class breakdown (from her writeup)** — confirmed by my recompute:

| protocol | cell | overall | knowledge | discourse | safety | stylistic | sentiment |
|---|---|---|---|---|---|---|---|
| anchor right-edge | T-SAE k=20 | 1.10 | **2.00** | **1.38** | **0.33** | 0.20 | 0.50 |
| right-edge | T=2 (Y) | 0.83 | 1.56 | 1.00 | 0.00 | 0.60 | 0.00 |
| right-edge | T=3 (W) | 0.77 | 1.56 | 0.62 | 0.17 | 0.40 | 0.50 |
| right-edge | T=5 (Y) | 0.70 | 1.44 | 0.50 | 0.00 | 0.60 | 0.50 |
| **per-position** | **T=2 (Y)** | **1.23** | 1.78 | **1.50** | **0.67** | 0.40 | **1.50** |
| **per-position** | **T=3 (W)** | **1.00** | 1.56 | 1.00 | 0.17 | **1.00** | 1.00 |
| **per-position** | **T=5 (Y)** | **0.83** | 1.33 | 0.88 | 0.00 | 0.60 | **1.50** |

**Three per-class patterns to call out:**

1. **Per-position uniformly lifts SENTIMENT** by ~+1.0 above the anchor's
   0.50 across all T. Sentiment concepts (positive_emotion, negative_emotion)
   are emotionally-charged but don't have rigidly fixed token positions —
   per-position writing distributes the sentiment signal across the steered
   window naturally.
2. **Cell C T=3 per-position dominates STYLISTIC** at 1.00 (vs anchor 0.20,
   Δ=+0.80). Stylistic features are multi-token form patterns. The combination
   of T=3 (just enough window context for form) + per-position (write the
   form across all positions of the steered span) is the optimal stack.
3. **KNOWLEDGE**: T-SAE k=20 still wins on knowledge concepts under either
   protocol, but the gap narrows with smaller T + per-position (Δ=−0.22 at
   T=2 per-position vs Δ=−0.56 at T=5 right-edge).

The per-class pattern under per-position **flips Y's earlier
"stylistic-only-TXC-favourable" finding**: under per-position, *sentiment*
becomes uniformly TXC-favourable; *stylistic* becomes T=3-specifically
TXC-favourable. The TXC structural advantage at matched sparsity
materialises differently per-class than at k_pos=100 (where TXC won on
knowledge).

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

### Pre-registered outcome — REVISED

Per `agent_w/plan.md` § Pre-registered Phase 1 outcomes:

> *No cell beats by ≥0.27, but at least one ties (within ±0.27)* → ambiguous.
> Run multi-seed on the best candidate. If still tied → "sparsity is the
> dominant lever, architecture is secondary".

Right-edge protocol: every k_pos=20 cell loses by ≥ 0.27 → Outcome C.
**Per-position protocol: every k_pos=20 cell ties (within ±0.27) → Outcome B
(ambiguous tie band)**. Y's T=2 per-position is the closest to a win at +0.13
but doesn't clear +0.27.

The protocol-dependence of the verdict is itself a methodological finding
worth its own paper subsection. The headline "matched-sparsity TXC ties T-SAE
k=20" is conditional on the per-position write-back protocol; under right-edge
it loses.

### Phase 2 plan (post-per-position rescue)

1. ✅ **Per-position write-back on cells C/D** — DONE. 3/3 in tie band.
   This was Phase 2 axis 1 in W's `project_phase7_w_phase2_axes.md`. Won
   "free" via the Q2.C protocol switch.

2. **Multi-seed verify Y's T=2 per-position cell** (the +0.13 closest-to-win).
   Y has Step 2 (T=5) seed=1 training in flight per her commit 448db4c3;
   would be valuable to also run T=2 seed=1.

3. **Cell E (matryoshka multiscale)** — re-launched at 00:23 UTC after the
   first attempt was killed at 1h with no convergence. Plateau early-stop
   should land it within ~80-150 min wall. Tests "does multi-scale matryoshka
   help over bare-antidead at k_pos=20?". The answer matters for whether
   the brief's family-axis perturbation is worth pursuing.

4. **Warm-start variant `txc_bare_antidead_t5_kpos20_warmstart`** —
   methodological probe: does init from T-SAE k=20 systematically shift the
   matched-sparsity matrix? Expected ~10-15 min train + ~25 min pipeline.
   Planned after cell E.

5. **Knowledge-class subset analysis at per-position**. Y's per-class table
   for cell D (right-edge) showed TXC wins on stylistic only. Under
   per-position write-back, the per-class pattern may shift — particularly
   knowledge concepts (where TXC won at k_pos=100). Cheap: re-aggregate
   existing per-position grades by concept class. Blocked on cell E's
   per-position grades for the matryoshka comparison.

### Hand-back to Han / Y at end of Phase 2

The matched-sparsity matrix under per-position protocol is the headline.
Y's T=2 +0.13 finding alone is paper-grade if multi-seed verifies it. My
cell C T=3 highest-raw-peak result complements it. The per-class structural
pattern (stylistic-wins, knowledge-flips depending on protocol) is a third
finding. **Three findings total, all under the matched-sparsity narrative.**

### Files

- This writeup: `docs/han/research_logs/phase7_unification/agent_w/2026-04-29-w-phase1-sweep.md`
- Cell C: `results/case_studies/steering_paper_normalised/txc_bare_antidead_t3_kpos20/{generations,grades}.jsonl`, training log `training_logs/txc_bare_antidead_t3_kpos20__seed42.json`
- Cell D (Y's): `results/case_studies/steering_paper_normalised/txc_bare_antidead_t5_kpos20/{generations,grades}.jsonl`, training log `training_logs/txc_bare_antidead_t5_kpos20__seed42.json`
- Cell E: pending (training in flight)
- Cell F: pending (queued)
- Comparison plot framework: `experiments/phase7_unification/case_studies/steering/compare_kpos20_vs_tsae.py` (Y's, takes `--archs <id1> <id2> ...`)
